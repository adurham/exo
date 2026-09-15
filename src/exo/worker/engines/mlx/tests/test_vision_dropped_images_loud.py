"""Regression tests: a request WITH images must never silently degrade to
text-only with no discoverable signal.

BACKGROUND — this is the SAME failure shape as two other confirmed exo
incidents (DSpark's checkpoint-key-mismatch silently falling back to weak
MTP-1 drafting; the lm_head mxfp8-fallback fix missing DSpark's own direct
lm_head call site), and it is not hypothetical here either: a REAL production
incident of this exact mechanism already happened (2026-09-09, commit
f76a4da3, "fix(api): emit a real image source in multimodal
chat_template_messages" — see docs/DSV4_VISION_PORT_PHASE5_PROCEDURE.md and
src/exo/api/tests/test_chat_completions_image_blocks.py). EVERY image
request silently degraded to a text-only HTTP 200 completion ("There is no
image attached to your prompt") because the vendored encoder's ValueError
was swallowed by:

    except Exception:
        logger.warning("Vision processing failed, falling back to text-only")

That commit fixed the TRIGGER (the API adapter now emits a valid image
block), but did NOT touch the SWALLOW mechanism itself — the identical
except-block still exists today in both generate.py and batch_generate.py,
ready to silently reproduce the same bug shape for any OTHER future
vision-processing defect. There is also a second, previously entirely
untested sub-case: if `VisionProcessor.load()` itself fails at MODEL LOAD
time, `self.vision_processor` becomes `None` for the runner's whole
lifetime, and every subsequent image-bearing request then takes the
`if self.vision_processor is not None:` branch's ELSE path — dropping the
attached image(s) with ZERO log output at all, forever.

These tests pin the FIXED behavior: any code path that drops an attached
image (processing exception, missing chat_template_messages, or a runner
with no live vision processor) must emit a loud, distinctly-tagged,
ERROR-level log line naming the drop reason and the image count — not a
routine warning, and never total silence.
"""

from __future__ import annotations

import logging

import pytest
from loguru import logger as loguru_logger

from exo.shared.models.model_cards import ModelId
from exo.shared.types.text_generation import Base64Image, TextGenerationTaskParams
from exo.worker.engines.mlx.vision import prepare_vision

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def caplog_loguru(caplog: pytest.LogCaptureFixture):
    """Bridge loguru records into stdlib caplog for this test module only.

    This test tree has no shared conftest.py wiring loguru -> caplog (unlike
    src/exo/shared/tests/conftest.py), so each test that needs to assert on
    log output installs its own sink rather than relying on one.
    """
    handler_id = loguru_logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
    )
    caplog.set_level(logging.WARNING)
    yield caplog
    loguru_logger.remove(handler_id)


def _task_params_with_images(n_images: int = 1) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("test/vision-model"),
        input=[],
        images=[Base64Image(_TINY_PNG_B64) for _ in range(n_images)],
        chat_template_messages=[{"role": "user", "content": "describe this"}],
    )


def test_vision_processing_exception_logs_at_error_not_warning(caplog_loguru) -> None:
    """A vision-processing exception must be ERROR-level and clearly tagged.

    This is the exact 2026-09-09 incident's swallow mechanism. Regardless of
    WHAT breaks inside vision_processor.process (any future defect, not just
    the specific bad-image-block trigger already fixed), the drop itself
    must be loud.
    """

    class _ExplodingVisionProcessor:
        def process(self, **kwargs: object) -> None:
            raise ValueError("synthetic vision-processing failure")

    task_params = _task_params_with_images(n_images=2)

    result = prepare_vision(
        images=task_params.images,
        chat_template_messages=task_params.chat_template_messages,
        vision_processor=_ExplodingVisionProcessor(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        model=object(),  # type: ignore[arg-type]
        model_id=task_params.model,
        task_params=task_params,
    )

    assert result is None

    error_records = [r for r in caplog_loguru.records if r.levelname == "ERROR"]
    assert error_records, (
        "vision processing failure must log at ERROR, not just WARNING — "
        f"got levels: {[r.levelname for r in caplog_loguru.records]}"
    )
    combined = "\n".join(r.message for r in error_records)
    assert "VISION-DROPPED" in combined, "drop must be tagged for grep-ability"
    assert "2" in combined, "image count must be in the log for diagnosis"
    assert "synthetic vision-processing failure" in combined, (
        "the real exception detail must be visible, not swallowed"
    )


def test_no_vision_processor_but_images_present_logs_loudly(caplog_loguru) -> None:
    """`vision_processor=None` + images present must NOT be silent.

    Reproduces the load-time-failure sub-case: VisionProcessor.load() raised
    at model load, self.vision_processor is None for the runner's lifetime,
    and this is the first place that fact and an incoming image collide.
    Today (pre-fix) this path logs NOTHING at all.
    """
    task_params = _task_params_with_images(n_images=1)

    result = prepare_vision(
        images=task_params.images,
        chat_template_messages=task_params.chat_template_messages,
        vision_processor=None,
        tokenizer=object(),  # type: ignore[arg-type]
        model=object(),  # type: ignore[arg-type]
        model_id=task_params.model,
        task_params=task_params,
    )

    assert result is None
    error_records = [r for r in caplog_loguru.records if r.levelname == "ERROR"]
    assert error_records, (
        "images present with no live vision processor must log loudly — "
        f"got levels: {[r.levelname for r in caplog_loguru.records]} "
        f"(messages: {[r.message for r in caplog_loguru.records]})"
    )
    combined = "\n".join(r.message for r in error_records)
    assert "VISION-DROPPED" in combined
    assert "no vision processor" in combined.lower()


def test_missing_chat_template_messages_logs_at_error(caplog_loguru) -> None:
    """images present + chat_template_messages missing must also be loud.

    Same drop shape as the other two cases (images attached, model responds
    as if none were sent) — was previously only a WARNING with no distinct
    tag, easy to miss in a log stream alongside routine warnings.
    """
    task_params = TextGenerationTaskParams(
        model=ModelId("test/vision-model"),
        input=[],
        images=[Base64Image(_TINY_PNG_B64)],
        chat_template_messages=None,
    )

    class _NeverCalledVisionProcessor:
        def process(self, **kwargs: object) -> None:
            raise AssertionError("must not be called when chat_template_messages is None")

    result = prepare_vision(
        images=task_params.images,
        chat_template_messages=task_params.chat_template_messages,
        vision_processor=_NeverCalledVisionProcessor(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        model=object(),  # type: ignore[arg-type]
        model_id=task_params.model,
        task_params=task_params,
    )

    assert result is None
    error_records = [r for r in caplog_loguru.records if r.levelname == "ERROR"]
    assert error_records, (
        f"got levels: {[r.levelname for r in caplog_loguru.records]}"
    )
    combined = "\n".join(r.message for r in error_records)
    assert "VISION-DROPPED" in combined


def test_no_images_is_silent_and_returns_none(caplog_loguru) -> None:
    """The ordinary text-only case (no images at all) must stay silent.

    Loud-on-drop must not become loud-on-every-request — only requests that
    actually attached an image and then lost it should log at all.
    """
    task_params = TextGenerationTaskParams(
        model=ModelId("test/vision-model"),
        input=[],
        images=[],
        chat_template_messages=[{"role": "user", "content": "hello"}],
    )

    result = prepare_vision(
        images=task_params.images,
        chat_template_messages=task_params.chat_template_messages,
        vision_processor=None,
        tokenizer=object(),  # type: ignore[arg-type]
        model=object(),  # type: ignore[arg-type]
        model_id=task_params.model,
        task_params=task_params,
    )

    assert result is None
    assert not caplog_loguru.records, (
        "a plain text-only request (no images attached) must not log anything — "
        f"got: {[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )


def test_successful_vision_processing_is_silent_and_returns_result(
    caplog_loguru,
) -> None:
    """The healthy path (images present, processing succeeds) must stay quiet
    and return the real VisionResult unchanged — this fix must not add noise
    or change behavior on success."""
    from exo.worker.engines.mlx.vision import MediaRegion, VisionResult

    sentinel_result = VisionResult(
        prompt="<image>describe this",
        prompt_tokens=None,  # type: ignore[arg-type]
        embeddings=None,  # type: ignore[arg-type]
        media_regions=[MediaRegion(content_hash="abc", start_pos=0, end_pos=1)],
        image_token_id=999,
    )

    class _WorkingVisionProcessor:
        def process(self, **kwargs: object):
            return sentinel_result

    task_params = _task_params_with_images(n_images=1)

    result = prepare_vision(
        images=task_params.images,
        chat_template_messages=task_params.chat_template_messages,
        vision_processor=_WorkingVisionProcessor(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        model=object(),  # type: ignore[arg-type]
        model_id=task_params.model,
        task_params=task_params,
    )

    assert result is sentinel_result
    assert not caplog_loguru.records, (
        f"success must not log — got: {[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )
