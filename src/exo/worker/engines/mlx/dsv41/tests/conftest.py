"""Fixtures for the DSv4.1 (EXL3) engine tests.

Nothing here touches a GPU, a checkpoint or the network: the DSv4.1 text
plumbing is pure parsing/wiring, so the whole suite runs anywhere `exo` and its
mlx-lm dependency import. The real strings under test (the DSML sentinel, the
thinking markers) are pinned to the CHECKPOINT's values -- see
``test_dsv41_sentinels.py`` for the byte-exact assertions and the provenance.

The checkpoint's own tokenizer cannot be used here (it lives on the Mac
studios, and its 6 MB vocab would make every test depend on it), so
``FakeTokenizer`` reproduces just the surface the engine and parsers touch:
``.encode``/``.decode``, a streaming ``detokenizer``, ``eos_token_ids``, the
thinking-marker attributes, and a ``_tokenizer`` with the DSML sentinel in its
vocab so real-vs-quoted DSML detection is exercised for real.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Any

import pytest

#: Pin MLX to the CPU for the WHOLE suite, before anything else imports exo or
#: mlx_lm. The DSv4.1 tests here use only tiny mock arrays (the argmax of a
#: one-hot row), so there is no reason to touch the GPU -- and on a cluster node
#: a Metal default device is a real hazard: the suite would take the GPU from
#: whatever is serving, which is exactly what an unlocked test run must never do.
#: This has to happen before exo's engine modules are imported because some of
#: them create arrays at import time, and a Metal stream created there keeps the
#: process on the GPU even if the default device changes later.
try:  # pragma: no cover - environment dependent
    import mlx.core as _mx

    _mx.set_default_device(_mx.cpu)
    MX_ON_CPU = True
except Exception:  # noqa: BLE001 - a build without MLX still runs the parsing tests
    MX_ON_CPU = False

from exo.shared.models.model_cards import ModelId
from exo.shared.types.worker.runner_response import GenerationResponse

#: Sentinel and markers as verified against the checkpoint (see
#: test_dsv41_sentinels.py): utf-8 for the sentinel is efbd9c44534d4cefbd9c and
#: both thinking markers are plain ASCII.
#:
#: NOTE the marker is spelled by CONCATENATION, not as one literal. The
#: toolchain that writes source files here eats a "<" immediately followed by a
#: letter (it treats it as a tag), which silently produced " think" instead of
#: the real marker in an earlier revision of this file. The byte-exact
#: assertions in test_dsv41_sentinels.py are what catch that class of damage.
DSML_SENTINEL = "\uff5cDSML\uff5c"
THINK_START = "<" + "think" + ">"
THINK_END = "<" + "/think" + ">"

#: Vocabulary ids from the checkpoint's tokenizer.
DSML_SENTINEL_ID = 128825
THINK_START_ID = 128821
THINK_END_ID = 128822

#: Token id -> text, for the fake streaming detokenizer. Ids are small so the
#: fake logits tables in the engine tests stay tiny. 0/1 are the "no text"
#: tokens (a token that detokenizes to nothing, and EOS).
DEFAULT_CHUNKS: dict[int, str] = {
    1: "",
    2: "hi",
    3: "there",
    4: " and",
    5: " bye",
}


class FakeHFTokenizer:
    """Stands in for the HF tokenizer behind ``TokenizerWrapper``."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class FakeDetokenizer:
    """One chunk per token, from a caller-supplied id -> text table."""

    def __init__(self, text_of: Mapping[int, str]) -> None:
        self.last_segment = ""
        self.tokens: list[int] = []
        self._text_of = text_of

    def add_token(self, token: int) -> None:
        self.tokens.append(token)
        self.last_segment = self._text_of.get(token, f"<{token}>")


class FakeTokenizer:
    """The subset of ``TokenizerWrapper`` the engine + parsers use."""

    def __init__(
        self,
        *,
        vocab: dict[str, int] | None = None,
        text_of: Mapping[int, str] | None = None,
        has_thinking: bool = True,
        think_start: str | None = THINK_START,
        think_end: str | None = THINK_END,
        eos_token_ids: tuple[int, ...] = (1,),
    ) -> None:
        self._tokenizer = FakeHFTokenizer(
            vocab
            if vocab is not None
            else {
                DSML_SENTINEL: DSML_SENTINEL_ID,
                THINK_START: THINK_START_ID,
                THINK_END: THINK_END_ID,
            }
        )
        self._detokenizer = FakeDetokenizer(text_of or DEFAULT_CHUNKS)
        # ``has_thinking`` is derived from the marker, exactly as the real
        # TokenizerWrapper's property is -- so installing markers in a test
        # flips it, like production.
        self._think_start = think_start if has_thinking else None
        self._think_end = think_end if has_thinking else None
        # What TokenizerWrapper.__init__ records from _infer_thinking: the
        # single-token ids for each marker (present iff the markers resolved).
        self._think_start_tokens = (THINK_START_ID,) if has_thinking else None
        self._think_end_tokens = (THINK_END_ID,) if has_thinking else None
        self._eos_token_ids = set(eos_token_ids)

    # --- wrapper surface -------------------------------------------------
    @property
    def detokenizer(self) -> FakeDetokenizer:
        return self._detokenizer

    @property
    def has_thinking(self) -> bool:
        return self._think_start is not None

    @property
    def think_start(self) -> str | None:
        return self._think_start

    @property
    def think_end(self) -> str | None:
        return self._think_end

    @property
    def think_start_tokens(self) -> tuple[int, ...] | None:
        # What ``_infer_thinking`` records for this checkpoint: one token each.
        # Kept in the instance dict (not derived) so the install test can see
        # the attributes the patch writes, and so the guard test can delete them.
        return self.__dict__.get("_think_start_tokens")

    @property
    def think_end_tokens(self) -> tuple[int, ...] | None:
        return self.__dict__.get("_think_end_tokens")

    @property
    def eos_token_ids(self) -> set[int]:
        return set(self._eos_token_ids)

    # --- tokenizer surface ----------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        # The fake model's logits do not depend on the prompt's ids, only on
        # the fact that the prompt is non-empty; three ids is enough to make
        # the engine's prompt-length checks real.
        return [10, 11, 12]

    def apply_chat_template(
        self,
        messages: Any,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        tools: Any = None,
        **kwargs: Any,
    ) -> str:
        # The engine's prompt for this model id is actually produced by exo's
        # vendored DeepSeek-V4 encoder (`_needs_v4_encoding`), not by this
        # method; it exists so a test that swaps the model id cannot explode.
        del messages, tokenize, add_generation_prompt, kwargs
        header = "<|Assistant|>"
        if tools:
            # tool_conditional: reasoning is kept when tools are present.
            header += THINK_START
        return f"<|User|>hello{header}"

    def decode(self, tokens: Any) -> str:
        return "".join(DEFAULT_CHUNKS.get(int(t), "") for t in tokens)


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def responses(
    texts: list[str],
    *,
    tokens: list[int] | None = None,
    finish_on_last: bool = True,
    separate_terminal: bool = True,
    interleave_none: bool = True,
) -> Generator[GenerationResponse | None]:
    """A token stream shaped like production (see test_dsml_e2e's fixture).

    Defaults reproduce what the real stream does: the terminal finish_reason
    arrives on a SEPARATE empty-text chunk after the last content chunk, and a
    bare ``None`` arrives between every pair of real responses (the engine's
    queue yields ``None`` when momentarily empty).
    """
    ids = tokens if tokens is not None else list(range(len(texts)))
    for i, text in enumerate(texts):
        is_last = i == len(texts) - 1
        attach = is_last and finish_on_last and not separate_terminal
        yield GenerationResponse(
            text=text,
            token=ids[i],
            finish_reason="stop" if attach else None,
            usage=None,
        )
        if interleave_none:
            yield None
    if separate_terminal and finish_on_last:
        yield GenerationResponse(text="", token=1, finish_reason="stop", usage=None)


def sentinel_tokenized(chunks: list[str]) -> tuple[list[str], list[int]]:
    """Zip text chunks with the token id the sentinel arrives on.

    A REAL tool call emits the sentinel as its dedicated vocab token, which is
    what ``_parse_dsml_stream``'s real-vs-quoted gate looks at. Any chunk that
    contains the sentinel is therefore reported with ``DSML_SENTINEL_ID`` (the
    id also appears on every chunk the sentinel's characters span); every other
    chunk gets a harmless filler id.
    """
    ids: list[int] = []
    for i, chunk in enumerate(chunks):
        ids.append(DSML_SENTINEL_ID if DSML_SENTINEL in chunk else 1000 + i)
    return chunks, ids


def model_id() -> ModelId:
    return ModelId("dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
