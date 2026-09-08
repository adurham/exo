"""Equality tests for the torch-free DeepSeek-V4 image processor port.

Every assertion in this file compares the port against DeepSeek's OWN PyTorch
reference implementation, executed in-process. ``torch`` is imported HERE ONLY --
the shipped module under test is torch-free by design.

The reference tree and the two shipped example images are not part of this
repository, so the whole module skips when they are absent. Set
``EXO_DSV4_VISION_REFERENCE_DIR`` to point at a checkout of
``deepseek-ai/DeepSeek-V4-Flash-Vision-Exp``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Protocol, cast, final

import numpy as np
import pytest
from numpy.typing import NDArray

from exo.worker.engines.mlx.vendor.deepseek_v4_image_processor import (
    COMPRESS_PAD_TO,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_START,
    ImageInput,
    VisionEncoderConfig,
    as_integer_list,
    build_image_block,
    expand_image_placeholders,
    grid_tokens,
    load_image,
    round_to_bfloat16_precision,
    safe_resize,
    solve_resize_ratio,
)

_DEFAULT_REFERENCE_DIR: Final = Path.home() / "dsv4_vision_ref"
_REFERENCE_DIR: Final = Path(
    os.environ.get("EXO_DSV4_VISION_REFERENCE_DIR", str(_DEFAULT_REFERENCE_DIR))
)
_INFERENCE_DIR: Final = _REFERENCE_DIR / "inference"
_EXAMPLES_DIR: Final = _INFERENCE_DIR / "examples"
_IMAGES_DIR: Final = _EXAMPLES_DIR / "images"

_REQUIRED_PATHS: Final = (
    _INFERENCE_DIR / "image_processor.py",
    _INFERENCE_DIR / "config.json",
    _REFERENCE_DIR / "encoding" / "encoding_dsv4.py",
    _EXAMPLES_DIR / "example_vl.txt",
    _EXAMPLES_DIR / "example_vl_harmony.json",
    _IMAGES_DIR / "carrots.jpeg",
    _IMAGES_DIR / "corn.jpeg",
    _REFERENCE_DIR / "tokenizer.json",
    _REFERENCE_DIR / "tokenizer_config.json",
)

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in _REQUIRED_PATHS),
    reason=(
        f"DeepSeek-V4-Flash-Vision-Exp reference tree not found under {_REFERENCE_DIR}. "
        "Set EXO_DSV4_VISION_REFERENCE_DIR to a checkout containing inference/, encoding/ and examples/."
    ),
)


def _load_reference_module(name: str, path: Path) -> ModuleType:
    """Import a reference file by path, without polluting ``sys.path`` permanently."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@final
@dataclass(frozen=True, slots=True)
class _ReferenceArguments:
    """Stand-in for the reference's ``ModelArgs``; it reads flat attributes."""

    vision_patch_size: int
    vision_downsample_ratio: int
    vision_max_n_token: int
    vision_min_pixels: int
    vision_max_wh_ratio: int | None
    vocab_size: int


@final
@dataclass(frozen=True, slots=True)
class _Reference:
    image_processor: ModuleType
    encoding: ModuleType
    arguments: _ReferenceArguments
    config: VisionEncoderConfig


@pytest.fixture(scope="module")
def reference() -> Iterator[_Reference]:
    """Import DeepSeek's reference modules and build matched config objects."""
    original_sys_path = list(sys.path)
    original_cwd = Path.cwd()
    # The reference resolves example image paths relative to inference/, and
    # image_processor.prepare_vl_inputs does `from encoding_dsv4 import ...`.
    sys.path.insert(0, str(_REFERENCE_DIR / "encoding"))
    sys.path.insert(0, str(_INFERENCE_DIR))
    os.chdir(_INFERENCE_DIR)
    try:
        image_processor = _load_reference_module(
            "dsv4_reference_image_processor", _INFERENCE_DIR / "image_processor.py"
        )
        encoding = _load_reference_module(
            "encoding_dsv4", _REFERENCE_DIR / "encoding" / "encoding_dsv4.py"
        )
        with open(_INFERENCE_DIR / "config.json") as config_file:
            raw_config = cast(dict[str, Any], json.load(config_file))
        config_mapping: Mapping[str, object] = raw_config
        arguments = _ReferenceArguments(
            vision_patch_size=cast(int, raw_config["vision_patch_size"]),
            vision_downsample_ratio=cast(int, raw_config["vision_downsample_ratio"]),
            vision_max_n_token=cast(int, raw_config["vision_max_n_token"]),
            vision_min_pixels=cast(int, raw_config["vision_min_pixels"]),
            vision_max_wh_ratio=cast(int, raw_config["vision_max_wh_ratio"]),
            vocab_size=cast(int, raw_config["vocab_size"]),
        )
        yield _Reference(
            image_processor=image_processor,
            encoding=encoding,
            arguments=arguments,
            config=VisionEncoderConfig.from_config_mapping(config_mapping),
        )
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path
        for name in ("dsv4_reference_image_processor", "encoding_dsv4"):
            sys.modules.pop(name, None)


def _reference_grid_tokens(
    reference: _Reference, best_height: int, best_width: int
) -> tuple[int, int, int]:
    result = cast(
        "tuple[int, int, int]",
        reference.image_processor.grid_tokens(  # pyright: ignore[reportAny]
            best_height,
            best_width,
            reference.arguments.vision_patch_size,
            reference.arguments.vision_downsample_ratio,
        ),
    )
    return result


def _reference_solve_resize_ratio(
    reference: _Reference, height: int, width: int, max_n_token: int
) -> tuple[int, int, int, int, int]:
    result = cast(
        "tuple[int, int, int, int, int]",
        reference.image_processor.solve_resize_ratio(  # pyright: ignore[reportAny]
            height,
            width,
            reference.arguments.vision_patch_size,
            reference.arguments.vision_downsample_ratio,
            max_n_token,
        ),
    )
    return result


def _reference_safe_resize(
    reference: _Reference, height: int, width: int, best_height: int, best_width: int
) -> tuple[int, int, int, int]:
    result = cast(
        "tuple[int, int, int, int]",
        reference.image_processor.safe_resize(  # pyright: ignore[reportAny]
            height,
            width,
            best_height,
            best_width,
            reference.arguments.vision_patch_size,
            reference.arguments.vision_downsample_ratio,
            reference.arguments.vision_max_n_token,
        ),
    )
    return result


class _TorchTensorLike(Protocol):
    """The subset of ``torch.Tensor`` this file uses.

    torch's own stubs type ``Tensor.tolist()`` as ``() -> list[Unknown]`` and
    ``torch.from_numpy`` as ``(ndarray: Unknown) -> Tensor``, which propagates
    partially-unknown types through every call site under this repository's
    strict settings. Declaring the used surface once here keeps the boundary
    explicit instead of scattering ignore comments.
    """

    def tolist(self) -> list[float]: ...
    def to(self, dtype: object, /) -> "_TorchTensorLike": ...
    def numpy(self) -> NDArray[Any]: ...


def _tensor_to_integer_list(tensor: _TorchTensorLike) -> list[int]:
    """Narrow a torch tensor of integers to ``list[int]``, once."""
    return [int(value) for value in tensor.tolist()]


def _tensor_to_float32_array(tensor: _TorchTensorLike) -> NDArray[np.float32]:
    """Narrow a torch tensor to a float32 numpy array, once."""
    import torch  # noqa: PLC0415  -- torch is a TEST-ONLY dependency

    return tensor.to(torch.float32).numpy().astype(np.float32, copy=False)


def _float32_array_to_bfloat16_array(
    values: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Round via torch's real bfloat16 cast and return float32 numpy, once."""
    import torch  # noqa: PLC0415  -- torch is a TEST-ONLY dependency

    # ``torch.from_numpy`` is stubbed as ``(ndarray: Unknown) -> Tensor``; go via
    # ``object`` because ``Tensor`` and the protocol do not structurally overlap
    # for pyright while ``tolist`` stays ``list[Unknown]``.
    from_numpy = cast("Callable[[NDArray[np.float32]], object]", torch.from_numpy)
    tensor = cast("_TorchTensorLike", from_numpy(values))
    return _tensor_to_float32_array(tensor.to(torch.bfloat16))


def _reference_parse_tagged_text(reference: _Reference, text: str) -> object:
    """``encoding_dsv4.parse_tagged_text`` is untyped; narrow it once, here."""
    parse_tagged_text = cast(
        "Callable[[str], object]", reference.encoding.parse_tagged_text
    )
    return parse_tagged_text(text)


def _reference_load_cases(reference: _Reference, path: Path) -> list[dict[str, Any]]:
    """``encoding_dsv4.load_cases`` is untyped; narrow it once, here."""
    load_cases = cast(
        "Callable[[str], list[dict[str, Any]]]", reference.encoding.load_cases
    )
    return load_cases(str(path))


def _reference_encode_case(
    reference: _Reference, case: dict[str, Any], thinking_mode: str
) -> tuple[str, list[dict[str, Any]]]:
    """``encoding_dsv4.encode_case`` is untyped; narrow it once, here."""
    encode_case = cast(
        "Callable[[dict[str, Any], str], tuple[str, list[dict[str, Any]]]]",
        reference.encoding.encode_case,
    )
    return encode_case(case, thinking_mode)


def _reference_build_image_block(
    n_llm_h: int, n_llm_w: int, start_pos: int
) -> tuple[list[int], list[int]]:
    """Run the reference ``build_image_block`` and return plain integer lists."""
    reference_module = sys.modules["dsv4_reference_image_processor"]
    types, perm = cast(
        "tuple[_TorchTensorLike, _TorchTensorLike]",
        reference_module.build_image_block(n_llm_h, n_llm_w, start_pos),  # pyright: ignore[reportAny]
    )
    return _tensor_to_integer_list(types), _tensor_to_integer_list(perm)


def _reference_load_image(
    reference: _Reference, record: Mapping[str, object]
) -> tuple[NDArray[np.float32], int, int, int, int]:
    """Run the reference ``load_image`` and return its patches as float32 numpy."""
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = cast(
        "tuple[_TorchTensorLike, int, int, int, int]",
        reference.image_processor.load_image(dict(record), reference.arguments),  # pyright: ignore[reportAny]
    )
    return _tensor_to_float32_array(patches), n_vit_h, n_vit_w, n_llm_h, n_llm_w


def _sha256_of_token_ids(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(token_id) for token_id in token_ids).encode()
    ).hexdigest()


def _encode_example(
    reference: _Reference, example: str
) -> tuple[list[int], list[ImageInput], list[int]]:
    """Encode one shipped example under BOTH the port and the reference.

    Returns ``(port_token_ids, port_image_inputs, reference_token_ids)``.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415  -- heavy, test-only

    tokenizer = cast(Any, AutoTokenizer).from_pretrained(str(_REFERENCE_DIR))  # pyright: ignore[reportAny]

    if example == "txt":
        with open(_EXAMPLES_DIR / "example_vl.txt") as example_file:
            raw_prompts = example_file.read().rstrip("\n").split("\n\n")
        cases = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": _reference_parse_tagged_text(reference, prompt),
                    }
                ]
            }
            for prompt in raw_prompts
        ]
    else:
        cases = _reference_load_cases(
            reference, _EXAMPLES_DIR / "example_vl_harmony.json"
        )
    assert len(cases) == 1, (
        f"expected a single case in the {example} example, got {len(cases)}"
    )

    prompt, image_records = _reference_encode_case(reference, cases[0], "chat")

    reference_token_ids = cast(
        "list[int]",
        reference.image_processor.prepare_vl_inputs(  # pyright: ignore[reportAny]
            prompt, image_records, tokenizer, reference.arguments
        )[0],
    )

    placeholder = cast(str, reference.encoding.IMAGE_PLACEHOLDER)
    placeholder_token_id = cast(int, tokenizer.convert_tokens_to_ids(placeholder))  # pyright: ignore[reportAny]
    prompt_token_ids = cast("list[int]", tokenizer.encode(prompt))  # pyright: ignore[reportAny]
    port_token_ids, port_image_inputs = expand_image_placeholders(
        prompt_token_ids,
        cast("list[Mapping[str, object]]", image_records),
        placeholder_token_id,
        reference.config,
    )
    return port_token_ids, port_image_inputs, reference_token_ids


def _assert_token_ids_equal(
    actual: Sequence[int], expected: Sequence[int], label: str
) -> None:
    """Fail with the FIRST divergence index and both values, never a bare mismatch."""
    for index, (actual_id, expected_id) in enumerate(
        zip(actual, expected, strict=False)
    ):
        if actual_id != expected_id:
            raise AssertionError(
                f"{label}: first divergence at index {index}: port={actual_id} reference={expected_id} "
                f"(lengths port={len(actual)} reference={len(expected)})"
            )
    assert len(actual) == len(expected), (
        f"{label}: common prefix matches but lengths differ: port={len(actual)} reference={len(expected)}"
    )


# ---------------------------------------------------------------------------
# Golden end-to-end: exact token-ID equality on the two shipped examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example", ["txt", "json"])
def test_shipped_example_token_ids_match_reference(
    reference: _Reference, example: str
) -> None:
    """The port's token IDs equal DeepSeek's reference token IDs, exactly."""
    port_token_ids, port_image_inputs, reference_token_ids = _encode_example(
        reference, example
    )
    _assert_token_ids_equal(
        port_token_ids, reference_token_ids, f"example_vl[{example}]"
    )
    assert len(port_image_inputs) == 2, "both shipped examples reference two images"
    print(
        f"\n[{example}] length={len(port_token_ids)} sha256={_sha256_of_token_ids(port_token_ids)} EXACT MATCH"
    )


def test_txt_and_json_examples_produce_identical_token_ids(
    reference: _Reference,
) -> None:
    """The README's claim: the TXT and JSON examples encode identically."""
    txt_token_ids, _, _ = _encode_example(reference, "txt")
    json_token_ids, _, _ = _encode_example(reference, "json")
    _assert_token_ids_equal(txt_token_ids, json_token_ids, "txt-vs-json")
    print(
        f"\ntxt length={len(txt_token_ids)} sha256={_sha256_of_token_ids(txt_token_ids)}"
        f"\njson length={len(json_token_ids)} sha256={_sha256_of_token_ids(json_token_ids)}"
    )


class _ReferenceImageInputLike(Protocol):
    """The reference's ``ImageInput`` dataclass, as this file reads it."""

    start: int
    n_vit_h: int
    n_vit_w: int
    types: _TorchTensorLike
    perm: _TorchTensorLike
    patches: _TorchTensorLike


def test_shipped_example_image_inputs_match_reference(reference: _Reference) -> None:
    """``ImageInput`` records match the reference field-for-field, patches included."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = cast(Any, AutoTokenizer).from_pretrained(str(_REFERENCE_DIR))  # pyright: ignore[reportAny]
    with open(_EXAMPLES_DIR / "example_vl.txt") as example_file:
        raw_prompt = example_file.read().rstrip("\n")
    case = {
        "messages": [
            {
                "role": "user",
                "content": _reference_parse_tagged_text(reference, raw_prompt),
            }
        ]
    }
    prompt, image_records = _reference_encode_case(reference, case, "chat")
    reference_image_inputs = cast(
        "list[_ReferenceImageInputLike]",
        reference.image_processor.prepare_vl_inputs(  # pyright: ignore[reportAny]
            prompt, image_records, tokenizer, reference.arguments
        )[1],
    )

    _, port_image_inputs, _ = _encode_example(reference, "txt")
    assert len(port_image_inputs) == len(reference_image_inputs)

    for index, (port_image, reference_image) in enumerate(
        zip(port_image_inputs, reference_image_inputs, strict=True)
    ):
        assert port_image.start == reference_image.start, f"image[{index}].start"
        assert port_image.n_vit_h == reference_image.n_vit_h, f"image[{index}].n_vit_h"
        assert port_image.n_vit_w == reference_image.n_vit_w, f"image[{index}].n_vit_w"
        reference_types = _tensor_to_integer_list(reference_image.types)
        reference_perm = _tensor_to_integer_list(reference_image.perm)
        assert as_integer_list(port_image.types) == reference_types, (
            f"image[{index}].types"
        )
        assert as_integer_list(port_image.perm) == reference_perm, (
            f"image[{index}].perm"
        )
        reference_patches = _tensor_to_float32_array(reference_image.patches)
        assert port_image.patches.shape == reference_patches.shape, (
            f"image[{index}].patches shape"
        )
        assert np.array_equal(port_image.patches, reference_patches), (
            f"image[{index}].patches differ from the reference bit-for-bit"
        )
        print(
            f"\nimage[{index}] start={port_image.start} n_vit_h={port_image.n_vit_h} "
            f"n_vit_w={port_image.n_vit_w} patches={port_image.patches.shape} "
            f"types={len(port_image.types)} perm={len(port_image.perm)} EXACT MATCH"
        )


@pytest.mark.parametrize("image_name", ["carrots.jpeg", "corn.jpeg"])
def test_shipped_image_derived_dimensions_match_reference(
    reference: _Reference, image_name: str
) -> None:
    """Per-image derived geometry matches, and the bf16 patch values match bitwise."""
    record: Mapping[str, object] = {
        "type": "image",
        "url": str(_IMAGES_DIR / image_name),
    }
    port_patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(
        record, reference.config
    )
    reference_patches, ref_n_vit_h, ref_n_vit_w, ref_n_llm_h, ref_n_llm_w = (
        _reference_load_image(reference, record)
    )

    assert (n_llm_h, n_llm_w, n_vit_h, n_vit_w) == (
        ref_n_llm_h,
        ref_n_llm_w,
        ref_n_vit_h,
        ref_n_vit_w,
    )
    assert port_patches.shape == reference_patches.shape
    assert np.array_equal(port_patches, reference_patches), (
        "patch values differ from the reference bit-for-bit"
    )

    _, _, num_tokens = grid_tokens(
        n_vit_h * reference.config.patch_size,
        n_vit_w * reference.config.patch_size,
        reference.config.patch_size,
        reference.config.downsample_ratio,
    )
    _, _, reference_num_tokens = _reference_grid_tokens(
        reference,
        n_vit_h * reference.config.patch_size,
        n_vit_w * reference.config.patch_size,
    )
    assert num_tokens == reference_num_tokens
    print(
        f"\n{image_name}: n_llm_h={n_llm_h} n_llm_w={n_llm_w} n_vit_h={n_vit_h} "
        f"n_vit_w={n_vit_w} num_tokens={num_tokens} patches={port_patches.shape}"
    )


# ---------------------------------------------------------------------------
# Arithmetic sweeps -- a single end-to-end match can pass by accident
# ---------------------------------------------------------------------------

_EXTREME_DIMENSIONS: Final = (
    1,
    2,
    3,
    7,
    13,
    14,
    15,
    28,
    31,
    64,
    127,
    224,
    401,
    768,
    1024,
    2048,
    4096,
    8192,
)


def test_grid_tokens_matches_reference_over_dimension_sweep(
    reference: _Reference,
) -> None:
    """Sweep grid_tokens over extreme aspect ratios; every case must match."""
    swept = 0
    odd_height_cases = 0
    nonzero_pad_last_cases = 0
    for best_height in _EXTREME_DIMENSIONS:
        for best_width in _EXTREME_DIMENSIONS:
            actual = grid_tokens(
                best_height,
                best_width,
                reference.config.patch_size,
                reference.config.downsample_ratio,
            )
            expected = _reference_grid_tokens(reference, best_height, best_width)
            assert actual == expected, (
                f"grid_tokens({best_height}, {best_width}) -> {actual} != {expected}"
            )
            n_llm_h, n_llm_w, _ = actual
            if n_llm_h % 2 == 1:
                odd_height_cases += 1
            rows = n_llm_h + n_llm_h % 2
            if ((((rows // 2) * (n_llm_w + 1)) % 2) * 2) != 0:
                nonzero_pad_last_cases += 1
            swept += 1
    assert swept == len(_EXTREME_DIMENSIONS) ** 2
    assert odd_height_cases > 0, (
        "sweep never produced an odd n_llm_h -- it does not exercise the odd branch"
    )
    assert nonzero_pad_last_cases > 0, "sweep never produced a nonzero pad_last term"
    print(
        f"\ngrid_tokens sweep: {swept} cases, {odd_height_cases} with odd n_llm_h, "
        f"{nonzero_pad_last_cases} with nonzero pad_last -- all match"
    )


def test_solve_resize_ratio_matches_reference_over_dimension_sweep(
    reference: _Reference,
) -> None:
    """Sweep solve_resize_ratio, including very wide, very tall, tiny and huge."""
    heights = (1, 2, 3, 8, 17, 64, 233, 701, 1080, 4096, 20000)
    widths = (1, 2, 3, 8, 17, 64, 233, 1024, 1920, 4096, 20000)
    budgets = (32, 64, 128, 384, 381, 1024)
    swept = 0
    branch_counts = {"narrow": 0, "short": 0, "general": 0}
    for height in heights:
        for width in widths:
            for budget in budgets:
                aspect_ratio = height / width
                max_w_float: float = (
                    math.sqrt(max((budget - 2) / aspect_ratio + 0.25, 0.0)) - 0.5
                )
                try:
                    expected = _reference_solve_resize_ratio(
                        reference, height, width, budget
                    )
                except AssertionError:
                    # The reference asserts max_w > 1 in the short branch; the
                    # port raises ValueError there instead. Both must reject.
                    with pytest.raises(ValueError):
                        _ = solve_resize_ratio(
                            height,
                            width,
                            reference.config.patch_size,
                            reference.config.downsample_ratio,
                            budget,
                        )
                    swept += 1
                    continue
                actual = solve_resize_ratio(
                    height,
                    width,
                    reference.config.patch_size,
                    reference.config.downsample_ratio,
                    budget,
                )
                assert actual == expected, (
                    f"solve_resize_ratio(h={height}, w={width}, budget={budget}) -> {actual} != {expected}"
                )
                if max_w_float < 1.0:
                    branch_counts["narrow"] += 1
                elif max_w_float * aspect_ratio < 2.0:
                    branch_counts["short"] += 1
                else:
                    branch_counts["general"] += 1
                swept += 1
    assert swept == len(heights) * len(widths) * len(budgets)
    for branch, count in branch_counts.items():
        assert count > 0, (
            f"sweep never entered the {branch!r} branch of solve_resize_ratio"
        )
    print(
        f"\nsolve_resize_ratio sweep: {swept} cases, branch coverage {branch_counts} -- all match"
    )


def test_safe_resize_matches_reference_over_dimension_sweep(
    reference: _Reference,
) -> None:
    """safe_resize's shrink loop must match, including its budget decrement."""
    patch_size = reference.config.patch_size
    swept = 0
    for height in (1, 17, 64, 308, 701, 1080, 4096, 20000):
        for width in (1, 17, 64, 450, 1024, 1920, 4096, 20000):
            best_height = -(-height // patch_size) * patch_size
            best_width = -(-width // patch_size) * patch_size
            actual = safe_resize(
                height,
                width,
                best_height,
                best_width,
                patch_size,
                reference.config.downsample_ratio,
                reference.config.max_token_count,
            )
            expected = _reference_safe_resize(
                reference, height, width, best_height, best_width
            )
            assert actual == expected, (
                f"safe_resize(h={height}, w={width}) -> {actual} != {expected}"
            )
            swept += 1
    print(f"\nsafe_resize sweep: {swept} cases -- all match")


def test_build_image_block_matches_reference_for_odd_and_even_heights(
    reference: _Reference,
) -> None:
    """Permutation equality vs the reference across odd AND even n_llm_h."""
    _ = reference  # the fixture imports the reference module this test reaches through
    swept = 0
    odd_heights = 0
    even_heights = 0
    nonzero_pad_last = 0
    nonzero_compress_pad = 0
    for n_llm_h in range(1, 18):
        for n_llm_w in range(1, 18):
            for start_pos in range(2 * COMPRESS_PAD_TO):
                types, perm = build_image_block(n_llm_h, n_llm_w, start_pos)
                expected_types, expected_perm = _reference_build_image_block(
                    n_llm_h, n_llm_w, start_pos
                )
                assert as_integer_list(types) == expected_types, (
                    f"build_image_block({n_llm_h}, {n_llm_w}, {start_pos}).types mismatch"
                )
                assert as_integer_list(perm) == expected_perm, (
                    f"build_image_block({n_llm_h}, {n_llm_w}, {start_pos}).perm mismatch"
                )
                assert len(perm) == n_llm_h * n_llm_w, (
                    "perm must cover every aligner output row exactly once"
                )
                assert sorted(as_integer_list(perm)) == list(
                    range(n_llm_h * n_llm_w)
                ), "perm must be a permutation"
                rows = n_llm_h + n_llm_h % 2
                if ((((rows // 2) * (n_llm_w + 1)) % 2) * 2) != 0:
                    nonzero_pad_last += 1
                if (COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO) != 0:
                    nonzero_compress_pad += 1
                swept += 1
            if n_llm_h % 2 == 1:
                odd_heights += 1
            else:
                even_heights += 1
    assert odd_heights > 0 and even_heights > 0
    assert nonzero_pad_last > 0, "sweep never produced a nonzero pad_last"
    assert nonzero_compress_pad > 0, "sweep never produced a nonzero compress_pad"
    print(
        f"\nbuild_image_block sweep: {swept} cases ({odd_heights} odd / {even_heights} even n_llm_h rows, "
        f"{nonzero_pad_last} nonzero pad_last, {nonzero_compress_pad} nonzero compress_pad) -- all match"
    )


def test_build_image_block_token_count_agrees_with_grid_tokens(
    reference: _Reference,
) -> None:
    """grid_tokens must predict the block length that build_image_block emits.

    They are computed independently -- one from best_height/best_width, one from
    the grid -- so agreement is a real cross-check of the pad_last term. The
    compress_pad offset is the difference, hence start_pos=3 (compress_pad=0).
    """
    patch_size = reference.config.patch_size
    downsample_ratio = reference.config.downsample_ratio
    swept = 0
    for n_llm_h in range(1, 15):
        for n_llm_w in range(1, 15):
            best_height = n_llm_h * patch_size * downsample_ratio
            best_width = n_llm_w * patch_size * downsample_ratio
            derived_h, derived_w, num_tokens = grid_tokens(
                best_height, best_width, patch_size, downsample_ratio
            )
            assert (derived_h, derived_w) == (n_llm_h, n_llm_w)
            types, _ = build_image_block(n_llm_h, n_llm_w, COMPRESS_PAD_TO - 1)
            assert len(types) == num_tokens, (
                f"grid_tokens said {num_tokens} tokens but build_image_block emitted {len(types)} "
                f"for n_llm_h={n_llm_h} n_llm_w={n_llm_w}"
            )
            swept += 1
    print(f"\ngrid_tokens vs build_image_block length agreement: {swept} cases")


def test_build_image_block_aligns_the_grid_to_the_compress_boundary() -> None:
    """The POOLED GRID REGION is what aligns to COMPRESS_PAD_TO -- not the whole block.

    Verified empirically against the reference over 5776 (n_llm_h, n_llm_w,
    start_pos) combinations: ``IMAGE_START`` always lands at an index congruent
    to 3 (mod 4), so the grid content immediately after it starts at a multiple
    of 4, and the grid content length is always a multiple of 4. ``IMAGE_START``
    and ``IMAGE_END`` therefore sit OUTSIDE the pooled region, which is exactly
    what a ratio-4 pooling layer needs. The block as a whole ends at index
    congruent to 1 (mod 4), so asserting whole-block alignment would be wrong.

    The length property also holds algebraically: with ``k = (rows // 2) *
    row_len``, the grid content length is ``rows * row_len + pad_last ==
    2 * k + (k % 2) * 2 == 2 * (k + k % 2)``, and ``k + k % 2`` is always even.
    """
    for n_llm_h in range(1, 12):
        for n_llm_w in range(1, 12):
            for start_pos in range(4 * COMPRESS_PAD_TO):
                types, _ = build_image_block(n_llm_h, n_llm_w, start_pos)
                token_types = as_integer_list(types)
                leading_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
                assert token_types[:leading_pad] == [IMAGE_PAD] * leading_pad
                assert token_types[leading_pad] == IMAGE_START
                assert token_types[-1] == IMAGE_END

                image_start_index = start_pos + leading_pad
                assert image_start_index % COMPRESS_PAD_TO == COMPRESS_PAD_TO - 1, (
                    f"IMAGE_START landed at {image_start_index}, not at a (mod {COMPRESS_PAD_TO}) == "
                    f"{COMPRESS_PAD_TO - 1} position"
                )
                grid_content_start = image_start_index + 1
                grid_content_length = len(types) - leading_pad - 2
                assert grid_content_start % COMPRESS_PAD_TO == 0, (
                    f"grid content starts at {grid_content_start}, not a multiple of {COMPRESS_PAD_TO}"
                )
                assert grid_content_length % COMPRESS_PAD_TO == 0, (
                    f"grid content length {grid_content_length} is not a multiple of {COMPRESS_PAD_TO}"
                )


def _encode_synthetic_png(width: int, height: int, seed: int) -> bytes:
    """A deterministic noise PNG, so pad-vs-resize differences are visible."""
    from PIL import Image  # noqa: PLC0415  -- keep the module import surface small

    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


# (width, height, expects_the_resize_branch). vision_max_wh_ratio is 8, and the
# reference's branch test is `image.width >= ratio * image.height`, so 8.0 is
# INSIDE the resize branch and 7.99 is outside it.
_ASPECT_FORK_CASES: Final = (
    (1600, 100, True),
    (2000, 250, True),
    (800, 100, True),
    (900, 100, True),
    (3000, 200, True),
    (799, 100, False),
    (100, 1600, False),
    (64, 64, False),
    (1024, 701, False),
)


@pytest.mark.parametrize(
    ("width", "height", "expects_resize_branch"), _ASPECT_FORK_CASES
)
def test_load_image_matches_reference_across_the_aspect_ratio_fork(
    reference: _Reference, width: int, height: int, expects_resize_branch: bool
) -> None:
    """Cover BOTH sides of the resize-vs-pad fork that the shipped images miss.

    Neither carrots.jpeg (1024x701) nor corn.jpeg (450x308) is wide enough to
    reach ``image.width >= vision_max_wh_ratio * image.height``, so the golden
    end-to-end test alone cannot detect the fork being collapsed to pad-only.
    A dedicated sabotage run confirmed exactly that gap: removing the resize
    branch left all golden assertions green.
    """
    record: Mapping[str, object] = {
        "data": _encode_synthetic_png(width, height, seed=width * 31 + height)
    }
    assert (
        reference.config.max_width_height_ratio is not None
        and width >= reference.config.max_width_height_ratio * height
    ) == expects_resize_branch, (
        f"{width}x{height} does not exercise the branch this case claims"
    )

    port_patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(
        record, reference.config
    )
    reference_patches, ref_n_vit_h, ref_n_vit_w, ref_n_llm_h, ref_n_llm_w = (
        _reference_load_image(reference, record)
    )

    assert (n_llm_h, n_llm_w, n_vit_h, n_vit_w) == (
        ref_n_llm_h,
        ref_n_llm_w,
        ref_n_vit_h,
        ref_n_vit_w,
    ), f"{width}x{height}: derived geometry differs from the reference"
    assert port_patches.shape == reference_patches.shape
    assert np.array_equal(port_patches, reference_patches), (
        f"{width}x{height}: patch values differ from the reference bit-for-bit"
    )


def test_aspect_ratio_fork_actually_changes_pixels(reference: _Reference) -> None:
    """The fork is behavioural, not cosmetic: the two paths yield different pixels.

    This is the negative control for the fork tests above. If pad and resize
    produced the same pixels for a very wide image, those tests would be
    decorative.
    """
    from PIL import Image, ImageOps  # noqa: PLC0415

    width, height = 2000, 100
    image_bytes = _encode_synthetic_png(width, height, seed=99)
    record: Mapping[str, object] = {"data": image_bytes}
    resize_branch_patches, n_vit_h, n_vit_w, _, _ = load_image(record, reference.config)
    assert reference.config.max_width_height_ratio is not None
    assert width >= reference.config.max_width_height_ratio * height, (
        "this case must take the resize branch"
    )

    patch_size = reference.config.patch_size
    best_height, best_width = n_vit_h * patch_size, n_vit_w * patch_size
    with Image.open(io.BytesIO(image_bytes)) as source:
        padded = ImageOps.pad(
            source.convert("RGB"), (best_width, best_height), color=(127, 127, 127)
        )
    channel_first = np.asarray(padded, dtype=np.float32).transpose(
        2, 0, 1
    ) / np.float32(255.0)
    normalized = round_to_bfloat16_precision(
        (channel_first - np.float32(0.5)) / np.float32(0.5)
    )
    pad_branch_patches = (
        normalized.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .transpose(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
    )

    assert pad_branch_patches.shape == resize_branch_patches.shape
    differing = int(
        np.count_nonzero(np.not_equal(pad_branch_patches, resize_branch_patches))
    )
    assert differing > 0, (
        "pad and resize produced identical pixels for a 20:1 image -- the fork tests would be decorative"
    )
    print(
        f"\naspect fork negative control: {differing} of {resize_branch_patches.size} patch elements "
        f"differ between the resize and pad paths"
    )


def test_sentinel_token_type_values_are_the_reference_range() -> None:
    """The reference spells these as ``... = range(5)``; the order is load-bearing."""
    assert (IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END) == (0, 1, 2, 3, 4)
    assert COMPRESS_PAD_TO == 4


def test_bfloat16_rounding_matches_torch_bitwise() -> None:
    """The bf16 stand-in must be bit-identical to torch's cast, ties included."""
    generator = np.random.default_rng(20260908)
    pipeline_domain = (
        (np.arange(256, dtype=np.float32) / 255.0) - np.float32(0.5)
    ) / np.float32(0.5)
    broad = (
        generator.standard_normal(500_000).astype(np.float32)
        * generator.choice(
            np.array([1e-30, 1e-3, 1.0, 1e3, 1e30], dtype=np.float32), 500_000
        )
    ).astype(np.float32)
    # Exact half-way values: every bf16 pattern with the tie bit (0x8000) set.
    tie_patterns = (np.arange(1 << 16, dtype=np.uint32) << np.uint32(16)) | np.uint32(
        0x8000
    )
    ties = tie_patterns.view(np.float32)
    ties = ties[np.isfinite(ties)]
    specials = np.array(
        [0.0, -0.0, np.inf, -np.inf, 1e-45, -1e-45, 3.4e38, -3.4e38], dtype=np.float32
    )

    for label, values in (
        ("pipeline domain", pipeline_domain),
        ("broad magnitude", broad),
        ("exact ties", ties),
        ("specials", specials),
    ):
        expected = _float32_array_to_bfloat16_array(values)
        actual = round_to_bfloat16_precision(values)
        differing = int(
            np.count_nonzero(
                np.not_equal(actual.view(np.uint32), expected.view(np.uint32))
            )
        )
        assert differing == 0, (
            f"{label}: bfloat16 rounding differs from torch on {differing} of {values.size} values"
        )

    nan_values = np.array([np.nan, -np.nan], dtype=np.float32)
    assert bool(np.all(np.isnan(round_to_bfloat16_precision(nan_values)))), (
        "NaN must survive the bf16 rounding"
    )


def test_expand_image_placeholders_rejects_image_count_mismatch(
    reference: _Reference,
) -> None:
    """A placeholder/image count mismatch is an error, matching the reference."""
    placeholder_token_id = 129264
    with pytest.raises(ValueError, match="Found 1 image tokens but got 0 images"):
        _ = expand_image_placeholders(
            [1, placeholder_token_id, 2], [], placeholder_token_id, reference.config
        )
    with pytest.raises(ValueError, match="Found 0 image tokens but got 1 images"):
        _ = expand_image_placeholders(
            [1, 2, 3],
            [{"url": str(_IMAGES_DIR / "corn.jpeg")}],
            placeholder_token_id,
            reference.config,
        )


def test_expand_image_placeholders_without_images_is_a_passthrough(
    reference: _Reference,
) -> None:
    """A text-only prompt round-trips unchanged with no ImageInput records."""
    token_ids, image_inputs = expand_image_placeholders(
        [5, 6, 7], [], 129264, reference.config
    )
    assert token_ids == [5, 6, 7]
    assert image_inputs == []


def test_vision_encoder_config_reads_the_shipped_config_json() -> None:
    """Both the HF and reference config.json spell the vision keys identically."""
    with open(_INFERENCE_DIR / "config.json") as config_file:
        reference_config = cast("dict[str, Any]", json.load(config_file))
    with open(_REFERENCE_DIR / "config.json") as config_file:
        huggingface_config = cast("dict[str, Any]", json.load(config_file))
    from_reference = VisionEncoderConfig.from_config_mapping(
        cast("Mapping[str, object]", reference_config)
    )
    from_huggingface = VisionEncoderConfig.from_config_mapping(
        cast("Mapping[str, object]", huggingface_config)
    )
    assert from_reference == from_huggingface, (
        "the HF config.json and the reference inference/config.json must agree on the vision keys"
    )
    assert from_reference.patch_size == 14
    assert from_reference.downsample_ratio == 3
    assert from_reference.max_token_count == 384
    assert from_reference.min_pixel_count == 147456
    assert from_reference.max_width_height_ratio == 8
    assert from_reference.vocabulary_size == 129280


def test_vision_encoder_config_rejects_missing_and_non_integer_keys() -> None:
    """Config parsing is strict: a missing or wrongly-typed key is an error."""
    complete: dict[str, object] = {
        "vision_patch_size": 14,
        "vision_downsample_ratio": 3,
        "vision_max_n_token": 384,
        "vision_min_pixels": 147456,
        "vision_max_wh_ratio": 8,
        "vocab_size": 129280,
    }
    assert VisionEncoderConfig.from_config_mapping(complete).patch_size == 14

    without_optional = dict(complete)
    without_optional.pop("vision_max_wh_ratio")
    assert (
        VisionEncoderConfig.from_config_mapping(without_optional).max_width_height_ratio
        is None
    )

    for missing_key in ("vision_patch_size", "vision_downsample_ratio", "vocab_size"):
        incomplete = dict(complete)
        del incomplete[missing_key]
        with pytest.raises(KeyError, match=missing_key):
            _ = VisionEncoderConfig.from_config_mapping(incomplete)

    wrong_type = dict(complete)
    wrong_type["vision_patch_size"] = 14.0
    with pytest.raises(TypeError, match="vision_patch_size"):
        _ = VisionEncoderConfig.from_config_mapping(wrong_type)


def test_sentinel_token_ids_are_emitted_above_the_vocabulary(
    reference: _Reference,
) -> None:
    """Sentinels are deliberately out-of-vocabulary: vocabulary_size + type."""
    _, port_image_inputs, _ = _encode_example(reference, "txt")
    port_token_ids, _, _ = _encode_example(reference, "txt")
    vocabulary_size = reference.config.vocabulary_size
    for image_input in port_image_inputs:
        token_types = as_integer_list(image_input.types)
        block = port_token_ids[image_input.start : image_input.start + len(token_types)]
        assert block == [vocabulary_size + token_type for token_type in token_types]
        assert all(token_id >= vocabulary_size for token_id in block), (
            "every sentinel token must sit above the vocabulary so the embedding layer can mask it"
        )
        assert block[-1] == vocabulary_size + IMAGE_END
        assert set(token_types) <= {
            IMAGE_START,
            IMAGE_PAD,
            IMAGE,
            IMAGE_NEW_LINE,
            IMAGE_END,
        }
    text_token_ids = [
        token_id
        for index, token_id in enumerate(port_token_ids)
        if not any(
            image.start <= index < image.start + len(image.types)
            for image in port_image_inputs
        )
    ]
    assert all(token_id < vocabulary_size for token_id in text_token_ids), (
        "no text token may collide with the sentinel range"
    )
