#!/usr/bin/env python3
"""Hardware verification of the ``PrefillCancelled`` path (requirement 2).

WHY THIS SCRIPT EXISTS
----------------------
``PrefillCancelled`` (src/exo/worker/engines/mlx/generator/generate.py:368,
handled at src/exo/worker/runner/llm_inference/batch_generator.py:762,
:799 and :840) covers exactly one window: a client cancels a request while
that request is still PREFILLING, before any token has been decoded.

The pre-existing cancel harness (bench/section27_cancel_abort_test.py)
structurally CANNOT reach that window. It learns the CommandId to cancel
from the ``id`` field of the first streamed chunk
(section27_cancel_abort_test.py:146-148). During prefill no chunk has been
streamed, so by the time it can name the request, prefill is already over
and the cancel necessarily lands on the DECODE path, not this one.

The fix is the client-supplied ``correlation_id``: the client picks an id
before it sends the request, exo echoes it verbatim into
``/state -> tasks[...].task_params.correlationId`` as soon as the master
indexes the task -- which happens BEFORE prefill starts. The harness polls
/state for its own correlation id, resolves the real CommandId from the
same task object, and can then cancel at any moment it likes, including
deep inside prefill.

PROVING WE ACTUALLY HIT THE WINDOW (the whole point)
----------------------------------------------------
A harness that cancels and reports PASS without proving WHEN the cancel
landed is worthless here -- silently cancelling post-prefill is the exact
bug being fixed. So this script requires THREE independent pieces of
evidence and FAILS LOUDLY if any is missing:

  E1. No token was ever streamed on the request's own SSE stream before
      the cancel was issued. (Necessary, not sufficient: absence of a
      token could also mean the request never started.)
  E2. Positive proof the request was genuinely mid-prefill: at least one
      ``: prefill_progress`` SSE comment was observed with
      ``processed_tokens`` strictly between 0 and ``total_tokens``, i.e.
      prefill had demonstrably started and demonstrably not finished. The
      cancel is issued at that instant.
  E3. The runner actually took the PrefillCancelled branch: the marker
      string ``PREFILL_CANCELLED_PATH`` appears in ~/exo.log on at least
      one node, newly, after the cancel was issued. This is ground truth
      from the runner process itself, not an inference from client-side
      timing.

Anything less is reported as INCONCLUSIVE (exit 2), never PASS. Exit 0 =
PASS, 1 = FAIL, 2 = INCONCLUSIVE (could not establish the window).

USAGE (run from the repo root on the box that can ssh to both nodes):

    uv run python bench/section85_prefill_cancel_hardware_test.py

This script is READ-ONLY with respect to cluster lifecycle: it issues one
chat request, one cancel, and one health-check request. It never restarts,
relaunches or reconfigures anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from typing import Any, Final, cast

import httpx

# Must match the marker logged by the PrefillCancelled handlers in
# src/exo/worker/runner/llm_inference/batch_generator.py. If you rename it
# there, rename it here -- the whole hardware proof (E3) hangs off it.
PREFILL_CANCELLED_MARKER: Final[str] = "PREFILL_CANCELLED_PATH"


def decode_json(raw: str | bytes) -> object:
    """``json.loads`` typed honestly: decoded JSON is ``object``, not ``Any``."""
    return json.loads(raw)  # pyright: ignore[reportAny]


def json_object(value: object) -> dict[str, object]:
    """Narrow an arbitrary decoded-JSON value to a str-keyed object.

    Decoded JSON is inherently ``object``; this repo type-checks in strict
    mode with ``reportAny`` as an error, so every traversal step is
    narrowed explicitly rather than smuggled through ``Any``.
    """
    if isinstance(value, dict):
        narrowed: dict[str, object] = {}
        pairs: list[tuple[object, object]] = list(value.items())  # pyright: ignore[reportUnknownArgumentType]
        for key, item in pairs:
            narrowed[str(key)] = item
        return narrowed
    return {}


def json_array(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)  # pyright: ignore[reportUnknownArgumentType]
    return []


def json_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def json_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


EXIT_PASS: Final[int] = 0
EXIT_FAIL: Final[int] = 1
EXIT_INCONCLUSIVE: Final[int] = 2


def count_marker_occurrences(host: str, marker: str, log_path: str) -> int | None:
    """Count occurrences of `marker` in `log_path` on `host`.

    Returns None if the log could not be read at all (ssh failure, missing
    file) -- distinct from 0, which means "read fine, marker absent".
    """
    try:
        result = subprocess.run(
            ["ssh", host, f"grep -c -- {marker!r} {log_path} 2>/dev/null || echo 0"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[harness] WARNING: cannot read {host}:{log_path}: {exc}")
        return None
    if result.returncode != 0:
        print(f"[harness] WARNING: ssh {host} exited {result.returncode}")
        return None
    raw = result.stdout.strip().splitlines()
    if not raw:
        return None
    try:
        return int(raw[-1].strip())
    except ValueError:
        return None


def snapshot_marker_counts(hosts: list[str], log_path: str) -> dict[str, int | None]:
    return {
        host: count_marker_occurrences(host, PREFILL_CANCELLED_MARKER, log_path)
        for host in hosts
    }


def build_long_prefill_prompt(approximate_tokens: int) -> str:
    """A prompt long enough that prefill takes many seconds -- wide enough
    for the harness to observe partial prefill progress and act inside it.

    Deliberately NOT a repetition of one short sentence: a highly
    repetitive prompt is a prefix-cache magnet, and a cache hit would skip
    most of prefill and collapse the very window under test. A unique
    nonce is embedded for the same reason.
    """
    nonce = uuid.uuid4().hex
    paragraph = (
        f"[nonce {nonce}] Consider a distributed inference system in which "
        "model weights are sharded across heterogeneous accelerators "
        "connected by a high-bandwidth interconnect. Discuss scheduling, "
        "cache residency, collective-communication ordering, failure "
        "domains, backpressure, admission control, tail latency, and the "
        "interaction between speculative decoding and pipeline "
        "parallelism. Enumerate the tradeoffs precisely and at length. "
    )
    repeats = max(1, approximate_tokens // 60)
    return "".join(f"{paragraph} (section {i}) " for i in range(repeats))


async def resolve_command_id_from_state(
    base_url: str, correlation_id: str, deadline_seconds: float
) -> tuple[str | None, str | None]:
    """Poll /state until a task carrying our correlation_id appears.

    Returns (command_id, task_id). This is the capability the old harness
    lacked: it resolves the in-flight request's identity from cluster
    state, with no dependence on a streamed chunk.
    """
    deadline = time.perf_counter() + deadline_seconds
    async with httpx.AsyncClient(timeout=15.0) as client:
        while time.perf_counter() < deadline:
            try:
                response = await client.get(f"{base_url}/state")
                state = json_object(decode_json(response.content))
            except (httpx.HTTPError, json.JSONDecodeError):
                await asyncio.sleep(0.25)
                continue
            tasks = json_object(state.get("tasks"))
            for tagged_task_value in tasks.values():
                tagged_task = json_object(tagged_task_value)
                for tag, body_value in tagged_task.items():
                    if tag != "TextGeneration":
                        continue
                    body = json_object(body_value)
                    # Casing is asymmetric and this bit me during
                    # development: the Task envelope is camelCased by
                    # State's alias_generator ("taskParams", "commandId"),
                    # but TextGenerationTaskParams has no alias generator,
                    # so its own fields stay snake_case
                    # ("correlation_id"). Pinned by
                    # src/exo/api/tests/test_correlation_id.py. Both
                    # spellings are accepted here purely for robustness.
                    params = json_object(
                        body.get("taskParams") or body.get("task_params")
                    )
                    seen = json_text(
                        params.get("correlation_id") or params.get("correlationId")
                    )
                    if seen == correlation_id:
                        return (
                            json_text(body.get("commandId") or body.get("command_id")),
                            json_text(body.get("taskId") or body.get("task_id")),
                        )
            await asyncio.sleep(0.25)
    return (None, None)


class PrefillCancelRun:
    """State of one long-prefill request + the cancel fired into it."""

    def __init__(self) -> None:
        self.correlation_id: str = f"prefill-cancel-{uuid.uuid4().hex}"
        self.command_id_from_state: str | None = None
        self.task_id_from_state: str | None = None
        self.command_id_from_stream: str | None = None
        self.tokens_streamed_before_cancel: int = 0
        self.tokens_streamed_total: int = 0
        self.partial_prefill_observed: bool = False
        self.prefill_progress_at_cancel: tuple[int, int] | None = None
        self.cancel_issued: bool = False
        self.cancel_status: int | None = None
        self.cancel_elapsed_seconds: float | None = None
        self.stream_error: str | None = None
        self.finish_reason: str | None = None

    def evidence_e1_no_token_before_cancel(self) -> bool:
        return self.cancel_issued and self.tokens_streamed_before_cancel == 0

    def evidence_e2_partial_prefill(self) -> bool:
        return self.partial_prefill_observed


async def issue_cancel(base_url: str, command_id: str, run: PrefillCancelRun) -> None:
    started = time.perf_counter()
    run.cancel_issued = True
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{base_url}/v1/cancel/{command_id}")
            run.cancel_status = response.status_code
    except httpx.HTTPError as exc:
        run.stream_error = f"cancel POST failed: {type(exc).__name__}: {exc}"
    run.cancel_elapsed_seconds = time.perf_counter() - started
    print(
        f"[harness] cancel POST /v1/cancel/{command_id} -> "
        f"status={run.cancel_status} in {run.cancel_elapsed_seconds:.2f}s"
    )


async def stream_and_cancel_mid_prefill(
    base_url: str, model: str, prompt_tokens: int, run: PrefillCancelRun
) -> None:
    """Send the long-prefill request and cancel the instant we can prove
    prefill is in progress and unfinished."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": build_long_prefill_prompt(prompt_tokens)}
        ],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": 2000,
        "correlation_id": run.correlation_id,
    }

    resolver = asyncio.create_task(
        resolve_command_id_from_state(base_url, run.correlation_id, 120.0)
    )

    async with (
        httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=120.0, pool=30.0)
        ) as client,
        client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as response,
    ):
        if response.status_code != 200:
            await response.aread()
            run.stream_error = f"HTTP {response.status_code}: {response.text[:400]}"
            resolver.cancel()
            return
        async for line in response.aiter_lines():
            if not line or line.isspace():
                continue

            # Prefill progress arrives as an SSE COMMENT, not a data
            # frame -- see generate_chat_stream in
            # src/exo/api/adapters/chat_completions.py:239-241. This is
            # the positive mid-prefill signal (E2).
            if line.startswith(": prefill_progress"):
                payload_text = line[len(": prefill_progress") :].strip()
                try:
                    progress = json_object(decode_json(payload_text))
                except json.JSONDecodeError:
                    continue
                processed = json_int(progress.get("processed_tokens"))
                total = json_int(progress.get("total_tokens"))
                if 0 < processed < total and not run.cancel_issued:
                    run.partial_prefill_observed = True
                    run.prefill_progress_at_cancel = (processed, total)
                    command_id = run.command_id_from_state
                    if command_id is None and resolver.done():
                        resolved = resolver.result()
                        run.command_id_from_state, run.task_id_from_state = resolved
                        command_id = run.command_id_from_state
                    if command_id is None:
                        # State hasn't caught up yet; keep streaming
                        # progress and try again on the next tick
                        # rather than cancelling blind.
                        run.partial_prefill_observed = False
                        run.prefill_progress_at_cancel = None
                        continue
                    print(
                        f"[harness] mid-prefill confirmed "
                        f"({processed}/{total} tokens prefilled), "
                        f"cancelling command_id={command_id} NOW"
                    )
                    run.tokens_streamed_before_cancel = run.tokens_streamed_total
                    await issue_cancel(base_url, command_id, run)
                continue

            if not line.startswith("data:"):
                continue
            data_text = line[5:].lstrip()
            if data_text == "[DONE]":
                break
            try:
                chunk = json_object(decode_json(data_text))
            except json.JSONDecodeError:
                continue
            if run.command_id_from_stream is None:
                run.command_id_from_stream = json_text(chunk.get("id"))
            for choice_value in json_array(chunk.get("choices")):
                choice = json_object(choice_value)
                delta = json_object(choice.get("delta"))
                text = json_text(delta.get("content")) or json_text(
                    delta.get("reasoning_content")
                )
                if text:
                    run.tokens_streamed_total += 1
                finish_reason = json_text(choice.get("finish_reason"))
                if finish_reason:
                    run.finish_reason = finish_reason

    if not resolver.done():
        resolver.cancel()
    elif run.command_id_from_state is None:
        run.command_id_from_state, run.task_id_from_state = resolver.result()


async def verify_cluster_healthy(base_url: str, model: str) -> bool:
    """A trivial follow-up request must succeed -- proves the cancel did
    not leave the runner/session corrupted. Prompt and budget copied from
    section27_cancel_abort_test.py, which documents why a naive
    "say hello" probe gives a false negative on this thinking model."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{base_url}/v1/chat/completions", json=body)
            if response.status_code != 200:
                print(f"[harness] health check FAILED: HTTP {response.status_code}")
                return False
            data = json_object(decode_json(response.content))
            choices = json_array(data.get("choices"))
            if not choices:
                print("[harness] health check FAILED: no choices in response")
                return False
            choice = json_object(choices[0])
            content = json_text(json_object(choice.get("message")).get("content"))
            finish_reason = json_text(choice.get("finish_reason"))
            print(
                f"[harness] health check: content={content!r} "
                f"finish_reason={finish_reason}"
            )
            return finish_reason == "stop" and bool(content)
    except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"[harness] health check EXCEPTION: {type(exc).__name__}: {exc}")
        return False


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default="http://adams-mac-studio-m4-1.local:52415"
    )
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=60000,
        help=(
            "Approximate prompt length. Must be long enough that prefill "
            "lasts many seconds, otherwise prefill finishes before the "
            "harness can act and the run is INCONCLUSIVE, not PASS."
        ),
    )
    parser.add_argument("--hosts", default="macstudio-m4-1,macstudio-m4-2")
    parser.add_argument("--log-path", default="~/exo.log")
    parser.add_argument(
        "--marker-settle-seconds",
        type=float,
        default=25.0,
        help=(
            "How long to wait after the cancel before re-reading the node "
            "logs for the PREFILL_CANCELLED_PATH marker. The cancel is "
            "collective and the log write happens on the runner, so a "
            "small settle window is required."
        ),
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip the post-cancel follow-up request.",
    )
    arguments = parser.parse_args()
    # argparse's Namespace is untyped; bind every value to an explicitly
    # typed local once, so the rest of main() type-checks properly.
    base_url = cast(str, arguments.base_url)
    model = cast(str, arguments.model)
    prompt_tokens = cast(int, arguments.prompt_tokens)
    log_path = cast(str, arguments.log_path)
    marker_settle_seconds = cast(float, arguments.marker_settle_seconds)
    skip_health_check = cast(bool, arguments.skip_health_check)

    hosts = [host for host in cast(str, arguments.hosts).split(",") if host]

    print("=== PrefillCancelled hardware verification (mid-prefill cancel) ===")
    before_counts = snapshot_marker_counts(hosts, log_path)
    print(f"[harness] marker counts before run: {before_counts}")

    run = PrefillCancelRun()
    print(f"[harness] correlation_id={run.correlation_id}")
    try:
        await stream_and_cancel_mid_prefill(base_url, model, prompt_tokens, run)
    except httpx.HTTPError as exc:
        run.stream_error = f"{type(exc).__name__}: {exc}"

    print(
        f"\n[harness] command_id from /state: {run.command_id_from_state}\n"
        f"[harness] command_id from stream : {run.command_id_from_stream}\n"
        f"[harness] tokens streamed before cancel: "
        f"{run.tokens_streamed_before_cancel}\n"
        f"[harness] tokens streamed total        : {run.tokens_streamed_total}\n"
        f"[harness] prefill progress at cancel   : "
        f"{run.prefill_progress_at_cancel}\n"
        f"[harness] finish_reason: {run.finish_reason}\n"
        f"[harness] stream error : {run.stream_error}"
    )

    if not run.cancel_issued:
        print(
            "\n=== INCONCLUSIVE: never issued a cancel ===\n"
            "The harness could not confirm the request was mid-prefill "
            "(no partial prefill_progress observed, or /state never "
            "surfaced the correlation id in time). It deliberately did "
            "NOT cancel blindly, because a post-prefill cancel would "
            "exercise the DECODE cancel path and prove nothing about "
            "PrefillCancelled. Retry with a larger --prompt-tokens."
        )
        return EXIT_INCONCLUSIVE

    print(
        f"\n[harness] waiting {marker_settle_seconds:.0f}s for the "
        "runner log to settle..."
    )
    await asyncio.sleep(marker_settle_seconds)
    after_counts = snapshot_marker_counts(hosts, log_path)
    print(f"[harness] marker counts after run: {after_counts}")

    marker_readable = False
    marker_grew = False
    for host in hosts:
        before_count = before_counts[host]
        after_count = after_counts[host]
        if before_count is None or after_count is None:
            continue
        marker_readable = True
        if after_count > before_count:
            marker_grew = True

    evidence_1 = run.evidence_e1_no_token_before_cancel()
    evidence_2 = run.evidence_e2_partial_prefill()

    print("\n=== EVIDENCE ===")
    print(f"  E1 no token streamed before cancel : {evidence_1}")
    print(f"  E2 partial prefill observed        : {evidence_2}")
    print(f"  E3 PREFILL_CANCELLED_PATH marker   : {marker_grew}")
    print(f"  cancel HTTP status                 : {run.cancel_status}")

    if not marker_readable:
        print(
            "\n=== INCONCLUSIVE: could not read the runner logs ===\n"
            "E3 is the only ground-truth proof that the PrefillCancelled "
            "branch ran. Without it this run cannot be called a PASS, "
            "regardless of client-side timing. Check ssh access to "
            f"{hosts} and --log-path."
        )
        return EXIT_INCONCLUSIVE

    healthy = True
    if not skip_health_check:
        print("\n=== Post-cancel cluster health check ===")
        healthy = await verify_cluster_healthy(base_url, model)

    if (
        evidence_1
        and evidence_2
        and marker_grew
        and run.cancel_status == 200
        and healthy
    ):
        print(
            "\n=== PASS ===\n"
            "A cancel was issued while the request was demonstrably "
            "mid-prefill, the runner took the PrefillCancelled branch, and "
            "the cluster stayed healthy afterwards."
        )
        return EXIT_PASS

    if not marker_grew and evidence_1 and evidence_2:
        print(
            "\n=== FAIL ===\n"
            "The cancel was issued inside a confirmed mid-prefill window "
            "but the runner never logged PREFILL_CANCELLED_PATH. Either "
            "the cancel did not reach the prefill loop, or it was handled "
            "on a different path. This is the real regression signature."
        )
        return EXIT_FAIL

    print(
        "\n=== FAIL ===\n"
        "One or more required conditions did not hold; see the evidence "
        "table above. This is NOT reported as a pass -- a cancel that may "
        "have landed post-prefill proves nothing about PrefillCancelled."
    )
    return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
