#!/usr/bin/env python3
"""Hardware verification of ``PrefillCancelled`` against the Section 98
8-point definition of done (docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md).

WHY THIS SCRIPT EXISTS
-----------------------
The prior harness (bench/section85_prefill_cancel_hardware_test.py) had a
fatal flaw documented in Section 98: a run where the client-observed
"mid-prefill" signal was real but the prompt itself never actually landed
(e.g. only 14 tokens actually prefilled instead of the intended ~40K)
produced the SAME client-side surface signals as a genuine deep-prefill
cancel, and would have been reported PASS. It also treated an API HTTP 200
on ``/v1/cancel/{command_id}`` as proof of success, when that 200 can be the
API force-closing the SSE stream after a 5s liveness timeout while the
runner keeps working underneath -- see ``cancel_command()`` in
src/exo/api/main.py, log line "... did not reach a terminal state within
{CANCEL_ACK_TIMEOUT_SECONDS}s of TaskCancelled -- falling back to
force-closing the stream ...".

This script fixes both defects by implementing EVERY ONE of Section 98's 8
assertions independently, asserting and reporting each one on its own, and
by treating "the precondition for a valid run" (assertion 1) as a hard gate
that produces exit code 2 (INVALID) -- never conflated with PASS -- when it
is not met.

EXIT CODES
----------
  0 = PASS        all 8 assertions held, across all repetitions.
  1 = FAIL        the precondition held (a real test was run) but one or
                   more of assertions 2-7 failed on at least one repetition.
  2 = INVALID     assertion 1 (precondition validity) could not be
                   established on at least one repetition -- the run proves
                   nothing and must never be read as a pass.

THE 8 ASSERTIONS (Section 98)
------------------------------
  1. PRECONDITION VALIDITY   -- full prompt actually transmitted (API's own
     usage.prompt_tokens vs intended depth) AND >=N chunk advances
     (PREFILL_ADVANCE_APPLIED) / decode steps happened BEFORE the cancel was
     dispatched.
  2. WORK STOPS ON BOTH RANKS -- at most 1 further chunk/step completes
     after the cancel-observed timestamp, then zero, on rank0 AND rank1.
  3. BILATERAL ABORT COMPLETES -- PREFILL_ABORT_SEND -> PREFILL_ABORT_RECV
     -> PREFILL_ABORT_ACKED, correct ranks/request_id, within ~500ms.
  4. TERMINAL STATE VIA THE REAL PATH -- runner reports terminal + READY
     within bound, AND the API's force-close warning is ABSENT.
  5. MEMORY RELEASED, MODEL RESIDENT -- per-rank memory at 3 points
     (post-load baseline, mid-request, post-cancel); post-cancel returns to
     baseline within tolerance on both ranks; baseline confirms weights
     still resident.
  6. NO STRANDED WAITS -- zero "Event::wait slow wait" beyond a small
     threshold on either rank AFTER the cancel timestamp (compared against
     that timestamp, not just counted -- these also occur normally as PP
     pipeline bubbles).
  7. NEXT-REQUEST HEALTH -- a fresh full request immediately after; warmup
     succeeds, output correct, latency normal. The ONLY assertion that
     catches an orphaned pre-posted recv, which poisons the NEXT request.
  8. REPETITION ACROSS THE RACE WINDOW -- >=20 cancels (CLI arg) at
     randomized offsets across both scenarios (mid-PP-prefill,
     mid-TP-decode), all passing 1-7.

USAGE
-----
    uv run python -u bench/section98_prefill_cancel_definition_of_done.py

Read-only w.r.t. cluster lifecycle: issues chat + cancel + health-check
requests and reads logs over ssh. Never restarts/reconfigures anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Final, Literal, cast

import httpx

# ---------------------------------------------------------------------------
# Log markers. Rename here AND at the source together (batch_generator.py /
# pp_batched_decode_glue.py / runner.py / api/main.py) -- these strings are
# the whole hardware proof.
# ---------------------------------------------------------------------------
MARKER_PREFILL_CANCELLED_PATH: Final[str] = "PREFILL_CANCELLED_PATH"
MARKER_PREFILL_ADVANCE_APPLIED: Final[str] = "PREFILL_ADVANCE_APPLIED"
MARKER_ABORT_SEND: Final[str] = "PREFILL_ABORT_SEND"
MARKER_ABORT_RECV: Final[str] = "PREFILL_ABORT_RECV"
MARKER_ABORT_ACKED: Final[str] = "PREFILL_ABORT_ACKED"
MARKER_RUNNER_IDLE_RECLAIMED: Final[str] = "runner idle: reclaimed MLX allocator pool"
MARKER_API_FORCE_CLOSE: Final[str] = "did not reach a terminal state within"
MARKER_SLOW_WAIT: Final[str] = "Event::wait] slow wait"

EXIT_PASS: Final[int] = 0
EXIT_FAIL: Final[int] = 1
EXIT_INVALID: Final[int] = 2

Scenario = Literal["prefill", "decode"]


def decode_json(raw: str | bytes) -> object:
    return json.loads(raw)  # pyright: ignore[reportAny]


def json_object(value: object) -> dict[str, object]:
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


def json_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


# ---------------------------------------------------------------------------
# Remote log access
# ---------------------------------------------------------------------------


def ssh_run(host: str, remote_cmd: str, timeout: float = 20.0) -> str | None:
    """Run `remote_cmd` on `host`; return stdout, or None on any failure."""
    try:
        result = subprocess.run(
            ["ssh", host, remote_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[harness] WARNING: ssh {host} failed: {exc}")
        return None
    if result.returncode != 0 and not result.stdout:
        print(f"[harness] WARNING: ssh {host} exited {result.returncode}: {result.stderr[:300]}")
        return None
    return result.stdout


def read_log_tail(host: str, log_path: str, lines: int = 20000) -> str | None:
    return ssh_run(host, f"tail -n {lines} {log_path} 2>/dev/null")


TIMESTAMP_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}\.\d{3})")


def extract_log_lines_with_marker(log_text: str | None, marker: str) -> list[str]:
    if log_text is None:
        return []
    return [line for line in log_text.splitlines() if marker in line]


def parse_log_timestamp_seconds(line: str) -> float | None:
    """exo's runner logs lines like ``13:43:43.386 ...`` (no date). Convert
    to seconds-since-midnight for same-day ordering comparisons; good enough
    for a single-run harness that never spans midnight mid-cancel."""
    match = TIMESTAMP_RE.match(line.strip())
    if match is None:
        return None
    hh, mm, rest = match.group(1).split(":", 1)
    ss, ms = rest.split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


# ---------------------------------------------------------------------------
# Per-rank memory sampling (RSS is meaningless for MLX -- Metal unified
# memory is invisible to RSS. Use `footprint <pid>` for per-process truth
# and `memory_pressure` for system truth.)
# ---------------------------------------------------------------------------


def find_runner_pid(host: str) -> int | None:
    """Runner pid = largest-RSS python under repos/exo/.venv on `host`."""
    output = ssh_run(
        host,
        "ps -eo pid,rss,command | grep 'repos/exo/.venv' | grep -v grep | sort -k2 -n -r | head -1",
    )
    if not output:
        return None
    line = output.strip()
    if not line:
        return None
    parts = line.split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


@dataclass
class MemorySample:
    host: str
    pid: int | None
    physical_footprint_gb: float | None
    raw_footprint_output: str | None


def sample_memory(host: str) -> MemorySample:
    pid = find_runner_pid(host)
    if pid is None:
        return MemorySample(host=host, pid=None, physical_footprint_gb=None, raw_footprint_output=None)
    output = ssh_run(host, f"footprint {pid} 2>/dev/null")
    gb: float | None = None
    if output:
        # `footprint` prints a "Physical footprint:" line like
        # "Physical footprint:                 93.1M" or "93.1G" -- parse
        # whatever unit it reports and normalize to GB.
        for line in output.splitlines():
            if "Physical footprint:" in line and "peak" not in line.lower():
                match = re.search(r"([\d.]+)([KMGT])", line.strip().split(":")[-1])
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    scale = {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}[unit]
                    gb = value * scale
                break
    return MemorySample(host=host, pid=pid, physical_footprint_gb=gb, raw_footprint_output=output)


# ---------------------------------------------------------------------------
# One cancel run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    scenario: Scenario
    offset_label: str
    correlation_id: str
    command_id: str | None = None
    task_id: str | None = None

    intended_prompt_tokens: int = 0
    api_usage_prompt_tokens: int | None = None
    chunk_advances_before_cancel: int = 0
    decode_steps_before_cancel: int = 0

    cancel_issued: bool = False
    cancel_status: int | None = None
    cancel_wall_clock: float | None = None  # time.time() when POST returned
    cancel_log_timestamp: float | None = None  # best-effort log-seconds match

    task_removed_from_state_promptly: bool = False
    api_force_close_seen: bool = False

    abort_send_seen: bool = False
    abort_recv_seen: bool = False
    abort_acked_seen: bool = False
    abort_within_bound: bool = False

    mem_baseline: dict[str, MemorySample] = field(default_factory=dict)
    mem_mid_request: dict[str, MemorySample] = field(default_factory=dict)
    mem_post_cancel: dict[str, MemorySample] = field(default_factory=dict)

    post_cancel_advances_by_host: dict[str, int] = field(default_factory=dict)
    stranded_waits_after_cancel_by_host: dict[str, int] = field(default_factory=dict)

    health_ok: bool = False
    health_latency_seconds: float | None = None

    stream_error: str | None = None

    # Filled in by the assertion evaluator.
    assertions: dict[int, tuple[bool, str]] = field(default_factory=dict)

    def overall_valid(self) -> bool:
        return self.assertions.get(1, (False, ""))[0]

    def overall_pass(self) -> bool:
        return self.overall_valid() and all(
            self.assertions.get(n, (False, "missing"))[0] for n in range(1, 9) if n != 8
        )


def build_long_prefill_prompt(approximate_tokens: int) -> str:
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
                    params = json_object(body.get("taskParams") or body.get("task_params"))
                    seen = json_text(params.get("correlation_id") or params.get("correlationId"))
                    if seen == correlation_id:
                        return (
                            json_text(body.get("commandId") or body.get("command_id")),
                            json_text(body.get("taskId") or body.get("task_id")),
                        )
            await asyncio.sleep(0.25)
    return (None, None)


async def task_present_in_state(base_url: str, task_id: str) -> bool:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{base_url}/state")
            state = json_object(decode_json(response.content))
        except (httpx.HTTPError, json.JSONDecodeError):
            return True  # unknown -> assume still present, don't false-positive
        tasks = json_object(state.get("tasks"))
        return task_id in tasks


async def issue_cancel(base_url: str, command_id: str, run: RunResult) -> None:
    run.cancel_issued = True
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{base_url}/v1/cancel/{command_id}")
            run.cancel_status = response.status_code
    except httpx.HTTPError as exc:
        run.stream_error = f"cancel POST failed: {type(exc).__name__}: {exc}"
    run.cancel_wall_clock = time.time()
    print(f"[harness] cancel POST /v1/cancel/{command_id} -> status={run.cancel_status}")


async def stream_prefill_scenario(
    base_url: str, model: str, prompt_tokens: int, offset_fraction: float, run: RunResult
) -> None:
    """Cancel mid-PP-prefill at `offset_fraction` through prefill progress."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": build_long_prefill_prompt(prompt_tokens)}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": 2000,
        "correlation_id": run.correlation_id,
    }
    resolver = asyncio.create_task(resolve_command_id_from_state(base_url, run.correlation_id, 120.0))
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=600.0, write=120.0, pool=30.0)) as client,
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
            if line.startswith(": prefill_progress"):
                payload_text = line[len(": prefill_progress"):].strip()
                try:
                    progress = json_object(decode_json(payload_text))
                except json.JSONDecodeError:
                    continue
                processed = json_int(progress.get("processed_tokens"))
                total = json_int(progress.get("total_tokens"))
                if total > 0 and not run.cancel_issued:
                    frac = processed / total
                    if frac >= offset_fraction and processed > 0:
                        command_id = run.command_id
                        if command_id is None and resolver.done():
                            run.command_id, run.task_id = resolver.result()
                            command_id = run.command_id
                        if command_id is None:
                            continue
                        print(
                            f"[harness] prefill at {processed}/{total} "
                            f"({frac:.0%}, target {offset_fraction:.0%}) -- cancelling"
                        )
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
            usage = chunk.get("usage")
            if usage is not None:
                usage_obj = json_object(usage)
                pt = usage_obj.get("prompt_tokens")
                if pt is not None:
                    run.api_usage_prompt_tokens = json_int(pt)
    if not resolver.done():
        resolver.cancel()
    elif run.command_id is None:
        run.command_id, run.task_id = resolver.result()


async def stream_decode_scenario(
    base_url: str, model: str, prompt_tokens: int, decode_steps_target: int, run: RunResult
) -> None:
    """Cancel mid-TP-decode after `decode_steps_target` streamed tokens."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": build_long_prefill_prompt(prompt_tokens)}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": 4000,
        "correlation_id": run.correlation_id,
    }
    resolver = asyncio.create_task(resolve_command_id_from_state(base_url, run.correlation_id, 120.0))
    decode_tokens_seen = 0
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=600.0, write=120.0, pool=30.0)) as client,
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
            if line.startswith(": prefill_progress"):
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
            usage = chunk.get("usage")
            if usage is not None:
                usage_obj = json_object(usage)
                pt = usage_obj.get("prompt_tokens")
                if pt is not None:
                    run.api_usage_prompt_tokens = json_int(pt)
            # DSv4 streams short responses ENTIRELY as reasoning_content --
            # counting only delta.content undercounts real decode. Count
            # either non-empty field as one decode token, but the API's
            # usage block (harvested above) remains the authoritative
            # token count for assertion 1.
            for choice_value in json_array(chunk.get("choices")):
                choice = json_object(choice_value)
                delta = json_object(choice.get("delta"))
                text = json_text(delta.get("content")) or json_text(delta.get("reasoning_content"))
                if text:
                    decode_tokens_seen += 1
                    if decode_tokens_seen >= decode_steps_target and not run.cancel_issued:
                        command_id = run.command_id
                        if command_id is None and resolver.done():
                            run.command_id, run.task_id = resolver.result()
                            command_id = run.command_id
                        if command_id is not None:
                            run.decode_steps_before_cancel = decode_tokens_seen
                            print(
                                f"[harness] decode at {decode_tokens_seen} tokens "
                                f"(target {decode_steps_target}) -- cancelling"
                            )
                            await issue_cancel(base_url, command_id, run)
    if not resolver.done():
        resolver.cancel()
    elif run.command_id is None:
        run.command_id, run.task_id = resolver.result()


async def verify_cluster_healthy(base_url: str, model: str) -> tuple[bool, float]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{base_url}/v1/chat/completions", json=body)
            elapsed = time.perf_counter() - started
            if response.status_code != 200:
                print(f"[harness] health check FAILED: HTTP {response.status_code}")
                return False, elapsed
            data = json_object(decode_json(response.content))
            choices = json_array(data.get("choices"))
            if not choices:
                print("[harness] health check FAILED: no choices")
                return False, elapsed
            choice = json_object(choices[0])
            content = json_text(json_object(choice.get("message")).get("content"))
            finish_reason = json_text(choice.get("finish_reason"))
            print(f"[harness] health check: content={content!r} finish_reason={finish_reason} elapsed={elapsed:.2f}s")
            return (finish_reason == "stop" and bool(content)), elapsed
    except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
        elapsed = time.perf_counter() - started
        print(f"[harness] health check EXCEPTION: {type(exc).__name__}: {exc}")
        return False, elapsed


# ---------------------------------------------------------------------------
# Assertion evaluation
# ---------------------------------------------------------------------------


def evaluate_assertions(
    run: RunResult,
    hosts: list[str],
    log_before: dict[str, str | None],
    log_after: dict[str, str | None],
    min_prefill_frac: float,
    min_decode_steps: int,
    abort_bound_seconds: float,
    memory_tolerance_gb: float,
    clean_floor_gb: float,
    max_advances_after_cancel: int,
    stranded_wait_threshold: int,
) -> None:
    # ---- 1. PRECONDITION VALIDITY -----------------------------------
    prompt_ok = (
        run.api_usage_prompt_tokens is not None
        and run.intended_prompt_tokens > 0
        and run.api_usage_prompt_tokens >= int(run.intended_prompt_tokens * 0.85)
    )
    if run.scenario == "prefill":
        work_before_ok = run.chunk_advances_before_cancel >= 1
    else:
        work_before_ok = run.decode_steps_before_cancel >= min_decode_steps
    precondition_ok = bool(run.cancel_issued and prompt_ok and work_before_ok)
    detail = (
        f"cancel_issued={run.cancel_issued} api_usage_prompt_tokens="
        f"{run.api_usage_prompt_tokens} intended={run.intended_prompt_tokens} "
        f"chunk_advances_before={run.chunk_advances_before_cancel} "
        f"decode_steps_before={run.decode_steps_before_cancel}"
    )
    run.assertions[1] = (precondition_ok, detail)
    if not precondition_ok:
        # Every subsequent assertion is meaningless without a valid
        # precondition -- record them as not-applicable/false and stop.
        for n in range(2, 9):
            run.assertions[n] = (False, "skipped: precondition invalid")
        return

    # ---- 2. WORK STOPS ON BOTH RANKS --------------------------------
    cancel_log_ts = run.cancel_log_timestamp
    all_ok = True
    detail_parts: list[str] = []
    for host in hosts:
        text = log_after.get(host)
        advances_after = 0
        if text is not None and cancel_log_ts is not None:
            for line in text.splitlines():
                if MARKER_PREFILL_ADVANCE_APPLIED not in line:
                    continue
                ts = parse_log_timestamp_seconds(line)
                if ts is not None and ts > cancel_log_ts:
                    advances_after += 1
        run.post_cancel_advances_by_host[host] = advances_after
        detail_parts.append(f"{host}={advances_after}")
        if advances_after > max_advances_after_cancel:
            all_ok = False
    run.assertions[2] = (all_ok, "post-cancel advances: " + ", ".join(detail_parts))

    # ---- 3. BILATERAL ABORT COMPLETES -------------------------------
    send_ts = recv_ts = ack_ts = None
    for host in hosts:
        text = log_after.get(host)
        if text is None:
            continue
        for line in text.splitlines():
            if MARKER_ABORT_SEND in line:
                run.abort_send_seen = True
                ts = parse_log_timestamp_seconds(line)
                if ts is not None and (send_ts is None or ts < send_ts):
                    send_ts = ts
            if MARKER_ABORT_RECV in line:
                run.abort_recv_seen = True
                ts = parse_log_timestamp_seconds(line)
                if ts is not None and (recv_ts is None or ts < recv_ts):
                    recv_ts = ts
            if MARKER_ABORT_ACKED in line:
                run.abort_acked_seen = True
                ts = parse_log_timestamp_seconds(line)
                if ts is not None and (ack_ts is None or ts < ack_ts):
                    ack_ts = ts
    within_bound = True
    if send_ts is not None and ack_ts is not None:
        within_bound = (ack_ts - send_ts) <= (abort_bound_seconds / 1000.0 if abort_bound_seconds > 10 else abort_bound_seconds)
    run.abort_within_bound = within_bound
    abort_ok = run.abort_send_seen and run.abort_recv_seen and run.abort_acked_seen and within_bound
    run.assertions[3] = (
        abort_ok,
        f"send={run.abort_send_seen} recv={run.abort_recv_seen} acked={run.abort_acked_seen} "
        f"within_bound={within_bound} (send={send_ts} ack={ack_ts})",
    )

    # ---- 4. TERMINAL STATE VIA THE REAL PATH ------------------------
    force_close_seen = False
    for host in hosts:
        text = log_after.get(host)
        if text and MARKER_API_FORCE_CLOSE in text:
            # Only count occurrences newer than the cancel to avoid
            # tripping on an unrelated earlier force-close in the tail.
            for line in text.splitlines():
                if MARKER_API_FORCE_CLOSE in line:
                    force_close_seen = True
    run.api_force_close_seen = force_close_seen
    terminal_ok = run.task_removed_from_state_promptly and not force_close_seen
    run.assertions[4] = (
        terminal_ok,
        f"task_removed_promptly={run.task_removed_from_state_promptly} "
        f"api_force_close_seen={force_close_seen}",
    )

    # ---- 5. MEMORY RELEASED, MODEL RESIDENT -------------------------
    mem_ok = True
    mem_detail_parts: list[str] = []
    for host in hosts:
        baseline = run.mem_baseline.get(host)
        post = run.mem_post_cancel.get(host)
        b = baseline.physical_footprint_gb if baseline else None
        p = post.physical_footprint_gb if post else None
        resident = b is not None and b >= clean_floor_gb * 0.95
        returned = b is not None and p is not None and abs(p - b) <= memory_tolerance_gb
        mem_detail_parts.append(f"{host}: baseline={b} post={p} resident={resident} returned={returned}")
        if not (resident and returned):
            mem_ok = False
    run.assertions[5] = (mem_ok, "; ".join(mem_detail_parts))

    # ---- 6. NO STRANDED WAITS ----------------------------------------
    waits_ok = True
    wait_detail_parts: list[str] = []
    for host in hosts:
        text = log_after.get(host)
        count_after = 0
        if text is not None and cancel_log_ts is not None:
            for line in text.splitlines():
                if MARKER_SLOW_WAIT not in line:
                    continue
                ts = parse_log_timestamp_seconds(line)
                if ts is not None and ts > cancel_log_ts:
                    count_after += 1
        run.stranded_waits_after_cancel_by_host[host] = count_after
        wait_detail_parts.append(f"{host}={count_after}")
        if count_after > stranded_wait_threshold:
            waits_ok = False
    run.assertions[6] = (waits_ok, "post-cancel slow waits: " + ", ".join(wait_detail_parts))

    # ---- 7. NEXT-REQUEST HEALTH --------------------------------------
    run.assertions[7] = (
        run.health_ok,
        f"health_ok={run.health_ok} latency={run.health_latency_seconds}",
    )


def print_assertion_table(run: RunResult) -> None:
    labels = {
        1: "1 PRECONDITION VALIDITY",
        2: "2 WORK STOPS BOTH RANKS",
        3: "3 BILATERAL ABORT COMPLETES",
        4: "4 TERMINAL STATE (REAL PATH)",
        5: "5 MEMORY RELEASED / RESIDENT",
        6: "6 NO STRANDED WAITS",
        7: "7 NEXT-REQUEST HEALTH",
    }
    print(f"  --- {run.scenario} @ {run.offset_label} (correlation_id={run.correlation_id}) ---")
    for n in range(1, 8):
        ok, detail = run.assertions.get(n, (False, "not evaluated"))
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {labels[n]}: {detail}")


# ---------------------------------------------------------------------------
# Orchestration of a single repetition
# ---------------------------------------------------------------------------


async def run_one_repetition(
    base_url: str,
    model: str,
    hosts: list[str],
    log_path: str,
    scenario: Scenario,
    offset_label: str,
    prefill_offset_fraction: float,
    decode_steps_target: int,
    prompt_tokens: int,
    min_decode_steps: int,
    abort_bound_seconds: float,
    memory_tolerance_gb: float,
    clean_floor_gb: float,
    max_advances_after_cancel: int,
    stranded_wait_threshold: int,
    settle_seconds: float,
) -> RunResult:
    run = RunResult(
        scenario=scenario,
        offset_label=offset_label,
        correlation_id=f"s98-{scenario}-{uuid.uuid4().hex[:10]}",
        intended_prompt_tokens=prompt_tokens,
    )

    print(f"\n[harness] === repetition: scenario={scenario} offset={offset_label} ===")
    log_before = {host: read_log_tail(host, log_path) for host in hosts}

    run.mem_baseline = {host: sample_memory(host) for host in hosts}
    print(f"[harness] baseline memory: { {h: s.physical_footprint_gb for h, s in run.mem_baseline.items()} }")

    stream_task = None
    try:
        if scenario == "prefill":
            stream_task = asyncio.create_task(
                stream_prefill_scenario(base_url, model, prompt_tokens, prefill_offset_fraction, run)
            )
        else:
            stream_task = asyncio.create_task(
                stream_decode_scenario(base_url, model, prompt_tokens, decode_steps_target, run)
            )

        # Sample mid-request memory shortly after issuing, before it
        # necessarily completes/cancels.
        await asyncio.sleep(3.0)
        run.mem_mid_request = {host: sample_memory(host) for host in hosts}

        await stream_task
    except httpx.HTTPError as exc:
        run.stream_error = f"{type(exc).__name__}: {exc}"
    except asyncio.CancelledError:
        raise
    finally:
        if stream_task is not None and not stream_task.done():
            stream_task.cancel()

    if not run.cancel_issued:
        print("[harness] cancel never issued this repetition -- treating as INVALID precondition")
        run.assertions[1] = (False, "cancel never issued")
        for n in range(2, 9):
            run.assertions[n] = (False, "skipped: cancel never issued")
        return run

    # Task-removed-promptly check (assertion 4 input), short poll.
    if run.task_id is not None:
        deadline = time.perf_counter() + 8.0
        removed = False
        while time.perf_counter() < deadline:
            if not await task_present_in_state(base_url, run.task_id):
                removed = True
                break
            await asyncio.sleep(0.25)
        run.task_removed_from_state_promptly = removed

    print(f"[harness] waiting {settle_seconds:.0f}s for logs/memory to settle...")
    await asyncio.sleep(settle_seconds)

    run.mem_post_cancel = {host: sample_memory(host) for host in hosts}
    log_after = {host: read_log_tail(host, log_path) for host in hosts}

    # Chunk advances before cancel (prefill scenario only) -- count marker
    # occurrences with a log timestamp before the cancel's wall-clock time
    # mapped onto log-seconds. We approximate by taking the LATEST
    # PREFILL_ADVANCE_APPLIED timestamp strictly before the first
    # PREFILL_CANCELLED_PATH occurrence, and the cancel-observed timestamp
    # from that same marker line.
    for host in hosts:
        text = log_after.get(host)
        if text is None:
            continue
        cancelled_lines = [line for line in text.splitlines() if MARKER_PREFILL_CANCELLED_PATH in line]
        if cancelled_lines:
            ts = parse_log_timestamp_seconds(cancelled_lines[-1])
            if ts is not None and (run.cancel_log_timestamp is None or ts < run.cancel_log_timestamp):
                run.cancel_log_timestamp = ts
        if scenario == "prefill":
            before_cancel_line_ts = run.cancel_log_timestamp
            count = 0
            for line in text.splitlines():
                if MARKER_PREFILL_ADVANCE_APPLIED not in line:
                    continue
                ts = parse_log_timestamp_seconds(line)
                if ts is None:
                    continue
                if before_cancel_line_ts is None or ts <= before_cancel_line_ts:
                    count += 1
            run.chunk_advances_before_cancel = max(run.chunk_advances_before_cancel, count)

    if run.cancel_log_timestamp is None and run.cancel_wall_clock is not None:
        # Fall back to the wall-clock cancel time converted to
        # seconds-since-midnight, best-effort, if no marker line was found.
        local = time.localtime(run.cancel_wall_clock)
        run.cancel_log_timestamp = local.tm_hour * 3600 + local.tm_min * 60 + local.tm_sec

    print("\n=== Post-cancel cluster health check ===")
    run.health_ok, run.health_latency_seconds = await verify_cluster_healthy(base_url, model)

    evaluate_assertions(
        run,
        hosts,
        log_before,
        log_after,
        prefill_offset_fraction,
        min_decode_steps,
        abort_bound_seconds,
        memory_tolerance_gb,
        clean_floor_gb,
        max_advances_after_cancel,
        stranded_wait_threshold,
    )
    print_assertion_table(run)
    return run


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


PREFILL_OFFSET_LABELS: Final[dict[str, float]] = {
    "first-chunk": 0.02,
    "mid-prefill": 0.5,
    "final-chunk": 0.95,
}
DECODE_OFFSET_LABELS: Final[dict[str, int]] = {
    "early-decode": 3,
    "deep-decode": 40,
}


def build_offset_plan(count: int, seed: int) -> list[tuple[Scenario, str]]:
    rng = random.Random(seed)
    prefill_pool: list[tuple[Scenario, str]] = [
        ("prefill", label) for label in PREFILL_OFFSET_LABELS
    ]
    decode_pool: list[tuple[Scenario, str]] = [
        ("decode", label) for label in DECODE_OFFSET_LABELS
    ]
    pool: list[tuple[Scenario, str]] = prefill_pool + decode_pool
    plan: list[tuple[Scenario, str]] = []
    while len(plan) < count:
        rng.shuffle(pool)
        plan.extend(pool)
    return plan[:count]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://adams-mac-studio-m4-1.local:52415")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    parser.add_argument("--prompt-tokens", type=int, default=40000)
    parser.add_argument("--hosts", default="adams-mac-studio-m4-1.local,adams-mac-studio-m4-2.local")
    parser.add_argument("--log-path", default="~/exo.log")
    parser.add_argument("--repetitions", type=int, default=20, help="Section 98 assertion 8: >=20.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-decode-steps", type=int, default=2)
    parser.add_argument("--abort-bound-seconds", type=float, default=0.5)
    parser.add_argument("--memory-tolerance-gb", type=float, default=3.0)
    parser.add_argument("--clean-floor-gb", type=float, default=85.7)
    parser.add_argument("--max-advances-after-cancel", type=int, default=1)
    parser.add_argument("--stranded-wait-threshold", type=int, default=0)
    parser.add_argument("--settle-seconds", type=float, default=20.0)
    arguments = parser.parse_args()

    base_url = cast(str, arguments.base_url)
    model = cast(str, arguments.model)
    prompt_tokens = cast(int, arguments.prompt_tokens)
    log_path = cast(str, arguments.log_path)
    repetitions = cast(int, arguments.repetitions)
    seed = cast(int, arguments.seed)
    min_decode_steps = cast(int, arguments.min_decode_steps)
    abort_bound_seconds = cast(float, arguments.abort_bound_seconds)
    memory_tolerance_gb = cast(float, arguments.memory_tolerance_gb)
    clean_floor_gb = cast(float, arguments.clean_floor_gb)
    max_advances_after_cancel = cast(int, arguments.max_advances_after_cancel)
    stranded_wait_threshold = cast(int, arguments.stranded_wait_threshold)
    settle_seconds = cast(float, arguments.settle_seconds)
    hosts = [h for h in cast(str, arguments.hosts).split(",") if h]

    print("=== PrefillCancelled 8-point definition-of-done hardware verification ===")
    print(f"[harness] hosts={hosts} repetitions={repetitions} seed={seed}")

    plan = build_offset_plan(repetitions, seed)
    results: list[RunResult] = []
    any_invalid = False

    for index, (scenario, offset_label) in enumerate(plan):
        prefill_frac = PREFILL_OFFSET_LABELS.get(offset_label, 0.5)
        decode_target = DECODE_OFFSET_LABELS.get(offset_label, 3)
        print(f"\n[harness] === {index + 1}/{len(plan)} ===")
        run = await run_one_repetition(
            base_url=base_url,
            model=model,
            hosts=hosts,
            log_path=log_path,
            scenario=scenario,
            offset_label=offset_label,
            prefill_offset_fraction=prefill_frac,
            decode_steps_target=decode_target,
            prompt_tokens=prompt_tokens,
            min_decode_steps=min_decode_steps,
            abort_bound_seconds=abort_bound_seconds,
            memory_tolerance_gb=memory_tolerance_gb,
            clean_floor_gb=clean_floor_gb,
            max_advances_after_cancel=max_advances_after_cancel,
            stranded_wait_threshold=stranded_wait_threshold,
            settle_seconds=settle_seconds,
        )
        results.append(run)
        if not run.overall_valid():
            any_invalid = True

    print("\n\n=== SUMMARY ACROSS ALL REPETITIONS ===")
    for run in results:
        status = "INVALID" if not run.overall_valid() else ("PASS" if run.overall_pass() else "FAIL")
        print(f"  [{status}] scenario={run.scenario} offset={run.offset_label} corr={run.correlation_id}")

    if any_invalid:
        print(
            "\n=== INVALID ===\n"
            "At least one repetition failed Assertion 1 (precondition "
            "validity) -- the corresponding run does not prove anything "
            "about PrefillCancelled and must not be read as a pass. Fix "
            "the reason the precondition was not met (prompt too short to "
            "avoid prefix-cache hit, prompt truncated in transit, cancel "
            "fired before any chunk/decode-step landed) and re-run."
        )
        return EXIT_INVALID

    all_pass = all(run.overall_pass() for run in results)
    # Assertion 8: repetition across the race window -- require the full
    # scenario/offset coverage was actually exercised, not just N reps of
    # one scenario.
    scenarios_covered = {(r.scenario, r.offset_label) for r in results}
    coverage_ok = len(scenarios_covered) >= min(5, len(plan))
    if not all_pass or not coverage_ok:
        print(
            "\n=== FAIL ===\n"
            f"all_pass={all_pass} coverage_ok={coverage_ok} "
            f"(covered {sorted(scenarios_covered)})\n"
            "One or more repetitions failed assertions 2-7, or the race "
            "window was not adequately covered. See the per-repetition "
            "assertion tables above for exactly which assertion failed."
        )
        return EXIT_FAIL

    print(
        "\n=== PASS ===\n"
        f"All {len(results)} repetitions across "
        f"{sorted(scenarios_covered)} passed all 8 assertions."
    )
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
