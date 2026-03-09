#!/usr/bin/env python3
"""Analyze benchmark traces from benchmark_with_trace.sh.

Usage: python3 scripts/analyze_trace.py tmp/trace_<label>/
"""

import json
import sys
from pathlib import Path
from statistics import mean, median, stdev


def load_macmon(path: Path) -> list[dict]:
    """Load macmon JSONL file."""
    samples = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return samples


def analyze_macmon(samples: list[dict], node: str, ctx: str) -> dict:
    """Analyze macmon samples for a single node/context."""
    if not samples:
        return {}

    gpu_usages = [s["gpu_usage"][1] * 100 for s in samples]  # fraction -> percent
    gpu_freqs = [s["gpu_usage"][0] for s in samples]
    gpu_powers = [s["gpu_power"] for s in samples]
    gpu_temps = [s["temp"]["gpu_temp_avg"] for s in samples]
    cpu_temps = [s["temp"]["cpu_temp_avg"] for s in samples]
    sys_powers = [s["sys_power"] for s in samples]
    ram_usages = [s["memory"]["ram_usage"] / (1024**3) for s in samples]
    all_powers = [s["all_power"] for s in samples]

    # GPU max freq from SoC info (first sample with soc)
    max_gpu_freq = None
    for s in samples:
        if "soc" in s and s["soc"]:
            max_gpu_freq = max(s["soc"]["gpu_freqs"])
            break

    return {
        "node": node,
        "context": ctx,
        "samples": len(samples),
        "gpu_usage_pct": {"mean": mean(gpu_usages), "max": max(gpu_usages), "min": min(gpu_usages)},
        "gpu_freq_mhz": {"mean": mean(gpu_freqs), "max": max(gpu_freqs), "min": min(gpu_freqs), "max_possible": max_gpu_freq},
        "gpu_power_w": {"mean": mean(gpu_powers), "max": max(gpu_powers)},
        "gpu_temp_c": {"mean": mean(gpu_temps), "max": max(gpu_temps), "min": min(gpu_temps)},
        "cpu_temp_c": {"mean": mean(cpu_temps), "max": max(cpu_temps)},
        "sys_power_w": {"mean": mean(sys_powers), "max": max(sys_powers)},
        "ram_usage_gb": {"mean": mean(ram_usages), "max": max(ram_usages)},
        "soc_power_w": {"mean": mean(all_powers), "max": max(all_powers)},
    }


def analyze_decode_timing(log_path: Path) -> dict[str, list[float]]:
    """Extract per-token decode timing from [decode] stderr lines."""
    contexts: dict[str, list[float]] = {}
    current_ctx = "unknown"
    for line in log_path.read_text().splitlines():
        # Match prefill start to know context size
        if "Prefilling" in line and "tokens" in line:
            import re
            m = re.search(r"Prefilling (\d+) tokens", line)
            if m:
                current_ctx = m.group(1)
        # Match [decode] lines
        if "[decode]" in line:
            import re
            m = re.search(r"\[decode\] n=(\d+) ([\d.]+)ms", line)
            if m:
                ms = float(m.group(2))
                contexts.setdefault(current_ctx, []).append(ms)
    return contexts


def analyze_generation_complete(log_path: Path) -> list[dict]:
    """Extract Generation complete summaries."""
    import re
    results = []
    for line in log_path.read_text().splitlines():
        if "Generation complete:" not in line:
            continue
        m = re.search(
            r"prefill (\d+) tokens @ ([\d.]+) tok/s, decoded (\d+) tokens @ ([\d.]+) tok/s in ([\d.]+)ms \(([\d.]+)ms/tok\)",
            line,
        )
        if m:
            results.append({
                "prefill_tokens": int(m.group(1)),
                "prefill_toks": float(m.group(2)),
                "decode_tokens": int(m.group(3)),
                "decode_toks": float(m.group(4)),
                "decode_ms": float(m.group(5)),
                "ms_per_tok": float(m.group(6)),
            })
    return results


def print_report(trace_dir: Path):
    """Print full analysis report."""
    nodes = ["macstudio-m4-1", "macstudio-m4-2"]
    contexts = ["1000", "10000", "30000", "50000", "70000", "100000"]
    ctx_labels = ["~1K", "~10K", "~30K", "~50K", "~70K", "~100K"]

    # --- Hardware trace analysis ---
    print("=" * 100)
    print("HARDWARE TRACE ANALYSIS")
    print("=" * 100)

    for node in nodes:
        print(f"\n{'─' * 100}")
        print(f"  {node}")
        print(f"{'─' * 100}")
        print(f"{'Context':<8} {'GPU%':>8} {'GPU MHz':>10} {'MaxMHz':>8} {'Throttle?':>10} {'GPU W':>8} {'GPU C':>8} {'CPU C':>8} {'SysW':>8} {'RAM GB':>8}")
        print("-" * 90)

        for ctx, label in zip(contexts, ctx_labels):
            macmon_path = trace_dir / f"{node}_macmon_{ctx}.jsonl"
            if not macmon_path.exists():
                print(f"{label:<8} {'(no data)':>8}")
                continue

            samples = load_macmon(macmon_path)
            stats = analyze_macmon(samples, node, ctx)
            if not stats:
                print(f"{label:<8} {'(empty)':>8}")
                continue

            gpu = stats["gpu_usage_pct"]
            freq = stats["gpu_freq_mhz"]
            throttle = ""
            if freq["max_possible"] and freq["max"] < freq["max_possible"] * 0.95:
                throttle = f"YES ({freq['max']}/{freq['max_possible']})"
            elif freq["max_possible"]:
                throttle = "no"

            print(
                f"{label:<8} "
                f"{gpu['mean']:>7.1f} "
                f"{freq['mean']:>9.0f} "
                f"{freq['max_possible'] or 0:>7.0f} "
                f"{throttle:>10} "
                f"{stats['gpu_power_w']['mean']:>7.1f} "
                f"{stats['gpu_temp_c']['max']:>7.1f} "
                f"{stats['cpu_temp_c']['max']:>7.1f} "
                f"{stats['sys_power_w']['mean']:>7.1f} "
                f"{stats['ram_usage_gb']['mean']:>7.1f}"
            )

    # --- Decode timing analysis ---
    print()
    print("=" * 100)
    print("DECODE TIMING ANALYSIS (per-token wall-clock from [decode] stderr)")
    print("=" * 100)

    for node in nodes:
        log_path = trace_dir / f"{node}.log"
        if not log_path.exists():
            continue

        decode = analyze_decode_timing(log_path)
        if not decode:
            print(f"\n  {node}: no [decode] timing data found")
            continue

        print(f"\n  {node}:")
        print(f"  {'Context':<10} {'Mean ms':>10} {'Median':>10} {'Stdev':>8} {'Min':>8} {'Max':>8} {'tok/s':>8}")
        print(f"  {'-' * 65}")

        for ctx_tok, times in sorted(decode.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            if len(times) < 2:
                continue
            m = mean(times)
            print(
                f"  {ctx_tok:<10} "
                f"{m:>9.1f} "
                f"{median(times):>9.1f} "
                f"{stdev(times):>7.2f} "
                f"{min(times):>7.1f} "
                f"{max(times):>7.1f} "
                f"{1000 / m:>7.1f}"
            )

    # --- Generation complete summary ---
    print()
    print("=" * 100)
    print("GENERATION COMPLETE SUMMARY")
    print("=" * 100)

    for node in nodes:
        log_path = trace_dir / f"{node}.log"
        if not log_path.exists():
            continue

        results = analyze_generation_complete(log_path)
        if not results:
            continue

        print(f"\n  {node}:")
        print(f"  {'Prefill':>8} {'PF tok/s':>10} {'Decode':>8} {'Dec tok/s':>10} {'ms/tok':>8}")
        print(f"  {'-' * 50}")
        for r in results:
            # Skip warmup (very small prefill)
            if r["prefill_tokens"] < 100:
                continue
            print(
                f"  {r['prefill_tokens']:>8} "
                f"{r['prefill_toks']:>9.1f} "
                f"{r['decode_tokens']:>8} "
                f"{r['decode_toks']:>9.1f} "
                f"{r['ms_per_tok']:>7.2f}"
            )

    # --- Thermal correlation ---
    print()
    print("=" * 100)
    print("THERMAL CORRELATION: Does GPU temp affect decode speed?")
    print("=" * 100)

    for node in nodes:
        log_path = trace_dir / f"{node}.log"
        if not log_path.exists():
            continue

        results = analyze_generation_complete(log_path)
        temps = []
        for ctx in contexts:
            macmon_path = trace_dir / f"{node}_macmon_{ctx}.jsonl"
            if macmon_path.exists():
                samples = load_macmon(macmon_path)
                if samples:
                    temps.append(max(s["temp"]["gpu_temp_avg"] for s in samples))
                else:
                    temps.append(None)
            else:
                temps.append(None)

        gen_results = [r for r in results if r["prefill_tokens"] >= 100]
        if len(gen_results) == len(temps):
            print(f"\n  {node}:")
            print(f"  {'Context':<8} {'GPU Max C':>10} {'Decode tok/s':>13} {'ms/tok':>8}")
            print(f"  {'-' * 45}")
            for label, t, r in zip(ctx_labels, temps, gen_results):
                t_str = f"{t:.1f}" if t else "?"
                print(f"  {label:<8} {t_str:>10} {r['decode_toks']:>12.1f} {r['ms_per_tok']:>7.2f}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace_dir>")
        sys.exit(1)

    trace_dir = Path(sys.argv[1])
    if not trace_dir.exists():
        print(f"Error: {trace_dir} does not exist")
        sys.exit(1)

    print_report(trace_dir)
