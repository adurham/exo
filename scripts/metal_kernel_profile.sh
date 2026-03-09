#!/bin/bash
# Profile Metal GPU kernels during decode to get per-shader timing breakdown.
#
# Usage: ./scripts/metal_kernel_profile.sh [API_HOST] [MODEL_ID] [CONTEXT_SIZE]
#
# Profiles each Mac Studio SEQUENTIALLY (one per request) to avoid deadlocking
# TP all-reduce. Metal System Trace adds GPU instrumentation that can
# desynchronize distributed collective ops when applied to both nodes at once.
#
# Requirements:
#   - xctrace available on both Studio nodes
#   - exo cluster running
#
# Output: tmp/metal_profile_<timestamp>/
#   - <node>_gpu_utilization.txt (GPU Active/Idle analysis, CPU-GPU gaps)
#   - <node>_kernel_summary.txt (per-shader breakdown, if Shader Timeline enabled)
#   - comparison.txt (side-by-side kernel breakdown)

set -euo pipefail

API_HOST="${1:-192.168.86.201:52415}"
MODEL_ID="${2:-mlx-community/MiniMax-M2.5-5bit}"
CONTEXT_SIZE="${3:-10000}"

NODES=("macstudio-m4-1" "macstudio-m4-2")
MAX_TRACE_DURATION=600
DECODE_TOKENS=128

LABEL="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="tmp/metal_profile_${LABEL}"
mkdir -p "$OUT_DIR"

echo "=== Metal Kernel Profiler ==="
echo "Nodes:    ${NODES[*]} (profiled sequentially)"
echo "Context:  ~${CONTEXT_SIZE} tokens"
echo "Decode:   ${DECODE_TOKENS} tokens"
echo "Output:   $OUT_DIR/"
echo ""

# --- Generate prompt ---
generate_prompt() {
    local target_tokens=$1
    local chars=$((target_tokens * 4))
    local base="The quick brown fox jumps over the lazy dog. "
    local prompt=""
    while [ ${#prompt} -lt $chars ]; do
        prompt="${prompt}${base}"
    done
    echo "$prompt"
}

PROMPT=$(generate_prompt "$CONTEXT_SIZE")

# --- Find the GPU runner process on a node ---
# The runner is a multiprocessing spawn subprocess, not named 'exo.worker.runner'.
find_runner_pid() {
    local node=$1
    # The runner subprocess is the one using the most CPU (it runs GPU inference).
    # It shows up as: python -c "from multiprocessing.spawn import spawn_main; ..."
    ssh "$node" "pgrep -f 'multiprocessing.spawn' | head -1" 2>/dev/null || true
}

# --- Profile a single node ---
profile_node() {
    local node=$1
    echo ""
    echo "=== Profiling $node ==="

    # Find runner PID
    local runner_pid
    runner_pid=$(find_runner_pid "$node")
    local attach_flag
    if [ -z "$runner_pid" ]; then
        echo "  WARNING: No runner process found. Tracing all processes."
        attach_flag="--all-processes"
    else
        echo "  Found runner PID: $runner_pid"
        attach_flag="--attach $runner_pid"
    fi

    # Start trace (background)
    echo "  Starting Metal System Trace..."
    ssh "$node" "rm -rf /tmp/metal_trace.trace; xctrace record \
        --template 'Metal System Trace' \
        $attach_flag \
        --time-limit ${MAX_TRACE_DURATION}s \
        --output /tmp/metal_trace.trace \
        2>/dev/null" &
    local trace_pid=$!

    sleep 2  # let xctrace start

    # Fire request
    echo "  Sending ${CONTEXT_SIZE}-token request..."
    local t_start
    t_start=$(date +%s)

    local response
    response=$(curl -sf --max-time "$MAX_TRACE_DURATION" "http://$API_HOST/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg model "$MODEL_ID" \
            --arg prompt "$PROMPT" \
            --argjson max_tokens "$DECODE_TOKENS" \
            '{
                model: $model,
                messages: [{role: "user", content: ("Repeat the word hello exactly 128 times: " + $prompt)}],
                max_tokens: $max_tokens,
                temperature: 0
            }'
        )" 2>&1 || echo '{"error": "request failed"}')

    local t_end
    t_end=$(date +%s)
    echo "  Request complete in $(( t_end - t_start ))s."
    echo "$response" | jq -r '.usage // empty' 2>/dev/null || true
    echo "$response" > "$OUT_DIR/${node}_response.json"

    # Stop trace
    echo "  Stopping trace..."
    ssh "$node" "pkill -INT -f 'xctrace record' 2>/dev/null || true"
    wait "$trace_pid" 2>/dev/null || true

    # Pull trace
    echo "  Copying trace..."
    scp -r "$node:/tmp/metal_trace.trace" "$OUT_DIR/${node}_metal.trace" 2>/dev/null || {
        echo "  ERROR: Failed to copy trace from $node"
        return 1
    }

    # Export data
    echo "  Exporting..."
    xctrace export \
        --input "$OUT_DIR/${node}_metal.trace" \
        --xpath '//table[@schema="metal-shader-profiler-intervals"]' \
        > "$OUT_DIR/${node}_shaders.xml" 2>&1 || echo "  No shader data"

    xctrace export \
        --input "$OUT_DIR/${node}_metal.trace" \
        --xpath '//table[@schema="metal-gpu-state-intervals"]' \
        > "$OUT_DIR/${node}_gpu_state.xml" 2>&1 || echo "  No GPU state data"

    xctrace export \
        --input "$OUT_DIR/${node}_metal.trace" \
        --xpath '//table[@schema="gpu-performance-state-intervals"]' \
        > "$OUT_DIR/${node}_gpu_pstate.xml" 2>&1 || echo "  No P-state data"

    # Parse shader data (may be empty if Shader Timeline disabled)
    parse_shaders "$OUT_DIR/${node}_shaders.xml" "$OUT_DIR/${node}_kernel_summary.txt" "$node"

    # Parse GPU state intervals for utilization analysis
    parse_gpu_state "$OUT_DIR/${node}_gpu_state.xml" "$OUT_DIR/${node}_gpu_utilization.txt" "$node"

    echo "  Done with $node."
}

# --- Parse GPU state intervals for utilization analysis ---
parse_gpu_state() {
    local input_xml=$1
    local output_txt=$2
    local node_name=$3

python3 - "$input_xml" "$output_txt" "$node_name" << 'PYEOF'
import sys
import xml.etree.ElementTree as ET

input_file = sys.argv[1]
output_file = sys.argv[2]
node_name = sys.argv[3]

def parse_duration(s):
    s = s.strip()
    if not s: return 0
    if 'µs' in s: return float(s.replace('µs','').strip()) * 1000
    if 'ms' in s: return float(s.replace('ms','').strip()) * 1e6
    if 'ns' in s: return float(s.replace('ns','').strip())
    if 's' in s: return float(s.replace('s','').strip()) * 1e9
    try: return float(s)
    except: return 0

try:
    tree = ET.parse(input_file)
    root = tree.getroot()
except Exception as e:
    print(f"  {node_name}: Could not parse GPU state XML: {e}")
    with open(output_file, 'w') as f:
        f.write(f"Parse error: {e}\n")
    sys.exit(0)

active_ns = 0
idle_ns = 0
active_count = 0
idle_count = 0
idle_gaps = []
compute_durations = []

for row in root.iter('row'):
    cols = {col.tag: col.get('fmt', col.text or '') for col in row}
    state = cols.get('gpu-state', '').strip()
    dur_ns = parse_duration(cols.get('duration', ''))
    label = cols.get('formatted-label', '').strip()

    if state == 'Active' or (not state and dur_ns > 0 and label != 'Idle'):
        active_ns += dur_ns
        active_count += 1
        compute_durations.append(dur_ns)
    elif state == 'Idle' or label == 'Idle':
        idle_ns += dur_ns
        idle_count += 1
        idle_gaps.append(dur_ns)

total_ns = active_ns + idle_ns

lines = [f"=== {node_name} GPU Utilization Analysis ===", ""]

if total_ns == 0:
    lines.append("No GPU state data captured.")
    output = "\n".join(lines)
    print(output)
    with open(output_file, 'w') as f:
        f.write(output + "\n")
    sys.exit(0)

lines.append(f"Total tracked time:  {total_ns / 1e6:.1f} ms")
lines.append(f"GPU Active:          {active_ns / 1e6:.1f} ms ({100*active_ns/total_ns:.1f}%)")
lines.append(f"GPU Idle:            {idle_ns / 1e6:.1f} ms ({100*idle_ns/total_ns:.1f}%)")
lines.append(f"Active intervals:    {active_count}")
lines.append(f"Idle intervals:      {idle_count}")

if compute_durations:
    compute_durations.sort()
    n = len(compute_durations)
    lines.append("")
    lines.append("Compute dispatch durations:")
    lines.append(f"  Median:  {compute_durations[n//2] / 1000:.1f} µs")
    lines.append(f"  Mean:    {sum(compute_durations)/n / 1000:.1f} µs")
    lines.append(f"  Min:     {compute_durations[0] / 1000:.1f} µs")
    lines.append(f"  Max:     {compute_durations[-1] / 1000:.1f} µs")
    lines.append(f"  P95:     {compute_durations[int(n*0.95)] / 1000:.1f} µs")
    lines.append(f"  P99:     {compute_durations[int(n*0.99)] / 1000:.1f} µs")

if idle_gaps:
    idle_gaps.sort()
    n = len(idle_gaps)
    lines.append("")
    lines.append("CPU-GPU submission gaps:")
    lines.append(f"  Median:  {idle_gaps[n//2] / 1000:.1f} µs")
    lines.append(f"  Mean:    {sum(idle_gaps)/n / 1000:.1f} µs")
    lines.append(f"  Min:     {idle_gaps[0] / 1000:.1f} µs")
    lines.append(f"  Max:     {idle_gaps[-1] / 1000:.1f} µs")
    lines.append(f"  P95:     {idle_gaps[int(n*0.95)] / 1000:.1f} µs")
    lines.append(f"  P99:     {idle_gaps[int(n*0.99)] / 1000:.1f} µs")
    big = [g for g in idle_gaps if g > 100_000]
    very_big = [g for g in idle_gaps if g > 1_000_000]
    lines.append(f"  Gaps > 100µs: {len(big)} ({100*len(big)/n:.1f}%)")
    lines.append(f"  Gaps > 1ms:   {len(very_big)} ({100*len(very_big)/n:.1f}%)")

output = "\n".join(lines)
print(output)
with open(output_file, 'w') as f:
    f.write(output + "\n")

PYEOF
}

# --- Parse shader XML into summary ---
parse_shaders() {
    local input_xml=$1
    local output_txt=$2
    local node_name=$3

python3 - "$input_xml" "$output_txt" "$node_name" << 'PYEOF'
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

input_file = sys.argv[1]
output_file = sys.argv[2]
node_name = sys.argv[3]

try:
    tree = ET.parse(input_file)
    root = tree.getroot()
except Exception as e:
    print(f"  {node_name}: Could not parse shader XML: {e}")
    with open(output_file, 'w') as f:
        f.write(f"Parse error: {e}\n")
    sys.exit(0)

shader_stats = defaultdict(lambda: {"count": 0, "total_ns": 0, "min_ns": float('inf'), "max_ns": 0})

for row in root.iter('row'):
    cols = {}
    for col in row:
        cols[col.tag] = col.get('fmt', col.text or '')
    name = cols.get('name', cols.get('label', cols.get('pso-label', 'unknown')))
    duration_str = cols.get('duration', '0')

    try:
        if 'µs' in duration_str:
            duration_ns = float(duration_str.replace('µs', '').strip()) * 1000
        elif 'ms' in duration_str:
            duration_ns = float(duration_str.replace('ms', '').strip()) * 1_000_000
        elif 'ns' in duration_str:
            duration_ns = float(duration_str.replace('ns', '').strip())
        elif 's' in duration_str:
            duration_ns = float(duration_str.replace('s', '').strip()) * 1_000_000_000
        else:
            duration_ns = float(duration_str)
    except (ValueError, TypeError):
        continue

    if not name or name == 'unknown':
        continue

    s = shader_stats[name]
    s["count"] += 1
    s["total_ns"] += duration_ns
    s["min_ns"] = min(s["min_ns"], duration_ns)
    s["max_ns"] = max(s["max_ns"], duration_ns)

if not shader_stats:
    print(f"  {node_name}: No shader timing data found.")
    with open(output_file, 'w') as f:
        f.write("No shader timing data found.\n")
    sys.exit(0)

sorted_shaders = sorted(shader_stats.items(), key=lambda x: x[1]["total_ns"], reverse=True)
total_gpu_ns = sum(s["total_ns"] for _, s in sorted_shaders)

lines = []
lines.append(f"=== {node_name} Kernel Breakdown ===")
lines.append("")
lines.append(f"{'Kernel':<60} {'Count':>8} {'Total ms':>10} {'%GPU':>7} {'Avg µs':>10} {'Min µs':>10} {'Max µs':>10}")
lines.append("-" * 117)

for name, stats in sorted_shaders:
    total_ms = stats["total_ns"] / 1_000_000
    pct = 100 * stats["total_ns"] / total_gpu_ns if total_gpu_ns > 0 else 0
    avg_us = (stats["total_ns"] / stats["count"]) / 1000
    min_us = stats["min_ns"] / 1000
    max_us = stats["max_ns"] / 1000
    display_name = name[:58] + ".." if len(name) > 60 else name
    lines.append(f"{display_name:<60} {stats['count']:>8} {total_ms:>10.2f} {pct:>6.1f}% {avg_us:>10.1f} {min_us:>10.1f} {max_us:>10.1f}")

lines.append("-" * 117)
lines.append(f"{'TOTAL':<60} {sum(s['count'] for _, s in sorted_shaders):>8} {total_gpu_ns / 1_000_000:>10.2f}")
lines.append("")

lines.append("Top 5 kernels by GPU time:")
for i, (name, stats) in enumerate(sorted_shaders[:5]):
    pct = 100 * stats["total_ns"] / total_gpu_ns if total_gpu_ns > 0 else 0
    lines.append(f"  {i+1}. {name[:70]} — {pct:.1f}%")

output = "\n".join(lines)
print(output)
with open(output_file, 'w') as f:
    f.write(output + "\n")

PYEOF
}

# --- Profile each node sequentially (one request per node) ---
for node in "${NODES[@]}"; do
    profile_node "$node"
    sleep 3  # cooldown between profiles
done

# --- Generate side-by-side comparison ---
echo ""
echo "=== Node Comparison ==="
python3 - "$OUT_DIR/${NODES[0]}_shaders.xml" "$OUT_DIR/${NODES[1]}_shaders.xml" "$OUT_DIR/comparison.txt" "${NODES[0]}" "${NODES[1]}" << 'PYEOF'
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_xml(path):
    stats = defaultdict(lambda: {"count": 0, "total_ns": 0})
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        return stats
    for row in root.iter('row'):
        cols = {}
        for col in row:
            cols[col.tag] = col.get('fmt', col.text or '')
        name = cols.get('name', cols.get('label', 'unknown'))
        duration_str = cols.get('duration', '0')
        try:
            if 'µs' in duration_str:
                ns = float(duration_str.replace('µs', '').strip()) * 1000
            elif 'ms' in duration_str:
                ns = float(duration_str.replace('ms', '').strip()) * 1e6
            elif 'ns' in duration_str:
                ns = float(duration_str.replace('ns', '').strip())
            elif 's' in duration_str:
                ns = float(duration_str.replace('s', '').strip()) * 1e9
            else:
                ns = float(duration_str)
        except (ValueError, TypeError):
            continue
        if name and name != 'unknown':
            stats[name]["count"] += 1
            stats[name]["total_ns"] += ns
    return stats

s1 = parse_xml(sys.argv[1])
s2 = parse_xml(sys.argv[2])
output_file = sys.argv[3]
n1, n2 = sys.argv[4], sys.argv[5]

all_kernels = sorted(set(list(s1.keys()) + list(s2.keys())),
                     key=lambda k: max(s1[k]["total_ns"], s2[k]["total_ns"]),
                     reverse=True)

total1 = sum(s["total_ns"] for s in s1.values())
total2 = sum(s["total_ns"] for s in s2.values())

lines = []
lines.append(f"{'Kernel':<50} {n1+' ms':>12} {n1+' %':>8} {n2+' ms':>12} {n2+' %':>8} {'Diff':>8}")
lines.append("-" * 100)

for k in all_kernels:
    ms1 = s1[k]["total_ns"] / 1e6
    ms2 = s2[k]["total_ns"] / 1e6
    p1 = 100 * s1[k]["total_ns"] / total1 if total1 else 0
    p2 = 100 * s2[k]["total_ns"] / total2 if total2 else 0
    diff_pct = ((ms2 - ms1) / ms1 * 100) if ms1 > 0 else (100 if ms2 > 0 else 0)
    dk = k[:48] + ".." if len(k) > 50 else k
    lines.append(f"{dk:<50} {ms1:>12.2f} {p1:>7.1f}% {ms2:>12.2f} {p2:>7.1f}% {diff_pct:>+7.1f}%")

lines.append("-" * 100)
lines.append(f"{'TOTAL':<50} {total1/1e6:>12.2f} {'':>8} {total2/1e6:>12.2f}")

if total1 > 0 and total2 > 0:
    skew = abs(total1 - total2) / max(total1, total2) * 100
    lines.append("")
    if skew < 5:
        lines.append(f"Nodes are balanced (skew: {skew:.1f}%)")
    else:
        slower = n1 if total1 > total2 else n2
        lines.append(f"WARNING: {slower} is {skew:.1f}% slower — potential TP imbalance")
        lines.append("(Note: profiling overhead may account for some skew)")

output = "\n".join(lines)
print(output)
with open(output_file, 'w') as f:
    f.write(output + "\n")

PYEOF

echo ""
echo "=== Metal profile complete ==="
echo "Output: $OUT_DIR/"
echo ""
echo "Files:"
ls -lh "$OUT_DIR/" | grep -v "^total"
echo ""
echo "Per-node analysis:"
echo "  cat $OUT_DIR/macstudio-m4-1_gpu_utilization.txt"
echo "  cat $OUT_DIR/macstudio-m4-2_gpu_utilization.txt"
echo "  cat $OUT_DIR/macstudio-m4-1_kernel_summary.txt"
echo "  cat $OUT_DIR/macstudio-m4-2_kernel_summary.txt"
echo "  cat $OUT_DIR/comparison.txt"
