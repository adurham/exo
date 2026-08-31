// Reviewable kernel source (Metal), from item2_phaseB.py
// HEADER:

constant uint L_ = 1024;
static inline ushort dsv4_topk_key(bfloat16_t v) {
    ushort u = as_type<ushort>(v);
    return (u & 0x8000) ? ushort(~u) : ushort(u | 0x8000);
}

// SOURCE:

uint l = threadgroup_position_in_grid.y;
uint b = threadgroup_position_in_grid.z;
uint tid = thread_position_in_threadgroup.x;
uint simd_gid = tid / 32;
uint simd_lid = tid % 32;

const uint P = params[0];
const uint K = params[1];
constexpr uint T_ = 1024;

const size_t row = (size_t(b) * L_ + l) * P;
const size_t out_row = (size_t(b) * L_ + l) * K;

threadgroup atomic_uint hist[256];
threadgroup uint scan_buf[32];
threadgroup uint bcast[8];

// ---- phase 1: high-byte histogram (strided read pass #1) ----
for (uint i = tid; i < 256; i += T_) {
    atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < P; i += T_) {
    ushort key = dsv4_topk_key(scores[row + i]);
    atomic_fetch_add_explicit(&hist[key >> 8], 1u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    uint above = 0;
    uint hb = 0;
    for (int bin = 255; bin >= 0; bin--) {
        uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
        if (above + c >= K) { hb = uint(bin); break; }
        above += c;
    }
    bcast[0] = hb;
    bcast[1] = above;  // count of keys with high byte > hb
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint hb = bcast[0];
const uint above_hb = bcast[1];

// ---- phase 2: low-byte histogram within boundary high-byte bin (read pass #2) ----
for (uint i = tid; i < 256; i += T_) {
    atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < P; i += T_) {
    ushort key = dsv4_topk_key(scores[row + i]);
    if (uint(key >> 8) == hb) {
        atomic_fetch_add_explicit(&hist[key & 0xFF], 1u, memory_order_relaxed);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    uint above = 0;
    uint lb = 0;
    for (int bin = 255; bin >= 0; bin--) {
        uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
        if (above_hb + above + c >= K) { lb = uint(bin); break; }
        above += c;
    }
    bcast[2] = lb;
    bcast[3] = above_hb + above;      // n_gt: keys strictly > threshold
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const ushort thresh = ushort((hb << 8) | bcast[2]);
const uint n_gt = bcast[3];
const uint n_eq_need = K - n_gt;

// ---- phase 3: deterministic index-ordered compaction (read pass #3) ----
// thread t owns the contiguous chunk [t*chunk, min((t+1)*chunk, P))
const uint chunk = (P + T_ - 1) / T_;
const uint lo = min(tid * chunk, P);
const uint hi = min(lo + chunk, P);

uint my_gt = 0, my_eq = 0;
for (uint i = lo; i < hi; i++) {
    ushort key = dsv4_topk_key(scores[row + i]);
    if (key > thresh) my_gt++;
    else if (key == thresh) my_eq++;
}

// two-level exclusive scans over the 1024 per-thread counts
uint gt_pre, eq_pre;
{
    uint lane_ex = simd_prefix_exclusive_sum(my_gt);
    uint sg_tot = simd_sum(my_gt);
    if (simd_lid == 31) scan_buf[simd_gid] = sg_tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        uint v = scan_buf[simd_lid];
        scan_buf[simd_lid] = simd_prefix_exclusive_sum(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gt_pre = scan_buf[simd_gid] + lane_ex;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
{
    uint lane_ex = simd_prefix_exclusive_sum(my_eq);
    uint sg_tot = simd_sum(my_eq);
    if (simd_lid == 31) scan_buf[simd_gid] = sg_tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        uint v = scan_buf[simd_lid];
        scan_buf[simd_lid] = simd_prefix_exclusive_sum(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    eq_pre = scan_buf[simd_gid] + lane_ex;
}

uint gt_pos = gt_pre;
uint eq_rank = eq_pre;
for (uint i = lo; i < hi; i++) {
    ushort key = dsv4_topk_key(scores[row + i]);
    if (key > thresh) {
        out_idx[out_row + gt_pos] = i;
        gt_pos++;
    } else if (key == thresh) {
        if (eq_rank < n_eq_need) {
            out_idx[out_row + n_gt + eq_rank] = i;
        }
        eq_rank++;
    }
}
