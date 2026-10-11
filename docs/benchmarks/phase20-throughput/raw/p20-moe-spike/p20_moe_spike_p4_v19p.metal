
#define PACKED_U32 24
#define K_BITS 3

    uint b = threadgroup_position_in_grid.y;
    if (b >= nbr[0]) {
        return;
    }
    uint colb = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;     // 128 threads
    uint lane = tid & 31u;
    uint sgid = tid >> 5u;
    uint sgr = sgid >> 1u;                            // 2x2 simdgroup grid
    uint sgc = sgid & 1u;

    uint in_tiles = dims[0];
    uint tiles_per_e = dims[1];
    uint src_tiles = dims[2];
    uint tn_base = dims[3];
    uint out_e = dims[4];
    uint nb_max = dims[5];
    uint in_features = in_tiles * 16u;

    uint e = blk_tab[b];
    uint row0 = blk_tab[nb_max + b];
    uint blk_len = blk_tab[2u * nb_max + b];
    uint tn0 = tn_base + e * tiles_per_e + colb * 4u;

    // decode-position precompute (fixed per thread; decode_full_t mapping)
    uint dcol = tid >> 3u;        // out col 0..15 within tile
    uint rp = tid & 7u;           // in row-pair 0..7
    uint i0_[2];
    uint i1_[2];
    uint sh_[2];
    for (uint j = 0u; j < 2u; j++) {
        uint src = inv_perm[(2u * rp + j) * 16u + dcol];
        uint t = src >> 1u;
        int b0 = int(t) * 2 * K_BITS + K_BITS - 16 + 256 * K_BITS;
        int b2 = b0 + K_BITS + 16;
        i0_[j] = uint(b0 / 32) % PACKED_U32;
        i1_[j] = uint((b2 - 1) / 32) % PACKED_U32;
        sh_[j] = uint(((b2 - 1) / 32 + 1) * 32 - b2) + ((src & 1u) ? 0u : K_BITS);
    }

    uint rlive = 0u;
    if (blk_len > sgr * 32u) {
        uint rrem = blk_len - sgr * 32u;
        rlive = (rrem > 32u) ? 4u : ((rrem + 7u) >> 3u);
    }
    threadgroup half wblk[2u * 32u * 72u];   // v19p ping-pong: 2 x (2 in-tiles x 4 out-tiles)
    threadgroup float csc[4u * 64u];

    simdgroup_float8x8 C[4][4];
    for (uint r = 0u; r < 4u; r++) {
        for (uint c = 0u; c < 4u; c++) {
            C[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const device half* xrow = xh + (ulong)(row0 + sgr * 32u) * in_features;
    uint stages = in_tiles >> 1u;             // in_tiles is even (>=32)
    // v19p (P4): decode(ks+1) into buffer (ks+1)&1 overlaps mma(ks)
    // on buffer ks&1; ONE barrier per stage. Accumulation order and
    // the guarded mma block are v19d's, so the result is 0-diff.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint it2 = 0u; it2 < 2u; it2++) {
        uint tk = it2;   // stage 0
        for (uint lt = 0u; lt < 4u; lt++) {
            const device uint* words =
                trellis + ((ulong)tk * src_tiles + tn0 + lt) * PACKED_U32;
            for (uint j = 0u; j < 2u; j++) {
                ulong merged =
                    ((ulong)words[i0_[j]] << 32) | (ulong)words[i1_[j]];
                uint cw = uint(merged >> sh_[j]) & 0xFFFFu;
        uint dq_cw = cw * 0x83DCD12Du;
        uint dq_t = (dq_cw & 0x00FF00FFu) + ((dq_cw >> 8u) & 0x00FF00FFu);
        uint dq_sum = 0x6400u + (dq_t & 0xFFFFu) + (dq_t >> 16u);
        half dq_h = as_type<half>(ushort(dq_sum & 0xFFFFu));
        half dq_k_inv = as_type<half>(ushort(0x1EEEu));
        half dq_k_bias = as_type<half>(ushort(0xC931u));
        float dq_val = float(dq_h * dq_k_inv + dq_k_bias);
                wblk[(0u) * 32u * 72u + (it2 * 16u + 2u * rp + j) * 72u + lt * 16u + dcol] =
                    half(dq_val);
            }
        }
    }
    for (uint ks = 0u; ks < stages; ks++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks + 1u < stages) {
            for (uint it2 = 0u; it2 < 2u; it2++) {
                uint tk = (ks + 1u) * 2u + it2;
                for (uint lt = 0u; lt < 4u; lt++) {
                    const device uint* words =
                        trellis + ((ulong)tk * src_tiles + tn0 + lt) * PACKED_U32;
                    for (uint j = 0u; j < 2u; j++) {
                        ulong merged =
                            ((ulong)words[i0_[j]] << 32) | (ulong)words[i1_[j]];
                        uint cw = uint(merged >> sh_[j]) & 0xFFFFu;
                uint dq_cw = cw * 0x83DCD12Du;
                uint dq_t = (dq_cw & 0x00FF00FFu) + ((dq_cw >> 8u) & 0x00FF00FFu);
                uint dq_sum = 0x6400u + (dq_t & 0xFFFFu) + (dq_t >> 16u);
                half dq_h = as_type<half>(ushort(dq_sum & 0xFFFFu));
                half dq_k_inv = as_type<half>(ushort(0x1EEEu));
                half dq_k_bias = as_type<half>(ushort(0xC931u));
                float dq_val = float(dq_h * dq_k_inv + dq_k_bias);
                        wblk[(((ks + 1u) & 1u)) * 32u * 72u + (it2 * 16u + 2u * rp + j) * 72u + lt * 16u + dcol] =
                            half(dq_val);
                    }
                }
            }
        }

        if (rlive == 4u) {
            for (uint k2 = 0u; k2 < 4u; k2++) {
            simdgroup_half8x8 bf0;
            simdgroup_half8x8 bf1;
            simdgroup_half8x8 bf2;
            simdgroup_half8x8 bf3;
            const threadgroup half* wrow =
                &wblk[((ks & 1u)) * 32u * 72u + k2 * 8u * 72u + sgc * 32u];
            simdgroup_load(bf0, wrow, 72u);
            simdgroup_load(bf1, wrow + 8u, 72u);
            simdgroup_load(bf2, wrow + 16u, 72u);
            simdgroup_load(bf3, wrow + 24u, 72u);
            for (uint r = 0u; r < 4u; r++) {
                simdgroup_half8x8 a;
                simdgroup_load(
                    a, xrow + (ulong)(r * 8u) * in_features + ks * 32u + k2 * 8u,
                    in_features);
                simdgroup_multiply_accumulate(C[r][0], a, bf0, C[r][0]);
                simdgroup_multiply_accumulate(C[r][1], a, bf1, C[r][1]);
                simdgroup_multiply_accumulate(C[r][2], a, bf2, C[r][2]);
                simdgroup_multiply_accumulate(C[r][3], a, bf3, C[r][3]);
            }
        }
        } else if (rlive > 0u) {
            if (rlive > 0u) {
                        for (uint k2 = 0u; k2 < 4u; k2++) {
                            simdgroup_half8x8 bf0;
                            simdgroup_half8x8 bf1;
                            simdgroup_half8x8 bf2;
                            simdgroup_half8x8 bf3;
                            const threadgroup half* wrow =
                                &wblk[((ks & 1u)) * 32u * 72u + k2 * 8u * 72u + sgc * 32u];
                            simdgroup_load(bf0, wrow, 72u);
                            simdgroup_load(bf1, wrow + 8u, 72u);
                            simdgroup_load(bf2, wrow + 16u, 72u);
                            simdgroup_load(bf3, wrow + 24u, 72u);
                            simdgroup_half8x8 a;
                            simdgroup_load(
                                a, xrow + (ulong)(0u * 8u) * in_features + ks * 32u + k2 * 8u,
                                in_features);
                            simdgroup_multiply_accumulate(C[0][0], a, bf0, C[0][0]);
                            simdgroup_multiply_accumulate(C[0][1], a, bf1, C[0][1]);
                            simdgroup_multiply_accumulate(C[0][2], a, bf2, C[0][2]);
                            simdgroup_multiply_accumulate(C[0][3], a, bf3, C[0][3]);
                        }
                    }
            if (rlive > 1u) {
                        for (uint k2 = 0u; k2 < 4u; k2++) {
                            simdgroup_half8x8 bf0;
                            simdgroup_half8x8 bf1;
                            simdgroup_half8x8 bf2;
                            simdgroup_half8x8 bf3;
                            const threadgroup half* wrow =
                                &wblk[((ks & 1u)) * 32u * 72u + k2 * 8u * 72u + sgc * 32u];
                            simdgroup_load(bf0, wrow, 72u);
                            simdgroup_load(bf1, wrow + 8u, 72u);
                            simdgroup_load(bf2, wrow + 16u, 72u);
                            simdgroup_load(bf3, wrow + 24u, 72u);
                            simdgroup_half8x8 a;
                            simdgroup_load(
                                a, xrow + (ulong)(1u * 8u) * in_features + ks * 32u + k2 * 8u,
                                in_features);
                            simdgroup_multiply_accumulate(C[1][0], a, bf0, C[1][0]);
                            simdgroup_multiply_accumulate(C[1][1], a, bf1, C[1][1]);
                            simdgroup_multiply_accumulate(C[1][2], a, bf2, C[1][2]);
                            simdgroup_multiply_accumulate(C[1][3], a, bf3, C[1][3]);
                        }
                    }
            if (rlive > 2u) {
                        for (uint k2 = 0u; k2 < 4u; k2++) {
                            simdgroup_half8x8 bf0;
                            simdgroup_half8x8 bf1;
                            simdgroup_half8x8 bf2;
                            simdgroup_half8x8 bf3;
                            const threadgroup half* wrow =
                                &wblk[((ks & 1u)) * 32u * 72u + k2 * 8u * 72u + sgc * 32u];
                            simdgroup_load(bf0, wrow, 72u);
                            simdgroup_load(bf1, wrow + 8u, 72u);
                            simdgroup_load(bf2, wrow + 16u, 72u);
                            simdgroup_load(bf3, wrow + 24u, 72u);
                            simdgroup_half8x8 a;
                            simdgroup_load(
                                a, xrow + (ulong)(2u * 8u) * in_features + ks * 32u + k2 * 8u,
                                in_features);
                            simdgroup_multiply_accumulate(C[2][0], a, bf0, C[2][0]);
                            simdgroup_multiply_accumulate(C[2][1], a, bf1, C[2][1]);
                            simdgroup_multiply_accumulate(C[2][2], a, bf2, C[2][2]);
                            simdgroup_multiply_accumulate(C[2][3], a, bf3, C[2][3]);
                        }
                    }
            if (rlive > 3u) {
                        for (uint k2 = 0u; k2 < 4u; k2++) {
                            simdgroup_half8x8 bf0;
                            simdgroup_half8x8 bf1;
                            simdgroup_half8x8 bf2;
                            simdgroup_half8x8 bf3;
                            const threadgroup half* wrow =
                                &wblk[((ks & 1u)) * 32u * 72u + k2 * 8u * 72u + sgc * 32u];
                            simdgroup_load(bf0, wrow, 72u);
                            simdgroup_load(bf1, wrow + 8u, 72u);
                            simdgroup_load(bf2, wrow + 16u, 72u);
                            simdgroup_load(bf3, wrow + 24u, 72u);
                            simdgroup_half8x8 a;
                            simdgroup_load(
                                a, xrow + (ulong)(3u * 8u) * in_features + ks * 32u + k2 * 8u,
                                in_features);
                            simdgroup_multiply_accumulate(C[3][0], a, bf0, C[3][0]);
                            simdgroup_multiply_accumulate(C[3][1], a, bf1, C[3][1]);
                            simdgroup_multiply_accumulate(C[3][2], a, bf2, C[3][2]);
                            simdgroup_multiply_accumulate(C[3][3], a, bf3, C[3][3]);
                        }
                    }
        }
    }

    // store: fp32 frags -> tg scratch -> half device rows within the segment
    uint cb0 = colb * 64u + sgc * 32u;
    uint rbase = sgr * 32u;
    for (uint r = 0u; r < 4u; r++) {
        if (rbase + r * 8u >= blk_len) {
            break;
        }
        for (uint cf = 0u; cf < 4u; cf++) {
            simdgroup_store(C[r][cf], &csc[sgid * 64u], 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 64u; i += 32u) {
                uint rr = i >> 3u;
                uint cc = i & 7u;
                uint gr = rbase + r * 8u + rr;
                if (gr < blk_len) {
                    out[(ulong)(row0 + gr) * out_e + cb0 + cf * 8u + cc] =
                        half(csc[sgid * 64u + i]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
