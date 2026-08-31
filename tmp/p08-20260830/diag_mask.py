import mlx.core as mx
import numpy as np

# Emulate RotatingKVCache.make_mask with max_size=128, offset clamped to 127,
# N=2048 (create_causal_mask(N, offset, window_size=128) at base.py:24-42).
N, offset, window = 2048, 127, 128
rinds = mx.arange(offset + N)          # 0..2174  (2175 keys)
linds = mx.arange(offset, offset + N)  # 127..2174
mask = (linds[:, None] >= rinds[None]) & (linds[:, None] < rinds[None] + window)

for row in (0, 127, 1023, 1300, 2047):
    vis = np.nonzero(np.array(mask[row]))[0]
    print(f"query row {row}: visible key range [{vis.min()}..{vis.max()}], "
          f"count {len(vis)}")