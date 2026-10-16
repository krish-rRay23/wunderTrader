import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.experiment_calibration_ema import all_targets, all_raw_preds, mask
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP

print('Testing dynamic volatility-conditioned scaling...')
best_wp = 0.0
best_k = 0.0

for k in np.linspace(-0.5, 0.5, 21):
    acc = GlobalAccumulator()
    for targs, raw_p in zip(all_targets, all_raw_preds):
        p0 = raw_p[:, 0].copy()
        p1 = raw_p[:, 1].copy()

        # Compute online causal volatility estimate
        vol = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
        v_curr = 0.5
        for t in range(WARMUP, SEQUENCE_LENGTH):
            mag = (abs(p0[t]) + abs(p1[t])) / 2.0
            v_curr = 0.98 * v_curr + 0.02 * mag
            vol[t] = v_curr

        mean_v = np.mean(vol[WARMUP:]) + 1e-6
        vol_factor = np.clip(1.0 + k * (vol / mean_v - 1.0), 0.2, 3.0)

        p_new = np.column_stack([p0 * 0.25 * vol_factor, p1 * 0.40 * vol_factor])
        acc.add(targs, p_new, mask)
    res = acc.result()
    wp = res['weighted_pearson']
    if wp > best_wp:
        best_wp = wp
        best_k = k
    print(f"Vol Sensitivity k = {k:+.2f} | WP: {wp:.6f} (t0: {res['t0']:.6f}, t1: {res['t1']:.6f})")

print(f"\nBest Vol Sensitivity: k = {best_k:+.2f} | Best WP: {best_wp:.6f}")
