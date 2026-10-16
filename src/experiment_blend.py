import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.experiment_calibration_ema import all_targets, all_raw_preds, mask
from utils import GlobalAccumulator

print('Testing cross-target anti-symmetric blend...')
best_wp = 0.0
best_w = 0.0

for w in np.linspace(-0.2, 0.4, 25):
    acc = GlobalAccumulator()
    for targs, raw_p in zip(all_targets, all_raw_preds):
        p0 = raw_p[:, 0]
        p1 = raw_p[:, 1]
        p0_blend = (1.0 - w) * p0 - w * p1
        p1_blend = (1.0 - w) * p1 - w * p0
        p_new = np.column_stack([p0_blend * 0.25, p1_blend * 0.40])
        acc.add(targs, p_new, mask)
    res = acc.result()
    wp = res['weighted_pearson']
    if wp > best_wp:
        best_wp = wp
        best_w = w
    print(f"Weight w = {w:+.3f} | WP: {wp:.6f} (t0: {res['t0']:.6f}, t1: {res['t1']:.6f})")

print(f"\nBest cross-target blend weight: w = {best_w:.3f} | Best WP: {best_wp:.6f}")
