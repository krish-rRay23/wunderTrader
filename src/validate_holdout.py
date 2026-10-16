"""Validate optimized calibration on independent holdout sequences (sequences 20-50)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data_reader import FastSequenceReader
from src.solution_accelerated import PredictionModel
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, DataPoint

reader = FastSequenceReader('datasets/train.parquet')
model_opt = PredictionModel('baseline/baseline.onnx')

acc_base = GlobalAccumulator()
acc_opt = GlobalAccumulator()
mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
mask[WARMUP:] = True

print('Evaluating untouched sequence holdout (Sequences 20 to 50 = 600,000 steps)...')

# Target calibration parameters
s0, s1 = 0.25, 0.40
d0, d1 = 0.10, 0.15

for seq_idx, (seq_id, features, targets) in enumerate(reader.iter_sequences(max_sequences=50)):
    if seq_idx < 20:
        continue # Skip training/fit sequences!

    p_base = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
    p_opt = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
    ema_curr = None

    for step in range(SEQUENCE_LENGTH):
        dp = DataPoint(seq_id, step, (step >= WARMUP), features[step])
        out = model_opt.predict(dp)
        if step >= WARMUP:
            p_base[step] = out
            if ema_curr is None:
                ema_curr = out.copy()
            else:
                ema_curr[0] = (1.0 - d0) * out[0] + d0 * ema_curr[0]
                ema_curr[1] = (1.0 - d1) * out[1] + d1 * ema_curr[1]
            p_opt[step] = [ema_curr[0] * s0, ema_curr[1] * s1]

    acc_base.add(targets, p_base, mask)
    acc_opt.add(targets, p_opt, mask)

res_base = acc_base.result()
res_opt = acc_opt.result()

print('\n=== INDEPENDENT SEQUENCE HOLDOUT EVALUATION (30 Sequences, 597,030 Scored Rows) ===')
wp_base = res_base['weighted_pearson']
wp_opt = res_opt['weighted_pearson']
delta = wp_opt - wp_base
pct = (delta / wp_base) * 100

print(f"Uncalibrated Baseline WP: {wp_base:.6f} (t0: {res_base['t0']:.6f}, t1: {res_base['t1']:.6f})")
print(f"Optimized Solution WP:     {wp_opt:.6f} (t0: {res_opt['t0']:.6f}, t1: {res_opt['t1']:.6f})")
print(f"Independent Holdout Gain:  +{delta:.6f} (+{pct:.2f}%)")
