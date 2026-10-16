"""Systematic experimentation: Output scaling, Online EMA smoothing, and Non-linear calibration."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP
from src.data_reader import FastSequenceReader
from src.solution_accelerated import PredictionModel
from utils import DataPoint

# Step 1: Precompute raw baseline predictions for the benchmark sequences
# to allow lightning-fast testing of calibration and EMA variants
print('Precomputing baseline predictions for 20 sequences...')
reader = FastSequenceReader('datasets/train.parquet')
model = PredictionModel('baseline/baseline.onnx')

all_targets = []
all_raw_preds = []

for seq_idx, (seq_id, features, targets) in enumerate(reader.iter_sequences(max_sequences=20)):
    raw_p = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
    for step in range(SEQUENCE_LENGTH):
        dp = DataPoint(
            seq_ix=seq_id,
            step_in_seq=step,
            need_prediction=(step >= WARMUP),
            state=features[step]
        )
        p = model.predict(dp)
        if step >= WARMUP:
            raw_p[step] = p
    all_targets.append(targets)
    all_raw_preds.append(raw_p)

print(f'Done! Precomputed {len(all_targets)} sequences.')

mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
mask[WARMUP:] = True

def score_predictions(pred_transform_fn):
    acc = GlobalAccumulator()
    for targs, raw_p in zip(all_targets, all_raw_preds):
        p_transformed = pred_transform_fn(raw_p)
        acc.add(targs, p_transformed, mask)
    res = acc.result()
    return res['weighted_pearson'], res['t0'], res['t1']

# Baseline Score
base_wp, base_t0, base_t1 = score_predictions(lambda p: p)
print(f'\n--- BASELINE SCORE ---')
print(f'Global WP: {base_wp:.6f} (t0: {base_t0:.6f}, t1: {base_t1:.6f})')

# Experiment 1: Output Scaling (Alpha)
print(f'\n--- EXPERIMENT 1: OUTPUT SCALING FACTOR ---')
best_scale = 1.0
best_scale_wp = base_wp
for scale in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:
    wp, t0, t1 = score_predictions(lambda p, s=scale: p * s)
    marker = ' <-- BEST' if wp > best_scale_wp else ''
    if wp > best_scale_wp:
        best_scale_wp = wp
        best_scale = scale
    print(f'Scale: {scale:4.2f} | WP: {wp:.6f} (t0: {t0:.6f}, t1: {t1:.6f}) {marker}')

# Experiment 2: Online EMA Smoothing
print(f'\n--- EXPERIMENT 2: ONLINE CAUSAL EMA SMOOTHING ---')
best_lambda = 0.0
best_ema_wp = base_wp

def apply_ema(preds, decay):
    out = np.empty_like(preds)
    out[:WARMUP] = np.nan
    curr = preds[WARMUP].copy()
    out[WARMUP] = curr
    for t in range(WARMUP + 1, SEQUENCE_LENGTH):
        curr = (1.0 - decay) * preds[t] + decay * curr
        out[t] = curr
    return out

for decay in [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    if decay == 0.0:
        wp, t0, t1 = base_wp, base_t0, base_t1
    else:
        wp, t0, t1 = score_predictions(lambda p, d=decay: apply_ema(p, d))
    marker = ' <-- BEST' if wp > best_ema_wp else ''
    if wp > best_ema_wp:
        best_ema_wp = wp
        best_lambda = decay
    print(f'EMA Decay: {decay:4.2f} | WP: {wp:.6f} (t0: {t0:.6f}, t1: {t1:.6f}) {marker}')

# Experiment 3: Non-linear Power Transform (Tail Sharpening)
print(f'\n--- EXPERIMENT 3: POWER TRANSFORM / TAIL MAGNIFICATION ---')
best_gamma = 1.0
best_gamma_wp = base_wp

def apply_power(preds, gamma):
    out = np.empty_like(preds)
    out[:WARMUP] = np.nan
    p = preds[WARMUP:]
    signs = np.sign(p)
    out[WARMUP:] = signs * (np.abs(p) ** gamma)
    return out

for gamma in [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]:
    if gamma == 1.0:
        wp, t0, t1 = base_wp, base_t0, base_t1
    else:
        wp, t0, t1 = score_predictions(lambda p, g=gamma: apply_power(p, g))
    marker = ' <-- BEST' if wp > best_gamma_wp else ''
    if wp > best_gamma_wp:
        best_gamma_wp = wp
        best_gamma = gamma
    print(f'Gamma: {gamma:4.2f} | WP: {wp:.6f} (t0: {t0:.6f}, t1: {t1:.6f}) {marker}')

# Experiment 4: Joint Calibration (Best Scale + Best EMA + Best Gamma)
print(f'\n--- EXPERIMENT 4: COMBINED OPTIMIZED PIPELINE ---')
def combined_pipeline(preds, scale, decay, gamma):
    p = preds * scale
    if decay > 0:
        p = apply_ema(p, decay)
    if gamma != 1.0:
        p = apply_power(p, gamma)
    return p

comb_wp, comb_t0, comb_t1 = score_predictions(lambda p: combined_pipeline(p, best_scale, best_lambda, best_gamma))
print(f'Combined (Scale={best_scale}, Decay={best_lambda}, Gamma={best_gamma}):')
print(f'WP: {comb_wp:.6f} (vs Baseline: {base_wp:.6f}, delta = {comb_wp - base_wp:+.6f})')
print(f't0: {comb_t0:.6f} (delta = {comb_t0 - base_t0:+.6f})')
print(f't1: {comb_t1:.6f} (delta = {comb_t1 - base_t1:+.6f})')
