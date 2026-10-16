"""Quantitative research experiment: Ensembling Recurrent Baseline with Fast Microstructure Model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data_reader import FastSequenceReader
from src.solution_accelerated import PredictionModel
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, DataPoint

reader = FastSequenceReader('datasets/train.parquet')
seqs = list(reader.iter_sequences(max_sequences=20))
model = PredictionModel('baseline/baseline.onnx')

print('Preparing data and baseline predictions for 20 sequences...')
all_feats = []
all_targs = []
all_base_preds = []

for seq_idx, (seq_id, feats, targs) in enumerate(seqs):
    preds = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
    ema_0, ema_1 = None, None
    for step in range(SEQUENCE_LENGTH):
        dp = DataPoint(seq_id, step, (step >= WARMUP), feats[step])
        out = model.predict(dp)
        if step >= WARMUP:
            if ema_0 is None:
                ema_0, ema_1 = out[0], out[1]
            else:
                ema_0 = 0.90 * out[0] + 0.10 * ema_0
                ema_1 = 0.85 * out[1] + 0.15 * ema_1
            preds[step] = [ema_0 * 0.25, ema_1 * 0.40]
    all_feats.append(feats)
    all_targs.append(targs)
    all_base_preds.append(preds)

mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
mask[WARMUP:] = True

# Fit a fast causal linear feature model on sequences 0..10
X_train = np.concatenate([f[WARMUP:] for f in all_feats[:10]], axis=0)
Y_train = np.concatenate([t[WARMUP:] for t in all_targs[:10]], axis=0)

# Add cross-instrument spread and key features
def make_fast_features(feats):
    mid_i0 = (np.max(feats[:, :11], axis=1) + np.min(feats[:, 11:22], axis=1)) / 2.0
    mid_i1 = (np.max(feats[:, 52:63], axis=1) + np.min(feats[:, 63:74], axis=1)) / 2.0
    spread = (mid_i0 - mid_i1).reshape(-1, 1)
    a_feats = feats[:, 104:112] # a0..a7
    dp_i0 = feats[:, 44:48]     # trade prices i0
    dv_i0 = feats[:, 48:52]     # trade vols i0
    dp_i1 = feats[:, 96:100]    # trade prices i1
    dv_i1 = feats[:, 100:104]   # trade vols i1
    return np.hstack([spread, a_feats, dp_i0, dv_i0, dp_i1, dv_i1])

X_fast_train = np.concatenate([make_fast_features(f)[WARMUP:] for f in all_feats[:10]], axis=0)
X_bias_train = np.hstack([X_fast_train, np.ones((len(X_fast_train), 1))])

# Ridge fit
lam = 1000.0
W_ridge = np.linalg.solve(X_bias_train.T @ X_bias_train + lam * np.eye(X_bias_train.shape[1]), X_bias_train.T @ Y_train)

# Evaluate on holdout sequences 10..20
print('\nEvaluating ensemble on holdout sequences 10..20...')
acc_base = GlobalAccumulator()
acc_ens = GlobalAccumulator()

best_alpha = 0.0
best_wp = 0.0

for alpha in np.linspace(0.0, 0.5, 11):
    acc = GlobalAccumulator()
    for f, t, p_base in zip(all_feats[10:], all_targs[10:], all_base_preds[10:]):
        x_fast = make_fast_features(f)
        x_bias = np.hstack([x_fast, np.ones((len(x_fast), 1))])
        p_fast = x_bias @ W_ridge
        
        # Ensemble blend
        p_blend = (1.0 - alpha) * p_base + alpha * (p_fast * 0.3)
        acc.add(t, p_blend, mask)
    res = acc.result()
    wp = res['weighted_pearson']
    if wp > best_wp:
        best_wp = wp
        best_alpha = alpha
    print(f'Ensemble Weight (Fast Model={alpha:.2f}) | Holdout WP: {wp:.6f} (t0: {res["t0"]:.6f}, t1: {res["t1"]:.6f})')

print(f'\nBest Ensemble Weight: alpha = {best_alpha:.2f} | Holdout WP: {best_wp:.6f}')
