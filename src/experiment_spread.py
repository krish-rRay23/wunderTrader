import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data_reader import FastSequenceReader
from src.solution_accelerated import PredictionModel
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, DataPoint

reader = FastSequenceReader('datasets/train.parquet')
model = PredictionModel('baseline/baseline.onnx')

all_spreads = []
all_preds = []
all_targs = []

print('Computing cross-instrument spread on benchmark sequences...')
for seq_idx, (seq_id, feats, targs) in enumerate(reader.iter_sequences(max_sequences=20)):
    bids_i0 = feats[:, :11]
    asks_i0 = feats[:, 11:22]
    mid_i0 = (np.max(bids_i0, axis=1) + np.min(asks_i0, axis=1)) / 2.0

    bids_i1 = feats[:, 52:63]
    asks_i1 = feats[:, 63:74]
    mid_i1 = (np.max(bids_i1, axis=1) + np.min(asks_i1, axis=1)) / 2.0

    spread = mid_i0 - mid_i1
    all_spreads.append(spread)
    all_targs.append(targs)

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
    all_preds.append(preds)

mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
mask[WARMUP:] = True

print('Testing spread blending weights...')
best_wp = 0.0
best_beta = 0.0

for beta in np.linspace(-0.15, 0.15, 31):
    acc = GlobalAccumulator()
    for targs, preds, spread in zip(all_targs, all_preds, all_spreads):
        p_aug = preds.copy()
        # Spread mean reversion
        p_aug[:, 0] -= beta * spread
        p_aug[:, 1] += beta * spread
        acc.add(targs, p_aug, mask)
    res = acc.result()
    wp = res['weighted_pearson']
    if wp > best_wp:
        best_wp = wp
        best_beta = beta
    print(f"Beta {beta:+.3f} | WP: {wp:.6f} (t0: {res['t0']:.6f}, t1: {res['t1']:.6f})")

print(f"\nBest Beta: {best_beta:+.3f} | Best WP: {best_wp:.6f}")
