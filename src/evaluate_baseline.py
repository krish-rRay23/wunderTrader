import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP
from src.data_reader import FastSequenceReader
from src.solution_accelerated import PredictionModel
from utils import DataPoint

def evaluate(model, num_sequences: int = 50):
    reader = FastSequenceReader('datasets/train.parquet')
    accumulator = GlobalAccumulator()

    t0 = time.time()
    for seq_idx, (seq_id, features, targets) in enumerate(reader.iter_sequences(max_sequences=num_sequences)):
        predictions = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
        
        # In train data, all steps >= 99 are scored (need_prediction & scored)
        mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
        mask[WARMUP:] = True

        for step in range(SEQUENCE_LENGTH):
            dp = DataPoint(
                seq_ix=seq_id,
                step_in_seq=step,
                need_prediction=(step >= WARMUP),
                state=features[step]
            )
            p = model.predict(dp)
            if step >= WARMUP:
                predictions[step] = p

        accumulator.add(targets, predictions, mask)
        if (seq_idx + 1) % 10 == 0:
            print(f'Evaluated {seq_idx + 1}/{num_sequences} sequences... Current WP: {accumulator.result()["weighted_pearson"]:.6f}')

    t1 = time.time()
    res = accumulator.result()
    dt = t1 - t0
    res['elapsed_seconds'] = dt
    res['steps_per_sec'] = (num_sequences * SEQUENCE_LENGTH) / dt
    return res

if __name__ == '__main__':
    model = PredictionModel('baseline/baseline.onnx')
    print('Starting baseline evaluation on 20 sequences (400,000 steps)...')
    res = evaluate(model, num_sequences=20)
    print('\n=== BASELINE EVALUATION RESULT ===')
    print(f"Global Weighted Pearson (WP): {res['weighted_pearson']:.6f}")
    print(f"t0 WP: {res['t0']:.6f}")
    print(f"t1 WP: {res['t1']:.6f}")
    print(f"Selected rows: {res['selected_rows']}")
    print(f"Throughput: {res['steps_per_sec']:.0f} steps/s ({1e6/res['steps_per_sec']:.1f} µs/step)")
