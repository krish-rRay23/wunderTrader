"""Competition callback and masked, dataset-global Weighted Pearson scorer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

SEQUENCE_LENGTH = 20_000
WARMUP = 99
N_FEATURES = 112
FEATURE_COLUMNS = [f'i{i}_{g}{j}' for i in range(2)
                   for g, n in [('p', 22), ('v', 22), ('dp', 4), ('dv', 4)] for j in range(n)]
FEATURE_COLUMNS += [f'a{i}' for i in range(8)]
TARGET_COLUMNS = ('t0', 't1')
METRIC_CLIP = 2.0
METRIC_SPECIFICATION = 'global_wp'


@dataclass(frozen=True)
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def weighted_pearson(target, prediction):
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError('expected equal one-dimensional target and prediction arrays')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('target/prediction contains nonfinite values')
    # The metric clips both arrays. Stored targets and submitted predictions
    # remain untouched outside this calculation.
    target = np.clip(target, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    prediction = np.clip(prediction, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    weights = np.abs(target)
    total = weights.sum()
    if total < 1e-8:
        return 0.0
    target_mean = np.sum(weights * target) / total
    prediction_mean = np.sum(weights * prediction) / total
    target_centered = target - target_mean
    prediction_centered = prediction - prediction_mean
    covariance = np.sum(weights * target_centered * prediction_centered) / total
    target_std = np.sqrt(np.sum(weights * target_centered**2) / total)
    prediction_std = np.sqrt(np.sum(weights * prediction_centered**2) / total)
    if target_std <= 1e-8 or prediction_std <= 1e-8:
        return 0.0
    return float(np.clip(covariance / (target_std * prediction_std), -1.0, 1.0))


class GlobalAccumulator:
    """Merge centered weighted moments, never averages of block scores.

    add() consumes one sequence for bounded memory and contract checking.
    Parallel-variance merging preserves between-sequence covariance.
    """

    def __init__(self):
        self.blocks_seen = 0
        self.selected_rows = 0
        self.nonempty_blocks = 0
        self.weight = np.zeros(2, dtype=np.float64)
        self.mean_y = np.zeros(2, dtype=np.float64)
        self.mean_p = np.zeros(2, dtype=np.float64)
        self.m2_y = np.zeros(2, dtype=np.float64)
        self.m2_p = np.zeros(2, dtype=np.float64)
        self.cross = np.zeros(2, dtype=np.float64)

    def add(self, targets, predictions, mask):
        targets = np.asarray(targets, dtype=np.float32)
        predictions = np.asarray(predictions, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if targets.shape != (SEQUENCE_LENGTH, 2) or predictions.shape != targets.shape or mask.shape != (SEQUENCE_LENGTH,):
            raise ValueError('one accumulator call must contain exactly one complete 20k block')
        if np.any(mask[:WARMUP]):
            raise ValueError('warm-up rows cannot be scored')
        if not np.isfinite(targets).all() or not np.isfinite(predictions[WARMUP:]).all():
            raise ValueError('nonfinite labels or required predictions')
        self.blocks_seen += 1
        self.selected_rows += int(mask.sum())
        if not mask.any():
            return
        self.nonempty_blocks += 1
        y = np.clip(targets[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        p = np.clip(predictions[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        for side in range(2):
            w = np.abs(y[:, side])
            weight = w.sum()
            if weight == 0:
                continue
            mean_y = np.sum(w * y[:, side]) / weight
            mean_p = np.sum(w * p[:, side]) / weight
            yc, pc = y[:, side] - mean_y, p[:, side] - mean_p
            total = self.weight[side] + weight
            delta_y = mean_y - self.mean_y[side]
            delta_p = mean_p - self.mean_p[side]
            correction = self.weight[side] * weight / total
            self.m2_y[side] += np.sum(w * yc * yc) + delta_y**2 * correction
            self.m2_p[side] += np.sum(w * pc * pc) + delta_p**2 * correction
            self.cross[side] += np.sum(w * yc * pc) + delta_y * delta_p * correction
            self.mean_y[side] += delta_y * weight / total
            self.mean_p[side] += delta_p * weight / total
            self.weight[side] = total

    def result(self):
        if not self.selected_rows:
            raise ValueError('no rows selected by the scoring mask')
        per_side = np.zeros(2, dtype=np.float64)
        for side in range(2):
            total = self.weight[side]
            if total < 1e-8:
                continue
            sy = np.sqrt(max(0.0, self.m2_y[side] / total))
            sp = np.sqrt(max(0.0, self.m2_p[side] / total))
            if sy > 1e-8 and sp > 1e-8:
                per_side[side] = np.clip(self.cross[side] / total / (sy * sp), -1.0, 1.0)
        return {'t0': float(per_side[0]), 't1': float(per_side[1]),
                'weighted_pearson': float(per_side.mean()), 'blocks': self.blocks_seen,
                'blocks_with_scored_rows': self.nonempty_blocks,
                'selected_rows': self.selected_rows, 'effective_scored_rows': self.selected_rows,
                'metric_clip': [-METRIC_CLIP, METRIC_CLIP],
                'metric_specification': METRIC_SPECIFICATION,
                'aggregation': 'mean of two target-wise global clipped Weighted Pearson correlations over all masked rows'}


def validate_sequence(table):
    if table.num_rows != SEQUENCE_LENGTH:
        raise ValueError('each row group must be exactly one 20k sequence')
    ids = table['seq_ix'].to_numpy()
    if not np.all(ids == ids[0]):
        raise ValueError('multiple sequence IDs in one row group')
    if not np.array_equal(table['step_in_seq'].to_numpy(), np.arange(SEQUENCE_LENGTH)):
        raise ValueError('incorrect step order')
    need = table['need_prediction'].to_numpy().astype(bool)
    if not np.array_equal(need, np.arange(SEQUENCE_LENGTH) >= WARMUP):
        raise ValueError('incorrect warm-up contract')
    return int(ids[0]), need


class ScorerStepByStep:
    def __init__(self, dataset_path: str | Path):
        self.parquet = pq.ParquetFile(dataset_path)
        expected = ['seq_ix', 'step_in_seq', 'need_prediction', 'is_scored', *FEATURE_COLUMNS, *TARGET_COLUMNS]
        if self.parquet.schema_arrow.names != expected:
            raise ValueError('local scoring requires the labeled validation schema')

    def score(self, model):
        accumulator = GlobalAccumulator()
        seen = set()
        for group in range(self.parquet.num_row_groups):
            table = self.parquet.read_row_group(group, use_threads=False)
            seq, need = validate_sequence(table)
            if seq in seen:
                raise ValueError('duplicate sequence ID')
            seen.add(seq)
            features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
            targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
            predictions = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
            for step in range(SEQUENCE_LENGTH):
                point = DataPoint(seq, step, bool(need[step]), features[step])
                value = model.predict(point)
                if not need[step]:
                    if value is not None:
                        raise ValueError('return None during warm-up AFTER updating state')
                else:
                    value = np.asarray(value, dtype=np.float32)
                    if value.shape != (2,) or not np.isfinite(value).all():
                        raise ValueError('return two finite scores on every required row')
                    predictions[step] = value
            accumulator.add(targets, predictions, table['is_scored'].to_numpy().astype(bool) & need)
        result = accumulator.result()
        result['rows_seen'] = len(seen)*SEQUENCE_LENGTH
        result['predictions_required'] = len(seen)*(SEQUENCE_LENGTH-WARMUP)
        return result
