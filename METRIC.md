# Scoring reference: Global Weighted Pearson correlation

The metric is Global Weighted Pearson correlation (WP).
Compute one weighted correlation for each target over all
scoring rows in the evaluated dataset, then average the two correlations.
Both targets refer to instrument `i0`. Higher is better.

## Evaluation domain

Each sequence contains 20,000 rows. Steps 0 through 98 are warm-up;
every later step requires two finite, float32-compatible predictions.
Update model state on every row and reset it between sequences.
Scoring uses only rows where `need_prediction AND is_scored` is true.
The validation mask is provided; the test mask is hidden from submissions.

All selected rows from all sequences enter the global calculation. There
is no additional sequence eligibility condition. Sequence boundaries control
model-state resets, not metric averaging. Targets are scored separately,
not concatenated with each other.

Required predictions must be valid even on unscored rows. Missing outputs,
incorrect shapes and nonfinite values are submission errors. An evaluated
dataset with no selected rows is an evaluation error.

## Formula

Let `M` be the set of all rows in the evaluated dataset selected by
`need_prediction AND is_scored`. For target `k`, define

$$
y_{ki}=\operatorname{clip}(t_{ki},-2,2),\qquad
\hat y_{ki}=\operatorname{clip}(p_{ki},-2,2),\qquad
w_{ki}=|y_{ki}|,\qquad i\in M.
$$

With $W_k=\sum_{i\in M}w_{ki}$, the weighted means are

$$
\bar y_{k,w}=\frac{\sum_{i\in M}w_{ki}y_{ki}}{W_k},\qquad
\bar{\hat y}_{k,w}=\frac{\sum_{i\in M}w_{ki}\hat y_{ki}}{W_k}.
$$

The per-target correlation and final score are

$$
\rho_{k,w}=
\frac{\sum_{i\in M}w_{ki}(y_{ki}-\bar y_{k,w})
(\hat y_{ki}-\bar{\hat y}_{k,w})}
{\sqrt{\left[\sum_{i\in M}w_{ki}(y_{ki}-\bar y_{k,w})^2\right]
\left[\sum_{i\in M}w_{ki}(\hat y_{ki}-\bar{\hat y}_{k,w})^2\right]}},
\qquad S=\frac{\rho_{0,w}+\rho_{1,w}}{2}.
$$

Both targets and predictions are clipped to the same fixed range `[-2, 2]`.
Weights are the absolute values of the clipped targets; zero targets have
zero weight. Values are not rounded. Clipping is part of scoring only and
does not modify the stored dataset or submitted predictions.

Use float32-compatible inputs and float64 metric arithmetic. If a target's
total weight is less than `1e-8`, or either weighted standard deviation is
at most `1e-8`, its correlation is `0`. Constant predictions therefore
contribute `0`. The final score remains the mean of both target scores.
Correlation is bounded to `[-1, 1]` to remove numerical round-off.

## Reference calculation

This in-memory reference takes all rows of an evaluated dataset in matching
order. `weighted_pearson` is provided in `utils.py` and applies the clipping,
weights and numerical conventions above.

```python
import numpy as np
from utils import weighted_pearson


def global_wp(targets, predictions, need_prediction, is_scored):
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    need = np.asarray(need_prediction, dtype=bool)
    scored = np.asarray(is_scored, dtype=bool)
    if y.ndim != 2 or y.shape[1] != 2 or p.shape != y.shape:
        raise ValueError("expected targets and predictions of shape (N, 2)")
    if need.shape != (len(y),) or scored.shape != need.shape:
        raise ValueError("incorrect mask shape")
    if not np.isfinite(y).all() or not np.isfinite(p[need]).all():
        raise ValueError("nonfinite target or required prediction")
    mask = need & scored
    if not mask.any():
        raise ValueError("no rows selected by the scoring mask")
    return float(np.mean([
        weighted_pearson(y[mask, k], p[mask, k]) for k in range(2)
    ]))
```

`GlobalAccumulator` in `utils.py` computes the same result with bounded
memory by merging centered weighted moments as sequences are read.
`ScorerStepByStep` uses this accumulator for local validation. Its report
includes `t0`, `t1`, `weighted_pearson`, sequence counts and selected-row counts.
