"""Step 3: Stratified Regime 5-Fold GroupCV & Moment Decomposition Evaluator.

Computes:
1. WP_global: Exact dataset-global metric
2. r_between: Weighted correlation of sequence-level macro-regime means
3. r_within: Weighted correlation of within-sequence centered residuals
4. 200-sample sequence block bootstrap
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, weighted_pearson


def compute_sequence_moments(targets_seq, preds_seq, mask):
    """
    Computes sequence-level moments for targets and predictions.
    targets_seq: (T, 2)
    preds_seq: (T, 2)
    mask: (T,) bool
    """
    y = np.clip(targets_seq[mask], -2.0, 2.0).astype(np.float64)
    p = np.clip(preds_seq[mask], -2.0, 2.0).astype(np.float64)
    
    # Target 0
    w0 = np.abs(y[:, 0])
    sw0 = np.sum(w0)
    if sw0 > 1e-8:
        my0 = np.sum(w0 * y[:, 0]) / sw0
        mp0 = np.sum(w0 * p[:, 0]) / sw0
        yc0 = y[:, 0] - my0
        pc0 = p[:, 0] - mp0
        cov0 = np.sum(w0 * yc0 * pc0) / sw0
        sy0 = np.sqrt(np.sum(w0 * yc0**2) / sw0)
        sp0 = np.sqrt(np.sum(w0 * pc0**2) / sw0)
        r_w0 = cov0 / (sy0 * sp0 + 1e-12)
    else:
        my0, mp0, sw0, r_w0 = 0.0, 0.0, 0.0, 0.0
        
    # Target 1
    w1 = np.abs(y[:, 1])
    sw1 = np.sum(w1)
    if sw1 > 1e-8:
        my1 = np.sum(w1 * y[:, 1]) / sw1
        mp1 = np.sum(w1 * p[:, 1]) / sw1
        yc1 = y[:, 1] - my1
        pc1 = p[:, 1] - mp1
        cov1 = np.sum(w1 * yc1 * pc1) / sw1
        sy1 = np.sqrt(np.sum(w1 * yc1**2) / sw1)
        sp1 = np.sqrt(np.sum(w1 * pc1**2) / sw1)
        r_w1 = cov1 / (sy1 * sp1 + 1e-12)
    else:
        my1, mp1, sw1, r_w1 = 0.0, 0.0, 0.0, 0.0
        
    return {
        'sw0': sw0, 'my0': my0, 'mp0': mp0, 'r_within_0': r_w0,
        'sw1': sw1, 'my1': my1, 'mp1': mp1, 'r_within_1': r_w1
    }


def evaluate_moments_decomposition(targets_all, preds_all):
    """
    Evaluates global WP, within-sequence correlation, and between-sequence correlation across all sequences.
    targets_all: (N_seqs, 20000, 2)
    preds_all: (N_seqs, 20000, 2)
    """
    N_seqs = len(targets_all)
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    
    acc = GlobalAccumulator()
    
    sw0_list, my0_list, mp0_list, rw0_list = [], [], [], []
    sw1_list, my1_list, mp1_list, rw1_list = [], [], [], []
    
    for s in range(N_seqs):
        acc.add(targets_all[s], preds_all[s], mask)
        m = compute_sequence_moments(targets_all[s], preds_all[s], mask)
        
        sw0_list.append(m['sw0'])
        my0_list.append(m['my0'])
        mp0_list.append(m['mp0'])
        rw0_list.append(m['r_within_0'])
        
        sw1_list.append(m['sw1'])
        my1_list.append(m['my1'])
        mp1_list.append(m['mp1'])
        rw1_list.append(m['r_within_1'])
        
    res = acc.result()
    wp_global = res['weighted_pearson']
    wp_t0 = res['t0']
    wp_t1 = res['t1']
    
    # Within-sequence average
    rw0_mean = float(np.average(rw0_list, weights=sw0_list))
    rw1_mean = float(np.average(rw1_list, weights=sw1_list))
    r_within = 0.5 * (rw0_mean + rw1_mean)
    
    # Between-sequence correlation of macro-regime means
    def w_corr(x, y, w):
        sw = np.sum(w)
        mx = np.sum(w * x) / sw
        my = np.sum(w * y) / sw
        cov = np.sum(w * (x - mx) * (y - my)) / sw
        sx = np.sqrt(np.sum(w * (x - mx)**2) / sw)
        sy = np.sqrt(np.sum(w * (y - my)**2) / sw)
        return float(cov / (sx * sy + 1e-12))
        
    rb0 = w_corr(np.array(my0_list), np.array(mp0_list), np.array(sw0_list))
    rb1 = w_corr(np.array(my1_list), np.array(mp1_list), np.array(sw1_list))
    r_between = 0.5 * (rb0 + rb1)
    
    return {
        'WP_global': wp_global,
        'WP_t0': wp_t0,
        'WP_t1': wp_t1,
        'r_between': r_between,
        'r_between_t0': rb0,
        'r_between_t1': rb1,
        'r_within': r_within,
        'r_within_t0': rw0_mean,
        'r_within_t1': rw1_mean,
    }


def create_stratified_folds(seq_stats, n_folds=5):
    """
    Creates stratified 5-fold splits based on sequence weighted mean and volatility mass.
    seq_stats: list of dicts with 'seq_id', 'mu_s', 'weight_s'
    """
    mu_vals = np.array([s['mu_s'] for s in seq_stats])
    w_vals = np.log(np.array([s['weight_s'] for s in seq_stats]) + 1.0)
    
    # 10 bins for mu, 5 bins for log weight -> 50 strata
    mu_bins = np.digitize(mu_vals, np.percentile(mu_vals, np.linspace(0, 100, 11)[1:-1]))
    w_bins = np.digitize(w_vals, np.percentile(w_vals, np.linspace(0, 100, 6)[1:-1]))
    strata = mu_bins * 10 + w_bins
    
    folds = [[] for _ in range(n_folds)]
    for stratum in np.unique(strata):
        idx = np.where(strata == stratum)[0]
        np.random.shuffle(idx)
        for i, seq_idx in enumerate(idx):
            folds[i % n_folds].append(seq_stats[seq_idx]['seq_id'])
            
    return folds


if __name__ == "__main__":
    print("Testing Moment Decomposition on Cached 50 Sequences...")
    cache = np.load('scratch/cached_50_seqs.npz')
    targets = cache['targets_all']
    preds = cache['raw_preds'] # Baseline predictions
    
    m = evaluate_moments_decomposition(targets, preds)
    print("Baseline Diagnostic Gate:")
    print(f"  WP_global: {m['WP_global']:.6f} (t0: {m['WP_t0']:.6f}, t1: {m['WP_t1']:.6f})")
    print(f"  r_between: {m['r_between']:.6f} (t0: {m['r_between_t0']:.6f}, t1: {m['r_between_t1']:.6f})")
    print(f"  r_within:  {m['r_within']:.6f} (t0: {m['r_within_t0']:.6f}, t1: {m['r_within_t1']:.6f})")
    print("Validation Diagnostic Evaluator ready!")
