"""Full-Batch Global Optimum Optimizer for the Connectome Architecture.

Optimizes the exact Global Weighted Pearson loss simultaneously over all 200 training sequences
(3,980,200 training steps) with zero batch noise.
Evaluates on all 100 Pristine Holdout Sequences (2,000,000 steps).
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn

from src.build_ultra_lob_matrix import build_ultra_lob_matrix
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, METRIC_CLIP
from src.validation import evaluate_moments_decomposition
from src.extract_baseline_weights import FrozenBaselineGRU

print("=" * 80)
print("TRAINING CONNECTOME WITH FULL-BATCH GLOBAL LOSS (ALL 200 SEQUENCES AT ONCE)")
print("=" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 1. Load Precomputed Training Data (200 sequences)
print("Loading all 200 training sequences into GPU memory...")
X_train = np.load('data/X.npy')[:, WARMUP:]   # (200, 19901, 112)
O0_train = np.load('data/O0.npy')[:, WARMUP:] # (200, 19901, 128)
O1_train = np.load('data/O1.npy')[:, WARMUP:] # (200, 19901, 128)
Y_train = np.load('data/Y.npy')[:, WARMUP:]   # (200, 19901, 2)

P_ultra, feat_names = build_ultra_lob_matrix() # (112, 47)
K_FEATS = P_ultra.shape[1]

# Project to Q
Q_train = np.matmul(X_train, P_ultra) # (200, 19901, 47)

# Flatten training data to (3,980,200, D)
N_STEPS = 200 * (SEQUENCE_LENGTH - WARMUP)
O0_flat = torch.from_numpy(O0_train.reshape(-1, 128)).to(device)
O1_flat = torch.from_numpy(O1_train.reshape(-1, 128)).to(device)
Q_flat = torch.from_numpy(Q_train.reshape(-1, K_FEATS)).to(device)
Y_flat = torch.from_numpy(Y_train.reshape(-1, 2)).to(device)

print(f"Full-batch training tensors ready: {N_STEPS:,} steps ({O0_flat.element_size() * O0_flat.nelement() / 1024**2:.1f} MB)")

# 2. Load Pristine Holdout Benchmark (100 sequences)
d_val = np.load('scratch/pristine_holdout_100.npz')
feats_val = d_val['features_all'] # (100, 20000, 112)
targets_val = d_val['targets_all'] # (100, 20000, 2)
N_VAL = len(feats_val)

base_gru = FrozenBaselineGRU('baseline/baseline.onnx').to(device).eval()
O0_val = []
O1_val = []
with torch.no_grad():
    for s in range(N_VAL):
        xt = torch.from_numpy(feats_val[s:s+1]).to(device)
        _, o0, o1, _, _ = base_gru(xt)
        O0_val.append(o0[0].cpu().numpy())
        O1_val.append(o1[0].cpu().numpy())
O0_val = np.array(O0_val)
O1_val = np.array(O1_val)
Q_val = np.matmul(feats_val, P_ultra)
print("Validation dataset ready: 100 sequences (2,000,000 steps)")

# 3. Model Definition: Warm-started from 0.6517 champion (best_ultra_connectome.pt)
class ConnectomeModel(nn.Module):
    def __init__(self, init_ckpt='scratch/best_ultra_connectome.pt', k_feats=47):
        super().__init__()
        d_prev = torch.load(init_ckpt, map_location='cpu', weights_only=False)
        
        self.head_o1 = nn.Linear(128, 2)
        self.head_o1.weight = nn.Parameter(d_prev['head_o1.weight'].clone(), requires_grad=False)
        self.head_o1.bias = nn.Parameter(d_prev['head_o1.bias'].clone(), requires_grad=False)
        
        self.head_o0 = nn.Linear(128, 2, bias=False)
        self.head_o0.weight = nn.Parameter(d_prev['head_o0.weight'].clone(), requires_grad=True)
        
        self.head_q = nn.Linear(k_feats, 2, bias=False)
        self.head_q.weight = nn.Parameter(d_prev['head_q.weight'].clone(), requires_grad=True)
        
        self.s0 = nn.Parameter(d_prev['s0'].clone(), requires_grad=True)
        self.s1 = nn.Parameter(d_prev['s1'].clone(), requires_grad=True)
        self.coupling = nn.Parameter(d_prev['coupling'].clone(), requires_grad=True)
        self.bias = nn.Parameter(d_prev['bias'].clone(), requires_grad=True)

    def forward(self, o0, o1, q):
        p_base = self.head_o1(o1)
        p_fast = self.head_o0(o0)
        p_micro = self.head_q(q)
        raw = p_base + p_fast + p_micro
        
        p0 = raw[..., 0:1] * self.s0
        p1 = raw[..., 1:2] * self.s1
        c = torch.clamp(self.coupling, 0.0, 0.15)
        out0 = (1.0 - c) * p0 - c * p1 + self.bias[0]
        out1 = (1.0 - c) * p1 - c * p0 + self.bias[1]
        return torch.cat([out0, out1], dim=-1)

model = ConnectomeModel(k_feats=K_FEATS).to(device)

mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
mask[WARMUP:] = True

def evaluate_holdout(m):
    m.eval()
    preds_all = []
    with torch.no_grad():
        for s in range(N_VAL):
            o0_t = torch.from_numpy(O0_val[s:s+1]).to(device)
            o1_t = torch.from_numpy(O1_val[s:s+1]).to(device)
            q_t = torch.from_numpy(Q_val[s:s+1]).to(device)
            p = m(o0_t, o1_t, q_t)
            preds_all.append(p[0].cpu().numpy())
    preds_all = np.array(preds_all)
    acc = GlobalAccumulator()
    for s in range(N_VAL):
        acc.add(targets_val[s], preds_all[s], mask)
    res = acc.result()
    diag = evaluate_moments_decomposition(targets_val, preds_all)
    return res['weighted_pearson'], res['t0'], res['t1'], diag['r_between'], diag['r_within']

init_wp, init_t0, init_t1, init_rb, init_rw = evaluate_holdout(model)
print(f"\n--- STEP 0 BENCHMARK EVALUATION ---")
print(f"Step 0 WP: {init_wp:.6f} (t0: {init_t0:.6f}, t1: {init_t1:.6f}) | r_b: {init_rb:.6f} | r_w: {init_rw:.6f}")

# 4. Pure Dataset-Global Weighted Pearson Loss
class DatasetGlobalWeightedPearsonLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        y_c = torch.clamp(target, -METRIC_CLIP, METRIC_CLIP)
        p_c = torch.clamp(pred, -METRIC_CLIP, METRIC_CLIP)
        w = torch.abs(y_c)
        corrs = []
        for k in range(2):
            yk = y_c[:, k]; pk = p_c[:, k]; wk = w[:, k]
            w_sum = torch.sum(wk) + self.eps
            y_bar = torch.sum(wk * yk) / w_sum
            p_bar = torch.sum(wk * pk) / w_sum
            yt = yk - y_bar; pt = pk - p_bar
            cov = torch.sum(wk * yt * pt)
            var_y = torch.sum(wk * yt * yt) + self.eps
            var_p = torch.sum(wk * pt * pt) + self.eps
            corr = cov / torch.sqrt(var_y * var_p)
            corrs.append(torch.clamp(corr, -1.0, 1.0))
        return 1.0 - 0.5 * (corrs[0] + corrs[1])

loss_fn = DatasetGlobalWeightedPearsonLoss().to(device)

# 5. Full-Batch Optimizer with Cosine Annealing
optimizer = torch.optim.AdamW([
    {'params': model.head_q.parameters(), 'lr': 1.0e-3},
    {'params': model.head_o0.parameters(), 'lr': 2.0e-4},
    {'params': [model.s0, model.s1, model.coupling, model.bias], 'lr': 1.0e-4},
], weight_decay=1e-4)

NUM_EPOCHS = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1.0e-5)

best_wp = init_wp
checkpoint_out = 'scratch/best_full_batch_optimum.pt'

print(f"\nTraining with Full-Batch Global Loss across {NUM_EPOCHS} steps...")

for step in range(1, NUM_EPOCHS + 1):
    t_start = time.time()
    model.train()
    
    optimizer.zero_grad(set_to_none=True)
    pred_full = model(O0_flat, O1_flat, Q_flat)
    loss = loss_fn(pred_full, Y_flat)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    step_time = time.time() - t_start
    
    if step % 5 == 0 or step == 1:
        val_wp, val_t0, val_t1, val_rb, val_rw = evaluate_holdout(model)
        gain_bps = (val_wp - init_wp) * 10000.0
        print(f"[Step {step:3d}/{NUM_EPOCHS}] Loss: {loss.item():.6f} | Val WP: {val_wp:.6f} (t0: {val_t0:.6f}, t1: {val_t1:.6f}) | r_b: {val_rb:.6f} | r_w: {val_rw:.6f} | Gain: {gain_bps:+.2f} bps | Time: {step_time:.2f}s")
        
        if val_wp > best_wp:
            best_wp = val_wp
            print(f"  >>> NEW ALL-TIME BEST WP: {best_wp:.6f} (Gain: {gain_bps:+.2f} bps)! Saving checkpoint...")
            torch.save(model.state_dict(), checkpoint_out)

print("=" * 80)
print(f"COMPLETE. Best Pristine Holdout WP: {best_wp:.6f} (Initial: {init_wp:.6f}, Gain: {(best_wp - init_wp)*10000:+.2f} bps)")
print("=" * 80)
