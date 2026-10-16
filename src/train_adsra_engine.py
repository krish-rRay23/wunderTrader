"""Train ADSRA Head on Pre-Trained Recurrent Manifold.

- Backbone: FrozenBaselineGRU (100% frozen, zero drift)
- Head: Multi-Stream SwiGLU Residual Mode Decomposition
- Objective: Global Weighted Pearson Loss + Sequence-Mean Anchor Penalty
- Validation: 50 Holdout Sequences (evaluating WP_global, r_between, r_within)
"""
import sys
import time
import gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.extract_baseline_weights import FrozenBaselineGRU
from src.validation import evaluate_moments_decomposition
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP


class ADSRAHeadModule(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.proj_o0 = nn.Linear(128, hidden, bias=False)
        self.proj_o1 = nn.Linear(128, hidden, bias=False)
        self.proj_x = nn.Linear(112, hidden)
        
        # SwiGLU Block
        self.w_gate1 = nn.Linear(hidden, hidden)
        self.w_val1 = nn.Linear(hidden, hidden)
        
        # Residual Block
        self.res_fc = nn.Linear(hidden, hidden)
        
        # Mode Readout: [common m, spread d]
        self.out_head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        
        # Anchor scaling & residual coupling
        self.scale_base = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([0.20, 0.20], dtype=torch.float32))
        self.bias_cal = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

    def forward(self, p_base, o0, o1, x):
        h_mix = self.proj_o0(o0) + self.proj_o1(o1) + self.proj_x(x)
        h = self.w_val1(h_mix) * F.silu(self.w_gate1(h_mix))
        h = h + F.silu(self.res_fc(h))
        
        modes = self.out_head(h) # (..., 2)
        m = modes[..., 0:1]
        d = modes[..., 1:2]
        
        res_0 = m + d
        res_1 = m - d
        res = torch.cat([res_0, res_1], dim=-1)
        
        y_hat = self.scale_base * p_base + self.gamma * res + self.bias_cal
        return y_hat


class GlobalWPLoss(nn.Module):
    def __init__(self, mu_y0=-0.4810, mu_y1=-0.5325, eps=1e-8):
        super().__init__()
        self.register_buffer('mu_y', torch.tensor([mu_y0, mu_y1], dtype=torch.float32))
        self.eps = eps

    def forward(self, y_pred, y_true, p_base=None):
        y_c = torch.clamp(y_true, -2.0, 2.0)
        p_c = 2.0 * torch.tanh(y_pred / 2.0)
        w = torch.abs(y_c)
        
        corrs = []
        wmse = 0.0
        for s in range(2):
            ys = y_c[:, s]
            ps = p_c[:, s]
            ws = w[:, s]
            w_sum = torch.sum(ws) + self.eps
            
            dy = ys - self.mu_y[s]
            mp = torch.sum(ws * ps) / w_sum
            dp = ps - mp
            
            cov = torch.sum(ws * dy * dp)
            var_y = torch.sum(ws * dy * dy) + self.eps
            var_p = torch.sum(ws * dp * dp) + self.eps
            
            corr = cov / torch.sqrt(var_y * var_p)
            corrs.append(torch.clamp(corr, -1.0, 1.0))
            wmse += torch.sum(ws * (ys - ps)**2) / w_sum
            
        wp_loss = 1.0 - 0.5 * (corrs[0] + corrs[1])
        loss = wp_loss + 0.15 * wmse
        
        # Sequence-mean anchor penalty: penalizes residual divergence
        if p_base is not None:
            residual = y_pred - p_base
            anchor_pen = torch.mean(residual**2)
            loss = loss + 0.30 * anchor_pen
            
        return loss


def train_engine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("STARTING ADSRA TRAINING ENGINE (FROZEN BACKBONE + STREAMLINED HEAD)")
    print(f"Device: {device}")
    print("=" * 70)
    
    # 1. Load memory-mapped data
    print("Loading memory-mapped cached sequences...")
    d200 = np.load('scratch/cached_200_seqs.npz', mmap_mode='r')
    X_mmap = d200['features_all'] # (200, 20000, 112)
    Y_mmap = d200['targets_all']  # (200, 20000, 2)
    
    N_val = 50
    N_train = 150 # Sequences 50..199
    
    print(f"Train split: {N_train} sequences ({N_train * (SEQUENCE_LENGTH - WARMUP):,} scored steps)")
    print(f"Val split:   {N_val} sequences ({N_val * (SEQUENCE_LENGTH - WARMUP):,} scored steps)")
    
    # 2. Extract taps for Validation set
    print("\nExtracting frozen GRU representations for Validation (50 seqs)...")
    base_gru = FrozenBaselineGRU('baseline/baseline.onnx').to(device).eval()
    
    val_p = np.empty((N_val, SEQUENCE_LENGTH, 2), dtype=np.float32)
    val_o0 = np.empty((N_val, SEQUENCE_LENGTH, 128), dtype=np.float32)
    val_o1 = np.empty((N_val, SEQUENCE_LENGTH, 128), dtype=np.float32)
    val_X = np.array(X_mmap[:N_val], dtype=np.float32)
    val_Y = np.array(Y_mmap[:N_val], dtype=np.float32)
    
    with torch.no_grad():
        for i in range(N_val):
            xt = torch.from_numpy(val_X[i:i+1]).to(device)
            pb, o0, o1, _, _ = base_gru(xt)
            val_p[i] = pb[0].cpu().numpy()
            val_o0[i] = o0[0].cpu().numpy()
            val_o1[i] = o1[0].cpu().numpy()
            
    # Baseline validation evaluation
    base_diag = evaluate_moments_decomposition(val_Y, val_p)
    print("\n--- BASELINE HOLDOUT METRICS ---")
    print(f"  WP_global: {base_diag['WP_global']:.6f} (t0: {base_diag['WP_t0']:.6f}, t1: {base_diag['WP_t1']:.6f})")
    print(f"  r_between: {base_diag['r_between']:.6f}")
    print(f"  r_within:  {base_diag['r_within']:.6f}")
    print("---------------------------------")
    
    # 3. Extract taps for Train set directly onto GPU in chunks
    print(f"\nExtracting and caching {N_train} training sequences (scored steps >= 99)...")
    N_steps_per_seq = SEQUENCE_LENGTH - WARMUP
    total_tr_steps = N_train * N_steps_per_seq
    
    tr_p_flat = torch.empty((total_tr_steps, 2), dtype=torch.float32, device=device)
    tr_o0_flat = torch.empty((total_tr_steps, 128), dtype=torch.float32, device=device)
    tr_o1_flat = torch.empty((total_tr_steps, 128), dtype=torch.float32, device=device)
    tr_x_flat = torch.empty((total_tr_steps, 112), dtype=torch.float32, device=device)
    tr_y_flat = torch.empty((total_tr_steps, 2), dtype=torch.float32, device=device)
    
    t0 = time.time()
    with torch.no_grad():
        for s in range(N_train):
            seq_idx = 50 + s
            xt_cpu = np.array(X_mmap[seq_idx:seq_idx+1], dtype=np.float32)
            yt_cpu = np.array(Y_mmap[seq_idx, WARMUP:], dtype=np.float32)
            
            xt = torch.from_numpy(xt_cpu).to(device)
            pb, o0, o1, _, _ = base_gru(xt)
            
            st_idx = s * N_steps_per_seq
            end_idx = st_idx + N_steps_per_seq
            
            tr_p_flat[st_idx:end_idx] = pb[0, WARMUP:]
            tr_o0_flat[st_idx:end_idx] = o0[0, WARMUP:]
            tr_o1_flat[st_idx:end_idx] = o1[0, WARMUP:]
            tr_x_flat[st_idx:end_idx] = xt[0, WARMUP:]
            tr_y_flat[st_idx:end_idx] = torch.from_numpy(yt_cpu).to(device)
            
    print(f"Extracted {total_tr_steps:,} training steps to GPU in {time.time()-t0:.1f}s")
    
    # 4. Model & Optimizer
    head = ADSRAHeadModule(hidden=96).to(device)
    loss_fn = GlobalWPLoss().to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': [head.proj_o0.weight, head.proj_o1.weight, head.proj_x.weight, head.proj_x.bias], 'lr': 1e-3},
        {'params': [head.w_gate1.weight, head.w_gate1.bias, head.w_val1.weight, head.w_val1.bias], 'lr': 1e-3},
        {'params': [head.res_fc.weight, head.res_fc.bias], 'lr': 1e-3},
        {'params': [head.out_head.weight, head.out_head.bias], 'lr': 1e-3},
        {'params': [head.scale_base, head.gamma, head.bias_cal], 'lr': 2e-3},
    ], weight_decay=1e-4)
    
    BATCH_SIZE = 65536
    EPOCHS = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * (total_tr_steps // BATCH_SIZE), eta_min=5e-5)
    
    def eval_model():
        head.eval()
        preds_all = []
        with torch.no_grad():
            for i in range(N_val):
                pb = torch.from_numpy(val_p[i]).to(device)
                o0 = torch.from_numpy(val_o0[i]).to(device)
                o1 = torch.from_numpy(val_o1[i]).to(device)
                xt = torch.from_numpy(val_X[i]).to(device)
                pred = head(pb, o0, o1, xt).cpu().numpy()
                preds_all.append(pred)
        preds_np = np.array(preds_all)
        return evaluate_moments_decomposition(val_Y, preds_np), preds_np
        
    print(f"\n=== TRAINING ADSRA HEAD ({EPOCHS} EPOCHS, BATCH {BATCH_SIZE}) ===")
    best_wp = base_diag['WP_global']
    best_diag = base_diag
    
    for ep in range(1, EPOCHS + 1):
        head.train()
        perm = torch.randperm(total_tr_steps, device=device)
        total_loss = 0.0
        steps = 0
        t_ep = time.time()
        
        for idx in range(0, total_tr_steps - BATCH_SIZE + 1, BATCH_SIZE):
            b_idx = perm[idx:idx + BATCH_SIZE]
            pb_b = tr_p_flat[b_idx]
            o0_b = tr_o0_flat[b_idx]
            o1_b = tr_o1_flat[b_idx]
            x_b = tr_x_flat[b_idx]
            y_b = tr_y_flat[b_idx]
            
            optimizer.zero_grad(set_to_none=True)
            y_hat = head(pb_b, o0_b, o1_b, x_b)
            loss = loss_fn(y_hat, y_b, pb_b)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            steps += 1
            
        dt_ep = time.time() - t_ep
        diag, _ = eval_model()
        wp = diag['WP_global']
        r_b = diag['r_between']
        r_w = diag['r_within']
        
        mark = " *** NEW BEST ***" if wp > best_wp else ""
        print(f"Ep {ep:2d}/{EPOCHS} ({dt_ep:.1f}s) | Loss: {total_loss/max(1,steps):.4f} | WP: {wp:.6f} | r_between: {r_b:.6f} | r_within: {r_w:.6f}{mark}")
        
        if wp > best_wp:
            best_wp = wp
            best_diag = diag
            torch.save(head.state_dict(), 'scratch/best_adsra_head.pt')
            
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE! SUMMARY OF RESULTS:")
    print(f"  Baseline WP: {base_diag['WP_global']:.6f} | ADSRA Peak WP: {best_diag['WP_global']:.6f} (Gain: +{best_diag['WP_global']-base_diag['WP_global']:.6f})")
    print(f"  Baseline r_between: {base_diag['r_between']:.6f} | ADSRA r_between: {best_diag['r_between']:.6f}")
    print(f"  Baseline r_within:  {base_diag['r_within']:.6f} | ADSRA r_within:  {best_diag['r_within']:.6f} (Gain: +{best_diag['r_within']-base_diag['r_within']:.6f})")
    print("=" * 70)
    
    return head, best_diag


if __name__ == "__main__":
    train_engine()
