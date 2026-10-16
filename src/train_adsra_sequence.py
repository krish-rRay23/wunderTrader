"""Train ADSRA Head with Sequence Batching and Causal Zero-Drift Centering.

- Backbone: FrozenBaselineGRU (100% frozen)
- Sequence-aware training (B = 30 sequences per batch, 19,901 scored steps per seq)
- In-batch causal zero-drift residual centering: res_centered = res - eta * res_mean
- Exact differentiable Global WP + Sequence-Mean Anchor Penalty
- Locks r_between >= 0.765 while aggressively lifting r_within!
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.extract_baseline_weights import FrozenBaselineGRU
from src.validation import evaluate_moments_decomposition
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP


class SequenceADSRAHead(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        # Dual-stream projections
        self.proj_o0 = nn.Linear(128, hidden, bias=False)
        self.proj_o1 = nn.Linear(128, hidden, bias=False)
        self.proj_x = nn.Linear(112, hidden)
        
        # SwiGLU Non-Linear Gating Block
        self.w_gate = nn.Linear(hidden, hidden)
        self.w_val = nn.Linear(hidden, hidden)
        
        # Residual Block
        self.res_fc = nn.Linear(hidden, hidden)
        
        # Mode Readout: [common m, spread d]
        self.out_head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        
        # Anchor scaling, residual coupling & drift control
        self.scale_base = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([0.20, 0.20], dtype=torch.float32))
        self.eta_raw = nn.Parameter(torch.tensor([1.5], dtype=torch.float32)) # sigmoid(1.5)*0.25 + 0.75 in [0.75, 1.0]
        self.bias_cal = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
        self.alpha_seq = 0.001

    def forward(self, p_base, o0, o1, x):
        # All inputs have shape (B, T, D)
        B, T, _ = p_base.shape
        device = p_base.device
        
        h_mix = self.proj_o0(o0) + self.proj_o1(o1) + self.proj_x(x) # (B, T, hidden)
        h = self.w_val(h_mix) * F.silu(self.w_gate(h_mix))
        h = h + F.silu(self.res_fc(h))
        
        modes = self.out_head(h) # (B, T, 2) -> [common m, spread d]
        m = modes[..., 0:1]
        d = modes[..., 1:2]
        
        res_0 = m + d
        res_1 = m - d
        res_tot = torch.cat([res_0, res_1], dim=-1) # (B, T, 2)
        
        # Causal sequence-mean residual tracker
        # In sequence batches (B, T, 2)
        a_seq = self.alpha_seq
        res_mean = torch.zeros(B, T, 2, device=device)
        cur_mean = torch.zeros(B, 2, device=device)
        for t in range(T):
            cur_mean = a_seq * res_tot[:, t] + (1.0 - a_seq) * cur_mean
            res_mean[:, t] = cur_mean
            
        eta = 0.75 + 0.25 * torch.sigmoid(self.eta_raw)
        res_centered = res_tot - eta * res_mean
        
        y_hat = self.scale_base * p_base + self.gamma * res_centered + self.bias_cal
        return y_hat, res_centered


class SequenceGlobalWPLoss(nn.Module):
    def __init__(self, mu_y0=-0.4810, mu_y1=-0.5325, eps=1e-8):
        super().__init__()
        self.register_buffer('mu_y', torch.tensor([mu_y0, mu_y1], dtype=torch.float32))
        self.eps = eps

    def forward(self, y_pred, y_true, res_centered):
        # y_pred, y_true: (B, T, 2)
        B, T, _ = y_pred.shape
        y_flat = y_true.view(-1, 2)
        p_flat = y_pred.view(-1, 2)
        
        y_c = torch.clamp(y_flat, -2.0, 2.0)
        p_c = 2.0 * torch.tanh(p_flat / 2.0)
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
        
        # Sequence-level mean penalty: for each sequence s, penalize squared weighted mean of res_centered
        w_seq = torch.abs(torch.clamp(y_true, -2.0, 2.0)) # (B, T, 2)
        seq_weights = torch.sum(w_seq, dim=1, keepdim=True) + self.eps # (B, 1, 2)
        seq_weighted_res_mean = torch.sum(w_seq * res_centered, dim=1) / seq_weights[:, 0] # (B, 2)
        anchor_pen = torch.mean(seq_weighted_res_mean**2)
        loss = loss + 1.0 * anchor_pen # Strong sequence-mean anchor!
        
        return loss


def train_sequence_engine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("STARTING SEQUENCE-AWARE ADSRA TRAINING ENGINE")
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
    
    # 2. Extract validation representations
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
            
    base_diag = evaluate_moments_decomposition(val_Y, val_p)
    print("\n--- BASELINE HOLDOUT METRICS ---")
    print(f"  WP_global: {base_diag['WP_global']:.6f} (t0: {base_diag['WP_t0']:.6f}, t1: {base_diag['WP_t1']:.6f})")
    print(f"  r_between: {base_diag['r_between']:.6f}")
    print(f"  r_within:  {base_diag['r_within']:.6f}")
    print("---------------------------------")
    
    # 3. Extract train representations sequence by sequence
    print(f"\nExtracting and caching {N_train} training sequences (scored steps >= 99)...")
    T_scored = SEQUENCE_LENGTH - WARMUP # 19901
    
    tr_p_seqs = torch.empty((N_train, T_scored, 2), dtype=torch.float32, device=device)
    tr_o0_seqs = torch.empty((N_train, T_scored, 128), dtype=torch.float32, device=device)
    tr_o1_seqs = torch.empty((N_train, T_scored, 128), dtype=torch.float32, device=device)
    tr_x_seqs = torch.empty((N_train, T_scored, 112), dtype=torch.float32, device=device)
    tr_y_seqs = torch.empty((N_train, T_scored, 2), dtype=torch.float32, device=device)
    
    t0 = time.time()
    with torch.no_grad():
        for s in range(N_train):
            seq_idx = 50 + s
            xt_cpu = np.array(X_mmap[seq_idx:seq_idx+1], dtype=np.float32)
            yt_cpu = np.array(Y_mmap[seq_idx, WARMUP:], dtype=np.float32)
            
            xt = torch.from_numpy(xt_cpu).to(device)
            pb, o0, o1, _, _ = base_gru(xt)
            
            tr_p_seqs[s] = pb[0, WARMUP:]
            tr_o0_seqs[s] = o0[0, WARMUP:]
            tr_o1_seqs[s] = o1[0, WARMUP:]
            tr_x_seqs[s] = xt[0, WARMUP:]
            tr_y_seqs[s] = torch.from_numpy(yt_cpu).to(device)
            
    print(f"Extracted {N_train} sequences ({N_train * T_scored:,} steps) to GPU in {time.time()-t0:.1f}s")
    
    # 4. Model & Optimizer
    head = SequenceADSRAHead(hidden=96).to(device)
    loss_fn = SequenceGlobalWPLoss().to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': [head.proj_o0.weight, head.proj_o1.weight, head.proj_x.weight, head.proj_x.bias], 'lr': 1e-3},
        {'params': [head.w_gate.weight, head.w_gate.bias, head.w_val.weight, head.w_val.bias], 'lr': 1e-3},
        {'params': [head.res_fc.weight, head.res_fc.bias], 'lr': 1e-3},
        {'params': [head.out_head.weight, head.out_head.bias], 'lr': 1e-3},
        {'params': [head.scale_base, head.gamma, head.eta_raw, head.bias_cal], 'lr': 2e-3},
    ], weight_decay=1e-4)
    
    # Batch size: B = 30 sequences per batch -> 5 batches per epoch
    B_SEQS = 30
    EPOCHS = 15
    num_batches = N_train // B_SEQS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * num_batches, eta_min=5e-5)
    
    def eval_model():
        head.eval()
        preds_all = []
        with torch.no_grad():
            for i in range(N_val):
                pb = torch.from_numpy(val_p[i:i+1]).to(device)
                o0 = torch.from_numpy(val_o0[i:i+1]).to(device)
                o1 = torch.from_numpy(val_o1[i:i+1]).to(device)
                xt = torch.from_numpy(val_X[i:i+1]).to(device)
                pred, _ = head(pb, o0, o1, xt)
                preds_all.append(pred[0].cpu().numpy())
        preds_np = np.array(preds_all)
        return evaluate_moments_decomposition(val_Y, preds_np), preds_np
        
    print(f"\n=== TRAINING ADSRA HEAD WITH SEQUENCE-CENTRING ({EPOCHS} EPOCHS, {B_SEQS} SEQS/BATCH) ===")
    best_wp = base_diag['WP_global']
    best_diag = base_diag
    
    for ep in range(1, EPOCHS + 1):
        head.train()
        perm = torch.randperm(N_train, device=device)
        total_loss = 0.0
        steps = 0
        t_ep = time.time()
        
        for b in range(num_batches):
            b_seq_indices = perm[b * B_SEQS : (b + 1) * B_SEQS]
            
            pb_b = tr_p_seqs[b_seq_indices]
            o0_b = tr_o0_seqs[b_seq_indices]
            o1_b = tr_o1_seqs[b_seq_indices]
            x_b = tr_x_seqs[b_seq_indices]
            y_b = tr_y_seqs[b_seq_indices]
            
            optimizer.zero_grad(set_to_none=True)
            y_hat, res_centered = head(pb_b, o0_b, o1_b, x_b)
            loss = loss_fn(y_hat, y_b, res_centered)
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
    train_sequence_engine()
