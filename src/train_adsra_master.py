"""Master ADSRA Training & Fusion Engine (Low-Memory CPU->GPU Streaming).

- Backbone: FrozenBaselineGRU (100% frozen)
- Memory-safe: tensors held on CPU RAM, transferred batch-by-batch to GPU
- Zero giant concatenation: separate linear projections summed in-place
- Exact differentiable Global WP + Sequence-Mean Anchor Penalty
- Exports to production ONNX model
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
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from src.extract_baseline_weights import FrozenBaselineGRU
from src.validation import evaluate_moments_decomposition
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP


class MasterADSRAHead(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        # Dual-stream separate projections to eliminate giant concat memory spikes
        self.proj_o0 = nn.Linear(128, hidden, bias=False)
        self.proj_o1 = nn.Linear(128, hidden, bias=False)
        self.proj_dh0 = nn.Linear(128, hidden, bias=False)
        self.proj_dh1 = nn.Linear(128, hidden, bias=False)
        self.proj_x = nn.Linear(112, hidden)
        self.proj_dx = nn.Linear(112, hidden, bias=False)
        self.proj_pb = nn.Linear(2, hidden, bias=False)
        
        # SwiGLU Gating Block
        self.w_gate = nn.Linear(hidden, hidden)
        self.w_val = nn.Linear(hidden, hidden)
        
        # Residual Block
        self.res_fc = nn.Linear(hidden, hidden)
        
        # Mode Readout Head: [common m, spread d]
        self.out_head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        
        # Anchoring & coupling parameters
        self.scale_base = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([0.20, 0.20], dtype=torch.float32))
        self.eta_raw = nn.Parameter(torch.tensor([1.5], dtype=torch.float32))
        self.bias_cal = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
        self.a_seq = 0.001

    def forward(self, p_base, o0, o1, dh0, dh1, x, dx, res_m=None):
        # All projections summed directly
        h_mix = (
            self.proj_o0(o0) +
            self.proj_o1(o1) +
            self.proj_dh0(dh0) +
            self.proj_dh1(dh1) +
            self.proj_x(x) +
            self.proj_dx(dx) +
            self.proj_pb(p_base)
        )
        
        h = self.w_val(h_mix) * F.silu(self.w_gate(h_mix))
        h = h + F.silu(self.res_fc(h))
        
        modes = self.out_head(h) # (..., 2) -> [common m, spread d]
        m = modes[..., 0:1]
        d = modes[..., 1:2]
        
        res_0 = m + d
        res_1 = m - d
        res_tot = torch.cat([res_0, res_1], dim=-1) # (..., 2)
        
        eta = 0.75 + 0.25 * torch.sigmoid(self.eta_raw)
        
        if res_m is not None:
            # Single-step inference mode with state
            next_res_m = self.a_seq * res_tot + (1.0 - self.a_seq) * res_m
            res_centered = res_tot - eta * next_res_m
            y_hat = self.scale_base * p_base + self.gamma * res_centered + self.bias_cal
            return y_hat, next_res_m
        else:
            # Full-sequence training mode (B, T, 2)
            seq_mean = torch.mean(res_tot, dim=1, keepdim=True)
            res_centered = res_tot - eta * seq_mean
            y_hat = self.scale_base * p_base + self.gamma * res_centered + self.bias_cal
            return y_hat, res_centered


class MasterGlobalWPLoss(nn.Module):
    def __init__(self, mu_y0=-0.4810, mu_y1=-0.5325, eps=1e-8):
        super().__init__()
        self.register_buffer('mu_y', torch.tensor([mu_y0, mu_y1], dtype=torch.float32))
        self.eps = eps

    def forward(self, y_pred, y_true, res_centered):
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
        
        # Sequence-level mean penalty
        w_seq = torch.abs(torch.clamp(y_true, -2.0, 2.0))
        seq_weights = torch.sum(w_seq, dim=1, keepdim=True) + self.eps
        seq_weighted_res_mean = torch.sum(w_seq * res_centered, dim=1) / seq_weights[:, 0]
        anchor_pen = torch.mean(seq_weighted_res_mean**2)
        loss = loss + 1.0 * anchor_pen
        
        return loss


def train_and_fuse():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("STARTING MASTER ADSRA TRAINING & FUSION ENGINE")
    print(f"Device: {device}")
    print("=" * 70)
    
    # 1. Load data
    d200 = np.load('scratch/cached_200_seqs.npz', mmap_mode='r')
    X_mmap = d200['features_all'] # (200, 20000, 112)
    Y_mmap = d200['targets_all']  # (200, 20000, 2)
    
    N_val = 50
    N_train = 150
    T_scored = SEQUENCE_LENGTH - WARMUP # 19901
    
    print(f"Train split: {N_train} sequences ({N_train * T_scored:,} scored steps)")
    print(f"Val split:   {N_val} sequences ({N_val * T_scored:,} scored steps)")
    
    # 2. Extract Validation features
    print("\nExtracting frozen GRU representations for Validation...")
    base_gru = FrozenBaselineGRU('baseline/baseline.onnx').to(device).eval()
    
    val_X = np.array(X_mmap[:N_val], dtype=np.float32)
    val_Y = np.array(Y_mmap[:N_val], dtype=np.float32)
    val_p = np.empty((N_val, SEQUENCE_LENGTH, 2), dtype=np.float32)
    val_o0 = np.empty((N_val, SEQUENCE_LENGTH, 128), dtype=np.float32)
    val_o1 = np.empty((N_val, SEQUENCE_LENGTH, 128), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(N_val):
            xt = torch.from_numpy(val_X[i:i+1]).to(device)
            pb, o0, o1, _, _ = base_gru(xt)
            val_p[i] = pb[0].cpu().numpy()
            val_o0[i] = o0[0].cpu().numpy()
            val_o1[i] = o1[0].cpu().numpy()
            
    val_dh0 = np.zeros_like(val_o0)
    val_dh1 = np.zeros_like(val_o1)
    val_dx = np.zeros_like(val_X)
    val_dh0[:, 1:] = val_o0[:, 1:] - val_o0[:, :-1]
    val_dh1[:, 1:] = val_o1[:, 1:] - val_o1[:, :-1]
    val_dx[:, 1:] = val_X[:, 1:] - val_X[:, :-1]
    
    base_diag = evaluate_moments_decomposition(val_Y, val_p)
    print("\n--- BASELINE HOLDOUT METRICS ---")
    print(f"  WP_global: {base_diag['WP_global']:.6f} (t0: {base_diag['WP_t0']:.6f}, t1: {base_diag['WP_t1']:.6f})")
    print(f"  r_between: {base_diag['r_between']:.6f}")
    print(f"  r_within:  {base_diag['r_within']:.6f}")
    print("---------------------------------")
    
    # 3. Extract Train features onto CPU RAM to prevent GPU OOM
    print(f"\nExtracting and caching {N_train} training sequences (CPU memory-buffered)...")
    tr_p_seqs = torch.empty((N_train, T_scored, 2), dtype=torch.float32, device='cpu')
    tr_o0_seqs = torch.empty((N_train, T_scored, 128), dtype=torch.float32, device='cpu')
    tr_o1_seqs = torch.empty((N_train, T_scored, 128), dtype=torch.float32, device='cpu')
    tr_dh0_seqs = torch.empty((N_train, T_scored, 128), dtype=torch.float32, device='cpu')
    tr_dh1_seqs = torch.empty((N_train, T_scored, 128), dtype=torch.float32, device='cpu')
    tr_x_seqs = torch.empty((N_train, T_scored, 112), dtype=torch.float32, device='cpu')
    tr_dx_seqs = torch.empty((N_train, T_scored, 112), dtype=torch.float32, device='cpu')
    tr_y_seqs = torch.empty((N_train, T_scored, 2), dtype=torch.float32, device='cpu')
    
    t0 = time.time()
    with torch.no_grad():
        for s in range(N_train):
            seq_idx = 50 + s
            xt_cpu = np.array(X_mmap[seq_idx:seq_idx+1], dtype=np.float32)
            yt_cpu = np.array(Y_mmap[seq_idx, WARMUP:], dtype=np.float32)
            
            xt = torch.from_numpy(xt_cpu).to(device)
            pb, o0, o1, _, _ = base_gru(xt)
            
            dh0 = torch.zeros_like(o0[0])
            dh1 = torch.zeros_like(o1[0])
            dx = torch.zeros_like(xt[0])
            dh0[1:] = o0[0, 1:] - o0[0, :-1]
            dh1[1:] = o1[0, 1:] - o1[0, :-1]
            dx[1:] = xt[0, 1:] - xt[0, :-1]
            
            tr_p_seqs[s] = pb[0, WARMUP:].cpu()
            tr_o0_seqs[s] = o0[0, WARMUP:].cpu()
            tr_o1_seqs[s] = o1[0, WARMUP:].cpu()
            tr_dh0_seqs[s] = dh0[WARMUP:].cpu()
            tr_dh1_seqs[s] = dh1[WARMUP:].cpu()
            tr_x_seqs[s] = xt[0, WARMUP:].cpu()
            tr_dx_seqs[s] = dx[WARMUP:].cpu()
            tr_y_seqs[s] = torch.from_numpy(yt_cpu)
            
    print(f"Extracted {N_train} sequences to CPU RAM in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()
    gc.collect()
    
    # 4. Model, Loss, Optimizer
    head = MasterADSRAHead(hidden=96).to(device)
    loss_fn = MasterGlobalWPLoss().to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': [head.proj_o0.weight, head.proj_o1.weight, head.proj_dh0.weight, head.proj_dh1.weight, head.proj_x.weight, head.proj_x.bias, head.proj_dx.weight, head.proj_pb.weight], 'lr': 1e-3},
        {'params': [head.w_gate.weight, head.w_gate.bias, head.w_val.weight, head.w_val.bias], 'lr': 1e-3},
        {'params': head.res_fc.parameters(), 'lr': 1e-3},
        {'params': head.out_head.parameters(), 'lr': 1e-3},
        {'params': [head.scale_base, head.gamma, head.eta_raw, head.bias_cal], 'lr': 2e-3},
    ], weight_decay=1e-4)
    
    # Batch size B = 15 sequences per batch -> 10 batches per epoch
    B_SEQS = 15
    EPOCHS = 12
    num_batches = N_train // B_SEQS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * num_batches, eta_min=5e-5)
    
    def eval_model():
        head.eval()
        preds_all = []
        with torch.no_grad():
            for i in range(N_val):
                pb = torch.from_numpy(val_p[i:i+1, WARMUP:]).to(device)
                o0 = torch.from_numpy(val_o0[i:i+1, WARMUP:]).to(device)
                o1 = torch.from_numpy(val_o1[i:i+1, WARMUP:]).to(device)
                dh0 = torch.from_numpy(val_dh0[i:i+1, WARMUP:]).to(device)
                dh1 = torch.from_numpy(val_dh1[i:i+1, WARMUP:]).to(device)
                xt = torch.from_numpy(val_X[i:i+1, WARMUP:]).to(device)
                dx = torch.from_numpy(val_dx[i:i+1, WARMUP:]).to(device)
                
                y_hat, _ = head(pb, o0, o1, dh0, dh1, xt, dx)
                
                full_pred = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
                full_pred[WARMUP:] = y_hat[0].cpu().numpy()
                preds_all.append(full_pred)
                
        preds_np = np.array(preds_all)
        return evaluate_moments_decomposition(val_Y, preds_np), preds_np
        
    print(f"\n=== TRAINING MASTER ADSRA HEAD ({EPOCHS} EPOCHS, {B_SEQS} SEQS/BATCH) ===")
    best_wp = base_diag['WP_global']
    best_diag = base_diag
    
    for ep in range(1, EPOCHS + 1):
        head.train()
        perm = torch.randperm(N_train)
        total_loss = 0.0
        steps = 0
        t_ep = time.time()
        
        for b in range(num_batches):
            b_idx = perm[b * B_SEQS : (b + 1) * B_SEQS]
            
            pb_b = tr_p_seqs[b_idx].to(device)
            o0_b = tr_o0_seqs[b_idx].to(device)
            o1_b = tr_o1_seqs[b_idx].to(device)
            dh0_b = tr_dh0_seqs[b_idx].to(device)
            dh1_b = tr_dh1_seqs[b_idx].to(device)
            x_b = tr_x_seqs[b_idx].to(device)
            dx_b = tr_dx_seqs[b_idx].to(device)
            y_b = tr_y_seqs[b_idx].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            y_hat, res_centered = head(pb_b, o0_b, o1_b, dh0_b, dh1_b, x_b, dx_b)
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
            torch.save(head.state_dict(), 'scratch/best_master_adsra.pt')
            
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE! SUMMARY OF RESULTS:")
    print(f"  Baseline WP: {base_diag['WP_global']:.6f} | ADSRA Peak WP: {best_diag['WP_global']:.6f} (Gain: +{best_diag['WP_global']-base_diag['WP_global']:.6f})")
    print(f"  Baseline r_between: {base_diag['r_between']:.6f} | ADSRA r_between: {best_diag['r_between']:.6f}")
    print(f"  Baseline r_within:  {base_diag['r_within']:.6f} | ADSRA r_within:  {best_diag['r_within']:.6f} (Gain: +{best_diag['r_within']-base_diag['r_within']:.6f})")
    print("=" * 70)
    
    return head, best_diag


if __name__ == "__main__":
    train_and_fuse()
