"""Step 2: Vectorized Feature & Hidden-State Precomputation Pipeline.

Computes:
1. Frozen Baseline GRU outputs: p_base (2), h0 (128), h1 (128), dh0 (128), dh1 (128)
2. Permutation-invariant cross-sectional order book moments and lead-lag signals (32)
3. Zero-lag causal multi-scale EMA divergence bank (MACD band-pass filters)
4. Elementwise notional interaction features (44)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.extract_baseline_weights import FrozenBaselineGRU
from utils import SEQUENCE_LENGTH, WARMUP


def compute_invariants_torch(x):
    """
    Computes 32 permutation-invariant cross-sectional order book moments and cross-instrument lead-lag features.
    x: (..., 112)
    """
    # i0: p (0..21), v (22..43), dp (44..47), dv (48..51)
    # i1: p (52..73), v (74..95), dp (96..99), dv (100..103)
    p0 = x[..., 0:22]
    v0 = x[..., 22:44]
    dp0 = x[..., 44:48]
    dv0 = x[..., 48:52]
    
    p1 = x[..., 52:74]
    v1 = x[..., 74:96]
    dp1 = x[..., 96:100]
    dv1 = x[..., 100:104]
    
    # Positive volumes
    v0_pos = F.relu(v0)
    v1_pos = F.relu(v1)
    dv0_pos = F.relu(dv0)
    dv1_pos = F.relu(dv1)
    
    # 1. VWAP
    vwap0 = torch.sum(p0 * v0_pos, dim=-1, keepdim=True) / (torch.sum(v0_pos, dim=-1, keepdim=True) + 1e-5)
    vwap1 = torch.sum(p1 * v1_pos, dim=-1, keepdim=True) / (torch.sum(v1_pos, dim=-1, keepdim=True) + 1e-5)
    
    # 2. Price Mean & Std
    p0_mean = torch.mean(p0, dim=-1, keepdim=True)
    p1_mean = torch.mean(p1, dim=-1, keepdim=True)
    p0_std = torch.std(p0, dim=-1, keepdim=True)
    p1_std = torch.std(p1, dim=-1, keepdim=True)
    
    # 3. Book Skew
    skew0 = vwap0 - p0_mean
    skew1 = vwap1 - p1_mean
    
    # 4. Volume Depth & Imbalance
    v0_sum = torch.sum(v0, dim=-1, keepdim=True)
    v1_sum = torch.sum(v1, dim=-1, keepdim=True)
    v0_pos_sum = torch.sum(v0_pos, dim=-1, keepdim=True)
    v1_pos_sum = torch.sum(v1_pos, dim=-1, keepdim=True)
    v0_neg_sum = torch.sum(F.relu(-v0), dim=-1, keepdim=True)
    v1_neg_sum = torch.sum(F.relu(-v1), dim=-1, keepdim=True)
    v0_imb = (v0_pos_sum - v0_neg_sum) / (v0_pos_sum + v0_neg_sum + 1e-5)
    v1_imb = (v1_pos_sum - v1_neg_sum) / (v1_pos_sum + v1_neg_sum + 1e-5)
    
    # 5. Trade Features
    trade_vwap0 = torch.sum(dp0 * dv0_pos, dim=-1, keepdim=True) / (torch.sum(dv0_pos, dim=-1, keepdim=True) + 1e-5)
    trade_vwap1 = torch.sum(dp1 * dv1_pos, dim=-1, keepdim=True) / (torch.sum(dv1_pos, dim=-1, keepdim=True) + 1e-5)
    trade_disloc0 = trade_vwap0 - vwap0
    trade_disloc1 = trade_vwap1 - vwap1
    dv0_sum = torch.sum(dv0, dim=-1, keepdim=True)
    dv1_sum = torch.sum(dv1, dim=-1, keepdim=True)
    
    # 6. Cross-Instrument Lead-Lag Signals
    vwap_diff = vwap1 - vwap0
    skew_diff = skew1 - skew0
    v_imb_diff = v1_imb - v0_imb
    trade_disloc_diff = trade_disloc1 - trade_disloc0
    flow_diff = dv1_sum - dv0_sum
    
    # Extra moments: L2 norm of volume
    v0_l2 = torch.sqrt(torch.sum(v0**2, dim=-1, keepdim=True) + 1e-5)
    v1_l2 = torch.sqrt(torch.sum(v1**2, dim=-1, keepdim=True) + 1e-5)
    
    # Total: 10 features for i0, 10 features for i1, 7 cross-instrument features + 5 extras = 32 dims
    invariants = torch.cat([
        vwap0, p0_mean, p0_std, skew0, v0_sum, v0_pos_sum, v0_imb, trade_vwap0, trade_disloc0, dv0_sum, # 10
        vwap1, p1_mean, p1_std, skew1, v1_sum, v1_pos_sum, v1_imb, trade_vwap1, trade_disloc1, dv1_sum, # 10
        vwap_diff, skew_diff, v_imb_diff, trade_disloc_diff, flow_diff, # 5
        v0_l2, v1_l2, (vwap0 - trade_vwap1), (skew0 - trade_disloc1), (v0_sum - v1_sum), # 5
        (v0_pos_sum - v1_pos_sum), (trade_vwap0 - trade_vwap1) # 2
    ], dim=-1)
    
    return invariants # shape (..., 32)


class CausalStateBank(nn.Module):
    """
    Maintains causal multi-scale EMA state and state velocities across sequence time.
    """
    def __init__(self, d_s=144, d_h=256):
        super().__init__()
        self.d_s = d_s
        self.d_h = d_h
        self.a1 = 0.35
        self.a2 = 0.08
        self.a3 = 0.015

    def forward_sequence(self, s_seq, h_seq):
        """
        Runs sequence-level causal filter bank over (T, D_s) and (T, D_h).
        Returns:
            dx: (T, 144)
            div_s: (T, 144*3) = (T, 432)
            div_h: (T, 256*3) = (T, 768)
        """
        T = s_seq.shape[0]
        device = s_seq.device
        
        # In PyTorch, run causal IIR loop efficiently
        e1_s = torch.zeros(T, self.d_s, device=device)
        e2_s = torch.zeros(T, self.d_s, device=device)
        e3_s = torch.zeros(T, self.d_s, device=device)
        
        e1_h = torch.zeros(T, self.d_h, device=device)
        e2_h = torch.zeros(T, self.d_h, device=device)
        e3_h = torch.zeros(T, self.d_h, device=device)
        
        # Step 0 initialization to eliminate cold-start transients
        cur_e1_s = s_seq[0].clone()
        cur_e2_s = s_seq[0].clone()
        cur_e3_s = s_seq[0].clone()
        
        cur_e1_h = h_seq[0].clone()
        cur_e2_h = h_seq[0].clone()
        cur_e3_h = h_seq[0].clone()
        
        e1_s[0] = cur_e1_s; e2_s[0] = cur_e2_s; e3_s[0] = cur_e3_s
        e1_h[0] = cur_e1_h; e2_h[0] = cur_e2_h; e3_h[0] = cur_e3_h
        
        a1, a2, a3 = self.a1, self.a2, self.a3
        for t in range(1, T):
            st = s_seq[t]
            cur_e1_s = a1 * st + (1.0 - a1) * cur_e1_s
            cur_e2_s = a2 * st + (1.0 - a2) * cur_e2_s
            cur_e3_s = a3 * st + (1.0 - a3) * cur_e3_s
            e1_s[t] = cur_e1_s; e2_s[t] = cur_e2_s; e3_s[t] = cur_e3_s
            
            ht = h_seq[t]
            cur_e1_h = a1 * ht + (1.0 - a1) * cur_e1_h
            cur_e2_h = a2 * ht + (1.0 - a2) * cur_e2_h
            cur_e3_h = a3 * ht + (1.0 - a3) * cur_e3_h
            e1_h[t] = cur_e1_h; e2_h[t] = cur_e2_h; e3_h[t] = cur_e3_h
            
        dx = torch.zeros_like(s_seq)
        dx[1:] = s_seq[1:] - s_seq[:-1]
        
        div_fast_s = s_seq - e1_s
        div_med_s = e1_s - e2_s
        div_slow_s = e2_s - e3_s
        div_s = torch.cat([div_fast_s, div_med_s, div_slow_s], dim=-1) # (T, 432)
        
        div_fast_h = h_seq - e1_h
        div_med_h = e1_h - e2_h
        div_slow_h = e2_h - e3_h
        div_h = torch.cat([div_fast_h, div_med_h, div_slow_h], dim=-1) # (T, 768)
        
        return dx, div_s, div_h


def extract_full_features_sequence(x_seq, base_gru):
    """
    Extracts the full composite feature representation for a single sequence.
    x_seq: (T, 112) float32 tensor
    base_gru: FrozenBaselineGRU module
    """
    device = x_seq.device
    T = x_seq.shape[0]
    
    with torch.no_grad():
        p_base, o0, o1, _, _ = base_gru(x_seq.unsqueeze(0))
        p_base = p_base[0] # (T, 2)
        h0 = o0[0]         # (T, 128)
        h1 = o1[0]         # (T, 128)
        
    dh0 = torch.zeros_like(h0)
    dh1 = torch.zeros_like(h1)
    dh0[1:] = h0[1:] - h0[:-1]
    dh1[1:] = h1[1:] - h1[:-1]
    
    # Extended baseline signals: [p0, p1, p0 - p1, p0 + p1] (4 dims)
    p_spread = (p_base[:, 0:1] - p_base[:, 1:2])
    p_sum = (p_base[:, 0:1] + p_base[:, 1:2])
    p_ext = torch.cat([p_base, p_spread, p_sum], dim=-1) # (T, 4)
    
    # 32 Invariant features
    invariants = compute_invariants_torch(x_seq) # (T, 32)
    
    # Level-wise notional: 22 prices * 22 volumes for i0, 22 prices * 22 volumes for i1 -> 44 dims
    notional = torch.cat([
        x_seq[:, 0:22] * x_seq[:, 22:44],
        x_seq[:, 52:74] * x_seq[:, 74:96]
    ], dim=-1) # (T, 44)
    
    s_t = torch.cat([x_seq, invariants], dim=-1) # (T, 144)
    h_cat = torch.cat([h0, h1], dim=-1)          # (T, 256)
    
    state_bank = CausalStateBank().to(device)
    dx, div_s, div_h = state_bank.forward_sequence(s_t, h_cat)
    
    # Composite Feature Representation for Stream A (Gated Residual SwiGLU MLP):
    # s_t (144) + notional (44) + h0 (128) + h1 (128) + dh0 (128) + dh1 (128) +
    # dx (144) + div_s (432) + div_h (768) + p_ext (4) = 1,920 dims!
    feat_stream_a = torch.cat([
        s_t, notional, h0, h1, dh0, dh1, dx, div_s, div_h, p_ext
    ], dim=-1)
    
    # Composite Feature Representation for Stream B (High-Frequency Innovation Micro-GRU):
    # Strictly zero-mean innovations: dx (144) + div_fast_s (144) + div_med_s (144) + dh0 (128) + dh1 (128) + a0..a7 (8) = 696 dims!
    div_fast_s = div_s[:, 0:144]
    div_med_s = div_s[:, 144:288]
    a_alphas = x_seq[:, 104:112]
    feat_stream_b = torch.cat([
        dx, div_fast_s, div_med_s, dh0, dh1, a_alphas
    ], dim=-1)
    
    return p_base, feat_stream_a, feat_stream_b


if __name__ == "__main__":
    print("Testing Feature Pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_gru = FrozenBaselineGRU().to(device)
    dummy_x = torch.randn(SEQUENCE_LENGTH, 112, device=device)
    p_b, f_a, f_b = extract_full_features_sequence(dummy_x, base_gru)
    print(f"Extracted p_base:   {p_b.shape}")
    print(f"Extracted Stream A: {f_a.shape} (D_in = {f_a.shape[-1]})")
    print(f"Extracted Stream B: {f_b.shape} (D_in = {f_b.shape[-1]})")
    print("Feature Pipeline test SUCCESS!")
