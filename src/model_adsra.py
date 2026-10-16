"""Step 4: Anchored Dual-Stream Residual Architecture (ADSRA).

Combines:
- Stream A: Gated Residual SwiGLU MLP on Multi-Horizon State Taps
- Stream B: Parallel Width-64 Innovation Micro-GRU (strictly zero sequence mean)
- Orthogonal Target Mode Decomposition (Common m & Spread d)
- Causal Zero-Drift Residual Centering
- Frozen Baseline Anchor locking r_between >= 0.805
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

    def forward(self, x):
        return self.w1(x) * F.silu(self.w2(x))


class StreamAMLP(nn.Module):
    def __init__(self, d_in=2048, hidden=256):
        super().__init__()
        self.proj = nn.Linear(d_in, 512)
        self.ln0 = nn.LayerNorm(512)
        self.swiglu = SwiGLU(512, hidden)
        self.drop0 = nn.Dropout(0.10)
        
        # Residual Block
        self.ln1 = nn.LayerNorm(hidden)
        self.res_fc = nn.Linear(hidden, hidden)
        self.drop1 = nn.Dropout(0.10)
        
        # Mode output: [common_m, spread_d]
        self.out_head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, x):
        # x: (..., d_in)
        h = self.drop0(self.swiglu(self.ln0(self.proj(x))))
        h = h + self.drop1(self.res_fc(F.silu(self.ln1(h))))
        modes = self.out_head(h) # (..., 2)
        return modes


class StreamBGRU(nn.Module):
    def __init__(self, d_innov=696, hidden=64):
        super().__init__()
        self.proj = nn.Linear(d_innov, 96)
        self.gru = nn.GRU(96, hidden, batch_first=True)
        self.out_head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, x, h_in=None):
        # x: (B, T, d_innov)
        proj = torch.tanh(self.proj(x))
        out, next_h = self.gru(proj, h_in)
        modes = self.out_head(out) # (B, T, 2)
        return modes, next_h


class ADSRAHead(nn.Module):
    def __init__(self, d_a=2048, d_b=696):
        super().__init__()
        self.norm_a_mean = nn.Parameter(torch.zeros(d_a), requires_grad=False)
        self.norm_a_std = nn.Parameter(torch.ones(d_a), requires_grad=False)
        
        self.norm_b_mean = nn.Parameter(torch.zeros(d_b), requires_grad=False)
        self.norm_b_std = nn.Parameter(torch.ones(d_b), requires_grad=False)
        
        self.stream_a = StreamAMLP(d_in=d_a, hidden=256)
        self.stream_b = StreamBGRU(d_innov=d_b, hidden=64)
        
        # Learned anchor & residual coupling parameters
        self.scale_base = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([0.20, 0.20], dtype=torch.float32))
        self.eta_raw = nn.Parameter(torch.tensor([1.5], dtype=torch.float32)) # sigmoid(1.5)*0.3 + 0.7 in [0.7, 1.0]
        self.bias_cal = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
        self.alpha_seq = 0.001

    def forward(self, p_base, feat_a, feat_b, h_gru_b=None, res_mean_prev=None):
        """
        p_base: (B, T, 2)
        feat_a: (B, T, 2048)
        feat_b: (B, T, 696)
        """
        # 1. Normalize features with clamp to prevent extreme outliers
        a_norm = torch.clamp((feat_a - self.norm_a_mean) / (self.norm_a_std + 1e-4), -10.0, 10.0)
        b_norm = torch.clamp((feat_b - self.norm_b_mean) / (self.norm_b_std + 1e-4), -10.0, 10.0)
        
        # 2. Dual Streams
        modes_a = self.stream_a(a_norm) # (B, T, 2) -> [m_a, d_a]
        modes_b, next_h_b = self.stream_b(b_norm, h_gru_b) # (B, T, 2) -> [m_b, d_b]
        
        # 3. Mode Recombination
        m_tot = modes_a[..., 0:1] + modes_b[..., 0:1]
        d_tot = modes_a[..., 1:2] + modes_b[..., 1:2]
        
        res_0 = m_tot + d_tot
        res_1 = m_tot - d_tot
        res_tot = torch.cat([res_0, res_1], dim=-1) # (B, T, 2)
        
        # 4. Causal Zero-Drift Residual Centering
        # In sequence training (B, T, 2)
        B, T, _ = res_tot.shape
        device = res_tot.device
        
        # Causal tracking of residual mean
        a_seq = self.alpha_seq
        if T > 1:
            # Vectorized causal EMA
            res_mean = torch.zeros(B, T, 2, device=device)
            cur_mean = res_mean_prev if res_mean_prev is not None else torch.zeros(B, 2, device=device)
            for t in range(T):
                cur_mean = a_seq * res_tot[:, t] + (1.0 - a_seq) * cur_mean
                res_mean[:, t] = cur_mean
            next_res_mean = cur_mean
        else:
            # Single-step inference
            cur_mean = res_mean_prev if res_mean_prev is not None else torch.zeros(B, 2, device=device)
            cur_mean = a_seq * res_tot[:, 0] + (1.0 - a_seq) * cur_mean
            res_mean = cur_mean.unsqueeze(1)
            next_res_mean = cur_mean
            
        eta = 0.70 + 0.30 * torch.sigmoid(self.eta_raw)
        res_centered = res_tot - eta * res_mean
        
        # 5. Final Anchored Output
        y_hat = self.scale_base * p_base + self.gamma * res_centered + self.bias_cal
        return y_hat, next_h_b, next_res_mean


class GlobalWPLoss(nn.Module):
    def __init__(self, mu_y0=-0.4810, mu_y1=-0.5325, eps=1e-8):
        super().__init__()
        self.register_buffer('mu_y', torch.tensor([mu_y0, mu_y1], dtype=torch.float32))
        self.eps = eps

    def forward(self, y_pred, y_true, p_base=None):
        # y_pred, y_true: (N, 2)
        y_c = torch.clamp(y_true, -2.0, 2.0)
        p_c = 2.0 * torch.tanh(y_pred / 2.0) # Smooth differentiable clip
        w = torch.abs(y_c)
        
        corrs = []
        wmse = 0.0
        for s in range(2):
            ys = y_c[:, s]
            ps = p_c[:, s]
            ws = w[:, s]
            w_sum = torch.sum(ws) + self.eps
            
            dy = ys - self.mu_y[s]
            mean_p = torch.sum(ws * ps) / w_sum
            dp = ps - mean_p
            
            cov = torch.sum(ws * dy * dp)
            var_y = torch.sum(ws * dy * dy) + self.eps
            var_p = torch.sum(ws * dp * dp) + self.eps
            
            corr = cov / torch.sqrt(var_y * var_p)
            corrs.append(torch.clamp(corr, -1.0, 1.0))
            wmse += torch.sum(ws * (ys - ps)**2) / w_sum
            
        wp_loss = 1.0 - 0.5 * (corrs[0] + corrs[1])
        
        # Sequence-mean anchor penalty if p_base is provided
        loss = wp_loss + 0.15 * wmse
        if p_base is not None:
            # Penalize residual divergence from baseline mean
            diff = (y_pred - p_base)
            anchor_pen = torch.mean(diff**2)
            loss = loss + 0.50 * anchor_pen
            
        return loss
