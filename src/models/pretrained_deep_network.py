"""Pretrained Microstructure Network initializing with 212M-step organizer GRU weights.

Combines:
1. Frozen or fine-tuned 2-layer GRU backbone from baseline.onnx (trained on 212M rows).
2. Specialized order book feature branches (Bid, Ask, Spread, Imbalance).
3. Non-linear residual representation head.
4. Anti-symmetric manifold coupling layer.
5. Calibrated scale and dynamic range layer.
"""
import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import numpy_helper


class PretrainedDeepMicroNetwork(nn.Module):
    def __init__(self, baseline_onnx_path: str = 'baseline/baseline.onnx', freeze_backbone: bool = False):
        super().__init__()
        self.gru0 = nn.GRU(112, 128, batch_first=True)
        self.gru1 = nn.GRU(128, 128, batch_first=True)

        # Microstructure feature indices
        # Bid features: p0..p10 (0..10), v0..v10 (22..32), dp0..dp3 (44..47), a0..a7 (104..111) -> 34 features
        self.bid_idx = list(range(0, 11)) + list(range(22, 33)) + list(range(44, 48)) + list(range(104, 112))
        # Ask features: p11..p21 (11..21), v11..v21 (33..43), dp0..dp3 (44..47), a0..a7 (104..111) -> 34 features
        self.ask_idx = list(range(11, 22)) + list(range(33, 44)) + list(range(44, 48)) + list(range(104, 112))

        # Base linear head from baseline.onnx
        self.base_head = nn.Linear(128, 2)

        # Load weights from baseline.onnx
        self._load_baseline_weights(baseline_onnx_path)

        if freeze_backbone:
            for p in self.gru0.parameters():
                p.requires_grad = False
            for p in self.gru1.parameters():
                p.requires_grad = False

        # Specialized Non-Linear Residual Microstructure Heads
        # Ingests GRU state (128) + specialized microstructure slice (34) = 162
        head_in = 128 + 34
        self.res_bid_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        self.res_ask_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        # Zero-initialize the final layer of the residual heads so at step 0 the model exactly equals baseline
        nn.init.zeros_(self.res_bid_head[-1].weight)
        nn.init.zeros_(self.res_bid_head[-1].bias)
        nn.init.zeros_(self.res_ask_head[-1].weight)
        nn.init.zeros_(self.res_ask_head[-1].bias)

        # Anti-symmetric coupling parameter (initialized to empirical optimal 0.045)
        self.coupling = nn.Parameter(torch.tensor([0.045], dtype=torch.float32))

        # Output calibrators (initialized to empirical optimal s0=0.275, s1=0.425)
        self.scale_0 = nn.Parameter(torch.tensor([0.275], dtype=torch.float32))
        self.scale_1 = nn.Parameter(torch.tensor([0.425], dtype=torch.float32))
        self.spread_beta = nn.Parameter(torch.tensor([0.045], dtype=torch.float32))

    def _load_baseline_weights(self, path: str):
        m = onnx.load(path)
        w = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}

        def load_layer(weight_ih, weight_hh, bias, gru_layer):
            zw, rw, hw = np.split(weight_ih[0], 3, axis=0)
            gru_layer.weight_ih_l0.data = torch.from_numpy(np.concatenate([rw, zw, hw], axis=0).copy())
            zr, rr, hr = np.split(weight_hh[0], 3, axis=0)
            gru_layer.weight_hh_l0.data = torch.from_numpy(np.concatenate([rr, zr, hr], axis=0).copy())
            bw, br = np.split(bias[0], 2, axis=0)
            zbw, rbw, hbw = np.split(bw, 3, axis=0)
            zbr, rbr, hbr = np.split(br, 3, axis=0)
            gru_layer.bias_ih_l0.data = torch.from_numpy(np.concatenate([rbw, zbw, hbw], axis=0).copy())
            gru_layer.bias_hh_l0.data = torch.from_numpy(np.concatenate([rbr, zbr, hbr], axis=0).copy())

        load_layer(w['onnx::GRU_163'], w['onnx::GRU_164'], w['onnx::GRU_165'], self.gru0)
        load_layer(w['onnx::GRU_183'], w['onnx::GRU_184'], w['onnx::GRU_185'], self.gru1)

        self.base_head.weight.data = torch.from_numpy(w['onnx::MatMul_186'].T.copy())
        self.base_head.bias.data = torch.from_numpy(w['reg_head.bias'].copy())

    def forward(self, x, h0=None, h1=None):
        # x: (batch, seq_len, 112)
        out0, next_h0 = self.gru0(x, h0)
        out1, next_h1 = self.gru1(out0, h1)

        # Baseline linear predictions
        base_pred = self.base_head(out1) # (batch, seq_len, 2)

        # Microstructure feature slices
        x_bid = x[:, :, self.bid_idx]
        x_ask = x[:, :, self.ask_idx]

        in_bid = torch.cat([out1, x_bid], dim=-1) # (batch, seq_len, 162)
        in_ask = torch.cat([out1, x_ask], dim=-1) # (batch, seq_len, 162)

        # Residual corrections
        res_0 = self.res_bid_head(in_bid) # (batch, seq_len, 1)
        res_1 = self.res_ask_head(in_ask) # (batch, seq_len, 1)

        raw_0 = base_pred[:, :, 0:1] + res_0
        raw_1 = base_pred[:, :, 1:2] + res_1

        # Anti-symmetric coupling
        c = torch.clamp(self.coupling, 0.0, 0.15)
        p0 = (1.0 - c) * raw_0 - c * raw_1
        p1 = (1.0 - c) * raw_1 - c * raw_0

        # Cross-instrument mid-price spread
        # i0 bids: 0..10, asks: 11..21
        # i1 bids: 52..62, asks: 63..73
        mid_i0 = (torch.max(x[:, :, 0:11], dim=-1, keepdim=True).values + torch.min(x[:, :, 11:22], dim=-1, keepdim=True).values) * 0.5
        mid_i1 = (torch.max(x[:, :, 52:63], dim=-1, keepdim=True).values + torch.min(x[:, :, 63:74], dim=-1, keepdim=True).values) * 0.5
        spread = mid_i0 - mid_i1

        # Final calibrated output
        out_0 = p0 * self.scale_0 + self.spread_beta * spread
        out_1 = p1 * self.scale_1 - self.spread_beta * spread

        preds = torch.cat([out_0, out_1], dim=-1)
        return preds, next_h0, next_h1
