"""Fused Institutional Connectome + Quant Ensemble.

Fuses:
1. Wunder Fund Pretrained 2-Layer GRU backbone (shared single pass).
2. Institutional Connectome Head (alpha=0.60):
   - Microstructure extractor (19 dims: alphas a0..a7, trades dv0..dv3, spread, L1-3 imbalances)
   - Decoupled dual-stream GELU MLPs
   - Anti-symmetric cross-coupling and output calibration
3. Sub 6 Breakthrough Quant Head (alpha=0.40):
   - High between-sequence alpha projection
4. Fused weighted combination: 0.60 * p_inst + 0.40 * p_sub6
5. Exact single ONNX graph, single inference call per step or sequence.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import json
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper

from src.train_institutional_connectome import InstitutionalConnectome
from src.train_breakthrough_quant_ensemble import build_quant_microstructure_matrix


class FusedInstitutionalModel(nn.Module):
    def __init__(self, inst_checkpoint: str = 'scratch/best_institutional_connectome.pt',
                 sub6_checkpoint: str = 'scratch/best_breakthrough_quant.pt',
                 baseline_path: str = 'baseline/baseline.onnx',
                 layout_path: str = 'data/layout.json',
                 alpha: float = 0.60):
        super().__init__()
        self.alpha = float(alpha)
        
        # Load Institutional Connectome
        self.inst = InstitutionalConnectome(baseline_path=baseline_path, layout_path=layout_path)
        inst_weights = torch.load(inst_checkpoint, map_location='cpu', weights_only=False)
        self.inst.load_state_dict(inst_weights)
        
        # Extract Sub 6 weights
        d_sub6 = torch.load(sub6_checkpoint, map_location='cpu', weights_only=False)
        W_h = d_sub6['head_h.weight'].numpy()
        b_h = d_sub6['head_h.bias'].numpy()
        W_q = d_sub6['head_quant.weight'].numpy()
        s0_s6 = float(d_sub6['s0'])
        s1_s6 = float(d_sub6['s1'])
        c_s6 = float(torch.clamp(d_sub6['coupling'], 0.0, 0.15))
        
        M_c = np.array([
            [(1.0 - c_s6) * s0_s6, -c_s6 * s0_s6],
            [-c_s6 * s1_s6, (1.0 - c_s6) * s1_s6]
        ], dtype=np.float32)
        
        W_H1 = (W_h.T @ M_c).T # (2, 128)
        b_H1 = b_h @ M_c      # (2,)
        P_quant, _ = build_quant_microstructure_matrix()
        W_X = (P_quant @ (W_q.T @ M_c)).T # (2, 112)
        
        self.sub6_w_h = nn.Parameter(torch.from_numpy(W_H1.copy()), requires_grad=False)
        self.sub6_w_x = nn.Parameter(torch.from_numpy(W_X.copy()), requires_grad=False)
        self.sub6_b = nn.Parameter(torch.from_numpy(b_H1.copy()), requires_grad=False)

    def forward(self, x, h0=None, h1=None):
        # x: (batch, seq_len, 112)
        # 1. Shared GRU Backbone
        o0, next_h0 = self.inst.gru0(x, h0)
        o1, next_h1 = self.inst.gru1(o0, h1) # (B, L, 128)
        
        # 2. Institutional Connectome Head
        base_pred = torch.matmul(o1, self.inst.base_head_w.T) + self.inst.base_head_b # (B, L, 2)
        micro = self.inst.extract_micro_features(x) # (B, L, 19)
        combined = torch.cat([o1, micro], dim=-1) # (B, L, 147)
        
        delta_0 = self.inst.head_0(combined) # (B, L, 1)
        delta_1 = self.inst.head_1(combined) # (B, L, 1)
        
        raw_0 = base_pred[:, :, 0:1] + delta_0
        raw_1 = base_pred[:, :, 1:2] + delta_1
        
        c = torch.clamp(self.inst.coupling, 0.0, 0.15)
        p0_inst = (1.0 - c) * raw_0 - c * raw_1
        p1_inst = (1.0 - c) * raw_1 - c * raw_0
        
        out_0_inst = p0_inst * self.inst.scale[0] + self.inst.bias[0]
        out_1_inst = p1_inst * self.inst.scale[1] + self.inst.bias[1]
        p_inst = torch.cat([out_0_inst, out_1_inst], dim=-1) # (B, L, 2)
        
        # 3. Sub 6 Breakthrough Quant Head
        p_sub6 = torch.matmul(o1, self.sub6_w_h.T) + torch.matmul(x, self.sub6_w_x.T) + self.sub6_b # (B, L, 2)
        
        # 4. Optimal Linear Fusion
        pred = self.alpha * p_inst + (1.0 - self.alpha) * p_sub6
        
        return pred, next_h0, next_h1
