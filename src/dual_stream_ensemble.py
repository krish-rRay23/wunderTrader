"""Dual-Stream Institutional Ensemble.

Fuses:
Stream A: Full Pretrained Baseline GRU + Sub 6 Breakthrough Quant Linear Head (Weight: 0.40)
Stream B: Fine-Tuned Institutional GRU + Decoupled Dual-Stream Microstructure MLPs (Weight: 0.60)

Inputs:
  features: [batch, seq, 112]
  h_base0:  [1, batch, 128]
  h_base1:  [1, batch, 128]
  h_inst0:  [1, batch, 128]
  h_inst1:  [1, batch, 128]

Outputs:
  output:   [batch, seq, 2]
  next_h_base0, next_h_base1
  next_h_inst0, next_h_inst1
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper

from src.train_institutional_connectome import InstitutionalConnectome
from src.train_breakthrough_quant_ensemble import build_quant_microstructure_matrix


class DualStreamEnsemble(nn.Module):
    def __init__(self, inst_checkpoint: str = 'scratch/best_institutional_connectome.pt',
                 sub6_checkpoint: str = 'scratch/best_breakthrough_quant.pt',
                 baseline_path: str = 'baseline/baseline.onnx',
                 layout_path: str = 'data/layout.json',
                 alpha: float = 0.60):
        super().__init__()
        self.alpha = float(alpha)
        
        # --- STREAM A: Baseline GRU + Sub 6 Quant Head ---
        self.gru_base0 = nn.GRU(112, 128, batch_first=True)
        self.gru_base1 = nn.GRU(128, 128, batch_first=True)
        
        m_base = onnx.load(baseline_path)
        w_base = {init.name: numpy_helper.to_array(init) for init in m_base.graph.initializer}
        
        def load_gru(w_ih, w_hh, b, gru):
            zw, rw, hw = np.split(w_ih[0], 3, axis=0)
            gru.weight_ih_l0.data = torch.from_numpy(np.concatenate([rw, zw, hw], axis=0).copy())
            zr, rr, hr = np.split(w_hh[0], 3, axis=0)
            gru.weight_hh_l0.data = torch.from_numpy(np.concatenate([rr, zr, hr], axis=0).copy())
            bw, br = np.split(b[0], 2, axis=0)
            zbw, rbw, hbw = np.split(bw, 3, axis=0)
            zbr, rbr, hbr = np.split(br, 3, axis=0)
            gru.bias_ih_l0.data = torch.from_numpy(np.concatenate([rbw, zbw, hbw], axis=0).copy())
            gru.bias_hh_l0.data = torch.from_numpy(np.concatenate([rbr, zbr, hbr], axis=0).copy())
            
        load_gru(w_base['onnx::GRU_163'], w_base['onnx::GRU_164'], w_base['onnx::GRU_165'], self.gru_base0)
        load_gru(w_base['onnx::GRU_183'], w_base['onnx::GRU_184'], w_base['onnx::GRU_185'], self.gru_base1)
        
        # Sub 6 Linear Weights
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
        
        # --- STREAM B: Fine-Tuned Institutional Connectome ---
        self.inst = InstitutionalConnectome(baseline_path=baseline_path, layout_path=layout_path)
        inst_weights = torch.load(inst_checkpoint, map_location='cpu', weights_only=False)
        self.inst.load_state_dict(inst_weights)

    def forward(self, x, h_base0=None, h_base1=None, h_inst0=None, h_inst1=None):
        # 1. Stream A: Baseline Forward
        o_b0, next_hb0 = self.gru_base0(x, h_base0)
        o_b1, next_hb1 = self.gru_base1(o_b0, h_base1) # (B, L, 128)
        p_sub6 = torch.matmul(o_b1, self.sub6_w_h.T) + torch.matmul(x, self.sub6_w_x.T) + self.sub6_b
        
        # 2. Stream B: Institutional Connectome Forward
        p_inst, next_hi0, next_hi1 = self.inst(x, h_inst0, h_inst1)
        
        # 3. Fused Output
        pred = self.alpha * p_inst + (1.0 - self.alpha) * p_sub6
        
        return pred, next_hb0, next_hb1, next_hi0, next_hi1
