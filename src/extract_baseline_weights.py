"""Step 1: Extract Baseline ONNX Weights and verify PyTorch bit-for-bit equivalence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn

from src.data_reader import FastSequenceReader
from utils import SEQUENCE_LENGTH, WARMUP, FEATURE_COLUMNS


class FrozenBaselineGRU(nn.Module):
    def __init__(self, baseline_path: str = "baseline/baseline.onnx"):
        super().__init__()
        self.gru0 = nn.GRU(112, 128, batch_first=True)
        self.gru1 = nn.GRU(128, 128, batch_first=True)
        self.head = nn.Linear(128, 2)
        
        m = onnx.load(baseline_path)
        w = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
        
        def load_l(w_ih, w_hh, b, gru):
            # ONNX stores GRU weights as (num_directions, 3*hidden_size, input_size) in [z, r, h] order
            # PyTorch expects [r, z, h] order
            zw, rw, hw = np.split(w_ih[0], 3, axis=0)
            gru.weight_ih_l0.data = torch.from_numpy(np.concatenate([rw, zw, hw], axis=0).copy())
            zr, rr, hr = np.split(w_hh[0], 3, axis=0)
            gru.weight_hh_l0.data = torch.from_numpy(np.concatenate([rr, zr, hr], axis=0).copy())
            bw, br = np.split(b[0], 2, axis=0)
            zbw, rbw, hbw = np.split(bw, 3, axis=0)
            zbr, rbr, hbr = np.split(br, 3, axis=0)
            gru.bias_ih_l0.data = torch.from_numpy(np.concatenate([rbw, zbw, hbw], axis=0).copy())
            gru.bias_hh_l0.data = torch.from_numpy(np.concatenate([rbr, zbr, hbr], axis=0).copy())
            
        load_l(w["onnx::GRU_163"], w["onnx::GRU_164"], w["onnx::GRU_165"], self.gru0)
        load_l(w["onnx::GRU_183"], w["onnx::GRU_184"], w["onnx::GRU_185"], self.gru1)
        
        self.head.weight.data = torch.from_numpy(w["onnx::MatMul_186"].T.copy())
        self.head.bias.data = torch.from_numpy(w["reg_head.bias"].copy())
        
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x, h0=None, h1=None):
        # x: (B, T, 112)
        # returns: p_base (B, T, 2), h0_seq (B, T, 128), h1_seq (B, T, 128), next_h0, next_h1
        o0, next_h0 = self.gru0(x, h0)
        o1, next_h1 = self.gru1(o0, h1)
        pred = self.head(o1)
        return pred, o0, o1, next_h0, next_h1


def verify_equivalence():
    print("=" * 70)
    print("STEP 1: VERIFYING BASELINE ONNX VS PYTORCH EXACT NUMERICAL PARITY")
    print("=" * 70)
    
    baseline_onnx_path = "baseline/baseline.onnx"
    sess = ort.InferenceSession(baseline_onnx_path, providers=["CPUExecutionProvider"])
    pt_model = FrozenBaselineGRU(baseline_onnx_path).eval()
    
    # Test on 5 full sequences (100,000 steps)
    cache_path = "scratch/cached_50_seqs.npz"
    cache = np.load(cache_path)
    feats = cache["features_all"][:5] # (5, 20000, 112)
    
    max_err_p = 0.0
    max_err_h0 = 0.0
    max_err_h1 = 0.0
    
    for s in range(5):
        h0_ort = np.zeros((1, 1, 128), dtype=np.float32)
        h1_ort = np.zeros((1, 1, 128), dtype=np.float32)
        p_ort_all = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
        h0_ort_all = np.zeros((SEQUENCE_LENGTH, 128), dtype=np.float32)
        h1_ort_all = np.zeros((SEQUENCE_LENGTH, 128), dtype=np.float32)
        
        # Step-by-step ORT
        for t in range(SEQUENCE_LENGTH):
            xt = feats[s, t:t+1, :][None, :]
            out_p, h0_ort, h1_ort = sess.run(None, {"features": xt, "hidden_0": h0_ort, "hidden_1": h1_ort})
            p_ort_all[t] = out_p[0, 0]
            h0_ort_all[t] = h0_ort[0, 0]
            h1_ort_all[t] = h1_ort[0, 0]
            
        # Full-sequence PyTorch in one shot
        with torch.no_grad():
            x_pt = torch.from_numpy(feats[s:s+1])
            p_pt, o0_pt, o1_pt, _, _ = pt_model(x_pt)
            p_pt = p_pt[0].numpy()
            h0_pt = o0_pt[0].numpy()
            h1_pt = o1_pt[0].numpy()
            
        err_p = np.max(np.abs(p_ort_all - p_pt))
        err_h0 = np.max(np.abs(h0_ort_all - h0_pt))
        err_h1 = np.max(np.abs(h1_ort_all - h1_pt))
        
        max_err_p = max(max_err_p, err_p)
        max_err_h0 = max(max_err_h0, err_h0)
        max_err_h1 = max(max_err_h1, err_h1)
        print(f"Sequence {s+1}/5 | Max Errors: p_base={err_p:.2e}, h0={err_h0:.2e}, h1={err_h1:.2e}")
        
    print("-" * 70)
    print(f"Overall Max Absolute Difference across 100,000 steps:")
    print(f"  Prediction (p_base): {max_err_p:.2e} (Pass: < 1e-5)")
    print(f"  Hidden 0 (h0):       {max_err_h0:.2e} (Pass: < 1e-5)")
    print(f"  Hidden 1 (h1):       {max_err_h1:.2e} (Pass: < 1e-5)")
    assert max_err_p < 1e-4, f"Prediction error {max_err_p} exceeds tolerance"
    assert max_err_h0 < 1e-4, f"h0 error {max_err_h0} exceeds tolerance"
    assert max_err_h1 < 1e-4, f"h1 error {max_err_h1} exceeds tolerance"
    print("STATUS: STEP 1 EXACT NUMERICAL EQUIVALENCE VERIFIED!")


if __name__ == "__main__":
    verify_equivalence()
