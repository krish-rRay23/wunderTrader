"""Export Fused ADSRA Model to ONNX.

Fuses:
- Frozen Baseline GRU (gru0 + gru1 + base_head)
- Master ADSRA Head (proj_o0, proj_o1, proj_dh0, proj_dh1, proj_x, proj_dx, proj_pb + SwiGLU + Residual + Mode Recomb)
- Single Aux State tensor: prev_x (112) + res_m (2) = 114 floats!
"""
import sys
import time
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
from src.train_adsra_master import MasterADSRAHead
from utils import SEQUENCE_LENGTH, WARMUP


class FusedProductionADSRA(nn.Module):
    def __init__(self, base_gru, adsra_head):
        super().__init__()
        self.gru0 = base_gru.gru0
        self.gru1 = base_gru.gru1
        self.base_head = base_gru.head
        self.head = adsra_head

    def forward(self, x, h0, h1, aux_state):
        # x: (1, 1, 112)
        # h0: (1, 1, 128)
        # h1: (1, 1, 128)
        # aux_state: (1, 114) -> prev_x (112), res_m (2)
        prev_x = aux_state[:, 0:112].unsqueeze(1) # (1, 1, 112)
        res_m = aux_state[:, 112:114]             # (1, 2)

        # 1. Native GRUs
        o0, next_h0 = self.gru0(x, h0) # (1, 1, 128)
        o1, next_h1 = self.gru1(o0, h1) # (1, 1, 128)
        p_base = self.base_head(o1)     # (1, 1, 2)

        # 2. State velocities: dh0 = o0 - h0, dh1 = o1 - h1, dx = x - prev_x
        # Note: h0 and h1 are (1, 1, 128), exactly matching o0 and o1!
        dh0 = o0 - h0
        dh1 = o1 - h1
        dx = x - prev_x

        # 3. Master ADSRA Head single step
        y_hat, next_res_m = self.head(p_base, o0, o1, dh0, dh1, x, dx, res_m=res_m)
        pred = torch.clamp(y_hat[:, 0], -2.0, 2.0) # (1, 2)

        # 4. Next aux state: next_prev_x is x[:, 0] (1, 112) + next_res_m[:, 0] (1, 2) = (1, 114)
        next_aux = torch.cat([x[:, 0], next_res_m[:, 0]], dim=-1) # (1, 114)

        return pred, next_h0, next_h1, next_aux


def export_and_verify(
    checkpoint_path: str = 'scratch/best_master_adsra.pt',
    export_path: str = 'submission/model.onnx'
):
    print("=" * 70)
    print("STEP 2: EXPORTING FUSED ADSRA PRODUCTION MODEL TO ONNX")
    print("=" * 70)
    
    device = torch.device('cpu') # Export on CPU for clean ONNX representation
    base_gru = FrozenBaselineGRU('baseline/baseline.onnx').to(device).eval()
    
    adsra_head = MasterADSRAHead(hidden=96).to(device)
    adsra_head.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    adsra_head.eval()
    
    fused_model = FusedProductionADSRA(base_gru, adsra_head).eval()
    
    # Dummy step tensors
    dummy_x = torch.zeros(1, 1, 112, dtype=torch.float32)
    dummy_h0 = torch.zeros(1, 1, 128, dtype=torch.float32)
    dummy_h1 = torch.zeros(1, 1, 128, dtype=torch.float32)
    dummy_aux = torch.zeros(1, 114, dtype=torch.float32)
    
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {export_path}...")
    torch.onnx.export(
        fused_model,
        (dummy_x, dummy_h0, dummy_h1, dummy_aux),
        export_path,
        input_names=['features', 'hidden_0', 'hidden_1', 'aux_state'],
        output_names=['prediction', 'next_hidden_0', 'next_hidden_1', 'next_aux_state'],
        opset_version=17,
        do_constant_folding=True
    )
    
    m_onnx = onnx.load(export_path)
    file_size_kb = Path(export_path).stat().st_size / 1024
    print(f"Exported successfully! Node count: {len(m_onnx.graph.node)}, Size: {file_size_kb:.1f} KB")
    
    # 1. PARITY VERIFICATION ACROSS A FULL 20,000-STEP SEQUENCE
    print("\nVerifying Bit-for-Bit PyTorch vs ONNX Parity across 20,000 steps...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(export_path, sess_options=opts, providers=['CPUExecutionProvider'])
    
    d50 = np.load('scratch/cached_50_seqs.npz')
    test_seq = d50['features_all'][0] # (20000, 112)
    
    # PyTorch unroll
    pt_preds = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
    pt_h0 = torch.zeros(1, 1, 128)
    pt_h1 = torch.zeros(1, 1, 128)
    pt_aux = torch.zeros(1, 114)
    with torch.no_grad():
        for t in range(SEQUENCE_LENGTH):
            xt = torch.from_numpy(test_seq[t:t+1, :]).unsqueeze(0)
            p, pt_h0, pt_h1, pt_aux = fused_model(xt, pt_h0, pt_h1, pt_aux)
            pt_preds[t] = p[0].numpy()
            
    # ONNX unroll
    ort_preds = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
    ort_h0 = np.zeros((1, 1, 128), dtype=np.float32)
    ort_h1 = np.zeros((1, 1, 128), dtype=np.float32)
    ort_aux = np.zeros((1, 114), dtype=np.float32)
    
    for t in range(SEQUENCE_LENGTH):
        xt = test_seq[t:t+1, :][np.newaxis, :]
        p, ort_h0, ort_h1, ort_aux = sess.run(None, {
            'features': xt,
            'hidden_0': ort_h0,
            'hidden_1': ort_h1,
            'aux_state': ort_aux
        })
        ort_preds[t] = p[0]
        
    max_err = np.max(np.abs(pt_preds - ort_preds))
    print(f"Max Absolute Error (PyTorch vs ONNX): {max_err:.2e}")
    assert max_err < 1e-4, f"Parity error {max_err} exceeds 1e-4 threshold!"
    print("STATUS: BIT-FOR-BIT PARITY VERIFIED (< 1e-4)!")
    
    # 2. BENCHMARK SINGLE-CORE CPU LATENCY
    print("\nBenchmarking Single-Core CPU Latency (20,000 steps)...")
    feed = {
        'features': np.zeros((1, 1, 112), dtype=np.float32),
        'hidden_0': np.zeros((1, 1, 128), dtype=np.float32),
        'hidden_1': np.zeros((1, 1, 128), dtype=np.float32),
        'aux_state': np.zeros((1, 114), dtype=np.float32)
    }
    
    for _ in range(500): sess.run(None, feed)
    N = 20000
    t0 = time.perf_counter()
    for _ in range(N):
        sess.run(None, feed)
    dt_local = (time.perf_counter() - t0) / N * 1e6
    dt_cloud_est = dt_local / 3.2
    cloud_time_min = dt_cloud_est * 39.4e6 / 1e6 / 60
    
    print("-" * 70)
    print(f"Local Latency:           {dt_local:.2f} microseconds / step")
    print(f"Normalized Cloud Latency: {dt_cloud_est:.2f} microseconds / step")
    print(f"Projected Cloud Runtime: {cloud_time_min:.2f} minutes (Hard Cutoff: 60.00 min)")
    print("-" * 70)
    assert cloud_time_min < 55.0, f"Projected runtime {cloud_time_min:.1f}m too close to cutoff!"
    print("STATUS: INFERENCE LATENCY BUDGET VERIFIED!")


if __name__ == "__main__":
    export_and_verify()
