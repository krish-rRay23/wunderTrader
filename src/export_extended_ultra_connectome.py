"""Export Extended Ultra Connectome (81 signals) into fused 15-node ONNX and package submission.zip.

Mathematical Tensor Fusion:
    W_O1_fused = W_o1.T @ M_c          (128, 2)
    W_O0_fused = W_o0.T @ M_c          (128, 2)
    W_X_fused  = P_ext @ W_q.T @ M_c   (112, 2)
    b_fused    = b_o1 @ M_c + bias      (2,)

Resulting ONNX model has identical 15-node topology, zero latency overhead, ~30-min server runtime.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import zipfile

from src.build_extended_ultra_matrix import build_extended_ultra_matrix
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP
from src.validation import evaluate_moments_decomposition
from src.extract_baseline_weights import FrozenBaselineGRU


def export_extended_ultra(ckpt_path='scratch/best_extended_ultra.pt', export_path='submission/model.onnx'):
    print("=" * 80)
    print(f"EXPORTING EXTENDED ULTRA CONNECTOME: {ckpt_path}")
    print("=" * 80)
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    P_ext, feat_names = build_extended_ultra_matrix() # (112, 81)
    K_FEATS = P_ext.shape[1]
    print(f"Loaded Extended Ultra Matrix: {P_ext.shape} ({K_FEATS} signals)")
    
    # 1. Holdout Benchmark Verification
    print("\n1. Verifying PyTorch Pristine Holdout Benchmark...")
    d_val = np.load('scratch/pristine_holdout_100.npz')
    feats_val = d_val['features_all']
    targets_val = d_val['targets_all']
    N_VAL = len(feats_val)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_gru = FrozenBaselineGRU('baseline/baseline.onnx').to(device).eval()
    
    O0_val = []
    O1_val = []
    with torch.no_grad():
        for s in range(N_VAL):
            xt = torch.from_numpy(feats_val[s:s+1]).to(device)
            _, o0, o1, _, _ = base_gru(xt)
            O0_val.append(o0[0].cpu().numpy())
            O1_val.append(o1[0].cpu().numpy())
    O0_val = np.array(O0_val)
    O1_val = np.array(O1_val)
    Q_val = np.matmul(feats_val, P_ext)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_o1 = nn.Linear(128, 2)
            self.head_o1.weight.data = ckpt['head_o1.weight']
            self.head_o1.bias.data = ckpt['head_o1.bias']
            
            self.head_o0 = nn.Linear(128, 2, bias=False)
            self.head_o0.weight.data = ckpt['head_o0.weight']
            
            self.head_q = nn.Linear(K_FEATS, 2, bias=False)
            self.head_q.weight.data = ckpt['head_q.weight']
            
            self.s0 = nn.Parameter(ckpt['s0'])
            self.s1 = nn.Parameter(ckpt['s1'])
            self.coupling = nn.Parameter(ckpt['coupling'])
            self.bias = nn.Parameter(ckpt['bias'])
            
        def forward(self, o0, o1, q):
            p_base = self.head_o1(o1)
            p_fast = self.head_o0(o0)
            p_micro = self.head_q(q)
            raw = p_base + p_fast + p_micro
            
            p0 = raw[..., 0:1] * self.s0
            p1 = raw[..., 1:2] * self.s1
            c = torch.clamp(self.coupling, 0.0, 0.15)
            out0 = (1.0 - c) * p0 - c * p1 + self.bias[0]
            out1 = (1.0 - c) * p1 - c * p0 + self.bias[1]
            return torch.cat([out0, out1], dim=-1)
            
    m = Model().to(device).eval()
    
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    
    preds_all = []
    with torch.no_grad():
        for s in range(N_VAL):
            o0_t = torch.from_numpy(O0_val[s:s+1]).to(device)
            o1_t = torch.from_numpy(O1_val[s:s+1]).to(device)
            q_t = torch.from_numpy(Q_val[s:s+1]).to(device)
            p = m(o0_t, o1_t, q_t)
            preds_all.append(p[0].cpu().numpy())
    preds_all = np.array(preds_all)
    
    acc = GlobalAccumulator()
    for s in range(N_VAL):
        acc.add(targets_val[s], preds_all[s], mask)
    res = acc.result()
    diag = evaluate_moments_decomposition(targets_val, preds_all)
    
    print(f"PYTORCH BENCHMARK EVALUATION:")
    print(f"Holdout WP: {res['weighted_pearson']:.6f} (t0: {res['t0']:.6f}, t1: {res['t1']:.6f})")
    print(f"r_between:  {diag['r_between']:.6f}")
    print(f"r_within:   {diag['r_within']:.6f}")
    
    # 2. Closed-Form Mathematical Tensor Fusion
    print("\n2. Closed-Form Mathematical Tensor Fusion...")
    W_o1 = ckpt['head_o1.weight'].numpy()
    b_o1 = ckpt['head_o1.bias'].numpy()
    W_o0 = ckpt['head_o0.weight'].numpy()
    W_q = ckpt['head_q.weight'].numpy()
    
    s0 = float(ckpt['s0'])
    s1 = float(ckpt['s1'])
    c = float(torch.clamp(ckpt['coupling'], 0.0, 0.15))
    bias = ckpt['bias'].numpy()
    
    print(f"Parameters: s0={s0:.6f}, s1={s1:.6f}, c={c:.6f}, bias={bias}")
    
    M_c = np.array([
        [(1.0 - c) * s0, -c * s0],
        [-c * s1, (1.0 - c) * s1]
    ], dtype=np.float32)
    
    W_O1_fused = W_o1.T @ M_c         # (128, 2)
    W_O0_fused = W_o0.T @ M_c         # (128, 2)
    b_fused = b_o1 @ M_c + bias       # (2,)
    
    W_q_coupled = W_q.T @ M_c         # (81, 2)
    W_X_fused = P_ext @ W_q_coupled   # (112, 2)
    
    print(f"Fused O1 Recurrent Head: {W_O1_fused.shape}, Norm: {np.linalg.norm(W_O1_fused):.6f}")
    print(f"Fused O0 Recurrent Head: {W_O0_fused.shape}, Norm: {np.linalg.norm(W_O0_fused):.6f}")
    print(f"Fused Input Bypass:      {W_X_fused.shape}, Norm: {np.linalg.norm(W_X_fused):.6f}")
    print(f"Fused Output Bias:       {b_fused}")
    
    # 3. Build Fused PyTorch Model
    class FusedExtendedUltraNet(nn.Module):
        def __init__(self, gru0, gru1, Wo0, Wo1, Wx, b):
            super().__init__()
            self.gru0 = gru0
            self.gru1 = gru1
            self.W_o0 = nn.Parameter(torch.from_numpy(Wo0.astype(np.float32)), requires_grad=False)
            self.W_o1 = nn.Parameter(torch.from_numpy(Wo1.astype(np.float32)), requires_grad=False)
            self.W_x = nn.Parameter(torch.from_numpy(Wx.astype(np.float32)), requires_grad=False)
            self.bias = nn.Parameter(torch.from_numpy(b.astype(np.float32)), requires_grad=False)
            
        def forward(self, x, h0=None, h1=None):
            o0, next_h0 = self.gru0(x, h0)
            o1, next_h1 = self.gru1(o0, h1)
            pred = (
                torch.matmul(o0, self.W_o0) +
                torch.matmul(o1, self.W_o1) +
                torch.matmul(x, self.W_x) +
                self.bias
            )
            return pred, next_h0, next_h1

    base_gru_cpu = FrozenBaselineGRU('baseline/baseline.onnx').cpu().eval()
    fused_net = FusedExtendedUltraNet(
        base_gru_cpu.gru0,
        base_gru_cpu.gru1,
        W_O0_fused,
        W_O1_fused,
        W_X_fused,
        b_fused
    ).eval()
    
    # 4. Export to ONNX
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    x_dummy = torch.randn(1, 1, 112)
    h0_dummy = torch.zeros(1, 1, 128)
    h1_dummy = torch.zeros(1, 1, 128)
    
    torch.onnx.export(
        fused_net,
        (x_dummy, h0_dummy, h1_dummy),
        export_path,
        input_names=['features', 'hidden_0', 'hidden_1'],
        output_names=['prediction', 'next_hidden_0', 'next_hidden_1'],
        dynamic_axes={
            'features': {0: 'batch', 1: 'seq'},
            'hidden_0': {1: 'batch'},
            'hidden_1': {1: 'batch'},
            'prediction': {0: 'batch', 1: 'seq'},
            'next_hidden_0': {1: 'batch'},
            'next_hidden_1': {1: 'batch'},
        },
        opset_version=17
    )
    
    m_onnx = onnx.load(export_path)
    file_size_kb = Path(export_path).stat().st_size / 1024
    print(f"\nExported {export_path} ({len(m_onnx.graph.node)} ONNX nodes, size: {file_size_kb:.1f} KB)")
    
    # 5. Numerical Parity Check
    with torch.no_grad():
        pt_pred, _, _ = fused_net(x_dummy, h0_dummy, h1_dummy)
        
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(export_path, sess_options=sess_opts, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {
        'features': x_dummy.numpy(),
        'hidden_0': h0_dummy.numpy(),
        'hidden_1': h1_dummy.numpy()
    })
    parity_err = np.max(np.abs(pt_pred.numpy() - ort_out[0]))
    print(f"PyTorch vs ONNX Parity Error: {parity_err:.8e}")
    assert parity_err < 1e-5, f"Parity error too high: {parity_err}"
    
    # 6. Single-Step Latency Benchmark
    x_np = np.zeros((1, 1, 112), dtype=np.float32)
    h0_np = np.zeros((1, 1, 128), dtype=np.float32)
    h1_np = np.zeros((1, 1, 128), dtype=np.float32)
    for _ in range(500):
        sess.run(None, {'features': x_np, 'hidden_0': h0_np, 'hidden_1': h1_np})
    t0 = time.time()
    N_BENCH = 10000
    for _ in range(N_BENCH):
        sess.run(None, {'features': x_np, 'hidden_0': h0_np, 'hidden_1': h1_np})
    t1 = time.time()
    lat = (t1 - t0) / N_BENCH * 1e6
    est_server_min = (lat * 39.4e6 / 1e6 / 60) * (42.85 / 105.9)
    print(f"Single-Step Latency: {lat:.1f} us/step")
    print(f"Estimated Server Runtime: {est_server_min:.1f} min (Budget: 60.0m)")
    
    # 7. Package submission.zip
    zip_path = 'submission.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write('submission/solution.py', arcname='solution.py')
        z.write(export_path, arcname='model.onnx')
    zip_size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
    print(f"Packaged {zip_path} (size: {zip_size_mb:.2f} MB)")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        print("Archive contents:")
        for info in z.infolist():
            print(f"  - {info.filename} ({info.file_size / 1024:.1f} KB)")
            
    return res['weighted_pearson'], diag['r_between'], diag['r_within']


if __name__ == '__main__':
    export_extended_ultra()
