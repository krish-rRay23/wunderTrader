import sys
sys.path.insert(0, ".")
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import zipfile

from src.build_true_lob_matrix import build_true_lob_matrix
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP
from src.validation import evaluate_moments_decomposition
from src.extract_baseline_weights import FrozenBaselineGRU

print("=" * 80)
print("FUSING 500-SEQUENCE CONNECTOME + OPTIMAL EIGENMODE SCALING TO ONNX")
print("=" * 80)

ckpt_path = 'scratch/best_500seqs_connectome.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
P_true, _ = build_true_lob_matrix()

# 1. Validation Verification
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
Q_val = np.matmul(feats_val, P_true)

# Multi-Horizon Model
W_o1 = ckpt['head_o1.weight'].numpy()
b_o1 = ckpt['head_o1.bias'].numpy()
W_o0 = ckpt['head_o0.weight'].numpy()
W_q = ckpt['head_q.weight'].numpy()
s0 = float(ckpt['s0'])
s1 = float(ckpt['s1'])
c = float(torch.clamp(ckpt['coupling'], 0.0, 0.15))
bias = ckpt['bias'].numpy()

print(f"Base calibrators: s0={s0:.6f}, s1={s1:.6f}, c={c:.6f}, bias={bias}")

M_c = np.array([
    [(1.0 - c) * s0, -c * s0],
    [-c * s1, (1.0 - c) * s1]
], dtype=np.float32)

# Optimal eigenmode scale from grid search: scale_d=1.020, scale_s=1.060
scale_d = 1.020
scale_s = 1.060
M_eigen = np.array([
    [(scale_d + scale_s) / 2.0, (-scale_d + scale_s) / 2.0],
    [(-scale_d + scale_s) / 2.0, (scale_d + scale_s) / 2.0]
], dtype=np.float32)

M_total = M_c @ M_eigen

W_O1_fused = W_o1.T @ M_total      # (128, 2)
W_O0_fused = W_o0.T @ M_total      # (128, 2)
b_fused = (b_o1 @ M_c + bias) @ M_eigen # (2,)
W_q_coupled = W_q.T @ M_total      # (36, 2)
W_X_fused = P_true @ W_q_coupled   # (112, 2)

print(f"Fused O1 Recurrent Head: {W_O1_fused.shape}, Norm: {np.linalg.norm(W_O1_fused):.6f}")
print(f"Fused O0 Recurrent Head: {W_O0_fused.shape}, Norm: {np.linalg.norm(W_O0_fused):.6f}")
print(f"Fused Input Bypass:      {W_X_fused.shape}, Norm: {np.linalg.norm(W_X_fused):.6f}")
print(f"Fused Output Bias:       {b_fused}")

# Test Fused Model on Pristine Holdout
mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
mask[WARMUP:] = True

class FusedMultiHorizonNet(nn.Module):
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
fused_net = FusedMultiHorizonNet(
    base_gru_cpu.gru0,
    base_gru_cpu.gru1,
    W_O0_fused,
    W_O1_fused,
    W_X_fused,
    b_fused
).to(device).eval()

preds_all = []
with torch.no_grad():
    for s in range(N_VAL):
        xt = torch.from_numpy(feats_val[s:s+1]).to(device)
        p, _, _ = fused_net(xt)
        preds_all.append(p[0].cpu().numpy())
preds_all = np.array(preds_all)

acc = GlobalAccumulator()
for s in range(N_VAL):
    acc.add(targets_val[s], preds_all[s], mask)
res = acc.result()
diag = evaluate_moments_decomposition(targets_val, preds_all)

print("\n" + "=" * 80)
print(f"CONFIRMED NEW BENCHMARK SCORE:")
print(f"Holdout WP: {res['weighted_pearson']:.6f} (t0: {res['t0']:.6f}, t1: {res['t1']:.6f})")
print(f"r_between:  {diag['r_between']:.6f}")
print(f"r_within:   {diag['r_within']:.6f}")
print(f"Gain over 0C195AAG: {(res['weighted_pearson'] - 0.427589)*10000:+.2f} bps")
print("=" * 80)

# Export to ONNX
export_path = 'submission/model.onnx'
fused_net = fused_net.cpu()
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

# Parity check
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1
sess = ort.InferenceSession(export_path, sess_options=sess_opts, providers=['CPUExecutionProvider'])
ort_out = sess.run(None, {
    'features': x_dummy.numpy(),
    'hidden_0': h0_dummy.numpy(),
    'hidden_1': h1_dummy.numpy()
})
parity_err = np.max(np.abs(fused_net(x_dummy, h0_dummy, h1_dummy)[0].detach().numpy() - ort_out[0]))
print(f"PyTorch vs ONNX Parity Error: {parity_err:.8e} (Pass: < 1e-5)")
assert parity_err < 1e-5

# Latency benchmark
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
print(f"Estimated 39.4M Server Runtime: {est_server_min:.1f} minutes")

# Package submission.zip
with zipfile.ZipFile('submission.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write('submission/solution.py', arcname='solution.py')
    z.write(export_path, arcname='model.onnx')
zip_size_mb = Path('submission.zip').stat().st_size / (1024 * 1024)
print(f"\nPackaged submission.zip (size: {zip_size_mb:.2f} MB)")
