"""Benchmark inference latency for recurrent model cells."""
import time
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

torch.set_num_threads(1)

class PyTorchGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru0 = nn.GRUCell(112, 128)
        self.gru1 = nn.GRUCell(128, 128)
        self.head = nn.Linear(128, 2)

    def forward(self, x, h0, h1):
        h0 = self.gru0(x, h0)
        h1 = self.gru1(h0, h1)
        pred = self.head(h1)
        return pred, h0, h1

# 1. PyTorch eager
model = PyTorchGRU()
x_t = torch.zeros(1, 112)
h0_t = torch.zeros(1, 128)
h1_t = torch.zeros(1, 128)

for _ in range(100):
    p, h0_t, h1_t = model(x_t, h0_t, h1_t)

t0 = time.time()
with torch.no_grad():
    for _ in range(5000):
        p, h0_t, h1_t = model(x_t, h0_t, h1_t)
t1 = time.time()
print(f'PyTorch eager (1 thread): {(t1 - t0)/5000 * 1e6:.1f} µs/step ({5000/(t1-t0):.0f} steps/s)')

# 2. Pure NumPy implementation of GRUCell
from onnx import numpy_helper
import onnx

model_onnx = onnx.load('baseline/baseline.onnx')
weights = {init.name: numpy_helper.to_array(init) for init in model_onnx.graph.initializer}

# ONNX weights: W is (384, in_dim), R is (384, hidden_dim), B is (768,)
w0 = weights['onnx::GRU_163'][0]
r0 = weights['onnx::GRU_164'][0]
b0 = weights['onnx::GRU_165'][0]
bw0, br0 = b0[:384], b0[384:]
b_total0 = bw0 + br0

w1 = weights['onnx::GRU_183'][0]
r1 = weights['onnx::GRU_184'][0]
b1 = weights['onnx::GRU_185'][0]
bw1, br1 = b1[:384], b1[384:]
b_total1 = bw1 + br1

w_head = weights['onnx::MatMul_186'] # (128, 2)
b_head = weights['reg_head.bias']   # (2,)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def gru_step(x, h, W, R, B_tot, hidden_dim=128):
    # gates: [z, r, h]
    gates = W @ x + R @ h + B_tot
    z = sigmoid(gates[:hidden_dim])
    r = sigmoid(gates[hidden_dim:2*hidden_dim])
    # in ONNX linear_before_reset=1:
    # h_tilde = tanh(W_h @ x + B_wh + r * (R_h @ h + B_rh))
    # here B_wh and B_rh:
    h_gate = np.tanh(gates[2*hidden_dim:] - (1 - r) * (R[2*hidden_dim:] @ h))
    h_next = (1 - z) * h_gate + z * h
    return h_next

x_np = np.zeros(112, dtype=np.float32)
h0_np = np.zeros(128, dtype=np.float32)
h1_np = np.zeros(128, dtype=np.float32)

t0 = time.time()
for _ in range(5000):
    h0_np = gru_step(x_np, h0_np, w0, r0, b_total0)
    h1_np = gru_step(h0_np, h1_np, w1, r1, b_total1)
    pred = h1_np @ w_head + b_head
t1 = time.time()
print(f'Pure NumPy GRU: {(t1 - t0)/5000 * 1e6:.1f} µs/step ({5000/(t1-t0):.0f} steps/s)')

# 3. ONNX Runtime baseline
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession('baseline/baseline.onnx', sess_options=options, providers=['CPUExecutionProvider'])
x_in = np.zeros((1, 1, 112), dtype=np.float32)
h0_in = np.zeros((1, 1, 128), dtype=np.float32)
h1_in = np.zeros((1, 1, 128), dtype=np.float32)

t0 = time.time()
for _ in range(5000):
    pred, h0_in, h1_in = session.run(None, {'features': x_in, 'hidden_0': h0_in, 'hidden_1': h1_in})
t1 = time.time()
print(f'ONNX Runtime: {(t1 - t0)/5000 * 1e6:.1f} µs/step ({5000/(t1-t0):.0f} steps/s)')
