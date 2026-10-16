"""Final Production Training and ONNX Export Engine.

Trains FullGRUModel on all 200 cached sequences (4,000,000 steps)
using the exact Weighted Pearson Loss, AdamW optimizer with differential learning rates,
and exports the final production ONNX model.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import onnx
from onnx import numpy_helper
from torch.amp import autocast, GradScaler
from src.loss.weighted_pearson import WeightedPearsonLoss
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device.type.upper()}")

print("Loading cached 200 sequences (4,000,000 steps)...")
data = np.load('scratch/cached_200_seqs.npz')
feats_all = data['features_all'] # (200, 20000, 112)
targets_all = data['targets_all'] # (200, 20000, 2)
num_seqs = len(feats_all)

# Load baseline weights for warm-start initialization
print("Loading baseline.onnx backbone weights...")
m_base = onnx.load('baseline/baseline.onnx')
w = {init.name: numpy_helper.to_array(init) for init in m_base.graph.initializer}

class FullGRUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru0 = torch.nn.GRU(112, 128, batch_first=True)
        self.gru1 = torch.nn.GRU(128, 128, batch_first=True)
        self.head = torch.nn.Linear(128, 2)
        
        def load_l(w_ih, w_hh, b, gru):
            zw, rw, hw = np.split(w_ih[0], 3, axis=0)
            gru.weight_ih_l0.data = torch.from_numpy(np.concatenate([rw, zw, hw], axis=0).copy())
            zr, rr, hr = np.split(w_hh[0], 3, axis=0)
            gru.weight_hh_l0.data = torch.from_numpy(np.concatenate([rr, zr, hr], axis=0).copy())
            bw, br = np.split(b[0], 2, axis=0)
            zbw, rbw, hbw = np.split(bw, 3, axis=0)
            zbr, rbr, hbr = np.split(br, 3, axis=0)
            gru.bias_ih_l0.data = torch.from_numpy(np.concatenate([rbw, zbw, hbw], axis=0).copy())
            gru.bias_hh_l0.data = torch.from_numpy(np.concatenate([rbr, zbr, hbr], axis=0).copy())
        
        load_l(w['onnx::GRU_163'], w['onnx::GRU_164'], w['onnx::GRU_165'], self.gru0)
        load_l(w['onnx::GRU_183'], w['onnx::GRU_184'], w['onnx::GRU_185'], self.gru1)
        self.head.weight.data = torch.from_numpy(w['onnx::MatMul_186'].T.copy())
        self.head.bias.data = torch.from_numpy(w['reg_head.bias'].copy())
        
    def forward(self, x, h0=None, h1=None):
        o0, next_h0 = self.gru0(x, h0)
        o1, next_h1 = self.gru1(o0, h1)
        pred = self.head(o1)
        return pred, next_h0, next_h1

model = FullGRUModel().to(device)
loss_fn = WeightedPearsonLoss()

# Differential Learning Rates:
# GRU layers: 1e-5 to preserve 212M-step pre-training while adapting
# Linear Head: 1e-3 for rapid convergence to Weighted Pearson metric
optimizer = torch.optim.AdamW([
    {'params': model.gru0.parameters(), 'lr': 1e-5},
    {'params': model.gru1.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3},
], weight_decay=1e-4)

epochs = 6
scaler = GradScaler('cuda')

print(f"Starting Production Training on ALL {num_seqs} sequences ({num_seqs * 20000:,} steps) for {epochs} epochs...")
for ep in range(1, epochs + 1):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    perm = np.random.permutation(num_seqs)
    
    for s_idx in perm:
        x = torch.from_numpy(feats_all[s_idx:s_idx+1]).to(device)
        y = torch.from_numpy(targets_all[s_idx:s_idx+1]).to(device)
        mask = torch.ones(1, 20000, dtype=torch.bool, device=device)
        mask[:, :99] = False
        
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.float16):
            pred, _, _ = model(x)
            loss = loss_fn(pred, y, mask)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
    print(f"Epoch {ep}/{epochs} completed in {time.time()-t0:.1f}s | Avg Loss: {total_loss/num_seqs:.4f}")

torch.save(model.state_dict(), 'scratch/production_final_model.pt')
print("Model checkpoint saved to scratch/production_final_model.pt!")

# Now Export to Production ONNX
print("\nExporting to production ONNX (submission/model.onnx)...")
model.eval()

class ProductionONNXWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, features, hidden_0, hidden_1):
        pred, next_h0, next_h1 = self.m(features, hidden_0, hidden_1)
        return pred, next_h0, next_h1

wrapper = ProductionONNXWrapper(model)
wrapper.eval()

dummy_x = torch.zeros(1, 1, 112, dtype=torch.float32, device=device)
dummy_h0 = torch.zeros(1, 1, 128, dtype=torch.float32, device=device)
dummy_h1 = torch.zeros(1, 1, 128, dtype=torch.float32, device=device)

onnx_out_path = Path('submission/model.onnx')
onnx_out_path.parent.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    wrapper,
    (dummy_x, dummy_h0, dummy_h1),
    str(onnx_out_path),
    input_names=['features', 'hidden_0', 'hidden_1'],
    output_names=['prediction', 'next_hidden_0', 'next_hidden_1'],
    dynamic_axes={
        'features': {0: 'batch', 1: 'seq_len'},
        'hidden_0': {0: 'batch', 1: 'seq_len'},
        'hidden_1': {0: 'batch', 1: 'seq_len'},
        'prediction': {0: 'batch', 1: 'seq_len'},
        'next_hidden_0': {0: 'batch', 1: 'seq_len'},
        'next_hidden_1': {0: 'batch', 1: 'seq_len'},
    },
    opset_version=17,
    do_constant_folding=True
)

print(f"Exported successfully! Size: {onnx_out_path.stat().st_size / 1024:.1f} KB")

# Verify Parity with ONNX Runtime
import onnxruntime as ort
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1
sess = ort.InferenceSession(str(onnx_out_path), sess_options=sess_opts, providers=['CPUExecutionProvider'])

test_x = np.random.randn(1, 1, 112).astype(np.float32)
test_h0 = np.random.randn(1, 1, 128).astype(np.float32)
test_h1 = np.random.randn(1, 1, 128).astype(np.float32)

with torch.no_grad():
    py_out, py_h0, py_h1 = wrapper(
        torch.from_numpy(test_x).to(device),
        torch.from_numpy(test_h0).to(device),
        torch.from_numpy(test_h1).to(device)
    )

ort_out = sess.run(None, {'features': test_x, 'hidden_0': test_h0, 'hidden_1': test_h1})
diff = np.max(np.abs(py_out.cpu().numpy() - ort_out[0]))
print(f"ONNX Parity Verification Max Abs Diff: {diff:.2e}")
assert diff < 1e-4, f"Parity diff too high: {diff}"
print("PARITY VERIFICATION CONFIRMED!")
