"""Deep Quantitative Research: Training a Deep Residual Representation Head on top of Recurrent States."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np
from src.data_reader import FastSequenceReader
from src.loss.weighted_pearson import WeightedPearsonLoss
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP
import onnx
from onnx import numpy_helper

torch.set_num_threads(4)

# 1. Feature Engineering function
def extract_microstructure_features(x):
    # x: shape (N, 112)
    bids_i0 = x[:, :11]
    asks_i0 = x[:, 11:22]
    mid_i0 = (np.max(bids_i0, axis=1) + np.min(asks_i0, axis=1)) / 2.0
    spread_i0 = np.min(asks_i0, axis=1) - np.max(bids_i0, axis=1)

    bids_i1 = x[:, 52:63]
    asks_i1 = x[:, 63:74]
    mid_i1 = (np.max(bids_i1, axis=1) + np.min(asks_i1, axis=1)) / 2.0

    cross_spread = mid_i0 - mid_i1
    vol_bid_i0 = np.sum(x[:, 22:33], axis=1)
    vol_ask_i0 = np.sum(x[:, 33:44], axis=1)
    depth_imb_i0 = (vol_bid_i0 - vol_ask_i0) / (vol_bid_i0 + vol_ask_i0 + 1e-5)

    trade_vol_i0 = np.sum(x[:, 48:52], axis=1)
    trade_vol_i1 = np.sum(x[:, 100:104], axis=1)
    trade_imb = (trade_vol_i0 - trade_vol_i1) / (trade_vol_i0 + trade_vol_i1 + 1e-5)

    a_feats = x[:, 104:112] # a0..a7

    extra = np.column_stack([
        mid_i0, spread_i0, cross_spread, depth_imb_i0, trade_imb, a_feats
    ])
    return extra.astype(np.float32)

# 2. Extract frozen GRU hidden representations for training & validation sequences
print('Loading dataset sequences...')
reader = FastSequenceReader('datasets/train.parquet')
all_seqs = list(reader.iter_sequences(max_sequences=40))

# Load baseline ONNX weights
model_onnx = onnx.load('baseline/baseline.onnx')
weights = {init.name: numpy_helper.to_array(init) for init in model_onnx.graph.initializer}

class BaseGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru0 = nn.GRU(112, 128, batch_first=True)
        self.gru1 = nn.GRU(128, 128, batch_first=True)

    def forward(self, x):
        out0, _ = self.gru0(x)
        out1, _ = self.gru1(out0)
        return out1

base_gru = BaseGRU()
def load_gru(w, r, b, layer):
    zw, rw, hw = np.split(w[0], 3, axis=0)
    layer.weight_ih_l0.data = torch.from_numpy(np.concatenate([rw, zw, hw], axis=0).copy())
    zr, rr, hr = np.split(r[0], 3, axis=0)
    layer.weight_hh_l0.data = torch.from_numpy(np.concatenate([rr, zr, hr], axis=0).copy())
    bw, br = np.split(b[0], 2, axis=0)
    zbw, rbw, hbw = np.split(bw, 3, axis=0)
    zbr, rbr, hbr = np.split(br, 3, axis=0)
    layer.bias_ih_l0.data = torch.from_numpy(np.concatenate([rbw, zbw, hbw], axis=0).copy())
    layer.bias_hh_l0.data = torch.from_numpy(np.concatenate([rbr, zbr, hbr], axis=0).copy())

load_gru(weights['onnx::GRU_163'], weights['onnx::GRU_164'], weights['onnx::GRU_165'], base_gru.gru0)
load_gru(weights['onnx::GRU_183'], weights['onnx::GRU_184'], weights['onnx::GRU_185'], base_gru.gru1)
base_gru.eval()
print('Base GRU loaded successfully!')

# Extract representations
print('Extracting representations for 40 sequences (800,000 steps)...')
H_train, Y_train = [], []
H_val, Y_val = [], []

with torch.no_grad():
    for idx, (seq_id, feats, targs) in enumerate(all_seqs):
        x_t = torch.from_numpy(feats).unsqueeze(0)
        h_out = base_gru(x_t)[0].numpy() # (20000, 128)
        extra = extract_microstructure_features(feats) # (20000, 13)
        combined = np.hstack([h_out[WARMUP:], feats[WARMUP:], extra[WARMUP:]]) # (19901, 128+112+13 = 253)
        targets = targs[WARMUP:]

        if idx < 25:
            H_train.append(combined)
            Y_train.append(targets)
        else:
            H_val.append(combined)
            Y_val.append(targets)

H_train = torch.from_numpy(np.concatenate(H_train, axis=0)).float()
Y_train = torch.from_numpy(np.concatenate(Y_train, axis=0)).float()
H_val = torch.from_numpy(np.concatenate(H_val, axis=0)).float()
Y_val = torch.from_numpy(np.concatenate(Y_val, axis=0)).float()

print(f'Train samples: {H_train.shape[0]:,}, Val samples: {H_val.shape[0]:,}, Feature Dim: {H_train.shape[1]}')

# 3. Define Deep Residual Head Architecture
class DeepResidualHead(nn.Module):
    def __init__(self, in_dim=253, hidden_dim=256):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.res_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_act = nn.GELU()
        self.res_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Decoupled heads for t0 and t1
        self.head_t0 = nn.Linear(hidden_dim, 1)
        self.head_t1 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_norm(x)
        h = self.act1(self.fc1(h))
        h = self.norm1(h)
        # Residual block
        res = self.res_fc2(self.res_act(self.res_fc1(h)))
        h = self.norm2(h + res)
        # Decoupled heads
        out0 = self.head_t0(h)
        out1 = self.head_t1(h)
        return torch.cat([out0, out1], dim=-1)

model_head = DeepResidualHead(in_dim=H_train.shape[1], hidden_dim=256)
loss_fn = WeightedPearsonLoss()
optimizer = torch.optim.AdamW(model_head.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

def score_head(head, H_data, Y_data):
    head.eval()
    with torch.no_grad():
        preds = head(H_data).numpy()
        acc = GlobalAccumulator()
        # Evaluate in 15 holdout blocks
        num_blocks = 15
        block_len = len(H_data) // num_blocks
        mask = np.ones(block_len, dtype=bool)
        for b in range(num_blocks):
            s = b * block_len
            e = s + block_len
            acc.add(Y_data[s:e].numpy(), preds[s:e], mask)
        res = acc.result()
        return res['weighted_pearson'], res['t0'], res['t1']

# Baseline Score on Val set (using original linear head)
W_base = weights['onnx::MatMul_186'].T
b_base = weights['reg_head.bias']
p_base_val = H_val[:, :128].numpy() @ W_base.T + b_base
acc_base = GlobalAccumulator()
block_len = len(H_val) // 15
mask = np.ones(block_len, dtype=bool)
for b in range(15):
    s = b * block_len
    e = s + block_len
    acc_base.add(Y_val[s:e].numpy(), p_base_val[s:e] * np.array([0.25, 0.40]), mask)
base_res = acc_base.result()
print(f'\nOriginal Baseline Val WP: {base_res["weighted_pearson"]:.6f} (t0: {base_res["t0"]:.6f}, t1: {base_res["t1"]:.6f})')

# Training loop
print('\nTraining Deep Residual Head directly on Weighted Pearson Loss...')
batch_size = 8192
n_samples = len(H_train)
best_wp = 0.0

for epoch in range(1, 11):
    model_head.train()
    perm = torch.randperm(n_samples)
    total_loss = 0.0
    steps = 0
    t0 = time.time()

    for i in range(0, n_samples, batch_size):
        idx = perm[i:i+batch_size]
        if len(idx) < 1000: continue
        x_b = H_train[idx]
        y_b = Y_train[idx]

        optimizer.zero_grad()
        pred_b = model_head(x_b)
        loss = loss_fn(pred_b, y_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    scheduler.step()
    dt = time.time() - t0
    wp, t0_s, t1_s = score_head(model_head, H_val, Y_val)
    marker = ' *** NEW BEST ***' if wp > best_wp else ''
    if wp > best_wp:
        best_wp = wp
        torch.save(model_head.state_dict(), 'deep_head_best.pt')

    print(f'Epoch {epoch:2d} ({dt:.1f}s) | Loss: {total_loss/steps:.4f} | Val WP: {wp:.6f} (t0: {t0_s:.6f}, t1: {t1_s:.6f}){marker}')

print(f'\nDeep Residual Head Training Complete! Peak Val WP: {best_wp:.6f} (vs Baseline: {base_res["weighted_pearson"]:.6f})')
