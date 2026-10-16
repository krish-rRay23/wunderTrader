"""Dual-Stream Decoupled Recurrent Network for Alpha Connectome.

Key Architectural Innovations:
1. Bid-Stream Head (specialized for t0) receiving GRU hidden states + Bid microstructure levels.
2. Ask-Stream Head (specialized for t1) receiving GRU hidden states + Ask microstructure levels.
3. Trainable Anti-Symmetric Cross-Coupling Layer exploiting r(t0, t1) ≈ -0.82.
4. Trainable Scale & Bias calibration preventing clipping boundary saturation.
5. Differential Learning Rates: Frozen/Micro-LR (1e-6) for GRU body, Active LR (1e-3) for dual heads.
6. Combined Loss: Weighted Pearson Loss + Normalized Weighted MSE with gradient clipping.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np
from src.data_reader import FastSequenceReader
from src.loss.weighted_pearson import WeightedPearsonLoss, CombinedWeightedLoss
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, FEATURE_COLUMNS
import onnx
from onnx import numpy_helper

torch.set_num_threads(4)

class DualStreamNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # 1. Shared Recurrent Temporal Backbone (pre-initialized from baseline)
        self.gru0 = nn.GRU(112, hidden_dim, batch_first=True)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # 2. Bid-focused feature slice indices:
        # i0_p0..p10 (0..10), i0_v0..v10 (22..32), i0_dp0..dp3 (44..47), a0..a7 (104..111)
        # Total bid features: 11 + 11 + 4 + 8 = 34
        self.bid_indices = list(range(0, 11)) + list(range(22, 33)) + list(range(44, 48)) + list(range(104, 112))
        
        # 3. Ask-focused feature slice indices:
        # i0_p11..p21 (11..21), i0_v11..v21 (33..43), i0_dp0..dp3 (44..47), a0..a7 (104..111)
        # Total ask features: 11 + 11 + 4 + 8 = 34
        self.ask_indices = list(range(11, 22)) + list(range(33, 44)) + list(range(44, 48)) + list(range(104, 112))
        
        # Pretrained linear baseline heads (preserving full pretrained representations)
        self.base_linear_0 = nn.Linear(128, 1)
        self.base_linear_1 = nn.Linear(128, 1)

        # Decoupled Non-linear Residual Microstructure Heads:
        # Input: 128 (GRU out) + 34 (microstructure) = 162
        self.res_mlp_0 = nn.Sequential(
            nn.Linear(128 + 34, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
        self.res_mlp_1 = nn.Sequential(
            nn.Linear(128 + 34, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
        # Trainable anti-symmetric coupling weight
        self.coupling = nn.Parameter(torch.tensor([0.025], dtype=torch.float32))
        
        # Trainable output calibration scales (initialized to optimal regime)
        self.scale_0 = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
        self.scale_1 = nn.Parameter(torch.tensor([0.45], dtype=torch.float32))
        self.bias_0 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.bias_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, x, h0=None, h1=None):
        # x: (batch, seq_len, 112)
        out0, h0_next = self.gru0(x, h0)
        out1, h1_next = self.gru1(out0, h1) # (batch, seq_len, 128)
        
        # Slice specialized feature channels
        x_bid = x[:, :, self.bid_indices] # (batch, seq_len, 34)
        x_ask = x[:, :, self.ask_indices] # (batch, seq_len, 34)
        
        in_t0 = torch.cat([out1, x_bid], dim=-1)
        in_t1 = torch.cat([out1, x_ask], dim=-1)
        
        # Baseline projection + learned non-linear residual microstructure signal
        raw_0 = self.base_linear_0(out1) + self.res_mlp_0(in_t0)
        raw_1 = self.base_linear_1(out1) + self.res_mlp_1(in_t1)
        
        # Anti-symmetric cross coupling
        c = torch.clamp(self.coupling, 0.0, 0.20)
        p0 = (1.0 - c) * raw_0 - c * raw_1
        p1 = (1.0 - c) * raw_1 - c * raw_0
        
        # Calibrated output scaling
        out_0 = p0 * self.scale_0 + self.bias_0
        out_1 = p1 * self.scale_1 + self.bias_1
        
        pred = torch.cat([out_0, out_1], dim=-1) # (batch, seq_len, 2)
        return pred, h0_next, h1_next


def load_pretrained_gru(model: DualStreamNetwork):
    model_onnx = onnx.load('baseline/baseline.onnx')
    weights = {init.name: numpy_helper.to_array(init) for init in model_onnx.graph.initializer}

    def load_gru_layer(w, r, b, layer):
        zw, rw, hw = np.split(w[0], 3, axis=0)
        layer.weight_ih_l0.data = torch.from_numpy(np.concatenate([rw, zw, hw], axis=0).copy())
        zr, rr, hr = np.split(r[0], 3, axis=0)
        layer.weight_hh_l0.data = torch.from_numpy(np.concatenate([rr, zr, hr], axis=0).copy())
        bw, br = np.split(b[0], 2, axis=0)
        zbw, rbw, hbw = np.split(bw, 3, axis=0)
        zbr, rbr, hbr = np.split(br, 3, axis=0)
        layer.bias_ih_l0.data = torch.from_numpy(np.concatenate([rbw, zbw, hbw], axis=0).copy())
        layer.bias_hh_l0.data = torch.from_numpy(np.concatenate([rbr, zbr, hbr], axis=0).copy())

    load_gru_layer(weights['onnx::GRU_163'], weights['onnx::GRU_164'], weights['onnx::GRU_165'], model.gru0)
    load_gru_layer(weights['onnx::GRU_183'], weights['onnx::GRU_184'], weights['onnx::GRU_185'], model.gru1)
    
    # Initialize base_linear_0 and base_linear_1 EXACTLY from baseline head!
    W_base = weights['onnx::MatMul_186'].T # (2, 128)
    b_base = weights['reg_head.bias']       # (2,)
    
    with torch.no_grad():
        model.base_linear_0.weight.data = torch.from_numpy(W_base[0:1].copy())
        model.base_linear_0.bias.data = torch.from_numpy(b_base[0:1].copy())
        model.base_linear_1.weight.data = torch.from_numpy(W_base[1:2].copy())
        model.base_linear_1.bias.data = torch.from_numpy(b_base[1:2].copy())
        
        # Zero-initialize the final layer of both residual MLPs!
        # This guarantees that at epoch 0, the residual is exactly 0.0,
        # perfectly matching the baseline's high performance before fine-tuning!
        nn.init.zeros_(model.res_mlp_0[3].weight)
        nn.init.zeros_(model.res_mlp_0[3].bias)
        nn.init.zeros_(model.res_mlp_1[3].weight)
        nn.init.zeros_(model.res_mlp_1[3].bias)

    print('Pretrained GRU backbone and base linear heads loaded successfully!')


def evaluate_model(model: DualStreamNetwork, val_sequences):
    model.eval()
    acc = GlobalAccumulator()
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    
    with torch.no_grad():
        for seq_id, feats, targets in val_sequences:
            x_t = torch.from_numpy(feats).unsqueeze(0) # (1, 20000, 112)
            preds, _, _ = model(x_t)
            p_np = preds[0].numpy()
            acc.add(targets, p_np, mask)
            
    res = acc.result()
    return res['weighted_pearson'], res['t0'], res['t1']


if __name__ == '__main__':
    print('Initializing DualStreamNetwork...')
    model = DualStreamNetwork()
    load_pretrained_gru(model)
    
    print('Loading dataset sequences from train.parquet...')
    reader = FastSequenceReader('datasets/train.parquet')
    
    # Load 60 sequences: 45 train (900,000 steps), 15 holdout (300,000 steps)
    all_seqs = list(reader.iter_sequences(max_sequences=60))
    train_seqs = all_seqs[:45]
    val_seqs = all_seqs[45:60]
    
    print(f'Train set: {len(train_seqs)} seqs (900,000 steps), Val set: {len(val_seqs)} seqs (300,000 steps)')
    
    # Evaluate baseline equivalent
    loss_fn = CombinedWeightedLoss(wp_weight=0.85, wmse_weight=0.15)
    
    # Freeze GRU backbone first, train heads for 2 epochs, then unfreeze with micro-LR
    for p in model.gru0.parameters(): p.requires_grad = False
    for p in model.gru1.parameters(): p.requires_grad = False
    
    head_params = (
        list(model.base_linear_0.parameters()) +
        list(model.base_linear_1.parameters()) +
        list(model.res_mlp_0.parameters()) +
        list(model.res_mlp_1.parameters()) +
        [model.coupling, model.scale_0, model.scale_1, model.bias_0, model.bias_1]
    )
    optimizer = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    
    # Evaluate at initialization (Epoch 0)
    init_wp, init_t0, init_t1 = evaluate_model(model, val_seqs)
    print(f'\n--- INITIALIZATION (Epoch 0) ---')
    print(f'Holdout WP: {init_wp:.6f} (t0: {init_t0:.6f}, t1: {init_t1:.6f})')
    best_wp = init_wp

    print('\nStage 1: Training Decoupled Microstructure Residual Heads (GRU Backbone Frozen)...')
    chunk_size = 4000
    for epoch in range(1, 4):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_chunks = 0
        
        # Permute sequences for training diversity
        perm = np.random.permutation(len(train_seqs))
        for seq_i in perm:
            _, feats, targets = train_seqs[seq_i]
            h0, h1 = None, None
            for start in range(0, SEQUENCE_LENGTH, chunk_size):
                end = min(start + chunk_size, SEQUENCE_LENGTH)
                x_chunk = torch.from_numpy(feats[start:end]).unsqueeze(0)
                y_chunk = torch.from_numpy(targets[start:end]).unsqueeze(0)
                
                preds, h0, h1 = model(x_chunk, h0, h1)
                h0 = h0.detach()
                h1 = h1.detach()
                
                if end <= WARMUP:
                    continue
                step_idx = np.arange(start, end)
                mask_chunk = torch.from_numpy(step_idx >= WARMUP).unsqueeze(0)
                
                loss = loss_fn(preds, y_chunk, mask_chunk)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(head_params, 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_chunks += 1
                
        dt = time.time() - t0
        wp, t0_s, t1_s = evaluate_model(model, val_seqs)
        marker = ' *** NEW BEST ***' if wp > best_wp else ''
        if wp > best_wp:
            best_wp = wp
            torch.save(model.state_dict(), 'dual_stream_best.pt')
        print(f'Epoch {epoch:2d} ({dt:.1f}s) | Loss: {total_loss/n_chunks:.4f} | Val WP: {wp:.6f} (t0: {t0_s:.6f}, t1: {t1_s:.6f}){marker}')

    # Stage 2: Fine-tune backbone with micro-LR (1e-6)
    print('\nStage 2: End-to-End Fine-Tuning with Micro-LR on Backbone...')
    for p in model.gru0.parameters(): p.requires_grad = True
    for p in model.gru1.parameters(): p.requires_grad = True
    
    optimizer = torch.optim.AdamW([
        {'params': model.gru0.parameters(), 'lr': 1e-6},
        {'params': model.gru1.parameters(), 'lr': 1e-6},
        {'params': head_params, 'lr': 5e-4}
    ], weight_decay=1e-5)
    
    for epoch in range(4, 7):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_chunks = 0
        perm = np.random.permutation(len(train_seqs))
        for seq_i in perm:
            _, feats, targets = train_seqs[seq_i]
            h0, h1 = None, None
            for start in range(0, SEQUENCE_LENGTH, chunk_size):
                end = min(start + chunk_size, SEQUENCE_LENGTH)
                x_chunk = torch.from_numpy(feats[start:end]).unsqueeze(0)
                y_chunk = torch.from_numpy(targets[start:end]).unsqueeze(0)
                
                preds, h0, h1 = model(x_chunk, h0, h1)
                h0 = h0.detach()
                h1 = h1.detach()
                
                if end <= WARMUP:
                    continue
                step_idx = np.arange(start, end)
                mask_chunk = torch.from_numpy(step_idx >= WARMUP).unsqueeze(0)
                
                loss = loss_fn(preds, y_chunk, mask_chunk)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                total_loss += loss.item()
                n_chunks += 1
                
        dt = time.time() - t0
        wp, t0_s, t1_s = evaluate_model(model, val_seqs)
        marker = ' *** NEW BEST ***' if wp > best_wp else ''
        if wp > best_wp:
            best_wp = wp
            torch.save(model.state_dict(), 'dual_stream_best.pt')
        print(f'Epoch {epoch:2d} ({dt:.1f}s) | Loss: {total_loss/n_chunks:.4f} | Val WP: {wp:.6f} (t0: {t0_s:.6f}, t1: {t1_s:.6f}){marker}')

    print(f'\nTraining Complete! Peak Holdout WP: {best_wp:.6f}')
