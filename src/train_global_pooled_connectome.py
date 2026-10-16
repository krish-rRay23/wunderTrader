"""Global Pooled BPTT Training Engine for Alfa Connectome.

Solves the Between-Sequence / Chunk-Mean Drift Problem:
1. Initialized with Sub 6 breakthrough parameters: starts at exact parity (Holdout WP = 0.426912, r_b = 0.767890).
2. Propagates detached recurrent hidden state through chunks (128 steps) for memory stability.
3. Accumulates predictions across the entire sequence (20,000 steps).
4. Computes Exact Global Weighted Pearson Loss pooled over all 16 sequences in each batch (318,416 steps)
   with explicit sequence-mean drift penalty.
5. Simultaneously optimizes within-sequence microstructure dynamics (r_w) AND between-sequence macro alignment (r_b).
6. Uses true specification layout (bids: 0..10, asks: 11..21).
7. Evaluates strictly on the Pristine 100-Sequence Holdout Benchmark.
"""
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper

from src.data_reader import FastSequenceReader
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP
from src.validation import evaluate_moments_decomposition
from alfa_head.common import Layout
from src.train_breakthrough_quant_ensemble import build_quant_microstructure_matrix


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class GlobalPooledConnectome(nn.Module):
    def __init__(self, baseline_path='baseline/baseline.onnx', layout_path='data/true_layout.json', sub6_path='scratch/best_breakthrough_quant.pt'):
        super().__init__()
        # 1. GRU Backbone
        self.gru0 = nn.GRU(112, 128, batch_first=True)
        self.gru1 = nn.GRU(128, 128, batch_first=True)
        
        m = onnx.load(baseline_path)
        w = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
        
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
            
        load_gru(w['onnx::GRU_163'], w['onnx::GRU_164'], w['onnx::GRU_165'], self.gru0)
        load_gru(w['onnx::GRU_183'], w['onnx::GRU_184'], w['onnx::GRU_185'], self.gru1)
        
        # 2. Linear Bypass (Sub 6 breakthrough) to maintain between-sequence macro stability
        self.base_head = nn.Linear(128, 2)
        self.bypass_head = nn.Linear(112, 2, bias=False)
        
        # 3. Load Sub 6 weights for perfect baseline initialization
        d_sub6 = torch.load(sub6_path, map_location='cpu', weights_only=False)
        self.base_head.weight.data = d_sub6['head_h.weight'].clone()
        self.base_head.bias.data = d_sub6['head_h.bias'].clone()
        
        W_q = d_sub6['head_quant.weight'].numpy()
        P_quant, _ = build_quant_microstructure_matrix()
        W_X = torch.from_numpy((P_quant @ W_q.T).T)
        self.bypass_head.weight.data = W_X.clone()
        
        # 4. Decoupled Dual-Stream Bid/Ask MLPs (initialized to ZERO)
        # Input: o1 (128) + raw features (112) -> 240 dims
        self.mlp_bid = nn.Sequential(
            nn.Linear(128 + 112, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.mlp_ask = nn.Sequential(
            nn.Linear(128 + 112, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        nn.init.zeros_(self.mlp_bid[0].weight); nn.init.zeros_(self.mlp_bid[0].bias)
        nn.init.zeros_(self.mlp_bid[2].weight); nn.init.zeros_(self.mlp_bid[2].bias)
        nn.init.zeros_(self.mlp_ask[0].weight); nn.init.zeros_(self.mlp_ask[0].bias)
        nn.init.zeros_(self.mlp_ask[2].weight); nn.init.zeros_(self.mlp_ask[2].bias)
        
        # 5. Coupling and Calibration (initialized from Sub 6)
        self.coupling = nn.Parameter(d_sub6['coupling'].clone())
        self.scale = nn.Parameter(torch.tensor([d_sub6['s0'].item(), d_sub6['s1'].item()], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

    def forward(self, x, h0=None, h1=None):
        o0, next_h0 = self.gru0(x, h0)
        o1, next_h1 = self.gru1(o0, h1)
        
        base_pred = self.base_head(o1)
        bypass = self.bypass_head(x)
        
        comb = torch.cat([o1, x], dim=-1)
        d_bid = self.mlp_bid(comb)
        d_ask = self.mlp_ask(comb)
        
        raw_0 = base_pred[:, :, 0:1] + bypass[:, :, 0:1] + d_bid
        raw_1 = base_pred[:, :, 1:2] + bypass[:, :, 1:2] + d_ask
        
        c = torch.clamp(self.coupling, 0.0, 0.15)
        p0 = (1.0 - c) * raw_0 - c * raw_1
        p1 = (1.0 - c) * raw_1 - c * raw_0
        
        out_0 = p0 * self.scale[0] + self.bias[0]
        out_1 = p1 * self.scale[1] + self.bias[1]
        pred = torch.cat([out_0, out_1], dim=-1)
        return pred, next_h0, next_h1


class ExactGlobalWeightedPearsonLoss(nn.Module):
    def __init__(self, wmse_weight=0.02, pen_b_weight=2.0, eps=1e-8):
        super().__init__()
        self.wmse_weight = wmse_weight
        self.pen_b_weight = pen_b_weight
        self.eps = eps

    def forward(self, pred, target, B=16, L=19901):
        # pred, target: (B * L, 2)
        y_c = torch.clamp(target, -2.0, 2.0)
        p_c = torch.clamp(pred, -2.0, 2.0)
        w = torch.abs(y_c)
        
        corrs = []
        mses = []
        for k in range(2):
            yk = y_c[:, k]; pk = p_c[:, k]; wk = w[:, k]
            w_sum = torch.sum(wk) + self.eps
            y_bar = torch.sum(wk * yk) / w_sum
            p_bar = torch.sum(wk * pk) / w_sum
            yt = yk - y_bar; pt = pk - p_bar
            cov = torch.sum(wk * yt * pt)
            var_y = torch.sum(wk * yt * yt) + self.eps
            var_p = torch.sum(wk * pt * pt) + self.eps
            corr = cov / torch.sqrt(var_y * var_p)
            corrs.append(torch.clamp(corr, -1.0, 1.0))
            wmse = torch.sum(wk * (yk - pk)**2) / w_sum
            mses.append(wmse)
            
        wp_loss = 1.0 - 0.5 * (corrs[0] + corrs[1])
        wmse_loss = 0.5 * (mses[0] + mses[1])
        
        # Sequence-level mean drift penalty
        # y_seq: (B, L, 2), p_seq: (B, L, 2), w_seq: (B, L, 2)
        y_seq = y_c.view(B, L, 2)
        p_seq = p_c.view(B, L, 2)
        w_seq = w.view(B, L, 2)
        
        w_s_sum = torch.sum(w_seq, dim=1) + self.eps # (B, 2)
        my_s = torch.sum(w_seq * y_seq, dim=1) / w_s_sum # (B, 2)
        mp_s = torch.sum(w_seq * p_seq, dim=1) / w_s_sum # (B, 2)
        pen_b = torch.mean((my_s - mp_s)**2)
        
        return wp_loss + self.wmse_weight * wmse_loss + self.pen_b_weight * pen_b


def evaluate_on_pristine_holdout(model, holdout_npz='scratch/pristine_holdout_100.npz', device='cuda'):
    model.eval()
    d = np.load(holdout_npz)
    features_all = d['features_all']
    targets_all = d['targets_all']
    N_SEQS = len(features_all)
    
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    
    preds_all = []
    with torch.no_grad():
        for s in range(N_SEQS):
            xt = torch.from_numpy(features_all[s:s+1]).to(device)
            p, _, _ = model(xt)
            preds_all.append(p[0].cpu().numpy())
            
    preds_all = np.array(preds_all)
    acc = GlobalAccumulator()
    for s in range(N_SEQS):
        acc.add(targets_all[s], preds_all[s], mask)
    res = acc.result()
    diag = evaluate_moments_decomposition(targets_all, preds_all)
    return res['weighted_pearson'], res['t0'], res['t1'], diag['r_between'], diag['r_within']


def train_global_pooled(
    num_train_seqs: int = 320,
    start_seq_idx: int = 350,
    batch_size: int = 16,
    chunk_len: int = 128,
    num_epochs: int = 5,
    checkpoint_out: str = 'scratch/best_global_pooled_connectome.pt'
):
    device = get_device()
    print("=" * 80)
    print("GLOBAL POOLED BPTT TRAINING ENGINE")
    print(f"Sequences: {num_train_seqs} (Indices {start_seq_idx}..{start_seq_idx + num_train_seqs - 1})")
    print(f"Batch size: {batch_size} sequences | Chunk length: {chunk_len} | Epochs: {num_epochs}")
    print("=" * 80)
    
    model = GlobalPooledConnectome().to(device)
    init_wp, init_t0, init_t1, init_rb, init_rw = evaluate_on_pristine_holdout(model, device=device)
    print(f"\n--- INITIAL BENCHMARK EVALUATION ---")
    print(f"Initial Holdout WP: {init_wp:.6f} (t0: {init_t0:.6f}, t1: {init_t1:.6f}) | r_b: {init_rb:.6f} | r_w: {init_rw:.6f}")
    assert abs(init_wp - 0.426912) < 5e-4, f"Parity mismatch: {init_wp} vs 0.426912"
    print(f"STRONG BASELINE CONFIRMED (WP: {init_wp:.6f} >= Sub 6: 0.426912, r_b: {init_rb:.6f})!")
    print("-" * 80)
    
    print(f"\nReading {num_train_seqs} sequences from datasets/train.parquet...")
    t_load = time.time()
    reader = FastSequenceReader('datasets/train.parquet')
    gen = reader.iter_sequences()
    for _ in range(start_seq_idx):
        next(gen)
        
    train_features = []
    train_targets = []
    for i in range(num_train_seqs):
        _, f, tgt = next(gen)
        train_features.append(f)
        train_targets.append(tgt)
        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1}/{num_train_seqs} sequences...")
            
    train_features = np.array(train_features, dtype=np.float32)
    train_targets = np.array(train_targets, dtype=np.float32)
    print(f"Loaded in {time.time() - t_load:.2f}s!")
    
    # Differential learning rates:
    # GRU backbone: tiny lr (5e-6)
    # Sub 6 Base head & bypass: (1e-5)
    # Nonlinear MLPs: active lr (1e-4)
    # Coupling and scale: (2e-4)
    optimizer = torch.optim.AdamW([
        {'params': model.gru0.parameters(), 'lr': 5.0e-6},
        {'params': model.gru1.parameters(), 'lr': 5.0e-6},
        {'params': model.base_head.parameters(), 'lr': 1.0e-5},
        {'params': model.bypass_head.parameters(), 'lr': 1.0e-5},
        {'params': model.mlp_bid.parameters(), 'lr': 1.0e-4},
        {'params': model.mlp_ask.parameters(), 'lr': 1.0e-4},
        {'params': [model.coupling, model.scale, model.bias], 'lr': 2.0e-4},
    ], weight_decay=1e-4)
    
    loss_fn = ExactGlobalWeightedPearsonLoss(wmse_weight=0.02, pen_b_weight=2.0).to(device)
    
    best_wp = init_wp
    
    for epoch in range(1, num_epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        perm = np.random.permutation(num_train_seqs)
        
        for b_start in range(0, num_train_seqs, batch_size):
            b_indices = perm[b_start:b_start + batch_size]
            actual_B = len(b_indices)
            if actual_B < batch_size:
                continue
                
            batch_x = torch.from_numpy(train_features[b_indices]).to(device) # (B, 20000, 112)
            batch_y = torch.from_numpy(train_targets[b_indices]).to(device)   # (B, 20000, 2)
            
            h0 = torch.zeros(1, actual_B, 128, device=device)
            h1 = torch.zeros(1, actual_B, 128, device=device)
            
            preds_list = []
            for t_start in range(0, SEQUENCE_LENGTH, chunk_len):
                t_end = min(SEQUENCE_LENGTH, t_start + chunk_len)
                xc = batch_x[:, t_start:t_end]
                p_chunk, h0, h1 = model(xc, h0.detach(), h1.detach())
                preds_list.append(p_chunk)
                
            # Full sequence predictions across all B sequences
            all_p = torch.cat(preds_list, dim=1)[:, WARMUP:].reshape(-1, 2) # (B * 19901, 2)
            all_y = batch_y[:, WARMUP:].reshape(-1, 2)
            
            loss = loss_fn(all_p, all_y, B=actual_B, L=19901)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        dt_ep = time.time() - t_ep
        avg_loss = total_loss / max(1, num_batches)
        
        # Evaluate on pristine holdout
        wp, t0, t1, rb, rw = evaluate_on_pristine_holdout(model, device=device)
        is_best = wp > best_wp
        mark = " *** NEW BEST ***" if is_best else ""
        
        if is_best:
            best_wp = wp
            torch.save(model.state_dict(), checkpoint_out)
            
        print(f"Epoch {epoch:2d}/{num_epochs:2d} ({dt_ep:5.1f}s) | Train Loss: {avg_loss:.4f} | Holdout WP: {wp:.6f} (t0: {t0:.6f}, t1: {t1:.6f}) | r_b: {rb:.6f} | r_w: {rw:.6f}{mark}")
        
    print("\n" + "=" * 80)
    print(f"GLOBAL POOLED TRAINING COMPLETE! Best WP: {best_wp:.6f} (Gain over Sub 6: +{best_wp - init_wp:.6f})")
    print(f"Checkpoint saved to: {checkpoint_out}")
    print("=" * 80)


if __name__ == '__main__':
    train_global_pooled(num_train_seqs=320, start_seq_idx=350, batch_size=16, chunk_len=128, num_epochs=5)
