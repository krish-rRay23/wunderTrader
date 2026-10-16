"""DeepMind-grade fine-tuning harness directly optimizing Weighted Pearson loss."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np
from src.data_reader import FastSequenceReader
from src.loss.weighted_pearson import WeightedPearsonLoss, CombinedWeightedLoss
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, DataPoint
import onnx
from onnx import numpy_helper

torch.set_num_threads(4)

class CompetitionGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.gru0 = nn.GRU(112, hidden_dim, batch_first=True)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.reg_head = nn.Linear(hidden_dim, 2)

    def forward(self, x, h0=None, h1=None):
        out0, h0_next = self.gru0(x, h0)
        out1, h1_next = self.gru1(out0, h1)
        pred = self.reg_head(out1)
        return pred, h0_next, h1_next

def load_baseline_weights(net: CompetitionGRU):
    model_onnx = onnx.load('baseline/baseline.onnx')
    weights = {init.name: numpy_helper.to_array(init) for init in model_onnx.graph.initializer}

    def onnx_to_torch_gru(w_onnx, r_onnx, b_onnx, gru_layer):
        z_w, r_w, h_w = np.split(w_onnx[0], 3, axis=0)
        w_torch = np.concatenate([r_w, z_w, h_w], axis=0)
        z_r, r_r, h_r = np.split(r_onnx[0], 3, axis=0)
        r_torch = np.concatenate([r_r, z_r, h_r], axis=0)
        b_w_onnx, b_r_onnx = np.split(b_onnx[0], 2, axis=0)
        z_bw, r_bw, h_bw = np.split(b_w_onnx, 3, axis=0)
        b_w_torch = np.concatenate([r_bw, z_bw, h_bw], axis=0)
        z_br, r_br, h_br = np.split(b_r_onnx, 3, axis=0)
        b_r_torch = np.concatenate([r_br, z_br, h_br], axis=0)

        gru_layer.weight_ih_l0.data = torch.from_numpy(w_torch.copy())
        gru_layer.weight_hh_l0.data = torch.from_numpy(r_torch.copy())
        gru_layer.bias_ih_l0.data = torch.from_numpy(b_w_torch.copy())
        gru_layer.bias_hh_l0.data = torch.from_numpy(b_r_torch.copy())

    onnx_to_torch_gru(weights['onnx::GRU_163'], weights['onnx::GRU_164'], weights['onnx::GRU_165'], net.gru0)
    onnx_to_torch_gru(weights['onnx::GRU_183'], weights['onnx::GRU_184'], weights['onnx::GRU_185'], net.gru1)
    net.reg_head.weight.data = torch.from_numpy(weights['onnx::MatMul_186'].T.copy())
    net.reg_head.bias.data = torch.from_numpy(weights['reg_head.bias'].copy())
    print('Baseline weights loaded into PyTorch network successfully!')

def evaluate_pytorch_model(net: CompetitionGRU, sequences, s0=0.25, s1=0.40):
    net.eval()
    acc = GlobalAccumulator()
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True

    with torch.no_grad():
        for seq_id, feats, targs in sequences:
            x = torch.from_numpy(feats).unsqueeze(0) # (1, 20000, 112)
            preds, _, _ = net(x)
            p_np = preds[0].numpy().copy()
            p_np[:, 0] *= s0
            p_np[:, 1] *= s1
            acc.add(targs, p_np, mask)

    return acc.result()

if __name__ == '__main__':
    print('Loading dataset sequences...')
    reader = FastSequenceReader('datasets/train.parquet')
    
    # Load 50 training sequences and 20 holdout sequences
    all_seqs = list(reader.iter_sequences(max_sequences=70))
    train_seqs = all_seqs[:50]
    val_seqs = all_seqs[50:70]
    print(f'Train: {len(train_seqs)} sequences (1,000,000 steps), Val: {len(val_seqs)} sequences (400,000 steps)')

    net = CompetitionGRU()
    load_baseline_weights(net)

    # Initial score before training
    init_res = evaluate_pytorch_model(net, val_seqs)
    print(f'Initial Holdout WP: {init_res["weighted_pearson"]:.6f} (t0: {init_res["t0"]:.6f}, t1: {init_res["t1"]:.6f})')

    # Train setup: optimize with Weighted Pearson loss
    loss_fn = CombinedWeightedLoss(wp_weight=0.8, wmse_weight=0.2)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    print('\nStarting fine-tuning with Combined Weighted Pearson loss...')
    chunk_size = 2000 # Backprop through time window of 2000 steps

    for epoch in range(1, 4):
        net.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        for seq_idx, (seq_id, feats, targs) in enumerate(train_seqs):
            h0, h1 = None, None
            # Process sequence in chunks of 2000 steps
            for start in range(0, SEQUENCE_LENGTH, chunk_size):
                end = min(start + chunk_size, SEQUENCE_LENGTH)
                x_chunk = torch.from_numpy(feats[start:end]).unsqueeze(0)
                y_chunk = torch.from_numpy(targs[start:end]).unsqueeze(0)

                pred_chunk, h0, h1 = net(x_chunk, h0, h1)
                h0 = h0.detach()
                h1 = h1.detach()

                # Calculate mask
                if end <= WARMUP:
                    continue
                step_indices = np.arange(start, end)
                mask_chunk = torch.from_numpy(step_indices >= WARMUP).unsqueeze(0)

                loss = loss_fn(pred_chunk, y_chunk, mask_chunk)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        scheduler.step()
        dt = time.time() - t0
        val_res = evaluate_pytorch_model(net, val_seqs)
        print(f'Epoch {epoch:2d} ({dt:.1f}s) | Avg Loss: {total_loss/n_batches:.4f} | Val WP: {val_res["weighted_pearson"]:.6f} (t0: {val_res["t0"]:.6f}, t1: {val_res["t1"]:.6f})')

    print('\nTraining complete! Evaluating final gain...')
    final_res = evaluate_pytorch_model(net, val_seqs)
    delta = final_res['weighted_pearson'] - init_res['weighted_pearson']
    print(f'Final Validation WP: {final_res["weighted_pearson"]:.6f} (Gain over initial: {delta:+.6f})')
