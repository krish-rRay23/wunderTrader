"""High-Performance GPU Fine-Tuning Engine for PretrainedDeepMicroNetwork.

Architecture:
- 212M-step Pretrained GRU Backbone (from baseline.onnx).
- Specialized Non-Linear Residual Microstructure Heads (Bid/Ask).
- Anti-Symmetric Manifold Coupling Layer.
- Direct Weighted Pearson Loss.
- Cosine Annealing Learning Rate Schedule.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from src.models.pretrained_deep_network import PretrainedDeepMicroNetwork
from src.loss.weighted_pearson import WeightedPearsonLoss
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP

print("Loading cached 50 sequences from scratch/cached_50_seqs.npz...", flush=True)
data = np.load('scratch/cached_50_seqs.npz')
features_all = data['features_all'] # (50, 20000, 112)
targets_all = data['targets_all']   # (50, 20000, 2)

train_X = features_all[:40]
train_Y = targets_all[:40]
val_X = features_all[40:]
val_Y = targets_all[40:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training Device: {device.type.upper()}", flush=True)

model = PretrainedDeepMicroNetwork(freeze_backbone=False).to(device)
print(f"PretrainedDeepMicroNetwork initialized with {sum(p.numel() for p in model.parameters()):,} parameters.", flush=True)

# Evaluation function
def evaluate(net):
    net.eval()
    acc = GlobalAccumulator()
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    
    with torch.no_grad():
        for s in range(len(val_X)):
            x_t = torch.from_numpy(val_X[s]).unsqueeze(0).to(device)
            preds, _, _ = net(x_t)
            acc.add(val_Y[s], preds[0].cpu().numpy(), mask)
    res = acc.result()
    return res['weighted_pearson'], res['t0'], res['t1']

# Benchmark Epoch 0
print("\n--- BASELINE INITIALIZATION BENCHMARK ---", flush=True)
init_wp, init_t0, init_t1 = evaluate(model)
print(f"Epoch  0 (Initialization) Holdout WP: {init_wp:.6f} (t0: {init_t0:.6f}, t1: {init_t1:.6f})\n", flush=True)

# Separate learning rates:
# Backbone: very small lr (5e-6) to preserve 212M-step pre-training while adapting
# Heads: standard lr (1e-3) for rapid convergence
optimizer = torch.optim.AdamW([
    {'params': model.gru0.parameters(), 'lr': 5e-6},
    {'params': model.gru1.parameters(), 'lr': 5e-6},
    {'params': model.base_head.parameters(), 'lr': 1e-4},
    {'params': model.res_bid_head.parameters(), 'lr': 1e-3},
    {'params': model.res_ask_head.parameters(), 'lr': 1e-3},
    {'params': [model.coupling, model.scale_0, model.scale_1, model.spread_beta], 'lr': 1e-3},
], weight_decay=1e-4)

epochs = 10
chunk_size = 4000
total_steps = len(train_X) * (SEQUENCE_LENGTH // chunk_size) * epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
scaler = GradScaler(enabled=(device.type == 'cuda'))
loss_fn = WeightedPearsonLoss()

best_wp = init_wp

print("==================================================", flush=True)
print(f"STARTING GPU FINE-TUNING LOOP ({epochs} EPOCHS)", flush=True)
print("==================================================", flush=True)

for epoch in range(1, epochs + 1):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    n_batches = 0
    
    perm = np.random.permutation(len(train_X))
    for s_i, s_idx in enumerate(perm):
        feats = train_X[s_idx]
        targets = train_Y[s_idx]
        h0, h1 = None, None
        
        for start in range(0, SEQUENCE_LENGTH, chunk_size):
            end = min(start + chunk_size, SEQUENCE_LENGTH)
            x_chunk = torch.from_numpy(feats[start:end]).unsqueeze(0).to(device)
            y_chunk = torch.from_numpy(targets[start:end]).unsqueeze(0).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda')):
                preds, h0, h1 = model(x_chunk, h0, h1)
                
                if end <= WARMUP:
                    h0 = h0.detach()
                    h1 = h1.detach()
                    continue
                    
                step_range = np.arange(start, end)
                mask_chunk = torch.from_numpy(step_range >= WARMUP).unsqueeze(0).to(device)
                loss = loss_fn(preds, y_chunk, mask_chunk)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            h0 = h0.detach()
            h1 = h1.detach()
            
            total_loss += loss.item()
            n_batches += 1
            
        if (s_i + 1) % 10 == 0:
            print(f"  Epoch {epoch:2d} | Seq {s_i+1:2d}/{len(train_X)} | Running Loss: {total_loss/max(1, n_batches):.4f}", flush=True)
            
    dt = time.time() - t0
    val_wp, val_t0, val_t1 = evaluate(model)
    marker = " *** NEW BEST MODEL ***" if val_wp > best_wp else ""
    if val_wp > best_wp:
        best_wp = val_wp
        torch.save(model.state_dict(), 'best_pretrained_micro.pt')
        
    print(f">> Epoch {epoch:2d} ({dt:.1f}s) | Train Loss: {total_loss/n_batches:.4f} | Val WP: {val_wp:.6f} (t0: {val_t0:.6f}, t1: {val_t1:.6f}){marker}", flush=True)

print(f"\nTraining Complete! Peak Holdout Score: {best_wp:.6f} (vs Baseline: {init_wp:.6f})", flush=True)
