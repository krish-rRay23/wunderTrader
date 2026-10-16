"""Ultra-Fast In-Memory Training of DeepMicrostructureNetwork on Cached Sequences."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np

from src.models.deep_microstructure_network import DeepMicrostructureNetwork
from src.loss.weighted_pearson import WeightedPearsonLoss
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP

print("Loading cached 50 sequences from scratch/cached_50_seqs.npz...", flush=True)
data = np.load('scratch/cached_50_seqs.npz')
features_all = data['features_all'] # (50, 20000, 112)
targets_all = data['targets_all']   # (50, 20000, 2)
print(f"Loaded: features {features_all.shape}, targets {targets_all.shape}", flush=True)

# Train on 40 sequences (800,000 steps), Validate on 10 holdout sequences (200,000 steps)
train_X = features_all[:40]
train_Y = targets_all[:40]
val_X = features_all[40:]
val_Y = targets_all[40:]

torch.set_num_threads(10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device.type.upper()}", flush=True)

model = DeepMicrostructureNetwork(in_features=112, hidden_dim=192).to(device)
print(f"DeepMicrostructureNetwork initialized with {sum(p.numel() for p in model.parameters()):,} parameters.", flush=True)

def evaluate(model):
    model.eval()
    acc = GlobalAccumulator()
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    
    with torch.no_grad():
        for s in range(len(val_X)):
            x_t = torch.from_numpy(val_X[s]).unsqueeze(0).to(device)
            preds, _, _ = model(x_t)
            acc.add(val_Y[s], preds[0].cpu().numpy(), mask)
    res = acc.result()
    return res['weighted_pearson'], res['t0'], res['t1']

# Baseline evaluation on holdout set
from utils import weighted_pearson
raw_val_p = data['raw_preds'][40:] * np.array([0.275, 0.425], dtype=np.float32)
acc_base = GlobalAccumulator()
mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
mask[WARMUP:] = True
for s in range(len(val_Y)):
    acc_base.add(val_Y[s], raw_val_p[s], mask)
res_base = acc_base.result()
print(f"\n--- REFERENCE BASELINE ON THIS HOLDOUT SPLIT ---", flush=True)
print(f"Baseline (Scaled) Holdout WP: {res_base['weighted_pearson']:.6f} (t0: {res_base['t0']:.6f}, t1: {res_base['t1']:.6f})\n", flush=True)

# Loss and Optimizer
chunk_size = 4000
epochs = 8
total_steps = len(train_X) * (SEQUENCE_LENGTH // chunk_size) * epochs

loss_fn = WeightedPearsonLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-5)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

best_wp = -1.0

print("==================================================", flush=True)
print(f"STARTING GPU IN-MEMORY TRAINING LOOP ({epochs} EPOCHS)", flush=True)
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
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
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
    
    val_wp, t0_s, t1_s = evaluate(model)
    marker = " *** NEW BEST ***" if val_wp > best_wp else ""
    if val_wp > best_wp:
        best_wp = val_wp
        torch.save(model.state_dict(), 'deep_micro_best.pt')
        
    print(f">> Epoch {epoch:2d} ({dt:.1f}s) | Train Loss: {total_loss/n_batches:.4f} | Val WP: {val_wp:.6f} (t0: {t0_s:.6f}, t1: {t1_s:.6f}){marker}", flush=True)

print(f"\nTraining Complete! Peak Holdout WP: {best_wp:.6f}", flush=True)

