"""Production Training & Mathematical Tensor Fusion Engine for Alfa Connectome.

Trains the Multi-Horizon Residual Quant Connectome across 500 full sequences (10,000,000 steps):
- Anchor: 212M-step pretrained GRU Layer 1 macro manifold (locked r_between).
- Fast Tick Flow: Layer 0 GRU order-flow representation (128 dims).
- True LOB Depth Microstructure: 36-feature decoded physical permutation matrix P_true.
- Exact Target Eigenmode Scaling & Anti-Symmetric Coupling.
- Evaluated on the 100-sequence Pristine Holdout Benchmark (2,000,000 steps).
- Closed-Form Mathematical Tensor Fusion into a minimal 14-node ONNX model (< 35 min on server).
"""
import sys
import time
import os
import gc
import zipfile
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from src.build_true_lob_matrix import build_true_lob_matrix
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, METRIC_CLIP
from src.validation import evaluate_moments_decomposition
from src.extract_baseline_weights import FrozenBaselineGRU


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class MultiHorizonResidualModel(nn.Module):
    def __init__(self, sub6_path='scratch/best_breakthrough_quant.pt'):
        super().__init__()
        d_sub6 = torch.load(sub6_path, map_location='cpu', weights_only=False)
        
        # Fixed Baseline O1 Head (Anchor)
        self.head_o1 = nn.Linear(128, 2)
        self.head_o1.weight = nn.Parameter(d_sub6['head_h.weight'].clone(), requires_grad=False)
        self.head_o1.bias = nn.Parameter(d_sub6['head_h.bias'].clone(), requires_grad=False)
        
        # Trainable Fast Tick Head (O0 -> 2)
        self.head_o0 = nn.Linear(128, 2, bias=False)
        nn.init.zeros_(self.head_o0.weight)
        
        # Trainable True Depth Microstructure Head (Q_true -> 2)
        self.head_q = nn.Linear(36, 2, bias=False)
        nn.init.zeros_(self.head_q.weight)
        
        # Scaling and Coupling parameters
        self.s0 = nn.Parameter(torch.tensor([float(d_sub6['s0'])], dtype=torch.float32))
        self.s1 = nn.Parameter(torch.tensor([float(d_sub6['s1'])], dtype=torch.float32))
        self.coupling = nn.Parameter(d_sub6['coupling'].clone())
        self.bias = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

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


class ExactGlobalWeightedPearsonLoss(nn.Module):
    def __init__(self, wmse_weight=0.015, pen_b_weight=1.5, eps=1e-8):
        super().__init__()
        self.wmse_weight = wmse_weight
        self.pen_b_weight = pen_b_weight
        self.eps = eps

    def forward(self, pred, target, B=25, L=19901):
        y_c = torch.clamp(target, -METRIC_CLIP, METRIC_CLIP)
        p_c = torch.clamp(pred, -METRIC_CLIP, METRIC_CLIP)
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
        y_seq = y_c.view(B, L, 2)
        p_seq = p_c.view(B, L, 2)
        w_seq = w.view(B, L, 2)
        w_s_sum = torch.sum(w_seq, dim=1) + self.eps
        my_s = torch.sum(w_seq * y_seq, dim=1) / w_s_sum
        mp_s = torch.sum(w_seq * p_seq, dim=1) / w_s_sum
        pen_b = torch.mean((my_s - mp_s)**2)
        
        return wp_loss + self.wmse_weight * wmse_loss + self.pen_b_weight * pen_b


def export_fused_onnx(model, P_true_np, baseline_path='baseline/baseline.onnx', export_path='submission/model.onnx'):
    """Collapses the multi-horizon residual model and baseline GRU into a 14-node ONNX model."""
    print("\n========================================================")
    print("MATHEMATICAL TENSOR FUSION INTO 14-NODE ONNX GRAPH")
    print("========================================================")
    
    W_o1 = model.head_o1.weight.data.cpu().numpy() # (2, 128)
    b_o1 = model.head_o1.bias.data.cpu().numpy()   # (2,)
    W_o0 = model.head_o0.weight.data.cpu().numpy() # (2, 128)
    W_q = model.head_q.weight.data.cpu().numpy()   # (2, 36)
    
    s0 = float(model.s0.data.cpu())
    s1 = float(model.s1.data.cpu())
    c = float(torch.clamp(model.coupling, 0.0, 0.15).data.cpu())
    bias = model.bias.data.cpu().numpy()           # (2,)
    
    print(f"Learned Calibrators: s0={s0:.6f}, s1={s1:.6f}, c={c:.6f}, bias={bias}")
    print(f"Fast Tick Head Norm:  {np.linalg.norm(W_o0):.6f}")
    print(f"True Depth Head Norm: {np.linalg.norm(W_q):.6f}")
    
    M_c = np.array([
        [(1.0 - c) * s0, -c * s0],
        [-c * s1, (1.0 - c) * s1]
    ], dtype=np.float32)
    
    W_O1_fused = W_o1.T @ M_c # (128, 2)
    W_O0_fused = W_o0.T @ M_c # (128, 2)
    b_fused = b_o1 @ M_c + bias # (2,)
    
    W_q_coupled = W_q.T @ M_c # (36, 2)
    W_X_fused = P_true_np @ W_q_coupled # (112, 2)
    
    print(f"Fused O1 Recurrent Head: {W_O1_fused.shape}, Norm: {np.linalg.norm(W_O1_fused):.6f}")
    print(f"Fused O0 Recurrent Head: {W_O0_fused.shape}, Norm: {np.linalg.norm(W_O0_fused):.6f}")
    print(f"Fused Input Bypass:      {W_X_fused.shape}, Norm: {np.linalg.norm(W_X_fused):.6f}")
    print(f"Fused Output Bias:       {b_fused}")
    
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
            
    base_gru = FrozenBaselineGRU(baseline_path).cpu().eval()
    fused_net = FusedMultiHorizonNet(base_gru.gru0, base_gru.gru1, W_O0_fused, W_O1_fused, W_X_fused, b_fused).eval()
    
    # Export ONNX
    x_dummy = torch.randn(1, 1, 112)
    h0_dummy = torch.zeros(1, 1, 128)
    h1_dummy = torch.zeros(1, 1, 128)
    
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
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
    print(f"Exported {export_path} ({len(m_onnx.graph.node)} ONNX nodes, size: {file_size_kb:.1f} KB)")
    
    # Parity verification
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
    print(f"PyTorch vs ONNX Parity Error: {parity_err:.8e} (Pass: < 1e-5)")
    assert parity_err < 1e-5, f"Parity error too high: {parity_err}"
    
    # Latency benchmark
    x_np = np.zeros((1, 1, 112), dtype=np.float32)
    h0_np = np.zeros((1, 1, 128), dtype=np.float32)
    h1_np = np.zeros((1, 1, 128), dtype=np.float32)
    for _ in range(200):
        sess.run(None, {'features': x_np, 'hidden_0': h0_np, 'hidden_1': h1_np})
    t0 = time.time()
    N_BENCH = 5000
    for _ in range(N_BENCH):
        sess.run(None, {'features': x_np, 'hidden_0': h0_np, 'hidden_1': h1_np})
    t1 = time.time()
    lat = (t1 - t0) / N_BENCH * 1e6
    est_server_min = (lat * 39.4e6 / 1e6 / 60) * (42.85 / 105.9)
    print(f"ONNX Latency: {lat:.1f} us/step | Est. 39.4M Server Runtime: {est_server_min:.1f} minutes")
    
    # Package into submission.zip
    with zipfile.ZipFile('submission.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write('submission/solution.py', arcname='solution.py')
        z.write(export_path, arcname='model.onnx')
    zip_size_mb = Path('submission.zip').stat().st_size / (1024 * 1024)
    print(f"Packaged submission.zip (size: {zip_size_mb:.2f} MB)")
    
    return fused_net


def main():
    device = get_device()
    print("=" * 80)
    print("FULL PRODUCTION TRAINING ACROSS 500 SEQUENCES (10,000,000 STEPS)")
    print("=" * 80)
    
    # 1. Load Data
    print("Loading 200 sequences from data/...")
    X0 = np.load('data/X.npy')
    O0_0 = np.load('data/O0.npy')
    O1_0 = np.load('data/O1.npy')
    Y0 = np.load('data/Y.npy')
    
    print("Loading 300 sequences from scratch/precomputed_seqs_300_599.npz...")
    d300 = np.load('scratch/precomputed_seqs_300_599.npz')
    X1 = d300['X']
    O0_1 = d300['O0']
    O1_1 = d300['O1']
    Y1 = d300['Y']
    
    X_train = np.concatenate([X0, X1], axis=0)
    O0_train = np.concatenate([O0_0, O0_1], axis=0)
    O1_train = np.concatenate([O1_0, O1_1], axis=0)
    Y_train = np.concatenate([Y0, Y1], axis=0)
    N_TRAIN = len(X_train)
    print(f"Total Combined Training Set: {N_TRAIN} sequences ({N_TRAIN * 20000:,} steps)")
    
    P_true, _ = build_true_lob_matrix()
    print("Precomputing Q_train (36 True LOB features)...")
    Q_train = np.matmul(X_train, P_true)
    print(f"Q_train shape: {Q_train.shape}")
    
    # Clean up intermediate arrays
    del X0, O0_0, O1_0, Y0, X1, O0_1, O1_1, Y1
    gc.collect()
    
    # 2. Load Validation Set
    print("\nLoading Pristine Holdout Benchmark (scratch/pristine_holdout_100.npz)...")
    d_val = np.load('scratch/pristine_holdout_100.npz')
    feats_val = d_val['features_all']
    targets_val = d_val['targets_all']
    N_VAL = len(feats_val)
    
    base_gru = FrozenBaselineGRU('baseline/baseline.onnx').to(device).eval()
    print("Precomputing validation O0 and O1...")
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
    print("Validation precomputations complete.")
    
    # 3. Model & Baseline Check
    model = MultiHorizonResidualModel().to(device)
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    
    def evaluate_holdout(m):
        m.eval()
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
        return res['weighted_pearson'], res['t0'], res['t1'], diag['r_between'], diag['r_within']
        
    init_wp, init_t0, init_t1, init_rb, init_rw = evaluate_holdout(model)
    print(f"\n--- STEP 0 BENCHMARK EVALUATION ---")
    print(f"Step 0 WP: {init_wp:.6f} (t0: {init_t0:.6f}, t1: {init_t1:.6f}) | r_b: {init_rb:.6f} | r_w: {init_rw:.6f}")
    
    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW([
        {'params': model.head_o0.parameters(), 'lr': 1.5e-4},
        {'params': model.head_q.parameters(), 'lr': 2.5e-4},
        {'params': [model.s0, model.s1, model.coupling, model.bias], 'lr': 8.0e-5},
    ], weight_decay=1e-4)
    
    NUM_EPOCHS = 20
    BATCH_SEQS = 25
    total_steps = NUM_EPOCHS * (N_TRAIN // BATCH_SEQS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1.0e-5)
    loss_fn = ExactGlobalWeightedPearsonLoss(wmse_weight=0.015, pen_b_weight=1.5).to(device)
    
    best_wp = init_wp
    print(f"\nStarting training across {NUM_EPOCHS} epochs ({N_TRAIN} sequences / {BATCH_SEQS} per batch)...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        t_ep = time.time()
        model.train()
        perm = np.random.permutation(N_TRAIN)
        total_loss = 0.0
        n_batches = 0
        
        for b_start in range(0, N_TRAIN, BATCH_SEQS):
            b_idx = perm[b_start:b_start + BATCH_SEQS]
            actual_B = len(b_idx)
            if actual_B < BATCH_SEQS:
                continue
                
            o0_b = torch.from_numpy(O0_train[b_idx, WARMUP:]).to(device)
            o1_b = torch.from_numpy(O1_train[b_idx, WARMUP:]).to(device)
            q_b = torch.from_numpy(Q_train[b_idx, WARMUP:]).to(device)
            y_b = torch.from_numpy(Y_train[b_idx, WARMUP:]).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            pred_b = model(o0_b, o1_b, q_b)
            
            loss = loss_fn(pred_b.reshape(-1, 2), y_b.reshape(-1, 2), B=actual_B, L=19901)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        avg_loss = total_loss / max(1, n_batches)
        val_wp, val_t0, val_t1, val_rb, val_rw = evaluate_holdout(model)
        gain_bps = (val_wp - init_wp) * 10000.0
        ep_time = time.time() - t_ep
        
        print(f"[Epoch {epoch:2d}/{NUM_EPOCHS}] Loss: {avg_loss:.5f} | Val WP: {val_wp:.6f} (t0: {val_t0:.6f}, t1: {val_t1:.6f}) | r_b: {val_rb:.6f} | r_w: {val_rw:.6f} | Gain: {gain_bps:+.1f} bps | Time: {ep_time:.1f}s")
        
        if val_wp > best_wp:
            best_wp = val_wp
            print(f"  >>> NEW ALL-TIME BEST WP: {best_wp:.6f} (Gain over Sub 6: {gain_bps:+.1f} bps)! Saving checkpoint...")
            torch.save(model.state_dict(), 'scratch/best_fused_connectome.pt')
            export_fused_onnx(model, P_true, 'baseline/baseline.onnx', 'submission/model.onnx')
            model.to(device)
            
    print("=" * 80)
    print(f"COMPLETE. Best Pristine Holdout WP: {best_wp:.6f} (Initial: {init_wp:.6f}, Gain: {(best_wp - init_wp)*10000:+.1f} bps)")
    print("=" * 80)


if __name__ == "__main__":
    main()
