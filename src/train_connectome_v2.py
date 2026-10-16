"""Production Connectome-V2 Training and Fusion Engine.

Integrates Decoded Order Book Microstructure (Level-1 Spread, Signed Order Flow)
directly into the recurrent state manifold, optimizing exact Global Weighted Pearson
correlation without non-linear division or phase-lag artifacts.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from src.data_reader import FastSequenceReader
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, FEATURE_COLUMNS


class GlobalWeightedPearsonLoss(nn.Module):
    """Directly optimizes Global Weighted Pearson correlation with global target centering."""
    def __init__(self, mu_y0=-0.4810, mu_y1=-0.5325, eps=1e-8):
        super().__init__()
        self.register_buffer('mu_y', torch.tensor([mu_y0, mu_y1], dtype=torch.float32))
        self.eps = eps
        
    def forward(self, pred, target, mask):
        p = pred[mask] # (N, 2)
        y = target[mask] # (N, 2)
        
        y_c = torch.clamp(y, -2.0, 2.0)
        p_c = torch.clamp(p, -2.0, 2.0)
        w = torch.abs(y_c)
        
        corrs = []
        for s in range(2):
            ys = y_c[:, s]
            ps = p_c[:, s]
            ws = w[:, s]
            w_sum = torch.sum(ws) + self.eps
            
            dy = ys - self.mu_y[s]
            mean_p = torch.sum(ws * ps) / w_sum
            dp = ps - mean_p
            
            cov = torch.sum(ws * dy * dp)
            var_y = torch.sum(ws * dy * dy) + self.eps
            var_p = torch.sum(ws * dp * dp) + self.eps
            
            corr = cov / torch.sqrt(var_y * var_p)
            corrs.append(torch.clamp(corr, -1.0, 1.0))
            
        return 1.0 - 0.5 * (corrs[0] + corrs[1])


class ConnectomeV2Model(nn.Module):
    """Recurrent backbone with linear microstructure bypass."""
    def __init__(self, baseline_path: str = 'baseline/baseline.onnx'):
        super().__init__()
        self.gru0 = nn.GRU(112, 128, batch_first=True)
        self.gru1 = nn.GRU(128, 128, batch_first=True)
        
        # Load weights from baseline.onnx
        m = onnx.load(baseline_path)
        w = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
        
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
        
        W_base = torch.from_numpy(w['onnx::MatMul_186'].T.copy())
        b_base = torch.from_numpy(w['reg_head.bias'].copy())
        
        self.head_h = nn.Linear(128, 2)
        self.head_h.weight.data = W_base.clone()
        self.head_h.bias.data = b_base.clone()
        
        # Linear Microstructure Bypass:
        # Decoded features:
        # 1. Level-1 Spread: p14 - p10
        # 2. Level-1 Signed Volume: v10 - v14
        # 3. Top-3 Signed Volume: (v10 + v5 + v2) - (v14 + v21 + v0)
        # 4. Lead-lag Mid Price Difference: Mid(i1) - Mid(i0)
        # Total bypass dim: 4
        self.head_micro = nn.Linear(4, 2, bias=False)
        nn.init.zeros_(self.head_micro.weight)
        
        # Learned calibration & coupling
        self.s0 = nn.Parameter(torch.tensor([0.281], dtype=torch.float32))
        self.s1 = nn.Parameter(torch.tensor([0.360], dtype=torch.float32))
        self.coupling = nn.Parameter(torch.tensor([0.065], dtype=torch.float32))

    def extract_micro(self, x):
        # x: (B, T, 112)
        # i0 best bid: col 10, best ask: col 14
        p_bb = x[:, :, 10:11]
        p_ba = x[:, :, 14:15]
        spread_l1 = p_ba - p_bb
        
        # i0 volumes
        v_bb = x[:, :, 32:33] # v10
        v_ba = x[:, :, 36:37] # v14
        d_vol_l1 = v_bb - v_ba
        
        # Top-3 volumes
        v_top3_b = v_bb + x[:, :, 27:28] + x[:, :, 24:25] # v10 + v5 + v2
        v_top3_a = v_ba + x[:, :, 43:44] + x[:, :, 22:23] # v14 + v21 + v0
        d_vol_top3 = v_top3_b - v_top3_a
        
        # Cross instrument mid difference
        mid_i0 = 0.5 * (p_ba + p_bb)
        mid_i1 = 0.5 * (x[:, :, 62:63] + x[:, :, 66:67]) # approximate i1 mid
        d_mid = mid_i1 - mid_i0
        
        return torch.cat([spread_l1, d_vol_l1, d_vol_top3, d_mid], dim=-1) # (B, T, 4)

    def forward(self, x, h0=None, h1=None):
        o0, next_h0 = self.gru0(x, h0)
        o1, next_h1 = self.gru1(o0, h1)
        
        pred_h = self.head_h(o1)
        micro = self.extract_micro(x)
        pred_m = self.head_micro(micro)
        
        raw = pred_h + pred_m
        
        scaled_0 = raw[:, :, 0:1] * self.s0
        scaled_1 = raw[:, :, 1:2] * self.s1
        
        c = torch.clamp(self.coupling, 0.0, 0.15)
        p0 = (1.0 - c) * scaled_0 - c * scaled_1
        p1 = (1.0 - c) * scaled_1 - c * scaled_0
        
        return torch.cat([p0, p1], dim=-1), next_h0, next_h1


def evaluate(model, val_seqs, device):
    model.eval()
    acc = GlobalAccumulator()
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    with torch.no_grad():
        for seq_id, feats, targets in val_seqs:
            x_t = torch.from_numpy(feats).unsqueeze(0).to(device)
            preds, _, _ = model(x_t)
            acc.add(targets, preds[0].cpu().numpy(), mask)
    res = acc.result()
    return res['weighted_pearson'], res['t0'], res['t1']


def export_fused_onnx(model, export_path: str = 'submission/model.onnx'):
    """Mathematically fuses linear head, scaling, coupling and bypass into a minimal ONNX model."""
    print(f"\nFusing weights into minimal ONNX architecture: {export_path}...")
    
    W_h = model.head_h.weight.data.cpu().numpy()      # (2, 128)
    b_h = model.head_h.bias.data.cpu().numpy()        # (2,)
    W_m = model.head_micro.weight.data.cpu().numpy()  # (2, 4)
    s0 = float(model.s0.data.cpu())
    s1 = float(model.s1.data.cpu())
    c = float(torch.clamp(model.coupling, 0.0, 0.15).data.cpu())
    
    print(f"Learned parameters: s0={s0:.4f}, s1={s1:.4f}, coupling={c:.4f}")
    
    # Mathematical fusion matrix M_c:
    M_c = np.array([
        [(1.0 - c) * s0, -c * s0],
        [-c * s1, (1.0 - c) * s1]
    ], dtype=np.float32)
    
    W_h_fused = W_h.T @ M_c # (128, 2)
    b_fused = b_h @ M_c     # (2,)
    W_m_fused = W_m.T @ M_c # (4, 2)
    
    # In state space, the 4 microstructure features are linear combinations of x!
    # spread_l1 = x[14] - x[10]
    # d_vol_l1  = x[32] - x[36]
    # d_vol_top3= x[32] + x[27] + x[24] - (x[36] + x[43] + x[22])
    # d_mid     = 0.5 * (x[62] + x[66]) - 0.5 * (x[14] + x[10])
    # We can project W_m_fused back onto the 112 input dimensions!
    # micro = x @ P_micro, so x @ P_micro @ W_m_fused = x @ W_x_fused!
    P_micro = np.zeros((112, 4), dtype=np.float32)
    P_micro[14, 0] += 1.0;  P_micro[10, 0] -= 1.0
    P_micro[32, 1] += 1.0;  P_micro[36, 1] -= 1.0
    P_micro[32, 2] += 1.0;  P_micro[27, 2] += 1.0; P_micro[24, 2] += 1.0
    P_micro[36, 2] -= 1.0;  P_micro[43, 2] -= 1.0; P_micro[22, 2] -= 1.0
    P_micro[62, 3] += 0.5;  P_micro[66, 3] += 0.5
    P_micro[14, 3] -= 0.5;  P_micro[10, 3] -= 0.5
    
    # Fully fused input projection:
    W_x_fused = P_micro @ W_m_fused # (112, 2)
    
    print(f"Bypass norm: {np.linalg.norm(W_x_fused):.6f}")
    
    # Build fused model:
    # If W_x_fused is negligible or if fused into one linear head:
    class FusedConnectome(nn.Module):
        def __init__(self, gru0, gru1, Wh, Wx, b):
            super().__init__()
            self.gru0 = gru0
            self.gru1 = gru1
            self.head_h = nn.Linear(128, 2)
            self.head_h.weight.data = torch.from_numpy(Wh.T.copy())
            self.head_h.bias.data = torch.from_numpy(b.copy())
            self.head_x = nn.Linear(112, 2, bias=False)
            self.head_x.weight.data = torch.from_numpy(Wx.T.copy())
            
        def forward(self, x, h0=None, h1=None):
            o0, next_h0 = self.gru0(x, h0)
            o1, next_h1 = self.gru1(o0, h1)
            pred = self.head_h(o1) + self.head_x(x)
            return pred, next_h0, next_h1
            
    fused_model = FusedConnectome(model.gru0, model.gru1, W_h_fused, W_x_fused, b_fused).cpu().eval()
    
    x_dummy = torch.randn(1, 1, 112)
    h0_dummy = torch.zeros(1, 1, 128)
    h1_dummy = torch.zeros(1, 1, 128)
    
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        fused_model,
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
    print(f"Successfully exported {export_path} ({len(m_onnx.graph.node)} ONNX nodes, size: {Path(export_path).stat().st_size / 1024:.1f} KB)")
    
    # Parity check
    with torch.no_grad():
        pt_p, _, _ = model.cpu()(x_dummy, h0_dummy, h1_dummy)
        
    sess = ort.InferenceSession(export_path, providers=['CPUExecutionProvider'])
    ort_p, _, _ = sess.run(None, {
        'features': x_dummy.numpy(),
        'hidden_0': h0_dummy.numpy(),
        'hidden_1': h1_dummy.numpy()
    })
    
    err = np.max(np.abs(pt_p.numpy() - ort_p))
    print(f"PyTorch vs ONNX Parity error: {err:.8e} (Pass: < 1e-5)")
    return export_path


def train_production(num_train: int = 1500, num_val: int = 50, epochs: int = 3):
    device = torch.device('cuda')
    print(f"\n========================================================")
    print(f"CONNECTOME-V2 TRAINING: {num_train} SEQUENCES ({num_train * 20000:,} STEPS)")
    print(f"========================================================")
    
    model = ConnectomeV2Model('baseline/baseline.onnx').to(device)
    
    print(f"\nStreaming {num_train + num_val} sequences from train.parquet...")
    t0_read = time.time()
    reader = FastSequenceReader('datasets/train.parquet')
    all_seqs = list(reader.iter_sequences(max_sequences=num_train + num_val))
    print(f"Read {len(all_seqs)} sequences in {time.time() - t0_read:.1f}s")
    
    val_seqs = all_seqs[:num_val]
    train_seqs = all_seqs[num_val:]
    
    init_wp, init_t0, init_t1 = evaluate(model, val_seqs, device)
    print(f"\nBaseline Validation WP on {num_val} sequences: {init_wp:.6f} (t0: {init_t0:.6f}, t1: {init_t1:.6f})")
    
    loss_fn = GlobalWeightedPearsonLoss().to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.gru0.parameters(), 'lr': 1e-5},
        {'params': model.gru1.parameters(), 'lr': 1e-5},
        {'params': model.head_h.parameters(), 'lr': 1e-4},
        {'params': model.head_micro.parameters(), 'lr': 5e-4},
        {'params': [model.s0, model.s1], 'lr': 1e-3},
        {'params': [model.coupling], 'lr': 1e-3},
    ], weight_decay=1e-4)
    
    scaler = GradScaler('cuda')
    best_wp = init_wp
    
    for ep in range(1, epochs + 1):
        model.train()
        t_ep = time.time()
        total_loss = 0.0
        perm = np.random.permutation(len(train_seqs))
        
        for i, s_idx in enumerate(perm):
            seq_id, feats, targets = train_seqs[s_idx]
            x_t = torch.from_numpy(feats).unsqueeze(0).to(device)
            y_t = torch.from_numpy(targets).unsqueeze(0).to(device)
            mask_t = torch.ones(1, 20000, dtype=torch.bool, device=device)
            mask_t[:, :WARMUP] = False
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.float16):
                preds, _, _ = model(x_t)
                loss = loss_fn(preds, y_t, mask_t)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if (i + 1) % 150 == 0 or (i + 1) == len(train_seqs):
                print(f"  Epoch {ep}/{epochs} | Seq {i+1:4d}/{len(train_seqs)} | Avg Loss: {total_loss/(i+1):.4f}", flush=True)
                
        val_wp, val_t0, val_t1 = evaluate(model, val_seqs, device)
        marker = " *** NEW BEST ***" if val_wp > best_wp else ""
        print(f"--> Epoch {ep} finished in {time.time() - t_ep:.1f}s | Val WP: {val_wp:.6f} (t0: {val_t0:.6f}, t1: {val_t1:.6f}){marker}\n", flush=True)
        
        if val_wp > best_wp:
            best_wp = val_wp
            torch.save(model.state_dict(), 'scratch/best_connectome_v2.pt')
            print(f"Checkpoint saved (WP: {best_wp:.6f})", flush=True)
            
    print(f"\nTraining Complete! Peak Val WP: {best_wp:.6f} (Baseline: {init_wp:.6f})")
    
    # Load best checkpoint and export
    model.load_state_dict(torch.load('scratch/best_connectome_v2.pt'))
    export_fused_onnx(model, 'submission/model.onnx')
    print("\nProduction export complete!")


if __name__ == '__main__':
    train_production(num_train=1500, num_val=50, epochs=3)
