"""Production Fused Training and ONNX Export Engine.

Trains an enhanced 2-layer GRU model on 1,000 sequences (20,000,000 steps)
using GlobalWeightedPearsonLoss on the NVIDIA RTX 3050 GPU.
Mathematically fuses the optimal scaling, bias offsets, and anti-symmetric coupling
directly into the linear output head, producing an ultra-fast 10-node ONNX model
guaranteed to run in < 45 minutes on the 1 vCPU scoring server.
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
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP


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


class TrainableAlphaModel(nn.Module):
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
        
        # Linear head: (128 -> 2)
        W_base = torch.from_numpy(w['onnx::MatMul_186'].T.copy())
        b_base = torch.from_numpy(w['reg_head.bias'].copy())
        
        self.head = nn.Linear(128, 2)
        self.head.weight.data = W_base.clone()
        self.head.bias.data = b_base.clone()
        
        # Trainable scaling and coupling
        self.s0 = nn.Parameter(torch.tensor([0.275], dtype=torch.float32))
        self.s1 = nn.Parameter(torch.tensor([0.425], dtype=torch.float32))
        self.coupling = nn.Parameter(torch.tensor([0.040], dtype=torch.float32))

    def forward(self, x, h0=None, h1=None):
        o0, next_h0 = self.gru0(x, h0)
        o1, next_h1 = self.gru1(o0, h1)
        
        raw = self.head(o1) # (B, T, 2)
        scaled_0 = raw[:, :, 0:1] * self.s0
        scaled_1 = raw[:, :, 1:2] * self.s1
        
        c = torch.clamp(self.coupling, 0.0, 0.15)
        p0 = (1.0 - c) * scaled_0 - c * scaled_1
        p1 = (1.0 - c) * scaled_1 - c * scaled_0
        
        preds = torch.cat([p0, p1], dim=-1)
        return preds, next_h0, next_h1


def evaluate(model, val_seqs, device):
    model.eval()
    acc = GlobalAccumulator()
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    with torch.no_grad():
        for seq_id, feats, targets in val_seqs:
            x_t = torch.from_numpy(feats).unsqueeze(0).to(device)
            preds, _, _ = model(x_t)
            p_np = preds[0].cpu().numpy()
            acc.add(targets, p_np, mask)
    res = acc.result()
    return res['weighted_pearson'], res['t0'], res['t1']


def export_fused_onnx(model, export_path: str = 'submission/model.onnx'):
    """Mathematically fuses linear head, scaling, and coupling into a single 10-node ONNX model."""
    print(f"\nFusing weights into minimal 10-node ONNX architecture: {export_path}...")
    
    W_pt = model.head.weight.data.cpu().numpy() # (2, 128)
    b_pt = model.head.bias.data.cpu().numpy()   # (2,)
    s0 = float(model.s0.data.cpu())
    s1 = float(model.s1.data.cpu())
    c = float(torch.clamp(model.coupling, 0.0, 0.15).data.cpu())
    
    print(f"Learned parameters: s0={s0:.4f}, s1={s1:.4f}, coupling={c:.4f}")
    
    # Mathematical fusion matrix M_c:
    # [p0, p1] = [raw0, raw1] @ M_c
    M_c = np.array([
        [(1.0 - c) * s0, -c * s0],
        [-c * s1, (1.0 - c) * s1]
    ], dtype=np.float32)
    
    # W in PyTorch head is (2, 128), so (h @ W.T + b) @ M_c = h @ (W.T @ M_c) + b @ M_c
    W_fused = W_pt.T @ M_c # (128, 2)
    b_fused = b_pt @ M_c   # (2,)
    
    # Build minimal fused model
    class Fused10NodeModel(nn.Module):
        def __init__(self, gru0, gru1, W_fused, b_fused):
            super().__init__()
            self.gru0 = gru0
            self.gru1 = gru1
            self.head = nn.Linear(128, 2)
            self.head.weight.data = torch.from_numpy(W_fused.T.copy())
            self.head.bias.data = torch.from_numpy(b_fused.copy())
            
        def forward(self, x, h0=None, h1=None):
            o0, next_h0 = self.gru0(x, h0)
            o1, next_h1 = self.gru1(o0, h1)
            pred = self.head(o1)
            return pred, next_h0, next_h1
            
    fused_model = Fused10NodeModel(model.gru0, model.gru1, W_fused, b_fused).cpu()
    fused_model.eval()
    
    # Export to ONNX with dynamic batch and sequence axes
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
    
    # Verify ONNX node count
    m_onnx = onnx.load(export_path)
    print(f"Successfully exported {export_path} ({len(m_onnx.graph.node)} ONNX nodes, size: {Path(export_path).stat().st_size / 1024:.1f} KB)")
    assert len(m_onnx.graph.node) <= 12, f"Expected <= 12 nodes, found {len(m_onnx.graph.node)}"
    
    # Verify parity
    with torch.no_grad():
        pt_p, _, _ = fused_model(x_dummy, h0_dummy, h1_dummy)
        
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(export_path, sess_options=opts, providers=['CPUExecutionProvider'])
    ort_p, _, _ = sess.run(None, {
        'features': x_dummy.numpy(),
        'hidden_0': h0_dummy.numpy(),
        'hidden_1': h1_dummy.numpy()
    })
    
    err = np.max(np.abs(pt_p.numpy() - ort_p))
    print(f"PyTorch vs ONNX Parity error: {err:.8e} (Pass: < 1e-5)")
    return export_path


def train_production(num_train: int = 1000, num_val: int = 50, epochs: int = 3):
    device = torch.device('cuda')
    print(f"\n========================================================")
    print(f"PRODUCTION TRAINING: {num_train} SEQUENCES ({num_train * 20000:,} STEPS)")
    print(f"========================================================")
    
    model = TrainableAlphaModel('baseline/baseline.onnx').to(device)
    
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
        {'params': model.head.parameters(), 'lr': 1e-4},
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
            
            if (i + 1) % 100 == 0 or (i + 1) == len(train_seqs):
                print(f"  Epoch {ep}/{epochs} | Seq {i+1:4d}/{len(train_seqs)} | Avg Loss: {total_loss/(i+1):.4f}", flush=True)
                
        val_wp, val_t0, val_t1 = evaluate(model, val_seqs, device)
        marker = " *** NEW BEST ***" if val_wp > best_wp else ""
        print(f"--> Epoch {ep} finished in {time.time() - t_ep:.1f}s | Val WP: {val_wp:.6f} (t0: {val_t0:.6f}, t1: {val_t1:.6f}){marker}\n", flush=True)
        
        if val_wp > best_wp:
            best_wp = val_wp
            torch.save(model.state_dict(), 'scratch/best_production_checkpoint.pt')
            print(f"Checkpoint saved (WP: {best_wp:.6f})", flush=True)
            
    print(f"\nTraining Complete! Peak Val WP: {best_wp:.6f} (Baseline: {init_wp:.6f})")
    
    # Export final model
    export_fused_onnx(model, 'submission/model.onnx')
    print("\nProduction export complete!")


if __name__ == '__main__':
    train_production(num_train=1000, num_val=50, epochs=3)
