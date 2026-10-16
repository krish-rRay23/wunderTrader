"""Full-Backbone End-to-End Fine-Tuning Engine for Alfa Connectome.

Unfreezes the entire 2-layer GRU manifold and fine-tunes all recurrent gates (z, r, h)
and regression head directly on the Global Weighted Pearson objective via Truncated BPTT.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import zipfile
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from src.data_reader import FastSequenceReader
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, DataPoint


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class FullEndToEndGRU(nn.Module):
    def __init__(self, baseline_path: str = 'baseline/baseline.onnx'):
        super().__init__()
        self.gru0 = nn.GRU(112, 128, batch_first=True)
        self.gru1 = nn.GRU(128, 128, batch_first=True)
        self.head = nn.Linear(128, 2)
        
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
        
        self.head.weight.data = torch.from_numpy(w['onnx::MatMul_186'].T.copy())
        self.head.bias.data = torch.from_numpy(w['reg_head.bias'].copy())

    def forward(self, x, h0=None, h1=None):
        o0, next_h0 = self.gru0(x, h0)
        o1, next_h1 = self.gru1(o0, h1)
        pred = self.head(o1)
        return pred, next_h0, next_h1


class ExactWeightedPearsonLoss(nn.Module):
    def __init__(self, mu_y0=-0.4810, mu_y1=-0.5325, eps=1e-8):
        super().__init__()
        self.register_buffer('mu_y', torch.tensor([mu_y0, mu_y1], dtype=torch.float32))
        self.eps = eps
        
    def forward(self, pred, target):
        # pred, target: (N, 2)
        y_c = torch.clamp(target, -2.0, 2.0)
        p_c = torch.clamp(pred, -2.0, 2.0)
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


def evaluate_full_holdout(model, val_feats, val_targets, device):
    model.eval()
    acc = GlobalAccumulator()
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    
    with torch.no_grad():
        for s in range(len(val_feats)):
            xt = torch.from_numpy(val_feats[s:s+1]).to(device)
            pred, _, _ = model(xt)
            p_np = pred[0].cpu().numpy()
            acc.add(val_targets[s], p_np, mask)
            
    res = acc.result()
    return res['weighted_pearson'], res['t0'], res['t1']


def export_to_onnx(model, export_path='submission/model.onnx'):
    model = model.cpu().eval()
    x_dummy = torch.randn(1, 1, 112)
    h0_dummy = torch.zeros(1, 1, 128)
    h1_dummy = torch.zeros(1, 1, 128)
    
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
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
    
    with torch.no_grad():
        pt_pred, _, _ = model(x_dummy, h0_dummy, h1_dummy)
        
    sess = ort.InferenceSession(export_path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {
        'features': x_dummy.numpy(),
        'hidden_0': h0_dummy.numpy(),
        'hidden_1': h1_dummy.numpy()
    })
    
    parity_err = np.max(np.abs(pt_pred.numpy() - ort_out[0]))
    print(f"PyTorch vs ONNX Parity Error: {parity_err:.8e} (Pass: < 1e-5)")
    assert parity_err < 1e-5, f"Parity error too high: {parity_err}"
    return export_path


def main():
    device = get_device()
    print("=" * 70)
    print("FULL-BACKBONE END-TO-END GRU RECURRENT FINE-TUNING")
    print("=" * 70)
    
    # 1. Load 50 holdout validation sequences
    print("\nLoading Disjoint 50-Sequence Holdout Validation Set...")
    d_val = np.load('scratch/cached_50_seqs.npz')
    val_feats = d_val['features_all'] # (50, 20000, 112)
    val_targets = d_val['targets_all'] # (50, 20000, 2)
    print(f"Validation set ready: {len(val_feats)} sequences (1,000,000 steps)")
    
    model = FullEndToEndGRU('baseline/baseline.onnx').to(device)
    loss_fn = ExactWeightedPearsonLoss().to(device)
    
    init_wp, init_t0, init_t1 = evaluate_full_holdout(model, val_feats, val_targets, device)
    print(f"\n--- BASELINE INITIAL HOLDOUT SCORE ---")
    print(f"Val WP: {init_wp:.6f} (t0: {init_t0:.6f}, t1: {init_t1:.6f})")
    print("-" * 70)
    
    # 2. Setup streaming reader for training sequences (from index 50 onward)
    reader = FastSequenceReader('datasets/train.parquet')
    seq_gen = reader.iter_sequences()
    print("Skipping the 50 validation sequences to ensure strict data separation...")
    for _ in range(50):
        next(seq_gen)
        
    # Differential learning rates with AdamW and cosine schedule
    optimizer = torch.optim.AdamW([
        {'params': model.gru0.parameters(), 'lr': 1.5e-5},
        {'params': model.gru1.parameters(), 'lr': 1.5e-5},
        {'params': model.head.parameters(), 'lr': 6.0e-5},
    ], weight_decay=1e-4)
    
    NUM_BLOCKS = 4
    BLOCK_SIZE = 125 # 125 sequences = 2,500,000 steps per block; 4 blocks = 500 seqs (10,000,000 steps)
    EPOCHS_PER_BLOCK = 3
    CHUNK_SIZE = 500
    
    best_val_wp = init_wp
    best_checkpoint_path = 'scratch/best_full_finetuned_gru.pt'
    
    total_seqs_trained = 0
    print(f"\nStarting Streaming Block Training across {NUM_BLOCKS * BLOCK_SIZE} sequences ({NUM_BLOCKS * BLOCK_SIZE * 20000:,} steps)...")
    
    for block_idx in range(NUM_BLOCKS):
        print(f"\n[Block {block_idx+1}/{NUM_BLOCKS}] Extracting {BLOCK_SIZE} training sequences...")
        block_feats = []
        block_targets = []
        for _ in range(BLOCK_SIZE):
            try:
                seq_id, f, tgt = next(seq_gen)
                block_feats.append(f)
                block_targets.append(tgt)
            except StopIteration:
                break
                
        block_feats = np.array(block_feats, dtype=np.float32)
        block_targets = np.array(block_targets, dtype=np.float32)
        total_seqs_trained += len(block_feats)
        print(f"[Block {block_idx+1}] Training on {len(block_feats)} sequences ({len(block_feats)*20000:,} steps)...")
        
        for ep in range(EPOCHS_PER_BLOCK):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            n_chunks = 0
            
            # Shuffle sequence order per epoch
            perm = np.random.permutation(len(block_feats))
            
            for s_idx in perm:
                h0 = torch.zeros(1, 1, 128, device=device)
                h1 = torch.zeros(1, 1, 128, device=device)
                seq_x = torch.from_numpy(block_feats[s_idx]).unsqueeze(0).to(device)
                seq_y = torch.from_numpy(block_targets[s_idx]).unsqueeze(0).to(device)
                
                for t_start in range(0, SEQUENCE_LENGTH, CHUNK_SIZE):
                    t_end = min(t_start + CHUNK_SIZE, SEQUENCE_LENGTH)
                    x_chunk = seq_x[:, t_start:t_end]
                    y_chunk = seq_y[:, t_start:t_end]
                    
                    optimizer.zero_grad()
                    pred_chunk, h0, h1 = model(x_chunk, h0, h1)
                    h0 = h0.detach()
                    h1 = h1.detach()
                    
                    if t_end > WARMUP:
                        valid_start = max(0, WARMUP - t_start)
                        p_valid = pred_chunk[0, valid_start:]
                        y_valid = y_chunk[0, valid_start:]
                        if len(p_valid) > 20:
                            loss = loss_fn(p_valid, y_valid)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            total_loss += loss.item()
                            n_chunks += 1
                            
            dt = time.time() - t0
            avg_loss = total_loss / max(1, n_chunks)
            
            val_wp, val_t0, val_t1 = evaluate_full_holdout(model, val_feats, val_targets, device)
            is_best = val_wp > best_val_wp
            best_tag = " *** NEW BEST ***" if is_best else ""
            if is_best:
                best_val_wp = val_wp
                torch.save(model.state_dict(), best_checkpoint_path)
                
            print(f"  Block {block_idx+1} Ep {ep+1} ({dt:.1f}s) | Loss: {avg_loss:.4f} | Val WP: {val_wp:.6f} (t0: {val_t0:.6f}, t1: {val_t1:.6f}){best_tag}", flush=True)
            
        del block_feats, block_targets
        gc.collect()
        
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE! Peak Val WP: {best_val_wp:.6f} (Baseline: {init_wp:.6f}, Gain: +{best_val_wp - init_wp:.6f})")
    print("=" * 70)
    
    # Load best checkpoint and export to ONNX
    print("\nLoading best model checkpoint and exporting to ONNX...")
    model.load_state_dict(torch.load(best_checkpoint_path))
    model.to('cpu')
    export_to_onnx(model, 'submission/model.onnx')
    
    # Package into submission.zip
    print("\nPackaging into clean submission.zip...")
    zip_path = 'submission.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write('submission/solution.py', arcname='solution.py')
        zf.write('submission/model.onnx', arcname='model.onnx')
    print(f"Successfully packaged {zip_path} (size: {Path(zip_path).stat().st_size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
