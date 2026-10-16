"""End-to-End True Depth Connectome Training Engine.

Trains the recurrent manifold end-to-end:
1. Feeds the 36-feature True LOB microstructure vector directly into GRU Layer 0 recurrent gates.
2. Unfreezes the recurrent backbone with tuned learning rates to avoid catastrophic forgetting while unlocking deep alpha.
3. Multi-horizon pooling combines fast tick flow (Layer 0) and slow macro trend (Layer 1).
4. Whole-sequence pooled batches (16 full sequences = 318,416 steps per batch) with sequence-level drift penalty.
5. Strict evaluation on the official 100-sequence Pristine Holdout benchmark (2,000,000 steps).
6. Automatic ONNX export, parity verification, latency benchmark, and submission packaging on new best score.
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
import onnxruntime as ort

from src.models.e2e_true_depth_connectome import EndToEndTrueDepthConnectome
from src.data_reader import FastSequenceReader
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, METRIC_CLIP
from src.validation import evaluate_moments_decomposition


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class ExactGlobalWeightedPearsonLoss(nn.Module):
    def __init__(self, wmse_weight=0.02, pen_b_weight=2.0, eps=1e-8):
        super().__init__()
        self.wmse_weight = wmse_weight
        self.pen_b_weight = pen_b_weight
        self.eps = eps

    def forward(self, pred, target, B=16, L=19901):
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
        
        # Sequence-level mean drift penalty to preserve r_between
        y_seq = y_c.view(B, L, 2)
        p_seq = p_c.view(B, L, 2)
        w_seq = w.view(B, L, 2)
        
        w_s_sum = torch.sum(w_seq, dim=1) + self.eps
        my_s = torch.sum(w_seq * y_seq, dim=1) / w_s_sum
        mp_s = torch.sum(w_seq * p_seq, dim=1) / w_s_sum
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


def export_clean_onnx(model, export_path='submission/model.onnx'):
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
    print(f"\nExported {export_path} ({len(m_onnx.graph.node)} ONNX nodes, size: {file_size_kb:.1f} KB)")
    
    with torch.no_grad():
        pt_pred, _, _ = model(x_dummy, h0_dummy, h1_dummy)
        
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
    
    # Quick latency benchmark
    x_np = np.zeros((1, 1, 112), dtype=np.float32)
    h0_np = np.zeros((1, 1, 128), dtype=np.float32)
    h1_np = np.zeros((1, 1, 128), dtype=np.float32)
    for _ in range(200):
        sess.run(None, {'features': x_np, 'hidden_0': h0_np, 'hidden_1': h1_np})
    t0 = time.time()
    N_BENCH = 3000
    for _ in range(N_BENCH):
        sess.run(None, {'features': x_np, 'hidden_0': h0_np, 'hidden_1': h1_np})
    t1 = time.time()
    lat = (t1 - t0) / N_BENCH * 1e6
    est_server_min = (lat * 39.4e6 / 1e6 / 60) * (42.85 / 105.9)
    print(f"ONNX Latency: {lat:.1f} us/step | Est. 39.4M Server Runtime: {est_server_min:.1f} minutes")
    
    return export_path


def package_submission(zip_path='submission.zip'):
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write('submission/solution.py', arcname='solution.py')
        z.write('submission/model.onnx', arcname='model.onnx')
    size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
    print(f"Packaged {zip_path} (size: {size_mb:.2f} MB)")


def train_e2e_connectome(
    num_train_seqs: int = 760,
    start_seq_idx: int = 300,
    buffer_size: int = 190,
    batch_size: int = 16,
    chunk_len: int = 128,
    num_epochs: int = 10,
    checkpoint_out: str = 'scratch/best_e2e_true_depth_connectome.pt'
):
    device = get_device()
    print("=" * 80)
    print("END-TO-END TRUE DEPTH CONNECTOME TRAINING")
    print(f"Total Sequences: {num_train_seqs} (Indices {start_seq_idx}..{start_seq_idx + num_train_seqs - 1})")
    print(f"Streaming Buffer: {buffer_size} sequences | Batch size: {batch_size} | Epochs: {num_epochs}")
    print("=" * 80)
    
    model = EndToEndTrueDepthConnectome().to(device)
    init_wp, init_t0, init_t1, init_rb, init_rw = evaluate_on_pristine_holdout(model, device=device)
    print(f"\n--- INITIAL BENCHMARK EVALUATION (Step 0 Parity) ---")
    print(f"Initial Holdout WP: {init_wp:.6f} (t0: {init_t0:.6f}, t1: {init_t1:.6f}) | r_b: {init_rb:.6f} | r_w: {init_rw:.6f}")
    assert abs(init_wp - 0.426912) < 5e-4, f"Parity mismatch: {init_wp} vs 0.426912"
    print(f"CONFIRMED SUB 6 PARITY (WP: {init_wp:.6f} >= Sub 6: 0.426912, r_b: {init_rb:.6f})!\n")
    
    # Differential learning rates:
    # gru0: 5.0e-5 (learns new 36 LOB signals while adapting layer 0)
    # gru1: 3.5e-5 (smooth adaptation of layer 1)
    # rec_head: 8.0e-5 (learns fast/slow multi-horizon combination)
    # bypass, mode parameters: 5.0e-5
    optimizer = torch.optim.AdamW([
        {'params': model.gru0.parameters(), 'lr': 5.0e-5},
        {'params': model.gru1.parameters(), 'lr': 3.5e-5},
        {'params': model.rec_head.parameters(), 'lr': 8.0e-5},
        {'params': model.bypass_head.parameters(), 'lr': 5.0e-5},
        {'params': [model.s0, model.s1, model.coupling, model.bias], 'lr': 5.0e-5},
    ], weight_decay=1e-4)
    
    total_steps = num_epochs * (num_train_seqs // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1.0e-6)
    loss_fn = ExactGlobalWeightedPearsonLoss(wmse_weight=0.02, pen_b_weight=2.0).to(device)
    
    best_wp = init_wp
    
    for epoch in range(1, num_epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Read data in streaming buffers
        reader = FastSequenceReader('datasets/train.parquet')
        gen = reader.iter_sequences()
        
        for _ in range(start_seq_idx):
            next(gen)
            
        seqs_read = 0
        buf_idx = 0
        while seqs_read < num_train_seqs:
            buf_idx += 1
            curr_buf_len = min(buffer_size, num_train_seqs - seqs_read)
            
            buf_x = []
            buf_y = []
            for _ in range(curr_buf_len):
                _, f, tgt = next(gen)
                buf_x.append(f)
                buf_y.append(tgt)
                
            seqs_read += curr_buf_len
            buf_x = np.array(buf_x, dtype=np.float32)
            buf_y = np.array(buf_y, dtype=np.float32)
            
            perm = np.random.permutation(curr_buf_len)
            
            for b_start in range(0, curr_buf_len, batch_size):
                b_indices = perm[b_start:b_start + batch_size]
                actual_B = len(b_indices)
                if actual_B < batch_size:
                    continue
                    
                batch_x = torch.from_numpy(buf_x[b_indices]).to(device)
                batch_y = torch.from_numpy(buf_y[b_indices]).to(device)
                
                h0 = torch.zeros(1, actual_B, 128, device=device)
                h1 = torch.zeros(1, actual_B, 128, device=device)
                
                preds_list = []
                for t_start in range(0, SEQUENCE_LENGTH, chunk_len):
                    t_end = min(SEQUENCE_LENGTH, t_start + chunk_len)
                    xc = batch_x[:, t_start:t_end]
                    p_chunk, h0, h1 = model(xc, h0.detach(), h1.detach())
                    preds_list.append(p_chunk)
                    
                all_p = torch.cat(preds_list, dim=1)[:, WARMUP:].reshape(-1, 2)
                all_y = batch_y[:, WARMUP:].reshape(-1, 2)
                
                loss = loss_fn(all_p, all_y, B=actual_B, L=19901)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            del buf_x, buf_y
            gc.collect()
            
        avg_loss = total_loss / max(1, num_batches)
        ep_time = time.time() - t_ep
        
        # Evaluate on pristine holdout
        val_wp, val_t0, val_t1, val_rb, val_rw = evaluate_on_pristine_holdout(model, device=device)
        gain_bps = (val_wp - init_wp) * 10000.0
        
        print(f"[Epoch {epoch:2d}/{num_epochs}] Loss: {avg_loss:.5f} | Val WP: {val_wp:.6f} (t0: {val_t0:.6f}, t1: {val_t1:.6f}) | r_b: {val_rb:.6f} | r_w: {val_rw:.6f} | Gain: {gain_bps:+.1f} bps | Time: {ep_time:.1f}s")
        
        if val_wp > best_wp:
            best_wp = val_wp
            print(f"  >>> NEW ALL-TIME BEST WP: {best_wp:.6f} (Gain over Sub 6: {gain_bps:+.1f} bps)! Saving checkpoint...")
            torch.save(model.state_dict(), checkpoint_out)
            export_clean_onnx(model, 'submission/model.onnx')
            package_submission('submission.zip')
            model.to(device)
            
    print("=" * 80)
    print(f"TRAINING COMPLETE. Best Pristine Holdout WP: {best_wp:.6f} (Initial: {init_wp:.6f}, Gain: {(best_wp - init_wp)*10000:+.1f} bps)")
    print("=" * 80)


if __name__ == "__main__":
    train_e2e_connectome()
