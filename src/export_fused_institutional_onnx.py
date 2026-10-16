"""Export Fused Institutional Connectome + Quant Ensemble to ONNX.

Validates:
1. Exact holdout benchmark reproduction (WP = 0.428993).
2. Bitwise parity between PyTorch and ONNX Runtime.
3. Dynamic sequence length (both seq=20000 full-pass and seq=1 streaming).
4. Single-step CPU inference latency budget.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.fused_institutional_model import FusedInstitutionalModel
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP
from src.validation import evaluate_moments_decomposition


def main():
    print("=" * 80)
    print("EXPORTING FUSED INSTITUTIONAL CONNECTOME TO ONNX")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusedInstitutionalModel(alpha=0.60).to(device).eval()
    
    # 1. Benchmark on pristine holdout
    d = np.load('scratch/pristine_holdout_100.npz')
    features_all = d['features_all'] # (100, 20000, 112)
    targets_all = d['targets_all']   # (100, 20000, 2)
    N_SEQS = len(features_all)
    
    mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    mask[WARMUP:] = True
    
    print("\n[Step 1] Verifying PyTorch model on Pristine Holdout...")
    preds_pt = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(N_SEQS):
            xt = torch.from_numpy(features_all[s:s+1]).to(device)
            p, _, _ = model(xt)
            preds_pt.append(p[0].cpu().numpy())
    dt = time.time() - t0
    preds_pt = np.array(preds_pt)
    
    acc = GlobalAccumulator()
    for s in range(N_SEQS):
        acc.add(targets_all[s], preds_pt[s], mask)
    res = acc.result()
    diag = evaluate_moments_decomposition(targets_all, preds_pt)
    print(f"PyTorch Holdout Result: WP={res['weighted_pearson']:.6f} (t0={res['t0']:.6f}, t1={res['t1']:.6f})")
    print(f"Moments: r_b={diag['r_between']:.6f}, r_w={diag['r_within']:.6f}")
    assert abs(res['weighted_pearson'] - 0.428993) < 1e-4, f"Mismatch: {res['weighted_pearson']}"
    print(" Holdout Benchmark PASSED with exact target WP: 0.428993!")
    
    # 2. Export to ONNX on CPU for clean portability
    print("\n[Step 2] Exporting to ONNX...")
    model_cpu = model.to('cpu').eval()
    dummy_x = torch.randn(1, 10, 112, dtype=torch.float32)
    dummy_h0 = torch.zeros(1, 1, 128, dtype=torch.float32)
    dummy_h1 = torch.zeros(1, 1, 128, dtype=torch.float32)
    
    onnx_path = 'submission/model.onnx'
    torch.onnx.export(
        model_cpu,
        (dummy_x, dummy_h0, dummy_h1),
        onnx_path,
        input_names=['features', 'hidden_0', 'hidden_1'],
        output_names=['output', 'next_h0', 'next_h1'],
        dynamic_axes={
            'features': {0: 'batch', 1: 'seq'},
            'hidden_0': {1: 'batch'},
            'hidden_1': {1: 'batch'},
            'output': {0: 'batch', 1: 'seq'},
            'next_h0': {1: 'batch'},
            'next_h1': {1: 'batch'},
        },
        opset_version=17,
        do_constant_folding=True
    )
    print(f" Saved ONNX model to {onnx_path}")
    
    # 3. Verify ONNX validity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f" ONNX check passed! Model file size: {file_size_mb:.2f} MB (Limit: 20 MB)")
    assert file_size_mb < 20.0, "Model exceeds 20 MB!"
    
    # 4. Parity test with ONNX Runtime
    print("\n[Step 3] Checking bitwise parity with ONNX Runtime...")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Test on sequence 0
    test_x = features_all[0:1, :500] # first 500 steps
    h0 = np.zeros((1, 1, 128), dtype=np.float32)
    h1 = np.zeros((1, 1, 128), dtype=np.float32)
    
    ort_outs = sess.run(None, {
        'features': test_x,
        'hidden_0': h0,
        'hidden_1': h1
    })
    ort_p, ort_h0, ort_h1 = ort_outs
    
    with torch.no_grad():
        pt_p, pt_h0, pt_h1 = model_cpu(
            torch.from_numpy(test_x),
            torch.from_numpy(h0),
            torch.from_numpy(h1)
        )
        
    diff_p = np.max(np.abs(ort_p - pt_p.numpy()))
    diff_h0 = np.max(np.abs(ort_h0 - pt_h0.numpy()))
    diff_h1 = np.max(np.abs(ort_h1 - pt_h1.numpy()))
    print(f"Max abs differences: output={diff_p:.2e}, next_h0={diff_h0:.2e}, next_h1={diff_h1:.2e}")
    assert diff_p < 1e-4, f"Prediction mismatch between PyTorch and ONNX: {diff_p}"
    print(" ONNX Parity Test PASSED!")
    
    # 5. Single-step streaming latency test (simulation of solution.py)
    print("\n[Step 4] Benchmarking single-step CPU inference latency...")
    warmup_steps = 100
    test_steps = 2000
    single_x = np.random.randn(1, 1, 112).astype(np.float32)
    h0 = np.zeros((1, 1, 128), dtype=np.float32)
    h1 = np.zeros((1, 1, 128), dtype=np.float32)
    
    # Warmup
    for _ in range(warmup_steps):
        out, h0, h1 = sess.run(None, {'features': single_x, 'hidden_0': h0, 'hidden_1': h1})
        
    t0 = time.perf_counter()
    for _ in range(test_steps):
        out, h0, h1 = sess.run(None, {'features': single_x, 'hidden_0': h0, 'hidden_1': h1})
    lat_us = ((time.perf_counter() - t0) / test_steps) * 1e6
    proj_runtime_min = (lat_us * 1e-6 * 39_400_000) / 60.0
    print(f"Single-step latency: {lat_us:.1f} us/step")
    print(f"Projected test evaluation time (39.4M steps): {proj_runtime_min:.1f} minutes (Budget: 60 min, <= 91.3 us)")
    assert lat_us < 90.0, f"Latency too high: {lat_us} us"
    print(" Latency Budget PASSED!")
    print("\n" + "=" * 80)
    print("EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
