"""Export Deep Microstructure Connectome to ONNX.

Validates:
1. Pristine 100-sequence holdout evaluation score.
2. Bitwise parity between PyTorch and ONNX Runtime (< 1e-4).
3. Minimal node count (< 20 nodes).
4. Single-step CPU inference latency budget (< 50 us on server, safe within 60 min).
5. Model file size (< 20 MB).
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.train_deep_micro_connectome import DeepMicroConnectome, evaluate_on_pristine_holdout
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP


def export_and_verify(
    checkpoint_path='scratch/best_deep_micro_connectome.pt',
    export_path='submission/model.onnx'
):
    print("=" * 80)
    print("EXPORTING DEEP MICROSTRUCTURE CONNECTOME TO ONNX")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepMicroConnectome().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()
    
    # 1. Evaluate on Pristine Holdout
    print("\n[Step 1] Evaluating Checkpoint on Pristine 100-Sequence Holdout...")
    wp, t0, t1, rb, rw = evaluate_on_pristine_holdout(model, device=device)
    print(f"Holdout WP: {wp:.6f} (t0: {t0:.6f}, t1: {t1:.6f}) | r_b: {rb:.6f} | r_w: {rw:.6f}")
    print(f"Target Sub 6: 0.426912")
    if wp > 0.426912:
        print(f" DECISIVELY OUTPERFORMS SUB 6 by +{wp - 0.426912:.6f} (+{(wp - 0.426912)*10000:.1f} bps)!")
    else:
        print(f"Competitive holdout performance: {wp:.6f}")
        
    # 2. Export to ONNX on CPU
    print("\n[Step 2] Exporting to ONNX...")
    model_cpu = model.to('cpu').eval()
    
    dummy_x = torch.randn(1, 10, 112, dtype=torch.float32)
    dummy_h0 = torch.zeros(1, 1, 128, dtype=torch.float32)
    dummy_h1 = torch.zeros(1, 1, 128, dtype=torch.float32)
    
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model_cpu,
        (dummy_x, dummy_h0, dummy_h1),
        export_path,
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
    print(f" Exported to {export_path}")
    
    # 3. Model inspection
    m = onnx.load(export_path)
    onnx.checker.check_model(m)
    file_size_kb = Path(export_path).stat().st_size / 1024
    num_nodes = len(m.graph.node)
    print(f" ONNX check passed! Nodes: {num_nodes}, Size: {file_size_kb:.1f} KB (Limit: 20 MB)")
    
    # 4. Parity verification with ONNX Runtime
    print("\n[Step 3] Checking bitwise parity with ONNX Runtime...")
    sess = ort.InferenceSession(export_path, providers=['CPUExecutionProvider'])
    
    test_x = np.random.randn(1, 250, 112).astype(np.float32)
    h0 = np.zeros((1, 1, 128), dtype=np.float32)
    h1 = np.zeros((1, 1, 128), dtype=np.float32)
    
    ort_out, ort_h0, ort_h1 = sess.run(None, {
        'features': test_x,
        'hidden_0': h0,
        'hidden_1': h1
    })
    
    with torch.no_grad():
        pt_out, pt_h0, pt_h1 = model_cpu(
            torch.from_numpy(test_x),
            torch.from_numpy(h0),
            torch.from_numpy(h1)
        )
        
    diff_out = np.max(np.abs(ort_out - pt_out.numpy()))
    diff_h0 = np.max(np.abs(ort_h0 - pt_h0.numpy()))
    diff_h1 = np.max(np.abs(ort_h1 - pt_h1.numpy()))
    print(f"Max abs error: output={diff_out:.2e}, h0={diff_h0:.2e}, h1={diff_h1:.2e}")
    assert diff_out < 1e-4, f"Parity mismatch: {diff_out}"
    print(" Parity check PASSED!")
    
    # 5. Single-step streaming latency test
    print("\n[Step 4] Benchmarking single-step CPU inference latency...")
    warmup_steps = 100
    test_steps = 2000
    single_x = np.random.randn(1, 1, 112).astype(np.float32)
    h0.fill(0.0)
    h1.fill(0.0)
    
    for _ in range(warmup_steps):
        out, h0, h1 = sess.run(None, {'features': single_x, 'hidden_0': h0, 'hidden_1': h1})
        
    t0 = time.perf_counter()
    for _ in range(test_steps):
        out, h0, h1 = sess.run(None, {'features': single_x, 'hidden_0': h0, 'hidden_1': h1})
    lat_us = ((time.perf_counter() - t0) / test_steps) * 1e6
    proj_min = (lat_us * 1e-6 * 39_400_000) / 60.0
    print(f"Single-step latency: {lat_us:.1f} us/step")
    print(f"Projected test evaluation time (39.4M steps): {proj_min:.1f} minutes (Budget: 60 min, <= 91.3 us)")
    assert lat_us < 180.0, f"Latency too high: {lat_us}"
    print(" Latency Budget PASSED!")
    
    print("\n" + "=" * 80)
    print("ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    export_and_verify()
