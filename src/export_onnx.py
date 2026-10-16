"""Export the trained DualStreamNetwork to a production-optimized ONNX model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from src.train_dual_stream import DualStreamNetwork

class DualStreamONNXWrapper(nn.Module):
    def __init__(self, core_model: DualStreamNetwork):
        super().__init__()
        self.core = core_model

    def forward(self, features, hidden_0, hidden_1):
        # features: (1, 1, 112)
        # hidden_0: (1, 1, 128)
        # hidden_1: (1, 1, 128)
        pred, next_h0, next_h1 = self.core(features, hidden_0, hidden_1)
        return pred, next_h0, next_h1

print("Loading DualStreamNetwork from dual_stream_best.pt...")
core = DualStreamNetwork()
state = torch.load('dual_stream_best.pt', map_location='cpu')
core.load_state_dict(state)
core.eval()

wrapper = DualStreamONNXWrapper(core)
wrapper.eval()

# Dummy inputs for tracing
dummy_feat = torch.zeros(1, 1, 112, dtype=torch.float32)
dummy_h0 = torch.zeros(1, 1, 128, dtype=torch.float32)
dummy_h1 = torch.zeros(1, 1, 128, dtype=torch.float32)

from torch.onnx.utils import export as legacy_export

export_path = Path('submission/model.onnx')
print(f"Exporting to {export_path}...")

legacy_export(
    wrapper,
    (dummy_feat, dummy_h0, dummy_h1),
    str(export_path),
    input_names=['features', 'hidden_0', 'hidden_1'],
    output_names=['prediction', 'next_hidden_0', 'next_hidden_1'],
    dynamic_axes={
        'features': {0: 'batch', 1: 'seq_len'},
        'hidden_0': {0: 'batch', 1: 'seq_len'},
        'hidden_1': {0: 'batch', 1: 'seq_len'},
        'prediction': {0: 'batch', 1: 'seq_len'},
        'next_hidden_0': {0: 'batch', 1: 'seq_len'},
        'next_hidden_1': {0: 'batch', 1: 'seq_len'},
    },
    opset_version=17,
    do_constant_folding=True
)

print(f"ONNX export completed! File size: {export_path.stat().st_size / 1024:.1f} KB")

# Verify model with onnx checker
m = onnx.load(str(export_path))
onnx.checker.check_model(m)
print("ONNX checker passed successfully!")

# Verify numerical equivalence between PyTorch and ONNX Runtime
print("Testing numerical parity with ONNX Runtime...")
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1
sess = ort.InferenceSession(str(export_path), sess_options=sess_opts, providers=['CPUExecutionProvider'])

test_x = torch.randn(1, 1, 112, dtype=torch.float32)
test_h0 = torch.randn(1, 1, 128, dtype=torch.float32)
test_h1 = torch.randn(1, 1, 128, dtype=torch.float32)

with torch.no_grad():
    py_pred, py_h0, py_h1 = wrapper(test_x, test_h0, test_h1)

ort_out = sess.run(
    None,
    {
        'features': test_x.numpy(),
        'hidden_0': test_h0.numpy(),
        'hidden_1': test_h1.numpy()
    }
)

diff_pred = np.max(np.abs(py_pred.numpy() - ort_out[0]))
diff_h0 = np.max(np.abs(py_h0.numpy() - ort_out[1]))
diff_h1 = np.max(np.abs(py_h1.numpy() - ort_out[2]))

print(f"Max absolute difference:")
print(f"  Prediction: {diff_pred:.2e}")
print(f"  Next H0:    {diff_h0:.2e}")
print(f"  Next H1:    {diff_h1:.2e}")

assert diff_pred < 1e-4, f"Prediction discrepancy too high: {diff_pred}"
assert diff_h0 < 1e-4, f"Hidden_0 discrepancy too high: {diff_h0}"
assert diff_h1 < 1e-4, f"Hidden_1 discrepancy too high: {diff_h1}"
print("VERIFICATION COMPLETE: PyTorch and ONNX Runtime outputs match within float precision!")
