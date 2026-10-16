"""Export trained DeepMicrostructureNetwork to production ONNX and verify numerical parity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from src.models.deep_microstructure_network import DeepMicrostructureNetwork

class DeepONNXWrapper(nn.Module):
    def __init__(self, core: DeepMicrostructureNetwork):
        super().__init__()
        self.core = core

    def forward(self, features, hidden_0, hidden_1):
        preds, next_h0, next_h1 = self.core(features, hidden_0, hidden_1)
        return preds, next_h0, next_h1

def export_model(checkpoint_path: str = 'best_deep_model.pt', output_path: str = 'submission/model.onnx'):
    print(f"Loading checkpoint from {checkpoint_path}...")
    model = DeepMicrostructureNetwork(in_features=112, hidden_dim=192)
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    wrapper = DeepONNXWrapper(model)
    wrapper.eval()

    dummy_feat = torch.zeros(1, 1, 112, dtype=torch.float32)
    dummy_h0 = torch.zeros(1, 1, 192, dtype=torch.float32)
    dummy_h1 = torch.zeros(1, 1, 192, dtype=torch.float32)

    from torch.onnx.utils import export as legacy_export

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX to {out_file}...")

    legacy_export(
        wrapper,
        (dummy_feat, dummy_h0, dummy_h1),
        str(out_file),
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

    size_kb = out_file.stat().st_size / 1024
    print(f"Export complete! Model size: {size_kb:.1f} KB")

    # Verify model with onnx checker
    m = onnx.load(str(out_file))
    onnx.checker.check_model(m)
    print("ONNX checker passed!")

    # Verify ONNX Runtime parity
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(out_file), sess_options=sess_opts, providers=['CPUExecutionProvider'])

    test_x = torch.randn(1, 1, 112, dtype=torch.float32)
    test_h0 = torch.randn(1, 1, 192, dtype=torch.float32)
    test_h1 = torch.randn(1, 1, 192, dtype=torch.float32)

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
    print(f"Parity Check: Max pred diff = {diff_pred:.2e}")
    assert diff_pred < 1e-4, f"Parity diff too high: {diff_pred}"
    print("Numerical parity confirmed successfully!")

if __name__ == '__main__':
    export_model()
