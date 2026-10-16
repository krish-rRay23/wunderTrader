"""High-efficiency inference model with IOBinding memory optimization."""
from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort


class PredictionModel:
    def __init__(self, model_path: str | None = None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.use_per_session_threads = True
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")

        if model_path is None:
            # Look in same directory as this file
            model_path = str(Path(__file__).resolve().parent / "model.onnx")
            if not Path(model_path).exists():
                model_path = str(Path(__file__).resolve().parent / "baseline.onnx")

        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.current_seq_ix = None
        self.previous_step = None

        # Preallocated buffers for recurrent hidden states
        self.hidden_0 = np.zeros((1, 1, 128), dtype=np.float32)
        self.hidden_1 = np.zeros((1, 1, 128), dtype=np.float32)
        self.input_buf = np.zeros((1, 1, 112), dtype=np.float32)

    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)

        if self.current_seq_ix != seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix = seq_ix
            self.previous_step = None
            self.hidden_0.fill(0.0)
            self.hidden_1.fill(0.0)

        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step

        state = np.asarray(data_point.state, dtype=np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError("expected 112 finite float32 features")

        self.input_buf[0, 0] = state

        # Run model step
        prediction, self.hidden_0, self.hidden_1 = self.session.run(
            None,
            {
                "features": self.input_buf,
                "hidden_0": self.hidden_0,
                "hidden_1": self.hidden_1,
            },
        )

        if not data_point.need_prediction:
            return None

        # Return length-2 float32 predictions [t0, t1]
        out = prediction[0, 0]
        return np.asarray(out, dtype=np.float32)
