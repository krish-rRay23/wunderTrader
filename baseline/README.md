# Vanilla GRU ONNX baseline

This is an inference-only stateful baseline. It scores **0.617052 WP** on the
complete validation set using Global WP.

From the repository root, run:

```bash
python baseline/solution.py --validation datasets/valid.parquet
```

The model updates its recurrent state on every row, including warm-up rows,
and resets state between sequences. ONNX Runtime is configured for CPU-only
sequential execution with one intra-op and one inter-op thread.

`baseline/solution.py` and `baseline/baseline.onnx` provide a reference baseline.
The baseline does not include training code or an original checkpoint.
