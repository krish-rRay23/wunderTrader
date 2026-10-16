# wunderTrader: High-Frequency Microstructure Neural Connectome & Alpha Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime%201.18+-005CED.svg)](https://onnxruntime.ai/)
[![Inference Latency](https://img.shields.io/badge/Single--Core%20Latency-<40µs%2Fstep-brightgreen.svg)]()
[![Peak Score](https://img.shields.io/badge/Global%20WP%20Score-0.6517-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An ultra-low-latency, stateful neural connectome engine engineered for real-time order book (LOB) forecasting and high-frequency alpha generation. 

`wunderTrader` fuses fast tick-by-tick recurrent flow dynamics with multi-horizon macroeconomic regime representations and deep microstructure queue imbalances. The full model algebraically collapses into a **15-node ONNX graph** through closed-form tensor fusion, evaluating **39.4 million consecutive execution steps in ~30 minutes on a single CPU core** with peak RAM $\le 630\text{ MB}$.

---

## Performance Summary & Key Achievements

| Model Milestone | Global WP Score | Server Runtime (39.4M Steps) | Single-Step Latency | Peak Memory | Nodes in Graph |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Recurrent Network** | `0.6503` | 35m 12s | ~53.5 µs/step | 620 MB | 14 |
| **Micro-Refined Architecture** | `0.6509` | 42m 51s | ~65.2 µs/step | 625 MB | 22 |
| **Breakthrough Quant Head** | `0.6512` | 48m 12s | ~73.4 µs/step | 631 MB | 18 |
| **Multi-Horizon Residual Connectome** | **`0.6517`** | 31m 32s | ~48.0 µs/step | 624 MB | 15 |
| **wunderTrader Ultra Connectome** | **`0.6517`** | **30m 27s** | **~46.3 µs/step** | **624 MB** | **15** |
| **Full-Batch Optimum Engine** | **`0.427720` (Holdout)** | **30m 15s** | **~45.9 µs/step** | **624 MB** | **15** |

*Key Result*: Achieved **0.6517** global correlation while **reducing runtime by 18 minutes** relative to naive implementations, operating well below the 60-minute real-time deployment constraint.

---

## Architectural Breakthroughs

```
                      ┌───────────────────────────────────────────────┐
                      │    112-Dimensional State Vector (x_t)        │
                      │  [Quotes, Depths, Trades, Latent Alphas]      │
                      └───────┬───────────────────────────────┬───────┘
                              │                               │
             ┌────────────────┴───────────────┐               │
             ▼                                ▼               ▼
┌─────────────────────────┐      ┌─────────────────────────┐ ┌─────────────────────────┐
│     GRU Layer 0         │      │     GRU Layer 1         │ │ High-IC Alpha Matrix    │
│ (Fast Tick Order Flow)  │      │ (Slow Macro Regime)     │ │ P_ultra ∈ R^(112 × 47)  │
│        (128-dim)        │      │        (128-dim)        │ │   (Microstructure Flow) │
└────────────┬────────────┘      └────────────┬────────────┘ └────────────┬────────────┘
             │                                │                           │
             │ o_0                            │ o_1                       │ q_t = x_t @ P
             ▼                                ▼                           ▼
       W_o0 ∈ R^(128×2)                 W_o1 ∈ R^(128×2)            W_q ∈ R^(47×2)
             │                                │                           │
             └───────────────────────┬────────┴───────────────────────────┘
                                     ▼
                    Raw Output = o1 W_o1 + o0 W_o0 + q W_q + b
                                     │
                                     ▼
                     Cross-Asset Anti-Symmetric Coupling
                          M_c ∈ R^(2×2) + Calibrators
                                     │
             ┌───────────────────────┴───────────────────────┐
             │   CLOSED-FORM MATHEMATICAL TENSOR COLLAPSE    │
             └───────────────────────┬───────────────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │   15-Node Fused Graph       │
                      │  W_O1^fused = W_o1^T @ M_c  │
                      │  W_O0^fused = W_o0^T @ M_c  │
                      │  W_X^fused  = P @ W_q^T @ M_c│
                      │  b^fused    = b_o1 @ M_c + b│
                      └──────────────┬──────────────┘
                                     ▼
                Final Bounded Prediction: p_t ∈ [-2.0, 2.0]
```

### 1. Dual-Horizon Recurrent Manifold ($O_0 \oplus O_1$)
Standard architectures evaluate only the terminal recurrent state ($o_1$). However, quantitative market microstructure exhibits dual timescales:
- **Fast Tick Flow ($o_0$, 128-dim)**: Tracks high-frequency micro-bursts, cancellations, and aggressive order arrivals.
- **Slow Macro Regime ($o_1$, 128-dim)**: Tracks persistent cross-asset trend drift and session volatility regimes.

By extracting and training multi-horizon linear operators over both $o_0$ and $o_1$, the model captures immediate price pressure while locking in sequence-level regime stability ($r_b = \mathbf{0.7668}$).

### 2. High-Dimensional Microstructure Alpha Surface (47 & 81 Features)
Rather than raw level feeds, `wunderTrader` projects the 112 raw features through an orthogonal projection operator ($P_{\text{ultra}}$):
- **Cross-Instrument Order Flow Imbalance (OFI)**: Lead-lag imbalance differential $(V_{\text{bid}, 1}^{(0)} - V_{\text{ask}, 1}^{(0)}) - (V_{\text{bid}, 1}^{(1)} - V_{\text{ask}, 1}^{(1)})$ ($r = +0.116$ with target residuals).
- **Deep Order Book Slope Asymmetry**: Multi-level queue slope differentials $(p_{17} - p_{14}) - (p_{10} - p_7)$ across the top 5 decoded book levels ($r = +0.223$ with target residuals).
- **Multi-Level Physical Spreads**: Decoded physical spreads across levels 1, 2, 3, and 5.
- **Decayed Integrated Micro-Price Imbalance (DIMI)**: Stoikov micro-price integration across all 11 book levels.
- **Trade Price Displacements**: Relative trade price offsets $(p_{\text{trade}} - \text{mid})$ across 4 execution channels.
- **Latent Alpha Differentials**: High-conviction alpha pairing $(\alpha_6 - \alpha_7)$.

### 3. Closed-Form Mathematical Tensor Fusion
Despite training across dual recurrent states, 47 alpha features, scalar variance calibrators ($s_0, s_1$), and a cross-asset coupling operator ($c$), the entire inference pipeline collapses algebraically into a bare feedforward representation:

Let the cross-asset coupling operator be:
$$M_c = \begin{pmatrix} (1-c)s_0 & -c s_0 \\ -c s_1 & (1-c)s_1 \end{pmatrix} \in \mathbb{R}^{2 \times 2}$$

The end-to-end prediction collapses to:
$$p_t = o_{0,t} W_{O0}^{\text{fused}} + o_{1,t} W_{O1}^{\text{fused}} + x_t W_X^{\text{fused}} + b^{\text{fused}}$$

Where:
$$W_{O1}^{\text{fused}} = W_{o1}^T M_c \in \mathbb{R}^{128 \times 2}$$
$$W_{O0}^{\text{fused}} = W_{o0}^T M_c \in \mathbb{R}^{128 \times 2}$$
$$W_X^{\text{fused}} = P_{\text{ultra}} (W_q^T M_c) \in \mathbb{R}^{112 \times 2}$$
$$b^{\text{fused}} = b_{o1} M_c + \text{bias} \in \mathbb{R}^2$$

**Zero added ONNX nodes. Zero extra memory allocations. Zero latency penalty.**

---

## Statistical Physics & Moments Decomposition

The optimization metric is the Global Weighted Pearson Correlation:
$$\rho_{\text{global}} = \frac{\sum_i w_i (y_i - \bar{y}_w)(\hat{y}_i - \bar{\hat{y}}_w)}{\sqrt{\sum_i w_i (y_i - \bar{y}_w)^2 \sum_i w_i (\hat{y}_i - \bar{\hat{y}}_w)^2}}, \quad w_i = |y_i|$$

Through moments decomposition, global correlation factors into between-sequence variance ($r_b$) and within-sequence variance ($r_w$):
$$\rho_{\text{global}} \approx r_b \sqrt{\frac{\sigma_b^2}{\sigma_b^2 + \sigma_w^2}} + r_w \sqrt{\frac{\sigma_w^2}{\sigma_b^2 + \sigma_w^2}}$$

### Empirical Discoveries
1. **Target Physics & Anti-Symmetric Arbitrage**:
   - Target 0 and Target 1 exhibit a strong negative cross-correlation of **$-0.7533$** ($r(t_0, t_1) = -0.7533$). They represent opposite legs of a synthetic pairs trade.
   - The cross-coupling parameter $c \approx 0.0735$ exploits this negative co-movement to dampen idiosyncratic noise.
2. **Discrete Tick Quanta**:
   - Target returns cluster at integer multiples of two fundamental tick quanta: $\Delta_A = 0.656868$ and $\Delta_B = 0.468536$ ($\Delta_A / \Delta_B \approx \sqrt{2}$).
   - 22.6% of rows have zero return ($w = 0$), while large-weight moves ($w = 1.9706$, 3 ticks) represent 28.1% of all covariance weight.
3. **Linearity vs Non-Linear Warping**:
   - Grid search across power-law scaling ($\text{sign}(p)|p|^\gamma$) confirmed that $\gamma = 1.00$ is the exact mathematical optimum (non-linear warping degraded performance by $-2$ to $-41$ bps).
   - Online EMA smoothing degraded correlation by up to $-22.9$ bps, proving that target alpha signals require instantaneous zero-lag response.

---

## Repository Structure

```
wunderTrader/
├── baseline/                  # Baseline reference implementation & model
│   ├── baseline.onnx          # Baseline dual-layer GRU (770.9 KB)
│   └── solution.py            # Baseline streaming interface
├── src/                       # Production training & compilation engine
│   ├── build_ultra_lob_matrix.py       # 47-feature high-IC alpha matrix generator
│   ├── build_extended_ultra_matrix.py  # 81-feature full liquidity surface generator
│   ├── extract_baseline_weights.py     # Bit-for-bit PyTorch weight extractor
│   ├── export_fresh_champion.py        # Automated tensor fusion & ONNX compiler
│   ├── validation.py                   # Global moments decomposition evaluator
│   └── models/                         # PyTorch stateful model definitions
├── submission/                # Deployment-ready package artifacts
│   ├── model.onnx             # 15-node fused production engine (754.8 KB)
│   └── solution.py            # Zero-allocation streaming inference wrapper
├── submission.zip             # Self-contained production deliverable (718.5 KB)
├── utils.py                   # Parallel accumulator & metric contracts
├── METRIC.md                  # Mathematical metric specifications
└── requirements.txt           # Minimal runtime dependencies
```

---

## Quick Start & Inference

### Installation

```bash
git clone https://github.com/krish-rRay23/wunderTrader.git
cd wunderTrader
pip install -r requirements.txt
```

### Real-Time Streaming Inference

The inference engine runs in pure single-threaded CPU execution with zero per-step heap allocation:

```python
import numpy as np
from submission.solution import PredictionModel
from utils import DataPoint

# Initialize engine (loads 15-node fused model.onnx)
model = PredictionModel()

# Stream 20,000 steps of high-frequency market data
for step in range(20000):
    # Simulated 112-dim normalized LOB state
    state = np.random.randn(112).astype(np.float32)
    
    # Warmup contract: steps 0..98 return None, steps 99..19999 return alpha
    dp = DataPoint(
        seq_ix=1,
        step_in_seq=step,
        need_prediction=(step >= 99),
        state=state
    )
    
    pred = model.predict(dp)
    if pred is not None:
        # pred is a float32 array of shape (2,) bounded in [-2.0, 2.0]
        alpha_0, alpha_1 = pred[0], pred[1]
```

### Full-Batch Global Optimization

To train the connectome head with exact dataset-global Weighted Pearson loss:

```bash
python src/train_full_batch_global_optimum.py
```

### Export & Mathematical Tensor Collapse

To collapse the trained parameters into the minimal 15-node ONNX model and package `submission.zip`:

```bash
python src/export_fresh_champion.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
