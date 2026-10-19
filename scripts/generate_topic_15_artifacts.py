"""Generator script for Category 04 Topic 15 reference safetensors checkpoint.

Topic: Grokking, Phase Transitions, Edge of Stability, and Implicit Bias.
"""

from pathlib import Path
import torch
from safetensors.torch import save_file

torch.manual_seed(42)

# 1. Grokfast EMA filter
# mu_next = alpha * mu_prev + (1 - alpha) * g_raw
# g_grokfast = g_raw + lambda_gf * mu_next
g_raw = torch.tensor([0.25, -0.4, 0.6, -0.15, 0.8, -0.5, 0.3, 0.1], dtype=torch.float32)
mu_prev = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.4, -0.2, 0.1, 0.05], dtype=torch.float32)
alpha_gf = 0.8
lambda_gf = 2.0
golden_mu = alpha_gf * mu_prev + (1.0 - alpha_gf) * g_raw
golden_g_grokfast = g_raw + lambda_gf * golden_mu

# 2. Edge of Stability (EOS) cubic oscillator step
# x_next = x - eta * (H * x + gamma * x^3)
x_eos = torch.tensor([0.5, -0.8, 0.3, -0.4], dtype=torch.float32)
h_eos = torch.tensor([2.5, 3.0, 2.2, 2.8], dtype=torch.float32)
gamma_eos = torch.tensor([0.4, 0.5, 0.3, 0.6], dtype=torch.float32)
eta_eos = 0.1
grad_eos = h_eos * x_eos + gamma_eos * (x_eos ** 3)
golden_x_eos = x_eos - eta_eos * grad_eos

# 3. Implicit Bias Logistic Loss Step on Linearly Separable Data
# L(w) = - 1/N sum_i log(1 + exp(- y_i * w^T x_i))
# grad_w = - 1/N sum_i (y_i * x_i) / (1 + exp(y_i * w^T x_i))
w_sep = torch.tensor([0.2, -0.1, 0.4, -0.3], dtype=torch.float32)
x_sep = torch.tensor([
    [1.0, 2.0, -1.0, 0.5],
    [2.0, -1.0, 1.5, -0.5],
    [-1.0, 1.0, -2.0, 1.0]
], dtype=torch.float32)
y_sep = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)
margins = y_sep * (x_sep @ w_sep)
sigmoids = 1.0 / (1.0 + torch.exp(margins))
grad_w_sep = - torch.mean((y_sep.unsqueeze(1) * x_sep) * sigmoids.unsqueeze(1), dim=0)
eta_sep = 0.2
golden_w_sep = w_sep - eta_sep * grad_w_sep

# 4. Linear Anchor Invariant (2D [in_dim, out_dim])
# anchor_x: [2, 8], anchor_w: [8, 4], anchor_b: [4]
anchor_x = torch.randn(2, 8, dtype=torch.float32)
anchor_w = torch.randn(8, 4, dtype=torch.float32)
anchor_b = torch.randn(4, dtype=torch.float32)
golden_anchor_out = anchor_x @ anchor_w + anchor_b

# Target save path
out_dir = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "01_grokking_phase_transitions_edge_of_stability_and_implicit_bias.safetensors"

tensor_dict = {
    # Standard Parity Keys for test_cross_track_parity.py
    "weight": anchor_w.clone().contiguous(),
    "bias": anchor_b.clone().contiguous(),
    "test_input": anchor_x.clone().contiguous(),
    "test_output": golden_anchor_out.clone().contiguous(),

    # Topic-specific anchors
    "anchor_x": anchor_x.clone().contiguous(),
    "anchor_w": anchor_w.clone().contiguous(),
    "anchor_b": anchor_b.clone().contiguous(),
    "golden_anchor_out": golden_anchor_out.clone().contiguous(),

    # Grokfast
    "g_raw": g_raw.clone().contiguous(),
    "mu_prev": mu_prev.clone().contiguous(),
    "golden_mu": golden_mu.clone().contiguous(),
    "golden_g_grokfast": golden_g_grokfast.clone().contiguous(),

    # Edge of Stability
    "x_eos": x_eos.clone().contiguous(),
    "h_eos": h_eos.clone().contiguous(),
    "gamma_eos": gamma_eos.clone().contiguous(),
    "golden_x_eos": golden_x_eos.clone().contiguous(),

    # Implicit Bias
    "w_sep": w_sep.clone().contiguous(),
    "x_sep": x_sep.clone().contiguous(),
    "y_sep": y_sep.clone().contiguous(),
    "golden_w_sep": golden_w_sep.clone().contiguous(),
}

save_file(tensor_dict, str(out_file))
print(f"Successfully generated golden checkpoint at {out_file}")
