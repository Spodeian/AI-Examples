"""Script to generate golden reference weights and verify PyTorch 2 / JAX for Topic 17.

Topic: 03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

# Set deterministic seed
torch.manual_seed(42)
np.random.seed(42)

# Dimensions
D = 8
B = 4

# 1. Hypergradient Descent (HD)
lr_init = torch.tensor([0.01], dtype=torch.float32)
hyper_lr = torch.tensor([0.001], dtype=torch.float32)
w_hd = torch.randn(D, dtype=torch.float32)
g_curr = torch.randn(D, dtype=torch.float32)
g_prev = torch.randn(D, dtype=torch.float32)

# HD Step: h_t = g_curr . g_prev, lr_next = lr + hyper_lr * h_t, w_next = w - lr_next * g_curr
h_t = torch.dot(g_curr, g_prev)
lr_next = lr_init + hyper_lr * h_t
w_next_hd = w_hd - lr_next * g_curr

# 2. Meta-Sample Reweighting
per_sample_grads = torch.randn(B, D, dtype=torch.float32)
# Corrupt sample 1 intentionally so its gradient opposes val_grad
val_grad = torch.randn(D, dtype=torch.float32)
per_sample_grads[1] = -val_grad * 1.5

# Alignment: s_i = G_i . val_grad
alignments = per_sample_grads @ val_grad
weights_relu = torch.clamp(alignments, min=0.0)
sum_weights = torch.sum(weights_relu) + 1e-8
normalized_weights = weights_relu / sum_weights
reweighted_grad = normalized_weights @ per_sample_grads

# 3. Learned Coordinate-Wise Optimizer MLP
param_learned = torch.randn(D, dtype=torch.float32)
grad_learned = torch.randn(D, dtype=torch.float32)
momentum_learned = torch.randn(D, dtype=torch.float32)

mlp_w1 = torch.randn(2, 4, dtype=torch.float32) * 0.5
mlp_b1 = torch.zeros(4, dtype=torch.float32)
mlp_w2 = torch.randn(4, 1, dtype=torch.float32) * 0.5
mlp_b2 = torch.zeros(1, dtype=torch.float32)

# Coordinate-wise MLP forward
# Input: [D, 2] stacked (grad, momentum)
coords_in = torch.stack([grad_learned, momentum_learned], dim=-1) # [D, 2]
h_mlp = torch.relu(coords_in @ mlp_w1 + mlp_b1) # [D, 4]
delta_w = (h_mlp @ mlp_w2 + mlp_b2).squeeze(-1) # [D]
p_learned_next = param_learned - delta_w
m_learned_next = 0.9 * momentum_learned + 0.1 * grad_learned

# 4. Standard Linear Anchor
anchor_x = torch.randn(2, 8, dtype=torch.float32)
anchor_w = torch.randn(8, 4, dtype=torch.float32)
anchor_b = torch.randn(4, dtype=torch.float32)
golden_anchor_out = anchor_x @ anchor_w + anchor_b

tensors = {
    # Hypergradient Descent
    "w_hd": w_hd.contiguous(),
    "g_curr": g_curr.contiguous(),
    "g_prev": g_prev.contiguous(),
    "lr_init": lr_init.contiguous(),
    "hyper_lr": hyper_lr.contiguous(),
    "golden_lr_next": lr_next.contiguous(),
    "golden_w_hd": w_next_hd.contiguous(),

    # Meta-Sample Reweighting
    "per_sample_grads": per_sample_grads.contiguous(),
    "val_grad": val_grad.contiguous(),
    "golden_sample_weights": normalized_weights.contiguous(),
    "golden_reweighted_grad": reweighted_grad.contiguous(),

    # Learned Optimizer MLP
    "param_learned": param_learned.contiguous(),
    "grad_learned": grad_learned.contiguous(),
    "momentum_learned": momentum_learned.contiguous(),
    "mlp_w1": mlp_w1.contiguous(),
    "mlp_b1": mlp_b1.contiguous(),
    "mlp_w2": mlp_w2.contiguous(),
    "mlp_b2": mlp_b2.contiguous(),
    "golden_delta_w": delta_w.contiguous(),
    "golden_param_learned": p_learned_next.contiguous(),
    "golden_momentum_learned": m_learned_next.contiguous(),

    # Anchor & root standard
    "anchor_x": anchor_x.clone().contiguous(),
    "anchor_w": anchor_w.clone().contiguous(),
    "anchor_b": anchor_b.clone().contiguous(),
    "golden_anchor_out": golden_anchor_out.clone().contiguous(),
    "test_input": anchor_x.clone().contiguous(),
    "weight": anchor_w.clone().contiguous(),
    "bias": anchor_b.clone().contiguous(),
    "test_output": golden_anchor_out.clone().contiguous(),
}

out_dir = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
out_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = out_dir / "03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting.safetensors"

save_file(tensors, str(ckpt_path))
print(f"Saved {len(tensors)} tensors to {ckpt_path}")
