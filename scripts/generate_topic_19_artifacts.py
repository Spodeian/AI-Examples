"""Script to generate golden reference weights and verify PyTorch 2 / JAX for Topic 19.

Topic: 05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian
Subfolder: 04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(42)
np.random.seed(42)

B = 4
D = 8
C = 4
eps = 1e-8

# 1. Forward-Forward (Hinton, 2022)
h_pos = torch.randn(B, D, dtype=torch.float32)
h_neg = torch.randn(B, D, dtype=torch.float32)
theta_ff = 2.0

g_pos = torch.sum(h_pos ** 2, dim=-1) # [B]
g_neg = torch.sum(h_neg ** 2, dim=-1) # [B]
loss_pos = F.softplus(theta_ff - g_pos)
loss_neg = F.softplus(g_neg - theta_ff)
golden_ff_loss = torch.mean(loss_pos + loss_neg)

# Forward-Forward layer normalization: h / (norm(h) + eps)
norm_pos = torch.norm(h_pos, p=2, dim=-1, keepdim=True) + eps
golden_h_norm = h_pos / norm_pos

# 2. Predictive Coding (Rao & Ballard, 1999)
pc_x = torch.randn(D, dtype=torch.float32)
pc_r = torch.randn(D, dtype=torch.float32)
pc_w = torch.randn(D, D, dtype=torch.float32) * 0.3
eta_pc = 0.1
lambda_pc = 0.01

# Error eps = x - r @ W
pc_pred = pc_r @ pc_w
pc_eps = pc_x - pc_pred
golden_pc_r_next = pc_r + eta_pc * (pc_eps @ pc_w.T - lambda_pc * pc_r)

# 3. Direct Feedback Alignment (DFA - Nøkland, 2016)
top_error = torch.randn(B, C, dtype=torch.float32)
dfa_feedback = torch.randn(D, C, dtype=torch.float32) # Fixed random feedback matrix
act_pre = torch.randn(B, D, dtype=torch.float32)
relu_mask = (act_pre > 0.0).float()
# Error projected via B rather than W_transpose: (top_error @ B^T) * relu_mask
golden_dfa_grad = (top_error @ dfa_feedback.T) * relu_mask

# 4. Oja's Hebbian Rule (Oja, 1982)
oja_x = torch.randn(D, dtype=torch.float32)
oja_w = torch.randn(D, dtype=torch.float32)
oja_w = oja_w / torch.norm(oja_w) # Normalized initial weight
eta_oja = 0.05
y_oja = torch.dot(oja_x, oja_w)
golden_oja_w_next = oja_w + eta_oja * (y_oja * oja_x - (y_oja ** 2) * oja_w)

# 5. Standard Linear Anchor
anchor_x = torch.randn(2, 8, dtype=torch.float32)
anchor_w = torch.randn(8, 4, dtype=torch.float32)
anchor_b = torch.randn(4, dtype=torch.float32)
golden_anchor_out = anchor_x @ anchor_w + anchor_b

tensors = {
    # Forward-Forward
    "h_pos": h_pos.contiguous(),
    "h_neg": h_neg.contiguous(),
    "golden_ff_loss": golden_ff_loss.reshape(1).contiguous(),
    "golden_h_norm": golden_h_norm.contiguous(),

    # Predictive Coding
    "pc_x": pc_x.contiguous(),
    "pc_r": pc_r.contiguous(),
    "pc_w": pc_w.contiguous(),
    "golden_pc_r_next": golden_pc_r_next.contiguous(),

    # DFA
    "top_error": top_error.contiguous(),
    "dfa_feedback": dfa_feedback.contiguous(),
    "act_pre": act_pre.contiguous(),
    "golden_dfa_grad": golden_dfa_grad.contiguous(),

    # Oja
    "oja_x": oja_x.contiguous(),
    "oja_w": oja_w.contiguous(),
    "golden_oja_w_next": golden_oja_w_next.contiguous(),

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
ckpt_path = out_dir / "05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian.safetensors"

save_file(tensors, str(ckpt_path))
print(f"Saved {len(tensors)} tensors to {ckpt_path}")
