"""Script to generate golden reference weights and verify PyTorch 2 / JAX for Topic 18.

Topic: 04_self_optimising_test_time_adaptation_tent_memo_star_and_spin
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
C = 4
K = 3
eps = 1e-8

# 1. TENT: Test-Time Entropy Minimization (Wang et al., 2021)
tent_logits = torch.randn(B, C, dtype=torch.float32)
tent_scale = torch.ones(C, dtype=torch.float32, requires_grad=True)
tent_shift = torch.zeros(C, dtype=torch.float32, requires_grad=True)

# Affine transformation on logits: logits * scale + shift
scaled_logits = tent_logits * tent_scale + tent_shift
p_tent = F.softmax(scaled_logits, dim=-1)
log_p_tent = F.log_softmax(scaled_logits, dim=-1)
tent_entropy = -torch.mean(torch.sum(p_tent * log_p_tent, dim=-1))
tent_entropy.backward()

golden_scale_grad = tent_scale.grad.clone().detach()
golden_shift_grad = tent_shift.grad.clone().detach()
# One step of SGD on affine parameters
lr_tent = 0.05
golden_scale_next = (tent_scale.detach() - lr_tent * golden_scale_grad).contiguous()
golden_shift_next = (tent_shift.detach() - lr_tent * golden_shift_grad).contiguous()

# 2. MEMO: Marginal Entropy Minimization with One test sample (Zhang et al., 2022)
memo_aug_logits = torch.randn(K, C, dtype=torch.float32)
memo_aug_probs = F.softmax(memo_aug_logits, dim=-1)
golden_marginal_p = torch.mean(memo_aug_probs, dim=0) # [C]
golden_memo_entropy = -torch.sum(golden_marginal_p * torch.log(golden_marginal_p + eps)) # scalar

# 3. SPIN: Self-Play Preference Loss (Chen et al., 2024)
logp_win = torch.randn(B, dtype=torch.float32)
logp_lose = logp_win - torch.abs(torch.randn(B, dtype=torch.float32)) - 0.5 # win is strictly higher log-prob
beta_spin = 0.5
golden_spin_loss = -torch.mean(F.logsigmoid(beta_spin * (logp_win - logp_lose)))

# 4. Standard Linear Anchor
anchor_x = torch.randn(2, 8, dtype=torch.float32)
anchor_w = torch.randn(8, 4, dtype=torch.float32)
anchor_b = torch.randn(4, dtype=torch.float32)
golden_anchor_out = anchor_x @ anchor_w + anchor_b

tensors = {
    # TENT
    "tent_logits": tent_logits.contiguous(),
    "tent_scale": tent_scale.detach().contiguous(),
    "tent_shift": tent_shift.detach().contiguous(),
    "golden_tent_entropy": tent_entropy.detach().reshape(1).contiguous(),
    "golden_scale_grad": golden_scale_grad.contiguous(),
    "golden_shift_grad": golden_shift_grad.contiguous(),
    "golden_scale_next": golden_scale_next.contiguous(),
    "golden_shift_next": golden_shift_next.contiguous(),

    # MEMO
    "memo_aug_logits": memo_aug_logits.contiguous(),
    "golden_marginal_p": golden_marginal_p.contiguous(),
    "golden_memo_entropy": golden_memo_entropy.reshape(1).contiguous(),

    # SPIN
    "logp_win": logp_win.contiguous(),
    "logp_lose": logp_lose.contiguous(),
    "golden_spin_loss": golden_spin_loss.reshape(1).contiguous(),

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
ckpt_path = out_dir / "04_self_optimising_test_time_adaptation_tent_memo_star_and_spin.safetensors"

save_file(tensors, str(ckpt_path))
print(f"Saved {len(tensors)} tensors to {ckpt_path}")
