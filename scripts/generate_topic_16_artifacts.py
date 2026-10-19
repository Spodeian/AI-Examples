"""Generator script for Category 04 Topic 16 reference safetensors checkpoint.

Topic: Training Optimisations: Progressive Stacking, Data Pruning, MRL, and Plasticity.
"""

from pathlib import Path
import torch
from safetensors.torch import save_file

torch.manual_seed(42)

# 1. Matryoshka Representation Learning (MRL) Nested Slices
# Input embedding Z: [4, 8]
# Weight W: [8, 4], Bias b: [4]
# Slices d in [2, 4, 8]
z_emb = torch.tensor([
    [0.5, -0.2, 0.8, -0.1, 0.4, 0.3, -0.6, 0.2],
    [-0.3, 0.7, -0.4, 0.5, -0.2, 0.8, 0.1, -0.5],
    [0.9, -0.8, 0.2, 0.1, 0.6, -0.4, 0.3, 0.7],
    [-0.1, 0.4, -0.7, 0.9, -0.5, 0.2, 0.8, -0.3]
], dtype=torch.float32)

mrl_w = torch.tensor([
    [0.2, -0.4, 0.1, 0.5],
    [-0.3, 0.6, -0.2, 0.1],
    [0.8, 0.1, -0.5, 0.4],
    [-0.5, 0.3, 0.7, -0.2],
    [0.4, -0.1, 0.3, 0.6],
    [-0.2, 0.5, -0.4, 0.1],
    [0.6, -0.3, 0.2, -0.5],
    [0.1, 0.4, -0.1, 0.3]
], dtype=torch.float32)
mrl_b = torch.tensor([0.1, -0.2, 0.05, -0.15], dtype=torch.float32)

# Nested logits for d=2, d=4, d=8
golden_logits_d2 = z_emb[:, :2] @ mrl_w[:2, :] + mrl_b
golden_logits_d4 = z_emb[:, :4] @ mrl_w[:4, :] + mrl_b
golden_logits_d8 = z_emb @ mrl_w + mrl_b

# 2. EL2N and RHO-Loss Data Pruning Metrics
# Probabilities P: [4, 3], Targets Y (one-hot): [4, 3]
p_probs = torch.tensor([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.3, 0.4, 0.3],
    [0.05, 0.05, 0.9]
], dtype=torch.float32)
y_targets = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=torch.float32)
# EL2N score: L2 norm of error vector per sample
golden_el2n = torch.norm(p_probs - y_targets, p=2, dim=-1)

# RHO-loss: L_model - L_ref
loss_model = torch.tensor([0.35, 0.22, 1.20, 0.10], dtype=torch.float32)
loss_ref = torch.tensor([0.15, 0.18, 0.95, 0.08], dtype=torch.float32)
golden_rho_loss = loss_model - loss_ref

# 3. ReDo Dormant Neuron Reset Operator
# Activations: [4, 6]
# Dormant threshold tau = 0.15
activations = torch.tensor([
    [0.8, 0.02, 0.6, 0.01, 0.4, 0.9],
    [0.5, 0.04, 0.3, 0.03, 0.7, 0.6],
    [0.9, 0.01, 0.8, 0.02, 0.3, 0.7],
    [0.4, 0.05, 0.5, 0.01, 0.8, 0.5]
], dtype=torch.float32)
mean_act = torch.mean(torch.abs(activations), dim=0) # [6]
dormant_mask = (mean_act < 0.15).to(torch.float32) # [6], indices 1 and 3 are dormant

# 4. Linear Anchor Invariant (2D [in_dim, out_dim])
anchor_x = torch.randn(2, 8, dtype=torch.float32)
anchor_w = torch.randn(8, 4, dtype=torch.float32)
anchor_b = torch.randn(4, dtype=torch.float32)
golden_anchor_out = anchor_x @ anchor_w + anchor_b

# Target save path
out_dir = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "02_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity.safetensors"

tensor_dict = {
    # Cross-track parity anchors
    "weight": anchor_w.clone().contiguous(),
    "bias": anchor_b.clone().contiguous(),
    "test_input": anchor_x.clone().contiguous(),
    "test_output": golden_anchor_out.clone().contiguous(),

    # Topic-specific anchors
    "anchor_x": anchor_x.clone().contiguous(),
    "anchor_w": anchor_w.clone().contiguous(),
    "anchor_b": anchor_b.clone().contiguous(),
    "golden_anchor_out": golden_anchor_out.clone().contiguous(),

    # MRL
    "z_emb": z_emb.clone().contiguous(),
    "mrl_w": mrl_w.clone().contiguous(),
    "mrl_b": mrl_b.clone().contiguous(),
    "golden_logits_d2": golden_logits_d2.clone().contiguous(),
    "golden_logits_d4": golden_logits_d4.clone().contiguous(),
    "golden_logits_d8": golden_logits_d8.clone().contiguous(),

    # Data Pruning: EL2N & RHO-Loss
    "p_probs": p_probs.clone().contiguous(),
    "y_targets": y_targets.clone().contiguous(),
    "golden_el2n": golden_el2n.clone().contiguous(),
    "loss_model": loss_model.clone().contiguous(),
    "loss_ref": loss_ref.clone().contiguous(),
    "golden_rho_loss": golden_rho_loss.clone().contiguous(),

    # ReDo Plasticity
    "activations": activations.clone().contiguous(),
    "mean_act": mean_act.clone().contiguous(),
    "dormant_mask": dormant_mask.clone().contiguous(),
}

save_file(tensor_dict, str(out_file))
print(f"Successfully generated golden checkpoint at {out_file}")
