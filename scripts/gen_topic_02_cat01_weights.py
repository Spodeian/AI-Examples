"""Generate golden weights and reference I/O tensors for Topic 02: Tree vs Kernel vs Neural.

Inductive Biases:
- Axis-aligned partitioning (Soft Decision Tree)
- Local isotropic smoothness (RBF Kernel Ridge)
- Adaptive continuous representation learning (MLP)
- Linear baseline (Canonical test_input / test_output anchor for cross-track test harness)
"""

from pathlib import Path

import torch
import torch.nn.functional as F

import core


def generate_topic_02_cat01_weights(output_path: Path) -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    in_dim = 4
    out_dim = 2
    num_leaves = 4  # Depth 2: 3 internal nodes (0, 1, 2), 4 leaves (0, 1, 2, 3)
    num_support = 8
    hidden_dim = 8

    # 1. Test input batch
    x = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)

    # 2. Canonical linear reference anchor (for core.parity.load_reference_checkpoint)
    weight = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)
    test_output = torch.matmul(x, weight) + bias

    # 3. Soft Decision Tree Parameters
    # Routing projection: [3, in_dim] where rows are one-hot selecting features [0, 1, 0]
    tree_routing_weights = torch.zeros(3, in_dim, dtype=torch.float32)
    tree_routing_weights[0, 0] = 1.0  # Node 0 splits on x[0]
    tree_routing_weights[1, 1] = 1.0  # Node 1 splits on x[1]
    tree_routing_weights[2, 0] = 1.0  # Node 2 splits on x[0]

    tree_thresholds = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    tree_temperature = torch.tensor([2.0], dtype=torch.float32)
    tree_leaf_weights = torch.randn(num_leaves, out_dim, generator=gen, dtype=torch.float32)

    # Compute Soft Decision Tree forward pass
    routing_logits = tree_temperature * (x @ tree_routing_weights.T - tree_thresholds)  # [B, 3]
    p_right = torch.sigmoid(routing_logits)  # [B, 3]
    p_left = 1.0 - p_right

    # Leaf probabilities:
    # Leaf 0: left at node 0, left at node 1
    # Leaf 1: left at node 0, right at node 1
    # Leaf 2: right at node 0, left at node 2
    # Leaf 3: right at node 0, right at node 2
    mu0 = p_left[:, 0:1] * p_left[:, 1:2]
    mu1 = p_left[:, 0:1] * p_right[:, 1:2]
    mu2 = p_right[:, 0:1] * p_left[:, 2:3]
    mu3 = p_right[:, 0:1] * p_right[:, 2:3]
    leaf_probs = torch.cat([mu0, mu1, mu2, mu3], dim=-1)  # [B, 4]

    tree_output = leaf_probs @ tree_leaf_weights  # [B, 2]

    # 4. RBF Kernel Ridge Parameters
    kernel_support = torch.randn(num_support, in_dim, generator=gen, dtype=torch.float32)
    kernel_dual_coeffs = torch.randn(num_support, out_dim, generator=gen, dtype=torch.float32)
    kernel_gamma = torch.tensor([0.5], dtype=torch.float32)
    kernel_bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)

    # Compute RBF Kernel forward pass
    x_sq = torch.sum(x**2, dim=-1, keepdim=True)  # [B, 1]
    s_sq = torch.sum(kernel_support**2, dim=-1, keepdim=True).T  # [1, M]
    dists_sq = torch.clamp(x_sq + s_sq - 2.0 * (x @ kernel_support.T), min=0.0)  # [B, M]
    rbf_sim = torch.exp(-kernel_gamma * dists_sq)  # [B, M]
    kernel_output = (rbf_sim @ kernel_dual_coeffs) + kernel_bias  # [B, 2]

    # 5. Neural MLP Parameters
    mlp_w1 = (
        torch.randn(in_dim, hidden_dim, generator=gen, dtype=torch.float32)
        * (2.0 / in_dim) ** 0.5
    )
    mlp_b1 = torch.zeros(hidden_dim, dtype=torch.float32)
    mlp_w2 = (
        torch.randn(hidden_dim, out_dim, generator=gen, dtype=torch.float32)
        * (2.0 / hidden_dim) ** 0.5
    )
    mlp_b2 = torch.zeros(out_dim, dtype=torch.float32)

    # Compute MLP forward pass
    h = F.relu(x @ mlp_w1 + mlp_b1)
    mlp_output = h @ mlp_w2 + mlp_b2

    # Assemble weights dict
    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "tree_routing_weights": tree_routing_weights.detach(),
        "tree_thresholds": tree_thresholds.detach(),
        "tree_temperature": tree_temperature.detach(),
        "tree_leaf_weights": tree_leaf_weights.detach(),
        "tree_output": tree_output.detach(),
        "kernel_support": kernel_support.detach(),
        "kernel_dual_coeffs": kernel_dual_coeffs.detach(),
        "kernel_gamma": kernel_gamma.detach(),
        "kernel_bias": kernel_bias.detach(),
        "kernel_output": kernel_output.detach(),
        "mlp_w1": mlp_w1.detach(),
        "mlp_b1": mlp_b1.detach(),
        "mlp_w2": mlp_w2.detach(),
        "mlp_b2": mlp_b2.detach(),
        "mlp_output": mlp_output.detach(),
    }

    core.parity.save_reference_checkpoint(
        path=output_path,
        weights_dict=weights_dict,
        input_tensor=x,
        output_tensor=test_output,
        metadata={
            "topic": "02_tree_vs_kernel_vs_neural_inductive_biases",
            "category": "01_paradigms_and_dichotomies",
            "batch_size": str(batch_size),
            "in_dim": str(in_dim),
            "out_dim": str(out_dim),
        },
    )

    print(f"Generated Topic 02 Category 01 weights at: {output_path}")
    print(f"Input shape: {x.shape}")
    print(f"Linear output shape: {test_output.shape}")
    print(f"Tree output shape: {tree_output.shape}")
    print(f"Kernel output shape: {kernel_output.shape}")
    print(f"MLP output shape: {mlp_output.shape}")


if __name__ == "__main__":
    out_file = (
        Path(__file__).resolve().parent.parent
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "02_tree_vs_kernel_vs_neural_inductive_biases.safetensors"
    )
    generate_topic_02_cat01_weights(out_file)
