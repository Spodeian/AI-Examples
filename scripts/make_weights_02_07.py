"""Generate golden weights and reference outputs for Topic 07:
07_differentiable_trees_node_soft_gbdt_and_gam_nam
"""

from __future__ import annotations
from pathlib import Path
import torch
import safetensors.torch

def pav_isotonic_regression(y: torch.Tensor) -> torch.Tensor:
    """Pool Adjacent Violators (PAV) algorithm for 1D isotonic regression."""
    y_iso = y.clone().tolist()
    # Each block is [start_idx, end_idx, weight, sum_val]
    blocks = [[i, i, 1.0, float(y_iso[i])] for i in range(len(y_iso))]
    
    i = 0
    while i < len(blocks) - 1:
        mean_curr = blocks[i][3] / blocks[i][2]
        mean_next = blocks[i + 1][3] / blocks[i + 1][2]
        if mean_curr > mean_next:
            # Merge block i and block i+1
            blocks[i][1] = blocks[i + 1][1]
            blocks[i][2] += blocks[i + 1][2]
            blocks[i][3] += blocks[i + 1][3]
            blocks.pop(i + 1)
            # Step back if possible to check monotonicity with previous block
            if i > 0:
                i -= 1
        else:
            i += 1
            
    out = torch.empty_like(y)
    for b in blocks:
        val = b[3] / b[2]
        out[b[0]:b[1] + 1] = val
    return out

def generate_weights() -> None:
    torch.manual_seed(42)
    N, D = 32, 4
    n_trees = 2
    depth = 2
    n_leaves = 2 ** depth # 4
    hidden_nam = 8

    # Synthetic observations X [32, 4]
    X = torch.randn(N, D)

    # 1. NODE Parameters
    # W_split: [n_trees, depth, D], b_split: [n_trees, depth]
    W_split = torch.randn(n_trees, depth, D) * 0.5
    b_split = torch.randn(n_trees, depth) * 0.1
    # leaf_values: [n_trees, n_leaves]
    leaf_values = torch.randn(n_trees, n_leaves)

    # Compute NODE forward pass:
    # For each tree: gating logits [N, depth]
    # For depth 2:
    # p0 = sigmoid(X @ W[t, 0] + b[t, 0])
    # p1 = sigmoid(X @ W[t, 1] + b[t, 1])
    # leaf_weights: [ (1-p0)*(1-p1), (1-p0)*p1, p0*(1-p1), p0*p1 ]
    node_out = torch.zeros(N)
    for t in range(n_trees):
        p0 = torch.sigmoid(X @ W_split[t, 0] + b_split[t, 0]) # [N]
        p1 = torch.sigmoid(X @ W_split[t, 1] + b_split[t, 1]) # [N]
        w0 = (1.0 - p0) * (1.0 - p1)
        w1 = (1.0 - p0) * p1
        w2 = p0 * (1.0 - p1)
        w3 = p0 * p1
        leaves_prob = torch.stack([w0, w1, w2, w3], dim=1) # [N, 4]
        tree_out = leaves_prob @ leaf_values[t] # [N]
        node_out = node_out + tree_out

    # 2. NAM Parameters (Neural Additive Model)
    # D independent scalar MLPs: f_d(x_d) = W2_d @ relu(W1_d * x_d + b1_d) + b2_d
    nam_w1 = torch.randn(D, hidden_nam, 1) * 0.5
    nam_b1 = torch.randn(D, hidden_nam) * 0.1
    nam_w2 = torch.randn(D, 1, hidden_nam) * 0.5
    nam_b2 = torch.randn(D, 1) * 0.1
    nam_bias0 = torch.tensor([0.25], dtype=torch.float32)

    nam_out = torch.zeros(N) + nam_bias0
    for d in range(D):
        x_d = X[:, d].unsqueeze(1) # [N, 1]
        h = torch.relu(x_d @ nam_w1[d].T + nam_b1[d]) # [N, hidden]
        out_d = (h @ nam_w2[d].T + nam_b2[d]).squeeze(1) # [N]
        nam_out = nam_out + out_d

    # 3. PAV Isotonic Regression
    # Sort X[:, 0] and take y_raw as noisy monotonic curve
    sorted_idx = torch.argsort(X[:, 0])
    y_noisy = torch.linspace(0.0, 5.0, N) + 0.5 * torch.randn(N)
    iso_fit = pav_isotonic_regression(y_noisy)

    # 4. Standard linear parity contract
    weight = torch.tensor([0.3, -0.5, 0.7, -0.4], dtype=torch.float32)
    bias = torch.tensor([0.2], dtype=torch.float32)
    test_out = X @ weight + bias

    # State dict
    state_dict = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": X.clone().contiguous(),
        "test_output": test_out.clone().contiguous(),
        "X": X.clone().contiguous(),
        "W_split": W_split.clone().contiguous(),
        "b_split": b_split.clone().contiguous(),
        "leaf_values": leaf_values.clone().contiguous(),
        "node_out": node_out.clone().contiguous(),
        "nam_w1": nam_w1.clone().contiguous(),
        "nam_b1": nam_b1.clone().contiguous(),
        "nam_w2": nam_w2.clone().contiguous(),
        "nam_b2": nam_b2.clone().contiguous(),
        "nam_bias0": nam_bias0.clone().contiguous(),
        "nam_out": nam_out.clone().contiguous(),
        "y_noisy": y_noisy.clone().contiguous(),
        "iso_fit": iso_fit.clone().contiguous(),
    }

    out_dir = Path("weights/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "07_differentiable_trees_node_soft_gbdt_and_gam_nam.safetensors"
    safetensors.torch.save_file(state_dict, out_file)
    print(f"Saved {len(state_dict)} tensors to {out_file}")

if __name__ == "__main__":
    generate_weights()
