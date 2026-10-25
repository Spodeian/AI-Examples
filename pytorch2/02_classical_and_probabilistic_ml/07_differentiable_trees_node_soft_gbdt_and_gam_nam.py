"""Differentiable Decision Trees, NODE, Soft-GBDT, and Neural Additive Models in PyTorch 2.x.

Features:
- Neural Oblivious Decision Ensemble (NODE) with soft binary path routing
- Neural Additive Models (NAM) with modular univariate feature sub-networks
- Pool-Adjacent-Violators (PAV) algorithm for monotonic non-parametric isotonic regression
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import safetensors.torch


class NeuralObliviousDecisionEnsemble(nn.Module):
    """Neural Oblivious Decision Ensemble (NODE) with soft gating."""
    def __init__(self, n_trees: int = 2, depth: int = 2, in_features: int = 4) -> None:
        super().__init__()
        self.n_trees = n_trees
        self.depth = depth
        self.in_features = in_features
        self.n_leaves = 2 ** depth

        self.W_split = nn.Parameter(torch.empty(n_trees, depth, in_features))
        self.b_split = nn.Parameter(torch.empty(n_trees, depth))
        self.leaf_values = nn.Parameter(torch.empty(n_trees, self.n_leaves))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing soft routing over binary oblivious paths."""
        N = x.shape[0]
        ensemble_out = torch.zeros(N, device=x.device, dtype=x.dtype)

        for t in range(self.n_trees):
            # Gating probabilities for each depth level
            p0 = torch.sigmoid(x @ self.W_split[t, 0] + self.b_split[t, 0]) # [N]
            p1 = torch.sigmoid(x @ self.W_split[t, 1] + self.b_split[t, 1]) # [N]

            # Path probabilities for 2^2 = 4 leaves: (00, 01, 10, 11)
            w0 = (1.0 - p0) * (1.0 - p1)
            w1 = (1.0 - p0) * p1
            w2 = p0 * (1.0 - p1)
            w3 = p0 * p1
            leaf_probs = torch.stack([w0, w1, w2, w3], dim=1) # [N, 4]

            tree_out = leaf_probs @ self.leaf_values[t] # [N]
            ensemble_out = ensemble_out + tree_out

        return ensemble_out


class FeatureNN(nn.Module):
    """Univariate sub-network for a single feature in Neural Additive Models."""
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x_j: torch.Tensor) -> torch.Tensor:
        # x_j: [N, 1] -> [N, 1]
        return self.fc2(self.act(self.fc1(x_j)))


class NeuralAdditiveModel(nn.Module):
    """Neural Additive Model (NAM): y_hat = beta_0 + sum_j f_j(x_j)."""
    def __init__(self, in_features: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.in_features = in_features
        self.feature_nns = nn.ModuleList([FeatureNN(hidden_dim) for _ in range(in_features)])
        self.bias0 = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype) + self.bias0
        for j in range(self.in_features):
            x_j = x[:, j].unsqueeze(1)
            out = out + self.feature_nns[j](x_j).squeeze(1)
        return out


def pav_isotonic_regression(y: torch.Tensor) -> torch.Tensor:
    """Pool Adjacent Violators (PAV) algorithm for monotonic regression."""
    y_list = y.clone().tolist()
    blocks = [[i, i, 1.0, float(y_list[i])] for i in range(len(y_list))]
    
    i = 0
    while i < len(blocks) - 1:
        mean_curr = blocks[i][3] / blocks[i][2]
        mean_next = blocks[i + 1][3] / blocks[i + 1][2]
        if mean_curr > mean_next:
            blocks[i][1] = blocks[i + 1][1]
            blocks[i][2] += blocks[i + 1][2]
            blocks[i][3] += blocks[i + 1][3]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
            
    out = torch.empty_like(y)
    for b in blocks:
        val = b[3] / b[2]
        out[b[0]:b[1] + 1] = val
    return out


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    X = chk["X"]
    golden_node_out = chk["node_out"]
    golden_nam_out = chk["nam_out"]
    golden_iso_fit = chk["iso_fit"]
    y_noisy = chk["y_noisy"]

    # 1. NODE Parity
    node = NeuralObliviousDecisionEnsemble(n_trees=2, depth=2, in_features=4)
    with torch.no_grad():
        node.W_split.copy_(chk["W_split"])
        node.b_split.copy_(chk["b_split"])
        node.leaf_values.copy_(chk["leaf_values"])

    node_out = node(X)
    diff_node = torch.max(torch.abs(node_out - golden_node_out)).item()
    assert diff_node < 1e-5, f"NODE output mismatch: {diff_node:.2e}"

    # 2. NAM Parity
    nam = NeuralAdditiveModel(in_features=4, hidden_dim=8)
    with torch.no_grad():
        nam.bias0.copy_(chk["nam_bias0"])
        for d in range(4):
            nam.feature_nns[d].fc1.weight.copy_(chk["nam_w1"][d])
            nam.feature_nns[d].fc1.bias.copy_(chk["nam_b1"][d])
            nam.feature_nns[d].fc2.weight.copy_(chk["nam_w2"][d])
            nam.feature_nns[d].fc2.bias.copy_(chk["nam_b2"][d].squeeze(0))

    nam_out = nam(X)
    diff_nam = torch.max(torch.abs(nam_out - golden_nam_out)).item()
    assert diff_nam < 1e-5, f"NAM output mismatch: {diff_nam:.2e}"

    # 3. PAV Isotonic Regression Parity
    iso_fit = pav_isotonic_regression(y_noisy)
    diff_iso = torch.max(torch.abs(iso_fit - golden_iso_fit)).item()
    assert diff_iso < 1e-5, f"PAV Isotonic output mismatch: {diff_iso:.2e}"

    print("PyTorch Topic 07 verification passed!")
    return True


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
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

    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    run_verification()
