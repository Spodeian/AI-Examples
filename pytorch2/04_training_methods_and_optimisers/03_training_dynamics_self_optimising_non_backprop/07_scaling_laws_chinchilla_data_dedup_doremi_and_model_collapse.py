"""PyTorch 2.x implementation for Category 04 Topic 21:
Scaling Laws, Data Deduplication (MinHash), DoReMi, and Model Collapse.

Provides production-grade implementations of:
1. Chinchilla parametric scaling laws and compute-optimal parameter/data allocation (Hoffmann et al., 2022).
2. DoReMi domain weight optimization via excess loss exponentiated descent (Xie et al., 2023).
3. MinHash Jaccard similarity estimation for scalable deduplication (Broder, 1997).
4. Model collapse recursive variance tracking and real-data accumulation cure (Shumailov et al., 2024).
5. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def compute_chinchilla_loss_and_opt(
    n: torch.Tensor,
    d: torch.Tensor,
    e_const: float = 1.69,
    a_const: float = 40.0,
    b_const: float = 50.0,
    alpha: float = 0.34,
    beta: float = 0.28,
    g_n: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes Chinchilla scaling loss and compute-optimal allocation (Hoffmann et al., 2022).

    n: [1], d: [1]
    Returns (loss, n_opt, d_opt).
    """
    loss = e_const + a_const * (n ** (-alpha)) + b_const * (d ** (-beta))

    exp_a = beta / (alpha + beta)
    exp_b = alpha / (alpha + beta)
    budget_c = 6.0 * n * d
    g_d = 1.0 / (6.0 * g_n)

    n_opt = g_n * (budget_c ** exp_a)
    d_opt = g_d * (budget_c ** exp_b)
    return loss, n_opt, d_opt


def doremi_update_step(
    w_init: torch.Tensor,
    losses_model: torch.Tensor,
    losses_ref: torch.Tensor,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DoReMi domain weight exponentiated gradient update (Xie et al., 2023).

    w_init: [K], losses_model: [K], losses_ref: [K]
    Returns (excess_loss, w_next).
    """
    excess_loss = losses_model - losses_ref
    log_w = torch.log(w_init) + eta * excess_loss
    w_next = F.softmax(log_w, dim=-1)
    return excess_loss, w_next


def minhash_pairwise_jaccard(
    signatures: torch.Tensor,
) -> torch.Tensor:
    """Pairwise Jaccard similarity estimation from MinHash signatures (Broder, 1997).

    signatures: [B, K] where K is number of hash functions.
    Returns jaccard_matrix: [B, B].
    """
    b, k = signatures.shape
    # Broadcast comparison: [B, 1, K] == [1, B, K] => [B, B, K]
    matches = (signatures.unsqueeze(1) == signatures.unsqueeze(0)).float()
    jaccard_matrix = torch.mean(matches, dim=-1)
    return jaccard_matrix


def model_collapse_variance_cure(
    var_synth: torch.Tensor,
    var_real: torch.Tensor,
    lambda_real: float = 0.25,
) -> torch.Tensor:
    """Computes mixture variance under real data accumulation cure (Shumailov et al., 2024).

    Returns cured_var.
    """
    return (1.0 - lambda_real) * var_synth + lambda_real * var_real


class LinearAnchor(nn.Module):
    """Anchor module for multi-track parity verification."""

    def __init__(self, in_features: int = 8, out_features: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ScalingLawsUnifiedModule:
    """Unified container for Category 04 Topic 21 verification."""

    def __init__(self, weights_path: Path):
        weights = load_file(str(weights_path))

        # Chinchilla
        self.n_val = weights["n_val"]
        self.d_val = weights["d_val"]
        self.golden_chinchilla_loss = weights["golden_chinchilla_loss"]
        self.golden_n_opt = weights["golden_n_opt"]
        self.golden_d_opt = weights["golden_d_opt"]

        # DoReMi
        self.w_doremi_init = weights["w_doremi_init"]
        self.losses_model = weights["losses_model"]
        self.losses_ref = weights["losses_ref"]
        self.golden_excess_loss = weights["golden_excess_loss"]
        self.golden_w_doremi_next = weights["golden_w_doremi_next"]

        # MinHash
        self.signatures = weights["signatures"]
        self.golden_jaccard_matrix = weights["golden_jaccard_matrix"]

        # Model collapse
        self.var_synth = weights["var_synth"]
        self.var_real = weights["var_real"]
        self.golden_cured_var = weights["golden_cured_var"]

        # Linear Anchor
        self.anchor = LinearAnchor(8, 4)
        self.anchor.fc.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.fc.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. Chinchilla invariant
        loss, n_opt, d_opt = compute_chinchilla_loss_and_opt(self.n_val, self.d_val)
        err_l = float(torch.abs(loss - self.golden_chinchilla_loss).item())
        err_n = float(torch.abs(n_opt - self.golden_n_opt).item())
        err_d = float(torch.abs(d_opt - self.golden_d_opt).item())
        err_chinchilla = max(err_l, err_n, err_d)
        assert err_chinchilla < atol, f"Chinchilla invariant failed: {err_chinchilla:.2e}"
        errors["chinchilla_err"] = err_chinchilla

        # 2. DoReMi invariant
        excess_l, w_next = doremi_update_step(self.w_doremi_init, self.losses_model, self.losses_ref, eta=1.0)
        err_ex = float(torch.max(torch.abs(excess_l - self.golden_excess_loss)).item())
        err_wn = float(torch.max(torch.abs(w_next - self.golden_w_doremi_next)).item())
        err_doremi = max(err_ex, err_wn)
        assert err_doremi < atol, f"DoReMi invariant failed: {err_doremi:.2e}"
        errors["doremi_err"] = err_doremi

        # 3. MinHash invariant
        jaccard_m = minhash_pairwise_jaccard(self.signatures)
        err_minhash = float(torch.max(torch.abs(jaccard_m - self.golden_jaccard_matrix)).item())
        assert err_minhash < atol, f"MinHash invariant failed: {err_minhash:.2e}"
        errors["minhash_err"] = err_minhash

        # 4. Model collapse invariant
        cured_v = model_collapse_variance_cure(self.var_synth, self.var_real, lambda_real=0.25)
        err_collapse = float(torch.abs(cured_v - self.golden_cured_var).item())
        assert err_collapse < atol, f"Model Collapse invariant failed: {err_collapse:.2e}"
        errors["collapse_err"] = err_collapse

        # 5. Linear Anchor invariant
        anchor_out = self.anchor(self.anchor_x)
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import math
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import save_file

    def generate_topic_21_artifacts():
        torch.manual_seed(42)
        np.random.seed(42)

        # 1. Chinchilla Scaling Law
        # L(N, D) = E + A / N^alpha + B / D^beta
        # Using Hoffmann et al. (2022) empirical coefficients
        # Normalized for numerical stability: N in millions, D in billions
        # Or in float64 / float32 scaled units:
        # Let N_val = 12.0 (e.g. 12M), D_val = 240.0 (e.g. 240M tokens)
        # E = 1.69, A = 406.4, B = 410.7, alpha = 0.34, beta = 0.28
        # For clean float32 operations:
        e_const = 1.69
        a_const = 40.0
        b_const = 50.0
        alpha = 0.34
        beta = 0.28

        n_val = torch.tensor([16.0], dtype=torch.float32) # e.g. 16M params
        d_val = torch.tensor([320.0], dtype=torch.float32) # e.g. 320M tokens

        # Loss = E + A * N^(-alpha) + B * D^(-beta)
        chinchilla_loss = e_const + a_const * (n_val ** (-alpha)) + b_const * (d_val ** (-beta))

        # Compute optimal allocation ratio given exponent a = beta / (alpha + beta), b = alpha / (alpha + beta)
        exp_a = beta / (alpha + beta)
        exp_b = alpha / (alpha + beta)
        budget_c = 6.0 * n_val * d_val # Compute FLOPs proxy
        # Optimal N* and D*
        # Scaling constant G_N = 0.6, G_D = 1/ (6 * G_N)
        g_n = 0.6
        g_d = 1.0 / (6.0 * g_n)
        n_opt = g_n * (budget_c ** exp_a)
        d_opt = g_d * (budget_c ** exp_b)

        # 2. DoReMi Domain Weight Optimization (Xie et al., 2023)
        # K = 4 domains
        w_doremi_init = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
        losses_model = torch.tensor([2.4, 3.1, 1.8, 2.9], dtype=torch.float32)
        losses_ref = torch.tensor([2.2, 2.5, 1.9, 2.6], dtype=torch.float32)
        eta_doremi = 1.0

        excess_loss = losses_model - losses_ref # [0.2, 0.6, -0.1, 0.3]
        # w_next = softmax(log(w_init) + eta * excess_loss)
        log_w = torch.log(w_doremi_init) + eta_doremi * excess_loss
        w_doremi_next = F.softmax(log_w, dim=-1)

        # 3. MinHash Jaccard Similarity (Broder, 1997)
        # 4 documents, signature size = 8
        # Signatures as float32 representation of hash indices
        signatures = torch.tensor([
            [1.0, 5.0, 9.0, 12.0, 3.0, 7.0, 15.0, 2.0],
            [1.0, 5.0, 9.0, 14.0, 3.0, 8.0, 15.0, 2.0], # 6 matches with doc 0 => 6/8 = 0.75
            [4.0, 6.0, 10.0, 14.0, 3.0, 8.0, 16.0, 2.0], # 3 matches with doc 1 => 3/8 = 0.375
            [1.0, 5.0, 9.0, 12.0, 3.0, 7.0, 15.0, 2.0], # exact duplicate of doc 0 => 8/8 = 1.0
        ], dtype=torch.float32)

        # Compute pairwise Jaccard matrix: [4, 4]
        # J_ij = 1/K sum_k (sig[i, k] == sig[j, k])
        b_docs = signatures.shape[0]
        k_hashes = signatures.shape[1]
        jaccard_matrix = torch.zeros(b_docs, b_docs, dtype=torch.float32)
        for i in range(b_docs):
            for j in range(b_docs):
                matches = torch.sum((signatures[i] == signatures[j]).float())
                jaccard_matrix[i, j] = matches / float(k_hashes)

        # 4. Model Collapse Variance Dynamics (Shumailov et al., 2024)
        # sigma_sq_init = 1.0
        # recursive generation without accumulation: variance decay per step by factor (M-1)/M
        # With accumulation of fraction lambda_real = 0.2:
        # sigma_sq_next = (1 - lambda_real) * (var_synthetic) + lambda_real * var_real
        var_synth = torch.tensor([0.85], dtype=torch.float32)
        var_real = torch.tensor([1.0], dtype=torch.float32)
        lambda_real = 0.25
        golden_cured_var = (1.0 - lambda_real) * var_synth + lambda_real * var_real

        # 5. Linear Anchor
        anchor_x = torch.randn(2, 8, dtype=torch.float32)
        anchor_w = torch.randn(8, 4, dtype=torch.float32) / math.sqrt(8)
        anchor_b = torch.randn(4, dtype=torch.float32) * 0.1
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

        # Construct tensors dictionary
        tensors = {
            # Chinchilla
            "n_val": n_val.clone().contiguous(),
            "d_val": d_val.clone().contiguous(),
            "golden_chinchilla_loss": chinchilla_loss.clone().contiguous(),
            "golden_n_opt": n_opt.clone().contiguous(),
            "golden_d_opt": d_opt.clone().contiguous(),

            # DoReMi
            "w_doremi_init": w_doremi_init.clone().contiguous(),
            "losses_model": losses_model.clone().contiguous(),
            "losses_ref": losses_ref.clone().contiguous(),
            "golden_excess_loss": excess_loss.clone().contiguous(),
            "golden_w_doremi_next": w_doremi_next.clone().contiguous(),

            # MinHash
            "signatures": signatures.clone().contiguous(),
            "golden_jaccard_matrix": jaccard_matrix.clone().contiguous(),

            # Model collapse
            "var_synth": var_synth.clone().contiguous(),
            "var_real": var_real.clone().contiguous(),
            "golden_cured_var": golden_cured_var.clone().contiguous(),

            # Linear Anchor
            "anchor_x": anchor_x.clone().contiguous(),
            "anchor_w": anchor_w.clone().contiguous(),
            "anchor_b": anchor_b.clone().contiguous(),
            "golden_anchor_out": golden_anchor_out.clone().contiguous(),

            # Standard test inputs/outputs for parity test runners
            "test_input": anchor_x.clone().contiguous(),
            "test_output": golden_anchor_out.clone().contiguous(),
            "weight": anchor_w.clone().contiguous(),
            "bias": anchor_b.clone().contiguous(),
        }

        out_dir = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "07_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse.safetensors"

        save_file(tensors, out_path)
        print(f"Saved {len(tensors)} tensors to {out_path}")

        # Verify self-consistency
        print(f"Chinchilla Loss: {chinchilla_loss.item():.6f}, N_opt: {n_opt.item():.4f}, D_opt: {d_opt.item():.4f}")
        print(f"DoReMi Next: {w_doremi_next.numpy().tolist()}")
        print(f"Jaccard Pair (0, 1): {jaccard_matrix[0, 1].item():.4f}, Pair (0, 3): {jaccard_matrix[0, 3].item():.4f}")
        print(f"Cured Variance: {golden_cured_var.item():.4f}")

    generate_topic_21_artifacts()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/07_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse.safetensors")
    module = ScalingLawsUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Topic 21 Invariant Parity Checks Passed: {errs}")
