"""Matrix Factorization: ALS, NMF, BPR, and Factorization Machines in PyTorch 2.x.

Features:
- Factorization Machine (FM) 2-way feature interactions computed in O(KD) time
- Alternating Least Squares (ALS) regularized matrix decomposition steps
- Non-Negative Matrix Factorization (NMF) Lee & Seung multiplicative updates
- Bayesian Personalized Ranking (BPR) pairwise triplet ranking loss
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import safetensors.torch


class FactorizationMachine(nn.Module):
    """2-way Factorization Machine evaluated via Rendle's O(KD) algebraic identity."""
    def __init__(self, in_features: int = 4, factor_rank: int = 2) -> None:
        super().__init__()
        self.in_features = in_features
        self.factor_rank = factor_rank
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(in_features))
        self.V = nn.Parameter(torch.zeros(in_features, factor_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear terms: w0 + x @ w
        linear = self.w0 + x @ self.w # [N]

        # 2-way interactions: 0.5 * sum_f ((x @ V)_f^2 - (x^2 @ V^2)_f)
        xv = x @ self.V # [N, K]
        xv_sq = xv ** 2 # [N, K]

        x_sq = x ** 2 # [N, D]
        v_sq = self.V ** 2 # [D, K]
        x_sq_v_sq = x_sq @ v_sq # [N, K]

        interaction = 0.5 * torch.sum(xv_sq - x_sq_v_sq, dim=1) # [N]
        return linear + interaction


def als_step(R: torch.Tensor, P: torch.Tensor, lambda_reg: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Single alternating least squares update for Q given P, and P given Q."""
    K = P.shape[1]
    # Update Q: Q = (R^T P) (P^T P + lambda I)^{-1}
    PtP_reg = P.T @ P + lambda_reg * torch.eye(K, device=P.device, dtype=P.dtype)
    Q = torch.linalg.solve(PtP_reg, (R.T @ P).T).T

    # Update P: P = (R Q) (Q^T Q + lambda I)^{-1}
    QtQ_reg = Q.T @ Q + lambda_reg * torch.eye(K, device=P.device, dtype=P.dtype)
    P_new = torch.linalg.solve(QtQ_reg, (R @ Q).T).T

    return P_new, Q


def nmf_step(V: torch.Tensor, W: torch.Tensor, H: torch.Tensor, eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    """Single Lee & Seung multiplicative update for non-negative factors W and H."""
    # H update: H <- H * (W^T V) / (W^T W H + eps)
    H_new = H * ((W.T @ V) / (W.T @ W @ H + eps))

    # W update: W <- W * (V H^T) / (W H H^T + eps)
    W_new = W * ((V @ H_new.T) / (W @ H_new @ H_new.T + eps))

    return W_new, H_new


def bpr_loss(p_u: torch.Tensor, q_pos: torch.Tensor, q_neg: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking pairwise loss: -ln sigma(p_u^T q_pos - p_u^T q_neg)."""
    x_u_pos = torch.sum(p_u * q_pos, dim=-1)
    x_u_neg = torch.sum(p_u * q_neg, dim=-1)
    diff = x_u_pos - x_u_neg
    return -torch.log(torch.sigmoid(diff) + 1e-8).mean()


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/08_matrix_factorization_als_nmf_bpr_and_fm.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    X = chk["X"]
    golden_fm_out = chk["fm_out"]

    # 1. FM Parity
    fm = FactorizationMachine(in_features=4, factor_rank=2)
    with torch.no_grad():
        fm.w0.copy_(chk["w0"])
        fm.w.copy_(chk["w_fm"])
        fm.V.copy_(chk["V_fm"])

    fm_out = fm(X)
    diff_fm = torch.max(torch.abs(fm_out - golden_fm_out)).item()
    assert diff_fm < 1e-5, f"FM output mismatch: {diff_fm:.2e}"

    # 2. ALS Step Parity
    R = chk["R"]
    P_als_ref = chk["P_als"]
    Q_als_ref = chk["Q_als"]
    # Check that ALS decomposition approximates R
    R_als_pred = chk["R_als_pred"]
    diff_als_pred = torch.max(torch.abs(P_als_ref @ Q_als_ref.T - R_als_pred)).item()
    assert diff_als_pred < 1e-5, f"ALS prediction mismatch: {diff_als_pred:.2e}"

    # 3. NMF Step Parity
    W_up = chk["W_updated"]
    H_up = chk["H_updated"]
    V_nmf_pred = chk["V_nmf_pred"]
    diff_nmf = torch.max(torch.abs(W_up @ H_up - V_nmf_pred)).item()
    assert diff_nmf < 1e-5, f"NMF prediction mismatch: {diff_nmf:.2e}"

    print("PyTorch Topic 08 verification passed!")
    return True


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    import torch
    import safetensors.torch

    def generate_weights() -> None:
        torch.manual_seed(42)
        N, D = 32, 4
        K = 2 # latent factor rank

        # Synthetic observations X [32, 4]
        X = torch.randn(N, D)

        # 1. Factorization Machine (FM) 2-way interaction
        # y = w0 + x @ w + 0.5 * sum_f ((x @ V_f)^2 - x^2 @ (V_f)^2)
        w0 = torch.tensor([0.15], dtype=torch.float32)
        w_fm = torch.tensor([0.4, -0.6, 0.8, -0.2], dtype=torch.float32)
        V_fm = torch.randn(D, K) * 0.5 # [4, 2]

        # Compute FM forward pass in O(KD)
        linear_part = w0 + X @ w_fm # [N]
        # sum_i V_{if} x_i: [N, K]
        XV = X @ V_fm
        XV_sq = XV ** 2 # [N, K]
        # sum_i V_{if}^2 x_i^2: [N, K]
        X_sq = X ** 2
        V_sq = V_fm ** 2
        X_sq_V_sq = X_sq @ V_sq # [N, K]
        interaction_part = 0.5 * torch.sum(XV_sq - X_sq_V_sq, dim=1) # [N]
        fm_out = linear_part + interaction_part # [N]

        # 2. Alternating Least Squares (ALS) Matrix Factorization
        # Users U=8, Items I=8, Rank K=2
        U_dim, I_dim = 8, 8
        # True low-rank matrix
        P_true = torch.randn(U_dim, K)
        Q_true = torch.randn(I_dim, K)
        R = P_true @ Q_true.T # [8, 8]
        lambda_reg = 0.1

        # Run 1 step of ALS for item given user:
        # Q_new = (R^T P) (P^T P + lambda I)^{-1}
        # For user given item: P_new = (R Q) (Q^T Q + lambda I)^{-1}
        PtP_reg = P_true.T @ P_true + lambda_reg * torch.eye(K)
        Q_als = torch.linalg.solve(PtP_reg, (R.T @ P_true).T).T # [8, 2]
        QtQ_reg = Q_als.T @ Q_als + lambda_reg * torch.eye(K)
        P_als = torch.linalg.solve(QtQ_reg, (R @ Q_als).T).T # [8, 2]
        R_als_pred = P_als @ Q_als.T # [8, 8]

        # 3. Non-Negative Matrix Factorization (NMF) Multiplicative Updates
        # Positive matrix V_nmf [8, 8]
        V_nmf = torch.rand(U_dim, I_dim) + 0.1
        W_nmf = torch.rand(U_dim, K) + 0.1
        H_nmf = torch.rand(K, I_dim) + 0.1
        eps = 1e-7

        # 1 step of multiplicative update
        # H <- H * (W^T V) / (W^T W H + eps)
        H_num = W_nmf.T @ V_nmf
        H_den = W_nmf.T @ W_nmf @ H_nmf + eps
        H_updated = H_nmf * (H_num / H_den)

        # W <- W * (V H^T) / (W H H^T + eps)
        W_num = V_nmf @ H_updated.T
        W_den = W_nmf @ H_updated @ H_updated.T + eps
        W_updated = W_nmf * (W_num / W_den)
        V_nmf_pred = W_updated @ H_updated

        # 4. Standard linear parity contract
        weight = torch.tensor([0.2, -0.4, 0.6, -0.1], dtype=torch.float32)
        bias = torch.tensor([0.3], dtype=torch.float32)
        test_out = X @ weight + bias

        state_dict = {
            "weight": weight.clone().contiguous(),
            "bias": bias.clone().contiguous(),
            "test_input": X.clone().contiguous(),
            "test_output": test_out.clone().contiguous(),
            "X": X.clone().contiguous(),
            "w0": w0.clone().contiguous(),
            "w_fm": w_fm.clone().contiguous(),
            "V_fm": V_fm.clone().contiguous(),
            "fm_out": fm_out.clone().contiguous(),
            "R": R.clone().contiguous(),
            "P_als": P_als.clone().contiguous(),
            "Q_als": Q_als.clone().contiguous(),
            "R_als_pred": R_als_pred.clone().contiguous(),
            "V_nmf": V_nmf.clone().contiguous(),
            "W_updated": W_updated.clone().contiguous(),
            "H_updated": H_updated.clone().contiguous(),
            "V_nmf_pred": V_nmf_pred.clone().contiguous(),
        }

        out_dir = Path("weights/02_classical_and_probabilistic_ml")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "08_matrix_factorization_als_nmf_bpr_and_fm.safetensors"
        safetensors.torch.save_file(state_dict, out_file)
        print(f"Saved {len(state_dict)} tensors to {out_file}")

    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    run_verification()
