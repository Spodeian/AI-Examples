"""Generate golden weights and reference outputs for Topic 08:
08_matrix_factorization_als_nmf_bpr_and_fm
"""

from __future__ import annotations
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

if __name__ == "__main__":
    generate_weights()
