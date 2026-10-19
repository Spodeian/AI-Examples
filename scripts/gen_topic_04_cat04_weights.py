"""Generate reference weights and golden test vectors for Category 04 Topic 04:
Orthogonal and Preconditioned Matrix Optimisers (Muon, Shampoo, GaLore).
"""

from pathlib import Path
import torch
from safetensors.torch import save_file

import sys
repo_root = Path.cwd().resolve()
while not (repo_root / "core").exists() and repo_root.parent != repo_root:
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import core


def newton_schulz_quintic(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton-Schulz iteration (Jordan et al. 2024) for orthogonalizing a 2D matrix.
    Computes an approximate polar decomposition O = UV^T via quintic polynomial iterations.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    m, n = G.shape
    X = G / (torch.norm(G) + eps)
    if m >= n:
        X = X.t()

    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if m >= n:
        X = X.t()
    return X


def step_muon(
    param: torch.Tensor,
    grad: torch.Tensor,
    m_buf: torch.Tensor,
    lr: float = 0.02,
    momentum: float = 0.95,
    nesterov: bool = True,
    weight_decay: float = 0.01,
    ns_steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Muon optimizer step: Momentum Orthogonalized by Newton-Schulz."""
    new_m = momentum * m_buf + grad
    if nesterov:
        update_cand = momentum * new_m + grad
    else:
        update_cand = new_m

    ortho_update = newton_schulz_quintic(update_cand, steps=ns_steps)
    shrunk_p = param * (1.0 - lr * weight_decay)
    new_p = shrunk_p - lr * ortho_update
    return new_p, new_m


def matrix_power_neg_fourth(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes M^(-1/4) via symmetric eigendecomposition."""
    eigvals, eigvecs = torch.linalg.eigh(M)
    inv_fourth = torch.pow(torch.clamp(eigvals, min=eps), -0.25)
    return eigvecs @ torch.diag(inv_fourth) @ eigvecs.t()


def step_shampoo(
    param: torch.Tensor,
    grad: torch.Tensor,
    l_cov: torch.Tensor,
    r_cov: torch.Tensor,
    lr: float = 0.005,
    beta2: float = 0.95,
    eps: float = 1e-6,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shampoo optimizer step (Gupta et al. 2018): Kronecker-factored 2D curvature preconditioning."""
    new_l = beta2 * l_cov + (1.0 - beta2) * (grad @ grad.t())
    new_r = beta2 * r_cov + (1.0 - beta2) * (grad.t() @ grad)

    l_inv = matrix_power_neg_fourth(new_l, eps=eps)
    r_inv = matrix_power_neg_fourth(new_r, eps=eps)

    precond_grad = l_inv @ grad @ r_inv
    shrunk_p = param * (1.0 - lr * weight_decay)
    new_p = shrunk_p - lr * precond_grad
    return new_p, new_l, new_r


def step_galore(
    param: torch.Tensor,
    grad: torch.Tensor,
    m_low: torch.Tensor,
    v_low: torch.Tensor,
    proj_matrix: torch.Tensor,
    step: int = 2,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GaLore optimizer step (Zhao et al. 2024): Low-rank gradient projection."""
    # grad: [m, n], proj_matrix: [m, r] (where r < m, n)
    grad_low = proj_matrix.t() @ grad  # [r, n]
    new_m = beta1 * m_low + (1.0 - beta1) * grad_low
    new_v = beta2 * v_low + (1.0 - beta2) * (grad_low * grad_low)

    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    m_hat = new_m / bc1
    v_hat = new_v / bc2

    adam_step = m_hat / (torch.sqrt(v_hat) + eps)
    full_update = proj_matrix @ adam_step  # [m, n]

    shrunk_p = param * (1.0 - lr * weight_decay)
    new_p = shrunk_p - lr * full_update
    return new_p, new_m, new_v


def main() -> None:
    core.setup_env(seed=42)
    device = torch.device("cpu")

    dim = 8
    # 1. Linear Anchor (y = x @ w + b)
    gen = torch.Generator().manual_seed(42)
    weight = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(dim, generator=gen, dtype=torch.float32)
    test_input = torch.randn(4, dim, generator=gen, dtype=torch.float32)
    test_output = test_input @ weight + bias

    # 2. 2D Parameter & Gradient Matrices for Matrix Optimisers (shape: [4, 4])
    m, n = 4, 4
    gen_opt = torch.Generator().manual_seed(1337)
    param_2d = torch.randn(m, n, generator=gen_opt, dtype=torch.float32)
    grad_2d = torch.randn(m, n, generator=gen_opt, dtype=torch.float32)

    # 3. Muon Golden Step
    m_init_muon = torch.randn(m, n, generator=gen_opt, dtype=torch.float32) * 0.1
    p_muon, m_muon = step_muon(
        param_2d, grad_2d, m_init_muon, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.01, ns_steps=5
    )

    # 4. Shampoo Golden Step
    l_init = torch.eye(m, dtype=torch.float32) * 0.1
    r_init = torch.eye(n, dtype=torch.float32) * 0.1
    p_shampoo, l_shampoo, r_shampoo = step_shampoo(
        param_2d, grad_2d, l_init, r_init, lr=0.005, beta2=0.95, eps=1e-6, weight_decay=0.01
    )

    # 5. GaLore Golden Step (rank r=2, m=4, n=4)
    r = 2
    # Semi-orthogonal projection matrix P: [m, r], P^T P = I_r
    proj_matrix = torch.zeros(m, r, dtype=torch.float32)
    proj_matrix[0, 0] = 1.0
    proj_matrix[1, 1] = 1.0
    m_init_galore = torch.randn(r, n, generator=gen_opt, dtype=torch.float32) * 0.05
    v_init_galore = torch.rand(r, n, generator=gen_opt, dtype=torch.float32) * 0.05 + 0.01

    p_galore, m_galore, v_galore = step_galore(
        param_2d, grad_2d, m_init_galore, v_init_galore, proj_matrix,
        step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
    )

    tensors = {
        # Linear Anchor
        "weight": weight,
        "bias": bias,
        "test_input": test_input,
        "test_output": test_output,
        # Shared 2D parameters
        "param_2d": param_2d,
        "grad_2d": grad_2d,
        # Muon
        "m_init_muon": m_init_muon,
        "p_muon": p_muon,
        "m_muon": m_muon,
        # Shampoo
        "l_init": l_init,
        "r_init": r_init,
        "p_shampoo": p_shampoo,
        "l_shampoo": l_shampoo,
        "r_shampoo": r_shampoo,
        # GaLore
        "proj_matrix": proj_matrix,
        "m_init_galore": m_init_galore,
        "v_init_galore": v_init_galore,
        "p_galore": p_galore,
        "m_galore": m_galore,
        "v_galore": v_galore,
    }

    out_dir = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "04_orthogonal_and_preconditioned_matrix_optimisers.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {len(tensors)} tensors to {out_path}")


if __name__ == "__main__":
    main()
