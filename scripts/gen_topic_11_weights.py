"""Generate deterministic reference safetensors checkpoint for Topic 11:
11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp
"""

from __future__ import annotations

from pathlib import Path

import torch

import core


def modified_gram_schmidt_qr(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes thin QR decomposition Y = Q R via Modified Gram-Schmidt with strictly positive diagonal R."""
    m, num_cols = y.shape
    q = y.clone()
    r = torch.zeros(num_cols, num_cols, dtype=y.dtype, device=y.device)

    for j in range(num_cols):
        v = q[:, j].clone()
        for i in range(j):
            rij = torch.dot(q[:, i], v)
            r[i, j] = rij
            v = v - rij * q[:, i]
        rjj = torch.linalg.norm(v)
        r[j, j] = rjj
        q[:, j] = v / (rjj + 1e-12)

    return q, r


def hutch_plus_plus_trace(
    a: torch.Tensor,
    s_sketch: torch.Tensor,
    g_mc: torch.Tensor,
) -> torch.Tensor:
    """Evaluates Hutch++ randomized trace estimator combining low-rank sketch with residual Monte Carlo."""
    # 1. Low-rank subspace sketch
    y = a @ s_sketch
    q, _ = modified_gram_schmidt_qr(y)

    # 2. Exact trace of projection on Q: tr(Q^T A Q)
    q_t_a_q = q.T @ a @ q
    tr_proj = torch.trace(q_t_a_q)

    # 3. Residual Monte Carlo on (I - Q Q^T) A
    a_g = a @ g_mc
    proj_g = q @ (q.T @ a_g)
    residual_g = a_g - proj_g

    # Tr(R) approx = (1/m) * sum_i g_i^T residual_g_i
    m_val = float(g_mc.shape[1])
    quad_forms = torch.sum(g_mc * residual_g, dim=0)
    tr_res = torch.sum(quad_forms) / m_val

    return tr_proj + tr_res


def lorentzian_broadened_f_matrix(
    singular_values: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Constructs skew-symmetric Lorentzian-broadened singular value coupling matrix F_ij.

    F_ij = (sigma_j - sigma_i) / ((sigma_j - sigma_i)^2 + epsilon^2) for i != j, 0 for i == j.
    Avoids 1 / (sigma_j - sigma_i) NaN blowup when singular values collide.
    """
    n_dim = singular_values.shape[0]
    s_i = singular_values.unsqueeze(1)
    s_j = singular_values.unsqueeze(0)
    delta = s_j - s_i

    f = delta / (delta**2 + (epsilon**2))
    f = f * (1.0 - torch.eye(n_dim, dtype=f.dtype, device=f.device))
    return f


def power_iteration_spectral_norm(
    w: torch.Tensor,
    v_init: torch.Tensor,
    num_iters: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluates dominant singular value (spectral norm) and singular vectors via Power Iteration."""
    v = v_init / (torch.linalg.norm(v_init) + 1e-12)
    u = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)

    for _ in range(num_iters):
        u_unnorm = w @ v
        u = u_unnorm / (torch.linalg.norm(u_unnorm) + 1e-12)
        v_unnorm = w.T @ u
        v = v_unnorm / (torch.linalg.norm(v_unnorm) + 1e-12)

    sigma = torch.dot(u, w @ v)
    return sigma, u, v


def generate_topic_11_weights() -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    in_dim = 6
    out_dim = 4

    # 1. Deterministic Linear Layer parameters and inputs
    weight = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)
    test_input = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)
    test_output = test_input @ weight + bias

    # 2. Symmetric Positive Semi-Definite Matrix A (8x8) with decaying spectrum for Hutch++
    m_mat = torch.randn(batch_size, batch_size, generator=gen, dtype=torch.float32)
    matrix_a = (m_mat @ m_mat.T) / float(batch_size)
    exact_trace = torch.trace(matrix_a).unsqueeze(0)

    # 3. Deterministic Sketch matrices
    s_sketch = torch.randn(batch_size, 3, generator=gen, dtype=torch.float32)
    g_rademacher = torch.where(
        torch.randn(batch_size, 4, generator=gen, dtype=torch.float32) > 0.0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )

    # Compute MGS QR basis of Y = A @ S
    y_sketch = matrix_a @ s_sketch
    q_basis, r_basis = modified_gram_schmidt_qr(y_sketch)

    # Compute Hutch++ Trace Estimate
    hutch_trace_est = hutch_plus_plus_trace(matrix_a, s_sketch, g_rademacher).unsqueeze(0)

    # 4. Power Iteration Spectral Norm for weight matrix
    v_init = torch.randn(out_dim, generator=gen, dtype=torch.float32)
    spectral_sigma, u_dom, v_dom = power_iteration_spectral_norm(weight, v_init, num_iters=5)

    # 5. Degenerate singular values with Lorentzian broadening
    colliding_sigma = torch.tensor([5.0, 3.0, 3.0, 1.0], dtype=torch.float32)
    lorentzian_f = lorentzian_broadened_f_matrix(colliding_sigma, epsilon=1e-2)

    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "matrix_a": matrix_a.detach(),
        "s_sketch": s_sketch.detach(),
        "g_rademacher": g_rademacher.detach(),
        "q_basis": q_basis.detach(),
        "r_basis": r_basis.detach(),
        "exact_trace": exact_trace.detach(),
        "hutch_trace_estimate": hutch_trace_est.detach(),
        "v_init": v_init.detach(),
        "spectral_sigma": spectral_sigma.detach().unsqueeze(0),
        "u_dom": u_dom.detach(),
        "v_dom": v_dom.detach(),
        "colliding_sigma": colliding_sigma.detach(),
        "lorentzian_f": lorentzian_f.detach(),
    }

    metadata = {
        "topic": "00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp",
        "description": "Randomized SVD, Hutch++ Trace Estimator, Lorentzian-Broadened Spectral Autograd",
        "batch_size": str(batch_size),
        "in_dim": str(in_dim),
        "out_dim": str(out_dim),
    }

    out_path = Path("weights/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.safetensors")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    core.parity.save_reference_checkpoint(
        path=out_path,
        weights_dict=weights_dict,
        input_tensor=test_input,
        output_tensor=test_output.detach(),
        metadata=metadata,
    )
    print(f"Successfully generated golden checkpoint at {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_topic_11_weights()
