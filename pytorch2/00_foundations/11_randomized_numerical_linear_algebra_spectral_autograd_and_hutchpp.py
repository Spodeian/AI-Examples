"""Production PyTorch 2.x implementation for Topic 11:
11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


def modified_gram_schmidt_qr(
    y: Float[Tensor, "m num_cols"],
) -> tuple[Tensor, Tensor]:
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


def randomized_range_finder(
    a: Float[Tensor, "m n_in"],
    omega: Float[Tensor, "n_in num_cols"],
) -> Tensor:
    """Forms an orthonormal basis Q spanning the dominant range of A via sketch Y = A Omega."""
    y = a @ omega
    q, _ = modified_gram_schmidt_qr(y)
    return q


def randomized_svd(
    a: Float[Tensor, "m n_in"],
    k: int,
    p: int = 2,
    n_iter: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes approximate low-rank SVD A approx U S V^T via Halko-Martinsson-Tropp randomized projection."""
    m, n_in = a.shape
    sketch_dim = k + p
    omega = torch.randn(n_in, sketch_dim, dtype=a.dtype, device=a.device)

    # Power iterations for sharper singular value decay
    y = a @ omega
    q, _ = modified_gram_schmidt_qr(y)
    for _ in range(n_iter):
        y_tilde = a.T @ q
        q_tilde, _ = modified_gram_schmidt_qr(y_tilde)
        y = a @ q_tilde
        q, _ = modified_gram_schmidt_qr(y)

    # Project A into low-dimensional subspace
    b = q.T @ a
    u_tilde, s, v_h = torch.linalg.svd(b, full_matrices=False)
    u = q @ u_tilde

    return u[:, :k], s[:k], v_h.T[:, :k]


def hutch_plus_plus_trace(
    a: Float[Tensor, "n_dim n_dim"],
    s_sketch: Float[Tensor, "n_dim l_cols"],
    g_mc: Float[Tensor, "n_dim m_cols"],
) -> Tensor:
    """Evaluates Meyer-Musco-Musco-Woodruff Hutch++ stochastic trace estimator with optimal O(1/eps) query complexity."""
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
    singular_values: Tensor,
    epsilon: float = 1e-3,
) -> Tensor:
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
    w: Tensor,
    v_init: Tensor,
    num_iters: int = 5,
) -> tuple[Tensor, Tensor, Tensor]:
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


class SpectralLinear(nn.Module):
    """Linear projection layer equipped with randomized spectral estimation routines."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / (self.in_features**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: Float[Tensor, "batch in_features"],
    ) -> Float[Tensor, "batch out_features"]:
        return x @ self.weight + self.bias

    def compute_spectral_norm(
        self,
        v_init: Tensor,
        num_iters: int = 5,
    ) -> Tensor:
        sigma, _, _ = power_iteration_spectral_norm(self.weight, v_init, num_iters=num_iters)
        return sigma

    def estimate_frobenius_sq_hutchpp(
        self,
        s_sketch: Float[Tensor, "in_features l_cols"],
        g_mc: Float[Tensor, "in_features m_cols"],
    ) -> Tensor:
        """Estimates ||W||_F^2 = tr(W W^T) via Hutch++."""
        ww_t = self.weight @ self.weight.T
        return hutch_plus_plus_trace(ww_t, s_sketch, g_mc)
