"""Losses: Divergences, Information Bottleneck, and Coding Rate in JAX / Functional NumPy (C03 Topic 09).

Pure functional implementations with transitive numerical parity:
1. Forward & Reverse KL Divergence + JSD.
2. alpha-Renyi Divergence.
3. Itakura-Saito Distance (Bregman divergence).
4. Variational Information Bottleneck (VIB) Gaussian Prior KL.
5. Maximal Coding Rate Reduction (MCR^2) Total Coding Rate.
6. Linear Anchor: X @ W + b.
"""

from pathlib import Path
from typing import Dict, Tuple
import math
import numpy as np
from safetensors.numpy import load_file


def linear_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.dot(x, weight) + bias


def kl_forward_fn(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Forward KL Divergence: D_KL(P || Q) = sum P * log(P / Q)."""
    p_safe = np.maximum(p, eps)
    q_safe = np.maximum(q, eps)
    kl_per_row = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=-1)
    return np.mean(kl_per_row)


def kl_reverse_fn(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Reverse KL Divergence: D_KL(Q || P) = sum Q * log(Q / P)."""
    p_safe = np.maximum(p, eps)
    q_safe = np.maximum(q, eps)
    kl_per_row = np.sum(q_safe * (np.log(q_safe) - np.log(p_safe)), axis=-1)
    return np.mean(kl_per_row)


def jsd_fn(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon Divergence: 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M), M = 0.5 * (P + Q)."""
    m = 0.5 * (p + q)
    return 0.5 * kl_forward_fn(p, m) + 0.5 * kl_forward_fn(q, m)


def renyi_divergence_fn(p: np.ndarray, q: np.ndarray, alpha: float = 0.5, eps: float = 1e-12) -> np.ndarray:
    """alpha-Renyi Divergence: 1/(alpha - 1) * log sum P^alpha * Q^(1-alpha)."""
    p_safe = np.maximum(p, eps)
    q_safe = np.maximum(q, eps)
    inner = np.sum((p_safe ** alpha) * (q_safe ** (1.0 - alpha)), axis=-1)
    renyi_per_row = np.log(np.maximum(inner, eps)) / (alpha - 1.0)
    return np.mean(renyi_per_row)


def itakura_saito_fn(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Itakura-Saito Distance: mean(x/y - log(x/y) - 1)."""
    x_safe = np.maximum(x, eps)
    y_safe = np.maximum(y, eps)
    ratio = x_safe / y_safe
    return np.mean(ratio - np.log(ratio) - 1.0)


def vib_kl_fn(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """VIB Gaussian Prior KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))."""
    kl_per_sample = -0.5 * np.sum(1.0 + logvar - (mu ** 2) - np.exp(logvar), axis=-1)
    return np.mean(kl_per_sample)


def mcr2_coding_rate_fn(z: np.ndarray, eps: float = 0.5) -> np.ndarray:
    """MCR^2 Total Coding Rate: 0.5 * log det(I + D / (N * eps^2) * Z^T Z)."""
    n, d = z.shape
    scale = float(d) / (float(n) * (eps ** 2))
    gram = np.dot(z.T, z)
    c = np.eye(d) + scale * gram
    _sign, logdet = np.linalg.slogdet(c)
    return 0.5 * logdet


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/09_losses_divergences_information_bottleneck_and_coding_rate.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    weights = load_file(str(weights_path))

    test_in = weights["test_input"]
    weight = weights["weight"]
    bias = weights["bias"]
    test_out_gold = weights["test_output"]

    p_dist = weights["p_dist"]
    q_dist = weights["q_dist"]
    r_alpha = float(weights["renyi_alpha"][0])

    is_x = weights["is_x"]
    is_y = weights["is_y"]

    vib_mu = weights["vib_mu"]
    vib_logvar = weights["vib_logvar"]

    mcr2_z = weights["mcr2_z"]
    mcr2_eps = float(weights["mcr2_eps"][0])

    # 1. Linear Anchor Parity
    y_anchor = linear_forward(test_in, weight, bias)
    diff_anchor = float(np.max(np.abs(y_anchor - test_out_gold)))
    assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

    # 2. Forward KL Parity
    loss_fwd_kl = kl_forward_fn(p_dist, q_dist)
    diff_fwd_kl = float(np.abs(loss_fwd_kl - weights["loss_fwd_kl"][0]))
    assert diff_fwd_kl < 1e-5, f"Forward KL mismatch: {diff_fwd_kl:.2e}"

    # 3. Reverse KL Parity
    loss_rev_kl = kl_reverse_fn(p_dist, q_dist)
    diff_rev_kl = float(np.abs(loss_rev_kl - weights["loss_rev_kl"][0]))
    assert diff_rev_kl < 1e-5, f"Reverse KL mismatch: {diff_rev_kl:.2e}"

    # 4. JSD Parity
    loss_jsd = jsd_fn(p_dist, q_dist)
    diff_jsd = float(np.abs(loss_jsd - weights["loss_jsd"][0]))
    assert diff_jsd < 1e-5, f"JSD mismatch: {diff_jsd:.2e}"

    # 5. Renyi Parity
    loss_renyi = renyi_divergence_fn(p_dist, q_dist, r_alpha)
    diff_renyi = float(np.abs(loss_renyi - weights["loss_renyi"][0]))
    assert diff_renyi < 1e-5, f"Renyi mismatch: {diff_renyi:.2e}"

    # 6. Itakura-Saito Parity
    loss_is = itakura_saito_fn(is_x, is_y)
    diff_is = float(np.abs(loss_is - weights["loss_is"][0]))
    assert diff_is < 1e-5, f"Itakura-Saito mismatch: {diff_is:.2e}"

    # 7. VIB KL Parity
    loss_vib = vib_kl_fn(vib_mu, vib_logvar)
    diff_vib = float(np.abs(loss_vib - weights["loss_vib"][0]))
    assert diff_vib < 1e-5, f"VIB mismatch: {diff_vib:.2e}"

    # 8. MCR^2 Coding Rate Parity
    loss_mcr2 = mcr2_coding_rate_fn(mcr2_z, mcr2_eps)
    diff_mcr2 = float(np.abs(loss_mcr2 - weights["loss_mcr2"][0]))
    assert diff_mcr2 < 1e-5, f"MCR^2 mismatch: {diff_mcr2:.2e}"

    print(
        f"All JAX / NumPy assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"FwdKL: {diff_fwd_kl:.2e}, RevKL: {diff_rev_kl:.2e}, JSD: {diff_jsd:.2e}, "
        f"Renyi: {diff_renyi:.2e}, IS: {diff_is:.2e}, VIB: {diff_vib:.2e}, MCR2: {diff_mcr2:.2e})"
    )


if __name__ == "__main__":
    main()
