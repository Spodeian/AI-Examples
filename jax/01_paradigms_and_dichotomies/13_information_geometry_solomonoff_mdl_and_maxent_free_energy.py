"""JAX Functional Implementation of Topic 13: Information Geometry, MDL & Free Energy."""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import load_file


def log_partition(theta: np.ndarray) -> np.ndarray:
    # Stable logsumexp
    c = np.max(theta, axis=0, keepdims=True)
    return c + np.log(np.sum(np.exp(theta - c), axis=0, keepdims=True))


def expectation_param(theta: np.ndarray) -> np.ndarray:
    c = np.max(theta, axis=0, keepdims=True)
    exp_t = np.exp(theta - c)
    return exp_t / np.sum(exp_t, axis=0, keepdims=True)


def dual_potential(eta: np.ndarray) -> np.ndarray:
    return np.array([np.sum(eta * np.log(eta))], dtype=np.float32)


def legendre_gap(theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
    psi = log_partition(theta)
    phi = dual_potential(eta)
    inner = np.sum(theta * eta)
    return np.abs(psi + phi - inner)


def fisher_rao_metric(eta: np.ndarray) -> np.ndarray:
    return np.diag(eta) - np.outer(eta, eta)


def e_geodesic(theta_1: np.ndarray, theta_2: np.ndarray, t: float = 0.5) -> np.ndarray:
    return (1.0 - t) * theta_1 + t * theta_2


def m_geodesic_eta(eta_1: np.ndarray, eta_2: np.ndarray, t: float = 0.5) -> np.ndarray:
    return (1.0 - t) * eta_1 + t * eta_2


def kl_divergence(eta_p: np.ndarray, eta_q: np.ndarray) -> np.ndarray:
    return np.array([np.sum(eta_p * np.log(eta_p / eta_q))], dtype=np.float32)


def bregman_divergence(theta_1: np.ndarray, theta_2: np.ndarray, eta_1: np.ndarray) -> np.ndarray:
    psi_1 = log_partition(theta_1)[0]
    psi_2 = log_partition(theta_2)[0]
    return np.array([psi_2 - psi_1 - np.sum(eta_1 * (theta_2 - theta_1))], dtype=np.float32)


def shtarkov_log_complexity(n: int) -> np.ndarray:
    comp_sum = 0.0
    for k in range(n + 1):
        comb = math.comb(n, k)
        if k == 0 or k == n:
            max_lik = 1.0
        else:
            p_hat = k / n
            max_lik = (p_hat ** k) * ((1.0 - p_hat) ** (n - k))
        comp_sum += comb * max_lik
    return np.array([math.log(comp_sum)], dtype=np.float32)


def prequential_codelength(stream_bits: np.ndarray) -> np.ndarray:
    n_seq = stream_bits.shape[0]
    preq_loss = 0.0
    s_prev = 0.0
    for t in range(1, n_seq + 1):
        bit = float(stream_bits[t - 1])
        p_pred = (s_prev + 0.5) / (t - 1 + 1.0)
        prob_t = p_pred if bit == 1.0 else (1.0 - p_pred)
        preq_loss -= math.log(prob_t)
        s_prev += bit
    return np.array([preq_loss], dtype=np.float32)


def compute_variational_free_energy(
    q_mu: np.ndarray,
    q_sigma: np.ndarray,
    p_prior_mu: np.ndarray,
    p_prior_sigma: np.ndarray,
    obs_x: np.ndarray,
    a_mat: np.ndarray,
    sigma_obs: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kl_qp = np.array([
        np.sum(
            np.log(p_prior_sigma / q_sigma)
            + (q_sigma ** 2 + (q_mu - p_prior_mu) ** 2) / (2.0 * p_prior_sigma ** 2)
            - 0.5
        )
    ], dtype=np.float32)

    obs_dim = obs_x.shape[0]
    res_mean = obs_x - np.matmul(a_mat, q_mu)
    sq_err_mean = np.sum(res_mean ** 2)
    tr_cov = np.sum(np.diagonal(np.matmul(a_mat.T, a_mat)) * (q_sigma ** 2))
    recon_loss = np.array([
        0.5 * obs_dim * math.log(2.0 * math.pi * (sigma_obs ** 2))
        + (sq_err_mean + tr_cov) / (2.0 * (sigma_obs ** 2))
    ], dtype=np.float32)

    free_energy = kl_qp + recon_loss
    return kl_qp, recon_loss, free_energy


def linear_anchor_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.matmul(x, weight) + bias


if __name__ == "__main__":
    ckpt = (
        Path(__file__).parents[2]
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "13_information_geometry_solomonoff_mdl_and_maxent_free_energy.safetensors"
    )
    tensors = load_file(str(ckpt))

    # 1. Linear Anchor
    out_anchor = linear_anchor_forward(tensors["x_anchor"], tensors["weight"], tensors["bias"])
    np.testing.assert_allclose(out_anchor, tensors["linear_anchor_out"], atol=1e-5)

    # 2. Amari Dually Flat Geometry
    psi = log_partition(tensors["theta"])
    np.testing.assert_allclose(psi, tensors["psi"], atol=1e-5)
    eta = expectation_param(tensors["theta"])
    np.testing.assert_allclose(eta, tensors["eta"], atol=1e-5)
    phi = dual_potential(eta)
    np.testing.assert_allclose(phi, tensors["phi"], atol=1e-5)
    gap = legendre_gap(tensors["theta"], eta)
    np.testing.assert_allclose(gap, tensors["legendre_gap"], atol=1e-5)
    fisher = fisher_rao_metric(eta)
    np.testing.assert_allclose(fisher, tensors["fisher_metric"], atol=1e-5)

    # 3. Dual Geodesics
    e_mid = e_geodesic(tensors["theta_1"], tensors["theta_2"], 0.5)
    np.testing.assert_allclose(e_mid, tensors["e_geodesic_mid"], atol=1e-5)
    m_mid_eta = m_geodesic_eta(tensors["eta_1"], tensors["eta_2"], 0.5)
    np.testing.assert_allclose(m_mid_eta, tensors["m_geodesic_mid_eta"], atol=1e-5)
    kl_fwd = kl_divergence(tensors["eta_1"], tensors["eta_2"])
    np.testing.assert_allclose(kl_fwd, tensors["kl_forward"], atol=1e-5)
    kl_rev = kl_divergence(tensors["eta_2"], tensors["eta_1"])
    np.testing.assert_allclose(kl_rev, tensors["kl_reverse"], atol=1e-5)
    bregman = bregman_divergence(tensors["theta_1"], tensors["theta_2"], tensors["eta_1"])
    np.testing.assert_allclose(bregman, tensors["bregman_div"], atol=1e-5)

    # 4. MDL & Shtarkov
    nml = shtarkov_log_complexity(8)
    np.testing.assert_allclose(nml, tensors["nml_log_comp"], atol=1e-5)
    preq = prequential_codelength(tensors["stream_bits"])
    np.testing.assert_allclose(preq, tensors["prequential_codelength"], atol=1e-5)

    # 5. Variational Free Energy
    kl_qp, recon, fe = compute_variational_free_energy(
        tensors["q_mu"],
        tensors["q_sigma"],
        tensors["p_prior_mu"],
        tensors["p_prior_sigma"],
        tensors["obs_x"],
        tensors["a_mat"],
    )
    np.testing.assert_allclose(kl_qp, tensors["kl_qp"], atol=1e-5)
    np.testing.assert_allclose(recon, tensors["recon_loss"], atol=1e-5)
    np.testing.assert_allclose(fe, tensors["free_energy"], atol=1e-5)

    print("JAX/NumPy Information Geometry, MDL & Free Energy verified successfully!")
