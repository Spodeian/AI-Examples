"""JAX / Functional NumPy: Proper Scoring Rules, Evidential Deep Learning, and Differentiable Ranking (C03 Topic 11).

Implements uncertainty quantification, calibrated scoring rules, and ranking objectives:
1. Continuous Ranked Probability Score (CRPS) for Gaussian predictive distributions (Matheson & Winkler 1976).
2. Brier Score: Proper scoring rule for categorical probability forecasts (Brier 1950).
3. Beta-NLL: Variance-discounted heteroscedastic regression loss (Seitzer et al. 2022).
4. Evidential Deep Learning (EDL MSE): Closed-form Dirichlet expected square error (Sensoy et al. 2018).
5. ListMLE Ranking Loss: Permutation probability objective over item lists (Xia et al. 2008).
6. Soft-Spearman Differentiable Ranks via pairwise sigmoid relaxation.
7. Linear Anchor: Baseline affine transform.
"""

import math
from pathlib import Path
from typing import Dict

import numpy as np
from safetensors.numpy import load_file

try:
    from scipy.special import erf
except ImportError:
    erf = np.vectorize(math.erf)


def crps_gaussian_np(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Continuous Ranked Probability Score for Gaussian predictions: CRPS(y, N(mu, sigma^2))."""
    sig = np.maximum(sigma, eps)
    z = (y - mu) / sig
    phi = np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    crps = sig * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return float(np.mean(crps))


def brier_score_np(
    p: np.ndarray,
    y_onehot: np.ndarray,
) -> float:
    """Brier Score: mean squared error between forecast probabilities and one-hot labels."""
    return float(np.mean(np.sum((p - y_onehot) ** 2, axis=-1)))


def beta_nll_np(
    y: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    beta: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Beta-NLL: (var^beta) * [0.5 * log(var) + 0.5 * (y - mu)^2 / var]."""
    v = np.maximum(var, eps)
    loss = 0.5 * (np.log(v) + (y - mu) ** 2 / v)
    weight = v ** beta
    return float(np.mean(weight * loss))


def edl_mse_np(
    alpha: np.ndarray,
    y_onehot: np.ndarray,
) -> float:
    """Evidential Deep Learning Expected MSE under Dirichlet distribution: sum (y - p_hat)^2 + Var(p)."""
    S = np.sum(alpha, axis=-1, keepdims=True)
    p_hat = alpha / S
    var_p = (alpha * (S - alpha)) / (S ** 2 * (S + 1.0))
    loss = np.sum((y_onehot - p_hat) ** 2 + var_p, axis=-1)
    return float(np.mean(loss))


def listmle_np(
    scores: np.ndarray,
    perm: np.ndarray,
) -> float:
    """ListMLE Ranking Loss: negative log-likelihood of ground-truth item permutation."""
    B, K = scores.shape
    total_loss = 0.0
    for b in range(B):
        s = scores[b]
        p = perm[b]
        s_ordered = s[p]
        for i in range(K):
            sub_scores = s_ordered[i:]
            max_s = np.max(sub_scores)
            log_denom = max_s + np.log(np.sum(np.exp(sub_scores - max_s)))
            total_loss -= (s_ordered[i] - log_denom)
    return float(total_loss / float(B * K))


def soft_ranks_np(
    scores: np.ndarray,
    tau: float = 0.5,
) -> np.ndarray:
    """Differentiable soft ranks via pairwise sigmoid relaxation: 0.5 + sum_j sigmoid((s_j - s_i)/tau)."""
    diff = (scores[:, :, None] - scores[:, None, :]) / tau
    sigmoid = 1.0 / (1.0 + np.exp(-diff))
    return 0.5 + np.sum(sigmoid, axis=-1)


def linear_anchor_np(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Linear baseline mapping."""
    return np.dot(x, weight) + bias


def load_checkpoint_numpy(checkpoint_path: Path) -> Dict[str, np.ndarray]:
    return load_file(str(checkpoint_path))


if __name__ == "__main__":
    ckpt = Path("weights/03_building_blocks_and_losses/11_losses_proper_scoring_evidential_and_differentiable_ranking.safetensors")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint at {ckpt}")

    data = load_checkpoint_numpy(ckpt)
    print(f"Loaded {len(data)} tensors from {ckpt.name}")

    # 1. Linear Anchor parity
    pred = linear_anchor_np(data["test_input"], data["weight"], data["bias"])
    err_anchor = float(np.max(np.abs(pred - data["test_output"])))
    assert err_anchor < 1e-5, f"Anchor parity error {err_anchor:.2e} >= 1e-5"

    # 2. CRPS parity
    crps_val = crps_gaussian_np(data["crps_y"], data["crps_mu"], data["crps_sigma"])
    err_crps = abs(crps_val - float(data["expected_crps"][0]))
    assert err_crps < 1e-5, f"CRPS parity error {err_crps:.2e} >= 1e-5"

    # 3. Brier parity
    brier_val = brier_score_np(data["brier_p"], data["brier_y"])
    err_brier = abs(brier_val - float(data["expected_brier"][0]))
    assert err_brier < 1e-5, f"Brier parity error {err_brier:.2e} >= 1e-5"

    # 4. Beta-NLL parity
    beta_nll_val = beta_nll_np(data["beta_nll_y"], data["beta_nll_mu"], data["beta_nll_var"], beta=0.5)
    err_beta_nll = abs(beta_nll_val - float(data["expected_beta_nll"][0]))
    assert err_beta_nll < 1e-5, f"Beta-NLL parity error {err_beta_nll:.2e} >= 1e-5"

    # 5. EDL MSE parity
    edl_val = edl_mse_np(data["edl_alpha"], data["edl_y"])
    err_edl = abs(edl_val - float(data["expected_edl"][0]))
    assert err_edl < 1e-5, f"EDL parity error {err_edl:.2e} >= 1e-5"

    # 6. ListMLE parity
    listmle_val = listmle_np(data["listmle_scores"], data["listmle_perm"])
    err_listmle = abs(listmle_val - float(data["expected_listmle"][0]))
    assert err_listmle < 1e-5, f"ListMLE parity error {err_listmle:.2e} >= 1e-5"

    # 7. Soft Ranks parity
    ranks_val = soft_ranks_np(data["soft_scores"], tau=0.5)
    err_ranks = float(np.max(np.abs(ranks_val - data["expected_soft_ranks"])))
    assert err_ranks < 1e-5, f"Soft ranks parity error {err_ranks:.2e} >= 1e-5"

    print(
        f"All 7 JAX/NumPy invariants verified! Anchor: {err_anchor:.2e}, CRPS: {err_crps:.2e}, Brier: {err_brier:.2e}, "
        f"BetaNLL: {err_beta_nll:.2e}, EDL: {err_edl:.2e}, ListMLE: {err_listmle:.2e}, SoftRanks: {err_ranks:.2e}"
    )
