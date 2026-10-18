"""Functional JAX/NumPy Implementation of Statistical Inference (Topic 11).

Functional implementations covering the 5 schools of inference:
1. Frequentist: Fisher Information Matrix, Cramér-Rao Lower Bound, Likelihood Ratio Test.
2. Bayesian: Conjugate Gaussian update, Laplace marginal likelihood, and Occam factor penalty.
3. James-Stein Shrinkage: Shrinkage estimator dominating MLE for K >= 3.
4. PAC-Bayes Generalization Bound: McAllester bound for stochastic models.
5. Anytime-Valid E-Values & Martingales: Sequential testing with Ville's inequality.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

from safetensors.numpy import load_file

try:
    import jax
    import jax.numpy as jnp
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
    _ = jax
except (ImportError, AttributeError):
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False


def frequentist_fisher_cramer_rao(
    x_data: Any,
    theta_0: Any,
    sigma2_noise: float = 0.5,
) -> Tuple[Any, Any, Any, Any]:
    """Computes Gaussian mean MLE, Fisher information diagonal, Cramér-Rao bound, and LRT statistic."""
    x = jnp.asarray(x_data, dtype=jnp.float32)
    th0 = jnp.asarray(theta_0, dtype=jnp.float32)

    num_obs = x.shape[0]
    obs_dim = x.shape[1]

    theta_mle = jnp.mean(x, axis=0)
    fisher_diag = jnp.full((obs_dim,), num_obs / sigma2_noise, dtype=jnp.float32)
    cramer_rao = jnp.full((obs_dim,), sigma2_noise / num_obs, dtype=jnp.float32)

    diff = theta_mle - th0
    lrt_stat = (num_obs / sigma2_noise) * jnp.sum(diff ** 2)

    return theta_mle, fisher_diag, cramer_rao, lrt_stat


def bayesian_conjugate_laplace(
    x_data: Any,
    mu_0: Any,
    sigma2_0: float = 1.0,
    sigma2_noise: float = 0.5,
) -> Tuple[Any, Any, Any, Any]:
    """Computes Gaussian conjugate posterior and Laplace marginal likelihood Occam factor."""
    x = jnp.asarray(x_data, dtype=jnp.float32)
    m0 = jnp.asarray(mu_0, dtype=jnp.float32)

    num_obs = x.shape[0]
    obs_dim = x.shape[1]
    theta_mle = jnp.mean(x, axis=0)

    prec_0 = 1.0 / sigma2_0
    prec_lik = num_obs / sigma2_noise
    prec_post = prec_0 + prec_lik
    var_post = 1.0 / prec_post
    sigma_post = math.sqrt(var_post)

    mu_post = var_post * (prec_0 * m0 + prec_lik * theta_mle)
    sigma_post_vec = jnp.full((obs_dim,), sigma_post, dtype=jnp.float32)

    log_det_h = float(obs_dim * math.log(prec_post))
    res = x - jnp.expand_dims(mu_post, axis=0)
    log_lik_map = (
        -0.5 * num_obs * obs_dim * math.log(2.0 * math.pi * sigma2_noise)
        - 0.5 * jnp.sum(res ** 2) / sigma2_noise
    )
    log_prior_map = (
        -0.5 * obs_dim * math.log(2.0 * math.pi * sigma2_0)
        - 0.5 * jnp.sum((mu_post - m0) ** 2) / sigma2_0
    )
    occam_penalty = -0.5 * log_det_h + 0.5 * obs_dim * math.log(2.0 * math.pi)
    laplace_log_marginal = log_lik_map + log_prior_map + occam_penalty

    return mu_post, sigma_post_vec, laplace_log_marginal, jnp.asarray(occam_penalty, dtype=jnp.float32)


def james_stein_estimator(
    y_obs: Any,
    sigma2: float = 1.0,
) -> Tuple[Any, Any, Any]:
    """Computes James-Stein shrinkage and positive-part shrinkage for group observations."""
    y = jnp.asarray(y_obs, dtype=jnp.float32)
    num_groups = y.shape[0]
    norm_sq = jnp.sum(y ** 2)

    c_shrink = 1.0 - ((num_groups - 2.0) * sigma2) / norm_sq
    c_shrink_pos = jnp.maximum(0.0, c_shrink)

    theta_js = c_shrink * y
    theta_js_pos = c_shrink_pos * y
    return theta_js, theta_js_pos, c_shrink


def pac_bayes_bound(
    mu_q: Any,
    sigma_q: Any,
    emp_risk: Any,
    num_obs: int,
    sigma_p: float = 1.0,
    delta: float = 0.05,
) -> Tuple[Any, Any]:
    """Computes McAllester PAC-Bayes bound given stochastic Gaussian hypothesis params."""
    mq = jnp.asarray(mu_q, dtype=jnp.float32)
    sq = jnp.asarray(sigma_q, dtype=jnp.float32)
    er = jnp.asarray(emp_risk, dtype=jnp.float32)

    kl_div = 0.5 * jnp.sum(
        (mq ** 2 + sq ** 2) / (sigma_p ** 2)
        - 1.0
        - jnp.log((sq ** 2) / (sigma_p ** 2))
    )
    complexity_term = jnp.sqrt(
        (kl_div + math.log(2.0 * math.sqrt(num_obs) / delta)) / (2.0 * num_obs)
    )
    pac_bound_val = er + complexity_term
    return kl_div, pac_bound_val


def anytime_valid_martingale(
    e_values: Any,
    alpha: float = 0.05,
) -> Tuple[Any, Any]:
    """Accumulates e-values into test martingale and provides Ville threshold."""
    ev = jnp.asarray(e_values, dtype=jnp.float32)
    martingale = jnp.cumprod(ev, axis=0)
    threshold = jnp.asarray(1.0 / alpha, dtype=jnp.float32)
    return martingale, threshold


def linear_anchor_forward(x: Any, weight: Any, bias: Any) -> Any:
    """Calculates linear forward pass: y = x @ weight + bias."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_arr = jnp.asarray(weight, dtype=jnp.float32)
    b_arr = jnp.asarray(bias, dtype=jnp.float32)
    # PyTorch linear layer has weight (out, in), reference weight is (in, out)
    if w_arr.shape[0] == x_arr.shape[-1]:
        return jnp.dot(x_arr, w_arr) + b_arr
    return jnp.dot(x_arr, w_arr.T) + b_arr


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    raw = load_file(checkpoint_path)
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in raw.items()}


if __name__ == "__main__":
    ckpt = (
        Path(__file__).parents[2]
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.safetensors"
    )
    tensors = load_checkpoint(str(ckpt))

    # 1. Linear Anchor
    out_anchor = linear_anchor_forward(tensors["x_anchor"], tensors["weight"], tensors["bias"])
    assert jnp.allclose(out_anchor, tensors["linear_anchor_out"], atol=1e-5), "Anchor mismatch"

    # 2. Frequentist Fisher & Cramér-Rao
    theta_0 = jnp.zeros_like(tensors["theta_mle"])
    mle, fisher, crlb, lrt = frequentist_fisher_cramer_rao(tensors["x_data"], theta_0)
    assert jnp.allclose(mle, tensors["theta_mle"], atol=1e-5), "MLE mismatch"
    assert jnp.allclose(fisher, tensors["fisher_diag"], atol=1e-5), "Fisher mismatch"
    assert jnp.allclose(crlb, tensors["cramer_rao_lower_bound"], atol=1e-5), "CRLB mismatch"
    assert jnp.allclose(lrt, tensors["lrt_stat"], atol=1e-5), "LRT mismatch"

    # 3. Bayesian Conjugate & Laplace
    mu_post, sigma_post, laplace_ll, occam = bayesian_conjugate_laplace(tensors["x_data"], tensors["mu_0"])
    assert jnp.allclose(mu_post, tensors["mu_post"], atol=1e-5), "Mu post mismatch"
    assert jnp.allclose(sigma_post, tensors["sigma_post_vec"], atol=1e-5), "Sigma post mismatch"
    assert jnp.allclose(laplace_ll, tensors["laplace_log_marginal"], atol=1e-5), "Laplace LL mismatch"
    assert jnp.allclose(occam, tensors["occam_penalty"], atol=1e-5), "Occam mismatch"

    # 4. James-Stein
    th_js, th_js_pos, c_sh = james_stein_estimator(tensors["y_obs"])
    assert jnp.allclose(th_js, tensors["theta_js"], atol=1e-5), "James-Stein mismatch"
    assert jnp.allclose(th_js_pos, tensors["theta_js_pos"], atol=1e-5), "James-Stein pos mismatch"
    assert jnp.allclose(c_sh, tensors["c_shrink"], atol=1e-5), "C shrink mismatch"

    # 5. PAC-Bayes
    kl, pac = pac_bayes_bound(tensors["mu_q"], tensors["sigma_q"], tensors["emp_risk"], num_obs=16)
    assert jnp.allclose(kl, tensors["kl_div"], atol=1e-5), "KL mismatch"
    assert jnp.allclose(pac, tensors["pac_bound"], atol=1e-5), "PAC bound mismatch"

    # 6. Anytime-Valid Martingale
    mart, thresh = anytime_valid_martingale(tensors["e_values"])
    assert jnp.allclose(mart, tensors["martingale"], atol=1e-5), "Martingale mismatch"
    assert jnp.allclose(thresh, tensors["ville_threshold"], atol=1e-5), "Threshold mismatch"

    print("JAX Statistical Inference operations successfully verified with zero errors!")
