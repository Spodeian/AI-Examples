"""JAX / Functional implementation for Category 04 Topic 13:
Sharpness-Aware Minimization (SAM), SWAD, and Weight Averaging Dynamics.

Provides pure functional, memory-efficient implementations of:
1. Sharpness-Aware Minimization (SAM) dual ascent gradient perturbation and update.
2. Lookahead optimizer synchronization between slow and fast weights.
3. Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA).
4. Uniform Model Soups weight-space averaging.
"""

from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def step_sam(
    param: np.ndarray,
    grad_at_w: np.ndarray,
    grad_at_pert: np.ndarray,
    rho: float = 0.05,
    lr: float = 0.01,
    weight_decay: float = 0.001,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Sharpness-Aware Minimization (SAM) update step."""
    grad_norm = float(np.linalg.norm(grad_at_w, ord=2)) + eps
    epsilon = rho * (grad_at_w / grad_norm)
    update = grad_at_pert + weight_decay * param
    param_next = param - lr * update
    return param_next, epsilon


def step_lookahead(
    slow_param: np.ndarray,
    fast_param: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Lookahead synchronization step."""
    slow_next = slow_param + alpha * (fast_param - slow_param)
    fast_next = slow_next.copy()
    return slow_next, fast_next


def step_swa_ema(
    swa_avg: np.ndarray,
    ema_avg: np.ndarray,
    current_param: np.ndarray,
    n_models: int,
    decay: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """SWA running average and EMA step."""
    new_swa = (swa_avg * n_models + current_param) / (n_models + 1)
    new_ema = decay * ema_avg + (1.0 - decay) * current_param
    return new_swa, new_ema


def uniform_model_soup(models: np.ndarray) -> np.ndarray:
    """Average a stack of M models: models shape [M, d]."""
    return np.mean(models, axis=0)


def linear_anchor_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear anchor forward pass: x @ w + b."""
    return np.matmul(x, w) + b


def verify_invariants(weights_path: str | Path, atol: float = 1e-5) -> dict[str, float]:
    """Verify parity against golden reference tensors."""
    weights = load_file(str(weights_path))
    errors: dict[str, float] = {}

    # 1. SAM Invariant
    p_sam, eps_sam = step_sam(
        weights["param_sam"],
        weights["grad_at_w"],
        weights["grad_at_pert"],
        rho=0.05,
        lr=0.01,
        weight_decay=0.001,
    )
    err_p_sam = float(np.max(np.abs(p_sam - weights["golden_param_sam"])))
    err_eps_sam = float(np.max(np.abs(eps_sam - weights["golden_epsilon_sam"])))
    err_sam = max(err_p_sam, err_eps_sam)
    assert err_sam < atol, f"SAM invariant failed: err={err_sam:.2e} >= {atol:.2e}"
    errors["sam_err"] = err_sam

    # 2. Lookahead Invariant
    slow_out, fast_out = step_lookahead(weights["slow_p"], weights["fast_p"], alpha=0.5)
    err_slow = float(np.max(np.abs(slow_out - weights["golden_slow_p"])))
    err_fast = float(np.max(np.abs(fast_out - weights["golden_fast_p"])))
    err_lookahead = max(err_slow, err_fast)
    assert err_lookahead < atol, f"Lookahead invariant failed: err={err_lookahead:.2e} >= {atol:.2e}"
    errors["lookahead_err"] = err_lookahead

    # 3. SWA & EMA Invariant
    new_swa, new_ema = step_swa_ema(
        weights["swa_avg"], weights["ema_avg"], weights["curr_param"], n_models=4, decay=0.99
    )
    err_swa = float(np.max(np.abs(new_swa - weights["golden_swa"])))
    err_ema = float(np.max(np.abs(new_ema - weights["golden_ema"])))
    err_avg = max(err_swa, err_ema)
    assert err_avg < atol, f"SWA/EMA invariant failed: err={err_avg:.2e} >= {atol:.2e}"
    errors["swa_ema_err"] = err_avg

    # 4. Model Soup Invariant
    soup_out = uniform_model_soup(weights["soup_models"])
    err_soup = float(np.max(np.abs(soup_out - weights["golden_soup"])))
    assert err_soup < atol, f"Model Soup invariant failed: err={err_soup:.2e} >= {atol:.2e}"
    errors["soup_err"] = err_soup

    # 5. Linear Anchor Invariant
    anchor_out = linear_anchor_forward(weights["anchor_x"], weights["anchor_w"], weights["anchor_b"])
    err_anchor = float(np.max(np.abs(anchor_out - weights["golden_anchor_out"])))
    assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
    errors["anchor_err"] = err_anchor

    return errors


if __name__ == "__main__":
    p = Path(
        "weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/06_sharpness_aware_swad_and_weight_averaging_dynamics.safetensors"
    )
    errs = verify_invariants(p)
    print(f"JAX Invariant Parity Checks Passed: {errs}")
