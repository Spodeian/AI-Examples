"""JAX / Functional implementation for Category 04 Topic 12:
Multi-Fidelity Hyperparameter Optimisation: Hyperband, ASHA, and BOED.

Provides pure functional, memory-efficient implementations of:
1. Successive Halving Algorithm (SHA) multi-fidelity configuration pruning.
2. Hyperband bracket resource allocation schedule calculation.
3. Tree-structured Parzen Estimator (TPE) non-parametric density ratio surrogate.
4. Asynchronous Successive Halving (ASHA) non-blocking promotion logic.
"""

import math
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def compute_sha_top_k(losses: np.ndarray, eta: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Select top k = n // eta candidates with lowest loss."""
    k = max(1, losses.shape[0] // eta)
    sorted_indices = np.argsort(losses)[:k]
    sorted_losses = losses[sorted_indices]
    return sorted_indices.astype(np.float32), sorted_losses


def compute_hyperband_schedule(R: int = 81, eta: int = 3) -> np.ndarray:
    """Compute (n_s, r_s) pairs across brackets s in [s_max, ..., 0]."""
    s_max = int(math.floor(math.log(R, eta)))
    B = (s_max + 1) * R
    schedule = []
    for s in range(s_max, -1, -1):
        n_s = int(math.ceil((B / R) * ((eta ** s) / (s + 1))))
        r_s = int(R * (eta ** (-s)))
        schedule.append([float(n_s), float(r_s)])
    return np.array(schedule, dtype=np.float32)


def compute_tpe_log_ratio(
    x_query: np.ndarray,
    good_samples: np.ndarray,
    bad_samples: np.ndarray,
    bandwidth: float = 0.5,
) -> np.ndarray:
    """Evaluate 1D TPE log-likelihood ratio: log l(x) - log g(x)."""
    def log_kde(x, samples):
        diff = (x[:, None] - samples[None, :]) / bandwidth
        log_kernel = -0.5 * diff ** 2 - 0.5 * math.log(2.0 * math.pi) - math.log(bandwidth)
        # logsumexp over axis -1
        max_val = np.max(log_kernel, axis=-1, keepdims=True)
        lse = max_val.squeeze(-1) + np.log(np.sum(np.exp(log_kernel - max_val), axis=-1))
        return lse - math.log(samples.shape[0])

    log_l = log_kde(x_query, good_samples)
    log_g = log_kde(x_query, bad_samples)
    return log_l - log_g


def check_asha_promotion(rung_losses: np.ndarray, candidate_loss: float, eta: int = 3) -> float:
    """Return 1.0 if candidate_loss is eligible for promotion (in top 1/eta), else 0.0."""
    k = max(1, rung_losses.shape[0] // eta)
    sorted_losses = np.sort(rung_losses)
    cutoff = sorted_losses[k - 1]
    return 1.0 if candidate_loss <= cutoff else 0.0


def linear_anchor_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear anchor forward pass: x @ w + b."""
    return np.matmul(x, w) + b


def verify_invariants(weights_path: str | Path, atol: float = 1e-5) -> dict[str, float]:
    """Verify parity against golden reference tensors."""
    weights = load_file(str(weights_path))
    errors: dict[str, float] = {}

    # 1. SHA Invariant
    idx, losses = compute_sha_top_k(weights["sha_losses"], eta=3)
    err_sha_idx = float(np.max(np.abs(idx - weights["golden_sha_idx"])))
    err_sha_l = float(np.max(np.abs(losses - weights["golden_sha_losses"])))
    err_sha = max(err_sha_idx, err_sha_l)
    assert err_sha < atol, f"SHA invariant failed: err={err_sha:.2e} >= {atol:.2e}"
    errors["sha_err"] = err_sha

    # 2. Hyperband Schedule Invariant
    schedule = compute_hyperband_schedule(R=81, eta=3)
    err_hb = float(np.max(np.abs(schedule - weights["golden_hb_schedule"])))
    assert err_hb < atol, f"Hyperband invariant failed: err={err_hb:.2e} >= {atol:.2e}"
    errors["hyperband_err"] = err_hb

    # 3. TPE Log Ratio Invariant
    tpe_ratio = compute_tpe_log_ratio(weights["x_query"], weights["good_samples"], weights["bad_samples"], bandwidth=0.5)
    err_tpe = float(np.max(np.abs(tpe_ratio - weights["golden_tpe_log_ratio"])))
    assert err_tpe < atol, f"TPE invariant failed: err={err_tpe:.2e} >= {atol:.2e}"
    errors["tpe_err"] = err_tpe

    # 4. ASHA Invariant
    promoted_0 = check_asha_promotion(weights["rung_losses"], float(weights["candidate_losses"][0]), eta=3)
    promoted_1 = check_asha_promotion(weights["rung_losses"], float(weights["candidate_losses"][1]), eta=3)
    asha_promoted = np.array([promoted_0, promoted_1], dtype=np.float32)
    err_asha = float(np.max(np.abs(asha_promoted - weights["golden_asha_promoted"])))
    assert err_asha < atol, f"ASHA invariant failed: err={err_asha:.2e} >= {atol:.2e}"
    errors["asha_err"] = err_asha

    # 5. Linear Anchor Invariant
    anchor_out = linear_anchor_forward(weights["anchor_x"], weights["anchor_w"], weights["anchor_b"])
    err_anchor = float(np.max(np.abs(anchor_out - weights["golden_anchor_out"])))
    assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
    errors["anchor_err"] = err_anchor

    return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed.safetensors")
    errs = verify_invariants(p)
    print(f"JAX Invariant Parity Checks Passed: {errs}")
