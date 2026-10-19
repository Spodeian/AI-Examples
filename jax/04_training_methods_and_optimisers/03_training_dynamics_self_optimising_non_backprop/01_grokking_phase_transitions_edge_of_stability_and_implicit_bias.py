"""JAX / Functional implementation for Category 04 Topic 15:
Grokking, Phase Transitions, Edge of Stability, and Implicit Bias.

Provides pure functional, memory-efficient implementations of:
1. Grokfast EMA gradient filtering for accelerating phase transition / delayed generalization.
2. Edge of Stability (EOS) cubic oscillator and self-stabilizing dynamics.
3. Implicit Bias logistic margin step on linearly separable data.
4. Linear anchor evaluation for transitive multi-track parity.
"""

from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def step_grokfast(
    g_raw: np.ndarray,
    mu_prev: np.ndarray,
    alpha: float = 0.8,
    lambda_gf: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Grokfast EMA gradient filtering step (Lee et al., 2024)."""
    mu_next = alpha * mu_prev + (1.0 - alpha) * g_raw
    g_filtered = g_raw + lambda_gf * mu_next
    return g_filtered, mu_next


def step_edge_of_stability(
    x: np.ndarray,
    H: np.ndarray,
    gamma: np.ndarray,
    eta: float = 0.1,
) -> np.ndarray:
    """Edge of Stability (EOS) cubic oscillator step (Cohen et al., 2021)."""
    grad = H * x + gamma * (x ** 3)
    x_next = x - eta * grad
    return x_next


def step_implicit_bias(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    eta: float = 0.2,
) -> np.ndarray:
    """Implicit bias gradient step on separable linear classification (Soudry et al., 2018)."""
    margins = y * np.matmul(X, w)
    sigmoids = 1.0 / (1.0 + np.exp(margins))
    grad = -np.mean((y[:, None] * X) * sigmoids[:, None], axis=0)
    w_next = w - eta * grad
    return w_next


class TrainingDynamicsFunctionalModule:
    """Functional module verifying training dynamics invariants against safetensors golden references."""

    def __init__(self, safetensors_path: str | Path) -> None:
        weights = load_file(str(safetensors_path))

        # Grokfast
        self.g_raw = weights["g_raw"]
        self.mu_prev = weights["mu_prev"]
        self.golden_mu = weights["golden_mu"]
        self.golden_g_grokfast = weights["golden_g_grokfast"]

        # EOS
        self.x_eos = weights["x_eos"]
        self.h_eos = weights["h_eos"]
        self.gamma_eos = weights["gamma_eos"]
        self.golden_x_eos = weights["golden_x_eos"]

        # Implicit Bias
        self.w_sep = weights["w_sep"]
        self.x_sep = weights["x_sep"]
        self.y_sep = weights["y_sep"]
        self.golden_w_sep = weights["golden_w_sep"]

        # Linear Anchor
        self.anchor_x = weights["anchor_x"]
        self.anchor_w = weights["anchor_w"]
        self.anchor_b = weights["anchor_b"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. Grokfast invariant
        g_gf, mu_next = step_grokfast(self.g_raw, self.mu_prev, alpha=0.8, lambda_gf=2.0)
        err_mu = float(np.max(np.abs(mu_next - self.golden_mu)))
        err_gf = float(np.max(np.abs(g_gf - self.golden_g_grokfast)))
        err_grok = max(err_mu, err_gf)
        assert err_grok < atol, f"Grokfast invariant failed: err={err_grok:.2e} >= {atol:.2e}"
        errors["grokfast_err"] = err_grok

        # 2. EOS invariant
        x_next = step_edge_of_stability(self.x_eos, self.h_eos, self.gamma_eos, eta=0.1)
        err_eos = float(np.max(np.abs(x_next - self.golden_x_eos)))
        assert err_eos < atol, f"EOS invariant failed: err={err_eos:.2e} >= {atol:.2e}"
        errors["eos_err"] = err_eos

        # 3. Implicit bias invariant
        w_next = step_implicit_bias(self.w_sep, self.x_sep, self.y_sep, eta=0.2)
        err_bias = float(np.max(np.abs(w_next - self.golden_w_sep)))
        assert err_bias < atol, f"Implicit bias invariant failed: err={err_bias:.2e} >= {atol:.2e}"
        errors["implicit_bias_err"] = err_bias

        # 4. Linear anchor invariant
        anchor_out = np.matmul(self.anchor_x, self.anchor_w) + self.anchor_b
        err_anchor = float(np.max(np.abs(anchor_out - self.golden_anchor_out)))
        assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/01_grokking_phase_transitions_edge_of_stability_and_implicit_bias.safetensors")
    module = TrainingDynamicsFunctionalModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"JAX / Functional Invariant Parity Checks Passed: {errs}")
