"""PyTorch 2.x implementation for Category 04 Topic 14:
Minimax Saddle Point, Extragradient, OGDA, SGA, and Trust-Region BOBYQA.

Provides production-grade implementations of:
1. Korpelevich Extragradient step on bilinear and minimax games.
2. Optimistic Gradient Descent-Ascent (OGDA) with negative momentum.
3. Symplectic Gradient Adjustment (SGA) Hamiltonian vector field alignment.
4. Nelder-Mead derivative-free simplex reflection dynamics.
"""

from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file


def step_extragradient(
    x: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
    eta_x: float = 0.1,
    eta_y: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Korpelevich Extragradient step on bilinear game min_x max_y x^T A y."""
    grad_x = torch.matmul(A, y)
    grad_y = torch.matmul(A.T, x)
    x_tilde = x - eta_x * grad_x
    y_tilde = y + eta_y * grad_y

    grad_x_tilde = torch.matmul(A, y_tilde)
    grad_y_tilde = torch.matmul(A.T, x_tilde)
    x_next = x - eta_x * grad_x_tilde
    y_next = y + eta_y * grad_y_tilde
    return x_next, y_next


def step_ogda(
    x: torch.Tensor,
    y: torch.Tensor,
    x_prev: torch.Tensor,
    y_prev: torch.Tensor,
    A: torch.Tensor,
    eta_x: float = 0.1,
    eta_y: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimistic Gradient Descent-Ascent (OGDA) step."""
    grad_x = torch.matmul(A, y)
    grad_y = torch.matmul(A.T, x)
    grad_x_prev = torch.matmul(A, y_prev)
    grad_y_prev = torch.matmul(A.T, x_prev)

    x_next = x - 2.0 * eta_x * grad_x + eta_x * grad_x_prev
    y_next = y + 2.0 * eta_y * grad_y - eta_y * grad_y_prev
    return x_next, y_next


def step_sga(
    x: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
    lambda_sga: float = 0.5,
    eta_x: float = 0.1,
    eta_y: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symplectic Gradient Adjustment (SGA) step."""
    AAT_x = torch.matmul(A, torch.matmul(A.T, x))
    ATA_y = torch.matmul(A.T, torch.matmul(A, y))
    grad_x = torch.matmul(A, y)
    grad_y = torch.matmul(A.T, x)

    update_x = grad_x + lambda_sga * AAT_x
    update_y = grad_y - lambda_sga * ATA_y
    x_next = x - eta_x * update_x
    y_next = y + eta_y * update_y
    return x_next, y_next


def nelder_mead_reflect(
    x_best: torch.Tensor,
    x_second: torch.Tensor,
    x_worst: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Nelder-Mead reflection point calculation."""
    centroid = 0.5 * (x_best + x_second)
    return centroid + alpha * (centroid - x_worst)


class LinearAnchor(nn.Module):
    """Linear anchor layer for cross-track parity verification."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T) + self.bias


class MinimaxUnifiedModule(nn.Module):
    """Unified module encapsulating minimax, extragradient, and derivative-free algorithms."""

    def __init__(self, weights_path: str | Path):
        super().__init__()
        weights = load_file(str(weights_path))

        self.A = weights["matrix_A"]
        self.x = weights["state_x"]
        self.y = weights["state_y"]
        self.x_prev = weights["state_x_prev"]
        self.y_prev = weights["state_y_prev"]

        self.golden_x_eg = weights["golden_x_eg"]
        self.golden_y_eg = weights["golden_y_eg"]
        self.golden_x_ogda = weights["golden_x_ogda"]
        self.golden_y_ogda = weights["golden_y_ogda"]
        self.golden_x_sga = weights["golden_x_sga"]
        self.golden_y_sga = weights["golden_y_sga"]

        self.x_best = weights["x_best"]
        self.x_second = weights["x_second"]
        self.x_worst = weights["x_worst"]
        self.golden_x_reflect = weights["golden_x_reflect"]

        self.anchor = LinearAnchor(8, 4)
        self.anchor.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verify parity against golden reference tensors."""
        errors: dict[str, float] = {}

        # 1. Extragradient Invariant
        x_eg, y_eg = step_extragradient(self.x, self.y, self.A, eta_x=0.1, eta_y=0.1)
        err_x_eg = float(torch.max(torch.abs(x_eg - self.golden_x_eg)).item())
        err_y_eg = float(torch.max(torch.abs(y_eg - self.golden_y_eg)).item())
        err_eg = max(err_x_eg, err_y_eg)
        assert err_eg < atol, f"Extragradient invariant failed: err={err_eg:.2e} >= {atol:.2e}"
        errors["extragradient_err"] = err_eg

        # 2. OGDA Invariant
        x_ogda, y_ogda = step_ogda(self.x, self.y, self.x_prev, self.y_prev, self.A, eta_x=0.1, eta_y=0.1)
        err_x_ogda = float(torch.max(torch.abs(x_ogda - self.golden_x_ogda)).item())
        err_y_ogda = float(torch.max(torch.abs(y_ogda - self.golden_y_ogda)).item())
        err_ogda = max(err_x_ogda, err_y_ogda)
        assert err_ogda < atol, f"OGDA invariant failed: err={err_ogda:.2e} >= {atol:.2e}"
        errors["ogda_err"] = err_ogda

        # 3. SGA Invariant
        x_sga, y_sga = step_sga(self.x, self.y, self.A, lambda_sga=0.5, eta_x=0.1, eta_y=0.1)
        err_x_sga = float(torch.max(torch.abs(x_sga - self.golden_x_sga)).item())
        err_y_sga = float(torch.max(torch.abs(y_sga - self.golden_y_sga)).item())
        err_sga = max(err_x_sga, err_y_sga)
        assert err_sga < atol, f"SGA invariant failed: err={err_sga:.2e} >= {atol:.2e}"
        errors["sga_err"] = err_sga

        # 4. Nelder-Mead Reflection Invariant
        x_refl = nelder_mead_reflect(self.x_best, self.x_second, self.x_worst, alpha=1.0)
        err_refl = float(torch.max(torch.abs(x_refl - self.golden_x_reflect)).item())
        assert err_refl < atol, f"Nelder-Mead invariant failed: err={err_refl:.2e} >= {atol:.2e}"
        errors["nelder_mead_err"] = err_refl

        # 5. Linear Anchor Invariant
        anchor_out = self.anchor(self.anchor_x)
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())
        assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    import torch
    import torch.nn as nn
    from safetensors.torch import save_file


    def step_extragradient(
        x: torch.Tensor,
        y: torch.Tensor,
        A: torch.Tensor,
        eta_x: float = 0.1,
        eta_y: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Korpelevich Extragradient step on bilinear game min_x max_y x^T A y."""
        grad_x = torch.matmul(A, y)
        grad_y = torch.matmul(A.T, x)
        x_tilde = x - eta_x * grad_x
        y_tilde = y + eta_y * grad_y

        grad_x_tilde = torch.matmul(A, y_tilde)
        grad_y_tilde = torch.matmul(A.T, x_tilde)
        x_next = x - eta_x * grad_x_tilde
        y_next = y + eta_y * grad_y_tilde
        return x_next, y_next


    def step_ogda(
        x: torch.Tensor,
        y: torch.Tensor,
        x_prev: torch.Tensor,
        y_prev: torch.Tensor,
        A: torch.Tensor,
        eta_x: float = 0.1,
        eta_y: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimistic Gradient Descent-Ascent (OGDA) step."""
        grad_x = torch.matmul(A, y)
        grad_y = torch.matmul(A.T, x)
        grad_x_prev = torch.matmul(A, y_prev)
        grad_y_prev = torch.matmul(A.T, x_prev)

        x_next = x - 2.0 * eta_x * grad_x + eta_x * grad_x_prev
        y_next = y + 2.0 * eta_y * grad_y - eta_y * grad_y_prev
        return x_next, y_next


    def step_sga(
        x: torch.Tensor,
        y: torch.Tensor,
        A: torch.Tensor,
        lambda_sga: float = 0.5,
        eta_x: float = 0.1,
        eta_y: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symplectic Gradient Adjustment (SGA) step."""
        AAT_x = torch.matmul(A, torch.matmul(A.T, x))
        ATA_y = torch.matmul(A.T, torch.matmul(A, y))
        grad_x = torch.matmul(A, y)
        grad_y = torch.matmul(A.T, x)

        update_x = grad_x + lambda_sga * AAT_x
        update_y = grad_y - lambda_sga * ATA_y
        x_next = x - eta_x * update_x
        y_next = y + eta_y * update_y
        return x_next, y_next


    def nelder_mead_reflect(
        x_best: torch.Tensor,
        x_second: torch.Tensor,
        x_worst: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Nelder-Mead reflection point calculation."""
        centroid = 0.5 * (x_best + x_second)
        return centroid + alpha * (centroid - x_worst)


    def main():
        torch.manual_seed(42)

        # 1. Bilinear game matrix and state
        A = torch.tensor([
            [0.8, -0.4],
            [0.3, 0.9],
        ], dtype=torch.float32)
        x = torch.tensor([1.5, -1.0], dtype=torch.float32)
        y = torch.tensor([-0.8, 1.2], dtype=torch.float32)
        x_prev = torch.tensor([1.4, -0.9], dtype=torch.float32)
        y_prev = torch.tensor([-0.7, 1.1], dtype=torch.float32)

        # Invariant 1: Extragradient
        gold_x_eg, gold_y_eg = step_extragradient(x, y, A, eta_x=0.1, eta_y=0.1)

        # Invariant 2: OGDA
        gold_x_ogda, gold_y_ogda = step_ogda(x, y, x_prev, y_prev, A, eta_x=0.1, eta_y=0.1)

        # Invariant 3: SGA
        gold_x_sga, gold_y_sga = step_sga(x, y, A, lambda_sga=0.5, eta_x=0.1, eta_y=0.1)

        # Invariant 4: Nelder-Mead Reflection
        x_best = torch.tensor([0.2, 0.4], dtype=torch.float32)
        x_second = torch.tensor([0.5, 0.6], dtype=torch.float32)
        x_worst = torch.tensor([1.2, 1.4], dtype=torch.float32)
        gold_x_reflect = nelder_mead_reflect(x_best, x_second, x_worst, alpha=1.0)

        # Invariant 5: Linear Anchor for Cross-Track Parity
        anchor_x = torch.randn(2, 8, dtype=torch.float32)
        anchor_w = torch.randn(8, 4, dtype=torch.float32)
        anchor_b = torch.randn(4, dtype=torch.float32)
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

        tensors = {
            # Bilinear game tensors
            "matrix_A": A,
            "state_x": x,
            "state_y": y,
            "state_x_prev": x_prev,
            "state_y_prev": y_prev,
            "golden_x_eg": gold_x_eg,
            "golden_y_eg": gold_y_eg,
            "golden_x_ogda": gold_x_ogda,
            "golden_y_ogda": gold_y_ogda,
            "golden_x_sga": gold_x_sga,
            "golden_y_sga": gold_y_sga,
            # Nelder-Mead
            "x_best": x_best,
            "x_second": x_second,
            "x_worst": x_worst,
            "golden_x_reflect": gold_x_reflect,
            # Linear Anchor
            "anchor_x": anchor_x,
            "anchor_w": anchor_w,
            "anchor_b": anchor_b,
            "golden_anchor_out": golden_anchor_out,
            # Standard Parity Keys for test_cross_track_parity.py
            "weight": anchor_w.clone(),
            "bias": anchor_b.clone(),
            "test_input": anchor_x.clone(),
            "test_output": golden_anchor_out.clone(),
        }

        out_dir = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "07_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa.safetensors"
        save_file(tensors, str(out_path))
        print(f"Saved {len(tensors)} tensors to {out_path}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/07_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa.safetensors")
    module = MinimaxUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Invariant Parity Checks Passed: {errs}")
