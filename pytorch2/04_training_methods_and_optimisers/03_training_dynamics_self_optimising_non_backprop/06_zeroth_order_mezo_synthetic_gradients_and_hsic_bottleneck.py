"""PyTorch 2.x implementation for Category 04 Topic 20:
Zeroth-Order Optimization: MeZO, Synthetic Gradients (DNI), and HSIC Bottleneck.

Provides production-grade implementations of:
1. MeZO (Memory-Efficient Zeroth-Order Optimizer) directional derivative and step (Malladi et al., 2023).
2. Decoupled Neural Interfaces (DNI) / Synthetic Gradients prediction and MSE loss (Jaderberg et al., 2017).
3. Hilbert-Schmidt Independence Criterion (HSIC) Bottleneck estimation (Ma et al., 2020).
4. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def mezo_step(
    theta: torch.Tensor,
    z: torch.Tensor,
    theta_star: torch.Tensor,
    eps: float = 1e-3,
    eta: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MeZO (Memory-Efficient Zeroth-Order) optimization step (Malladi et al., 2023).

    theta: [D], z: [D], theta_star: [D]
    Returns (loss_pos, loss_neg, dir_deriv, theta_next).
    """
    def loss_fn(p: torch.Tensor) -> torch.Tensor:
        diff = p - theta_star
        return 0.5 * torch.sum(diff ** 2)

    loss_pos = loss_fn(theta + eps * z)
    loss_neg = loss_fn(theta - eps * z)
    dir_deriv = (loss_pos - loss_neg) / (2.0 * eps)
    theta_next = theta - eta * dir_deriv * z
    return loss_pos, loss_neg, dir_deriv, theta_next


def dni_synthetic_gradient(
    h: torch.Tensor,
    w_synth: torch.Tensor,
    b_synth: torch.Tensor,
    target_grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoupled Neural Interfaces (DNI) synthetic gradient computation (Jaderberg et al., 2017).

    h: [B, D], w_synth: [D, D], b_synth: [D], target_grad: [B, D]
    Returns (pred_grad, dni_loss).
    """
    pred_grad = torch.matmul(h, w_synth) + b_synth
    dni_loss = 0.5 * torch.mean((pred_grad - target_grad) ** 2)
    return pred_grad, dni_loss


def hsic_bottleneck(
    h: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion (HSIC) with linear kernels (Ma et al., 2020).

    h: [B, D], y: [B, C]
    Returns scalar HSIC score.
    """
    b = h.shape[0]
    k_mat = torch.matmul(h, h.T)
    l_mat = torch.matmul(y, y.T)

    eye = torch.eye(b, dtype=h.dtype, device=h.device)
    ones = torch.ones(b, b, dtype=h.dtype, device=h.device) / float(b)
    h_c = eye - ones

    k_c = torch.matmul(torch.matmul(h_c, k_mat), h_c)
    l_c = torch.matmul(torch.matmul(h_c, l_mat), h_c)

    score = (1.0 / ((b - 1.0) ** 2)) * torch.trace(torch.matmul(k_c, l_c))
    return score


class LinearAnchor(nn.Module):
    """Anchor module for multi-track parity verification."""

    def __init__(self, in_features: int = 8, out_features: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ZerothOrderUnifiedModule:
    """Unified container for Category 04 Topic 20 verification."""

    def __init__(self, weights_path: Path):
        weights = load_file(str(weights_path))

        # MeZO tensors
        self.theta = weights["theta"]
        self.z = weights["z"]
        self.theta_star = weights["theta_star"]
        self.golden_loss_pos = weights["golden_loss_pos"]
        self.golden_loss_neg = weights["golden_loss_neg"]
        self.golden_dir_deriv = weights["golden_dir_deriv"]
        self.golden_theta_next = weights["golden_theta_next"]

        # DNI tensors
        self.h = weights["h"]
        self.w_synth = weights["w_synth"]
        self.b_synth = weights["b_synth"]
        self.target_grad = weights["target_grad"]
        self.golden_pred_grad = weights["golden_pred_grad"]
        self.golden_dni_loss = weights["golden_dni_loss"]

        # HSIC tensors
        self.hsic_h = weights["hsic_h"]
        self.hsic_y = weights["hsic_y"]
        self.golden_hsic_score = weights["golden_hsic_score"]

        # Linear Anchor
        self.anchor = LinearAnchor(8, 4)
        self.anchor.fc.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.fc.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. MeZO invariant
        loss_p, loss_n, dir_d, th_next = mezo_step(
            self.theta, self.z, self.theta_star, eps=1e-3, eta=0.05
        )
        err_lp = float(torch.abs(loss_p - self.golden_loss_pos[0]).item())
        err_ln = float(torch.abs(loss_n - self.golden_loss_neg[0]).item())
        err_dd = float(torch.abs(dir_d - self.golden_dir_deriv[0]).item())
        err_th = float(torch.max(torch.abs(th_next - self.golden_theta_next)).item())
        err_mezo = max(err_lp, err_ln, err_dd, err_th)
        assert err_mezo < atol, f"MeZO invariant failed: {err_mezo:.2e}"
        errors["mezo_err"] = err_mezo

        # 2. DNI invariant
        pred_g, dni_l = dni_synthetic_gradient(
            self.h, self.w_synth, self.b_synth, self.target_grad
        )
        err_pg = float(torch.max(torch.abs(pred_g - self.golden_pred_grad)).item())
        err_dl = float(torch.abs(dni_l - self.golden_dni_loss[0]).item())
        err_dni = max(err_pg, err_dl)
        assert err_dni < atol, f"DNI invariant failed: {err_dni:.2e}"
        errors["dni_err"] = err_dni

        # 3. HSIC invariant
        hsic_s = hsic_bottleneck(self.hsic_h, self.hsic_y)
        err_hsic = float(torch.abs(hsic_s - self.golden_hsic_score[0]).item())
        assert err_hsic < atol, f"HSIC invariant failed: {err_hsic:.2e}"
        errors["hsic_err"] = err_hsic

        # 4. Linear Anchor invariant
        anchor_out = self.anchor(self.anchor_x)
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import math
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import save_file

    def generate_topic_20_artifacts():
        torch.manual_seed(42)
        np.random.seed(42)

        # 1. MeZO setup
        # theta: [8], z: [8], theta_star: [8]
        theta = torch.randn(8, dtype=torch.float32)
        z = torch.randn(8, dtype=torch.float32)
        # Normalize z to unit sphere or keep standard normal
        z = z / torch.norm(z)
        theta_star = torch.randn(8, dtype=torch.float32)

        eps = 1e-3
        eta = 0.05

        def loss_fn(p: torch.Tensor) -> torch.Tensor:
            diff = p - theta_star
            return 0.5 * torch.sum(diff ** 2)

        loss_pos = loss_fn(theta + eps * z)
        loss_neg = loss_fn(theta - eps * z)
        dir_deriv = (loss_pos - loss_neg) / (2.0 * eps)
        theta_next = theta - eta * dir_deriv * z

        # 2. Decoupled Neural Interfaces (DNI) / Synthetic Gradients
        # h: [4, 8], w_synth: [8, 8], b_synth: [8], target_grad: [4, 8]
        h = torch.randn(4, 8, dtype=torch.float32)
        w_synth = torch.randn(8, 8, dtype=torch.float32) / math.sqrt(8)
        b_synth = torch.randn(8, dtype=torch.float32) * 0.1
        target_grad = torch.randn(4, 8, dtype=torch.float32)

        pred_grad = torch.matmul(h, w_synth) + b_synth # [4, 8]
        dni_loss = 0.5 * torch.mean((pred_grad - target_grad) ** 2) # scalar

        # 3. HSIC Bottleneck
        # Batch = 4, feature dim = 8, label dim = 4
        hsic_h = torch.randn(4, 8, dtype=torch.float32)
        hsic_y = torch.randn(4, 4, dtype=torch.float32)

        # Linear kernels: K = H H^T [4, 4], L = Y Y^T [4, 4]
        k_mat = torch.matmul(hsic_h, hsic_h.T) # [4, 4]
        l_mat = torch.matmul(hsic_y, hsic_y.T) # [4, 4]

        # Centering matrix H_c = I - 1/B * 1 1^T
        eye4 = torch.eye(4, dtype=torch.float32)
        ones4 = torch.ones(4, 4, dtype=torch.float32) / 4.0
        h_c = eye4 - ones4

        k_centered = torch.matmul(torch.matmul(h_c, k_mat), h_c)
        l_centered = torch.matmul(torch.matmul(h_c, l_mat), h_c)

        # HSIC = 1 / (B - 1)^2 * Tr(K_c L_c) = 1/9 * sum(K_c * L_c^T)
        hsic_score = (1.0 / 9.0) * torch.trace(torch.matmul(k_centered, l_centered))

        # 4. Linear Anchor
        anchor_x = torch.randn(2, 8, dtype=torch.float32)
        anchor_w = torch.randn(8, 4, dtype=torch.float32) / math.sqrt(8)
        anchor_b = torch.randn(4, dtype=torch.float32) * 0.1
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

        # Construct tensors dictionary
        tensors = {
            "theta": theta.clone().contiguous(),
            "z": z.clone().contiguous(),
            "theta_star": theta_star.clone().contiguous(),
            "golden_loss_pos": loss_pos.unsqueeze(0).clone().contiguous(),
            "golden_loss_neg": loss_neg.unsqueeze(0).clone().contiguous(),
            "golden_dir_deriv": dir_deriv.unsqueeze(0).clone().contiguous(),
            "golden_theta_next": theta_next.clone().contiguous(),

            "h": h.clone().contiguous(),
            "w_synth": w_synth.clone().contiguous(),
            "b_synth": b_synth.clone().contiguous(),
            "target_grad": target_grad.clone().contiguous(),
            "golden_pred_grad": pred_grad.clone().contiguous(),
            "golden_dni_loss": dni_loss.unsqueeze(0).clone().contiguous(),

            "hsic_h": hsic_h.clone().contiguous(),
            "hsic_y": hsic_y.clone().contiguous(),
            "golden_hsic_score": hsic_score.unsqueeze(0).clone().contiguous(),

            "anchor_x": anchor_x.clone().contiguous(),
            "anchor_w": anchor_w.clone().contiguous(),
            "anchor_b": anchor_b.clone().contiguous(),
            "golden_anchor_out": golden_anchor_out.clone().contiguous(),

            # Standard test inputs/outputs for parity test runners
            "test_input": anchor_x.clone().contiguous(),
            "test_output": golden_anchor_out.clone().contiguous(),
            "weight": anchor_w.clone().contiguous(),
            "bias": anchor_b.clone().contiguous(),
        }

        out_dir = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck.safetensors"

        save_file(tensors, out_path)
        print(f"Saved {len(tensors)} tensors to {out_path}")

        # Verify self-consistency
        print(f"MeZO Loss Pos: {loss_pos.item():.6f}, Neg: {loss_neg.item():.6f}, Deriv: {dir_deriv.item():.6f}")
        print(f"DNI Loss: {dni_loss.item():.6f}")
        print(f"HSIC Score: {hsic_score.item():.6f}")

    generate_topic_20_artifacts()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck.safetensors")
    module = ZerothOrderUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Topic 20 Invariant Parity Checks Passed: {errs}")
