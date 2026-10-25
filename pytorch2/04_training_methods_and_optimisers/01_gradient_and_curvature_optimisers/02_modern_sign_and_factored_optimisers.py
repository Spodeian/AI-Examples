"""PyTorch 2.x implementation for Category 04 Topic 02:
02_modern_sign_and_factored_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.
"""

from __future__ import annotations

import math
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file

import core


def step_lion(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float = 0.0001,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    update = torch.sign(beta1 * exp_avg + (1.0 - beta1) * grad)
    new_param = param * (1.0 - lr * weight_decay) - lr * update
    new_exp_avg = beta2 * exp_avg + (1.0 - beta2) * grad
    return new_param, new_exp_avg


def step_adafactor_2d(
    param: torch.Tensor,
    grad: torch.Tensor,
    r_buf: torch.Tensor,
    c_buf: torch.Tensor,
    step: int = 2,
    lr: float = 0.001,
    beta2: float = 0.8,
    eps1: float = 1e-30,
    eps2: float = 1e-3,
    clip_threshold: float = 1.0,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decay_rate = 1.0 - (step ** (-beta2))
    s = grad * grad + eps1
    new_r = decay_rate * r_buf + (1.0 - decay_rate) * torch.mean(s, dim=1)
    new_c = decay_rate * c_buf + (1.0 - decay_rate) * torch.mean(s, dim=0)

    r_mean = torch.mean(new_r)
    v_approx = (new_r.unsqueeze(1) * new_c.unsqueeze(0)) / (r_mean + eps1)

    u = grad / (torch.sqrt(v_approx) + eps2)
    rms_u = torch.sqrt(torch.mean(u * u))
    clip_factor = torch.clamp(rms_u / clip_threshold, min=1.0)
    u_clipped = u / clip_factor

    new_param = param * (1.0 - lr * weight_decay) - lr * u_clipped
    return new_param, new_r, new_c


def step_ademamix(
    param: torch.Tensor,
    grad: torch.Tensor,
    m1_buf: torch.Tensor,
    m2_buf: torch.Tensor,
    v_buf: torch.Tensor,
    step: int = 2,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta3: float = 0.9999,
    alpha: float = 5.0,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    new_m1 = beta1 * m1_buf + (1.0 - beta1) * grad
    new_m2 = beta3 * m2_buf + (1.0 - beta3) * grad
    new_v = beta2 * v_buf + (1.0 - beta2) * (grad * grad)

    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    bc3 = 1.0 - beta3**step

    m1_hat = new_m1 / bc1
    m2_hat = new_m2 / bc3
    v_hat = new_v / bc2

    combined_m = m1_hat + alpha * m2_hat
    u = combined_m / (torch.sqrt(v_hat) + eps)

    new_param = param * (1.0 - lr * weight_decay) - lr * u
    return new_param, new_m1, new_m2, new_v


class ModernSignOptimisersUnifiedModule(nn.Module):
    """Unified module validating Lion, Adafactor, and AdEMAMix updates."""

    def __init__(self, in_dim: int = 8, out_dim: int = 8, ckpt_path: str | Path | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        if ckpt_path is not None:
            self.load_reference_weights(ckpt_path)

    def load_reference_weights(self, ckpt_path: str | Path) -> None:
        weights = load_file(str(ckpt_path))
        with torch.no_grad():
            # weight shape is [8, 8] matching Linear(in_features=8, out_features=8)
            # weights["weight"] was saved as weight.t() for test_in @ weight, so linear.weight is weight.t()
            self.linear.weight.copy_(weights["weight"].t())
            self.linear.bias.copy_(weights["bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def verify_against_checkpoint(ckpt_path: str | Path) -> None:
    weights = load_file(str(ckpt_path))

    # 1. Lion (1D)
    param_1d = weights["param_1d"]
    grad_1d = weights["grad_1d"]
    m_1d = weights["m_1d"]
    p_lion, m_lion = step_lion(param_1d, grad_1d, m_1d, lr=0.0001, beta1=0.9, beta2=0.99, weight_decay=0.1)
    err_p_lion = float(torch.max(torch.abs(p_lion - weights["p_lion"])))
    err_m_lion = float(torch.max(torch.abs(m_lion - weights["m_lion"])))
    assert err_p_lion < 1e-6 and err_m_lion < 1e-6, f"Lion error: {err_p_lion}, {err_m_lion}"

    # 2. Adafactor (2D)
    param_2d = weights["param_2d"]
    grad_2d = weights["grad_2d"]
    r_buf = weights["r_buf"]
    c_buf = weights["c_buf"]
    p_adafactor, r_adafactor, c_adafactor = step_adafactor_2d(
        param_2d, grad_2d, r_buf, c_buf, step=2, lr=0.001, beta2=0.8, weight_decay=0.01
    )
    err_p_ada = float(torch.max(torch.abs(p_adafactor - weights["p_adafactor"])))
    err_r_ada = float(torch.max(torch.abs(r_adafactor - weights["r_adafactor"])))
    err_c_ada = float(torch.max(torch.abs(c_adafactor - weights["c_adafactor"])))
    assert err_p_ada < 1e-6 and err_r_ada < 1e-6 and err_c_ada < 1e-6, f"Adafactor error: {err_p_ada}"

    # 3. AdEMAMix (1D)
    m2_1d = weights["m2_1d"]
    v_1d = weights["v_1d"]
    p_ademamix, m1_ademamix, m2_ademamix, v_ademamix = step_ademamix(
        param_1d, grad_1d, m_1d, m2_1d, v_1d, step=2, lr=0.001, beta1=0.9, beta2=0.999, beta3=0.9999, alpha=5.0, eps=1e-8, weight_decay=0.01
    )
    err_p_adem = float(torch.max(torch.abs(p_ademamix - weights["p_ademamix"])))
    err_m1_adem = float(torch.max(torch.abs(m1_ademamix - weights["m1_ademamix"])))
    err_m2_adem = float(torch.max(torch.abs(m2_ademamix - weights["m2_ademamix"])))
    err_v_adem = float(torch.max(torch.abs(v_ademamix - weights["v_ademamix"])))
    assert err_p_adem < 1e-6 and err_m1_adem < 1e-6 and err_m2_adem < 1e-6 and err_v_adem < 1e-6, f"AdEMAMix error: {err_p_adem}"

    # 4. Linear Anchor
    test_in = weights["test_input"]
    test_out = weights["test_output"]
    model = ModernSignOptimisersUnifiedModule(ckpt_path=ckpt_path)
    pred_y = model(test_in)
    err_anchor = float(torch.max(torch.abs(pred_y.detach() - test_out)))
    assert err_anchor < 1e-6, f"Linear anchor error: {err_anchor}"

    print(f"All 4 invariants verified successfully! Max error: {max(err_p_lion, err_p_ada, err_p_adem, err_anchor):.2e}")


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import math
    from pathlib import Path
    import torch
    from safetensors.torch import save_file

    import core


    def step_lion(
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        lr: float = 0.0001,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Lion (Chen et al. 2023):
        # update = sign(beta1 * exp_avg + (1 - beta1) * grad)
        # param = param * (1 - lr * weight_decay) - lr * update
        # exp_avg = beta2 * exp_avg + (1 - beta2) * grad
        update = torch.sign(beta1 * exp_avg + (1.0 - beta1) * grad)
        new_param = param * (1.0 - lr * weight_decay) - lr * update
        new_exp_avg = beta2 * exp_avg + (1.0 - beta2) * grad
        return new_param, new_exp_avg


    def step_adafactor_2d(
        param: torch.Tensor,
        grad: torch.Tensor,
        r_buf: torch.Tensor,
        c_buf: torch.Tensor,
        step: int = 2,
        lr: float = 0.001,
        beta2: float = 0.8,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Adafactor (Shazeer & Stern 2018) for 2D matrix [R, C]:
        # Factored second moment: r in R^R, c in R^C
        # beta2_t = 1 - t^(-0.8)
        decay_rate = 1.0 - (step ** (-beta2))
        # s = grad^2 + eps1
        s = grad * grad + eps1
        # r_{t+1} = beta2_t * r_t + (1 - beta2_t) * mean(s, dim=1)
        # c_{t+1} = beta2_t * c_t + (1 - beta2_t) * mean(s, dim=0)
        new_r = decay_rate * r_buf + (1.0 - decay_rate) * torch.mean(s, dim=1)
        new_c = decay_rate * c_buf + (1.0 - decay_rate) * torch.mean(s, dim=0)

        # Reconstructed variance estimate: v = (r * c^T) / mean(r)
        # Outer product: [R, 1] * [1, C]
        r_mean = torch.mean(new_r)
        v_approx = (new_r.unsqueeze(1) * new_c.unsqueeze(0)) / (r_mean + eps1)

        # update = grad / (sqrt(v_approx) + eps2)
        u = grad / (torch.sqrt(v_approx) + eps2)

        # RMS clipping: d_t = max(1, RMS(u) / clip_threshold)
        rms_u = torch.sqrt(torch.mean(u * u))
        clip_factor = torch.clamp(rms_u / clip_threshold, min=1.0)
        u_clipped = u / clip_factor

        new_param = param * (1.0 - lr * weight_decay) - lr * u_clipped
        return new_param, new_r, new_c


    def step_ademamix(
        param: torch.Tensor,
        grad: torch.Tensor,
        m1_buf: torch.Tensor,
        m2_buf: torch.Tensor,
        v_buf: torch.Tensor,
        step: int = 2,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 0.9999,
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # AdEMAMix (Pagliardini et al. 2024):
        # m1: fast momentum EMA (beta1)
        # m2: ultra-slow momentum EMA (beta3)
        # v: second moment EMA (beta2)
        new_m1 = beta1 * m1_buf + (1.0 - beta1) * grad
        new_m2 = beta3 * m2_buf + (1.0 - beta3) * grad
        new_v = beta2 * v_buf + (1.0 - beta2) * (grad * grad)

        # Bias corrections
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
        bc3 = 1.0 - beta3**step

        m1_hat = new_m1 / bc1
        m2_hat = new_m2 / bc3
        v_hat = new_v / bc2

        # Mixed momentum update: m1_hat + alpha * m2_hat
        combined_m = m1_hat + alpha * m2_hat
        u = combined_m / (torch.sqrt(v_hat) + eps)

        new_param = param * (1.0 - lr * weight_decay) - lr * u
        return new_param, new_m1, new_m2, new_v


    def main() -> None:
        core.setup_env(seed=42)
        device = torch.device("cpu")

        # Dimensions
        dim = 8
        # 1. Linear Anchor (Test standard: test_input @ weight + bias = test_output)
        gen = torch.Generator().manual_seed(42)
        weight = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
        bias = torch.randn(dim, generator=gen, dtype=torch.float32)
        test_input = torch.randn(4, dim, generator=gen, dtype=torch.float32)
        test_output = test_input @ weight + bias

        # 2. Lion setup (1D vector of length 8)
        param_1d = torch.tensor([0.5, -0.3, 1.2, -0.8, 0.1, -1.5, 0.7, -0.2], dtype=torch.float32)
        grad_1d = torch.tensor([0.04, -0.02, 0.15, -0.08, 0.01, -0.25, 0.12, -0.05], dtype=torch.float32)
        m_1d = torch.tensor([0.01, -0.005, 0.03, -0.02, 0.005, -0.05, 0.02, -0.01], dtype=torch.float32)

        p_lion, m_lion = step_lion(param_1d, grad_1d, m_1d, lr=0.0001, beta1=0.9, beta2=0.99, weight_decay=0.1)

        # 3. Adafactor setup (2D matrix of shape [4, 4])
        param_2d = torch.tensor([
            [0.4, -0.2, 0.1, -0.5],
            [-0.3, 0.6, -0.1, 0.2],
            [0.5, -0.4, 0.3, -0.1],
            [-0.2, 0.1, -0.6, 0.4]
        ], dtype=torch.float32)
        grad_2d = torch.tensor([
            [0.05, -0.01, 0.02, -0.08],
            [-0.04, 0.09, -0.02, 0.03],
            [0.07, -0.05, 0.04, -0.01],
            [-0.03, 0.02, -0.09, 0.06]
        ], dtype=torch.float32)
        r_buf = torch.tensor([0.002, 0.004, 0.003, 0.005], dtype=torch.float32)
        c_buf = torch.tensor([0.003, 0.002, 0.004, 0.005], dtype=torch.float32)

        p_adafactor, r_adafactor, c_adafactor = step_adafactor_2d(
            param_2d, grad_2d, r_buf, c_buf, step=2, lr=0.001, beta2=0.8, weight_decay=0.01
        )

        # 4. AdEMAMix setup (1D vector of length 8)
        m2_1d = torch.tensor([0.005, -0.002, 0.015, -0.01, 0.002, -0.025, 0.01, -0.005], dtype=torch.float32)
        v_1d = torch.tensor([0.002, 0.001, 0.01, 0.005, 0.001, 0.025, 0.008, 0.003], dtype=torch.float32)

        p_ademamix, m1_ademamix, m2_ademamix, v_ademamix = step_ademamix(
            param_1d, grad_1d, m_1d, m2_1d, v_1d, step=2, lr=0.001, beta1=0.9, beta2=0.999, beta3=0.9999, alpha=5.0, eps=1e-8, weight_decay=0.01
        )

        tensors = {
            # Standard Linear Anchor
            "weight": weight,
            "bias": bias,
            "test_input": test_input,
            "test_output": test_output,
            # Lion
            "param_1d": param_1d,
            "grad_1d": grad_1d,
            "m_1d": m_1d,
            "p_lion": p_lion,
            "m_lion": m_lion,
            # Adafactor
            "param_2d": param_2d,
            "grad_2d": grad_2d,
            "r_buf": r_buf,
            "c_buf": c_buf,
            "p_adafactor": p_adafactor,
            "r_adafactor": r_adafactor,
            "c_adafactor": c_adafactor,
            # AdEMAMix
            "m2_1d": m2_1d,
            "v_1d": v_1d,
            "p_ademamix": p_ademamix,
            "m1_ademamix": m1_ademamix,
            "m2_ademamix": m2_ademamix,
            "v_ademamix": v_ademamix,
        }

        out_dir = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "02_modern_sign_and_factored_optimisers.safetensors"
        save_file(tensors, str(out_path))
        print(f"Saved {len(tensors)} tensors to {out_path}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.safetensors")
    verify_against_checkpoint(p)
