"""PyTorch 2.x implementation for Category 04 Topic 17:
Self-Optimising Hypergradients, Learned Optimisers, and Meta-Reweighting.

Provides production-grade implementations of:
1. Hypergradient Descent (HD) online learning rate adaptation (Baydin et al., 2018).
2. Meta-Sample Reweighting for noisy label robustness (Ren et al., 2018).
3. Coordinate-wise Learned Optimizer MLP step (Andrychowicz et al., 2016; Metz et al., 2022).
4. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file


def step_hypergradient_descent(
    w: torch.Tensor,
    g_curr: torch.Tensor,
    g_prev: torch.Tensor,
    lr: torch.Tensor,
    hyper_lr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hypergradient Descent (HD) update (Baydin et al., 2018).

    Updates learning rate using inner product of consecutive gradients:
    lr_{t+1} = lr_t + hyper_lr * <g_curr, g_prev>
    w_{t+1} = w_t - lr_{t+1} * g_curr
    """
    h_t = torch.dot(g_curr, g_prev)
    lr_next = lr + hyper_lr * h_t
    w_next = w - lr_next * g_curr
    return w_next, lr_next


def meta_sample_reweight(
    per_sample_grads: torch.Tensor,
    val_grad: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Meta-Sample Reweighting based on gradient alignment with validation set (Ren et al., 2018).

    Calculates sample alignment s_i = <G_i, val_grad>, rectifies negative alignment,
    normalizes across batch, and returns reweighted gradient.
    """
    alignments = per_sample_grads @ val_grad
    weights_relu = torch.clamp(alignments, min=0.0)
    sum_weights = torch.sum(weights_relu) + eps
    norm_weights = weights_relu / sum_weights
    reweighted_grad = norm_weights @ per_sample_grads
    return norm_weights, reweighted_grad


def step_learned_optimizer(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate-wise Learned Optimizer MLP step (Andrychowicz et al., 2016).

    Predicts coordinate update delta_w = MLP([g, m]) and updates momentum.
    """
    coords_in = torch.stack([grad, momentum], dim=-1) # [D, 2]
    h = torch.relu(coords_in @ w1 + b1) # [D, 4]
    delta_w = (h @ w2 + b2).squeeze(-1) # [D]
    p_next = param - delta_w
    m_next = 0.9 * momentum + 0.1 * grad
    return p_next, m_next, delta_w


class LinearAnchor(nn.Module):
    """Linear anchor module for cross-track parity verification."""

    def __init__(self, in_features: int = 8, out_features: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class HypergradientsUnifiedModule(nn.Module):
    """Unified module verifying self-optimising hypergradients against golden safetensors references."""

    def __init__(self, safetensors_path: str | Path) -> None:
        super().__init__()
        weights = load_file(str(safetensors_path))

        # Hypergradient Descent
        self.w_hd = weights["w_hd"]
        self.g_curr = weights["g_curr"]
        self.g_prev = weights["g_prev"]
        self.lr_init = weights["lr_init"]
        self.hyper_lr = weights["hyper_lr"]
        self.golden_lr_next = weights["golden_lr_next"]
        self.golden_w_hd = weights["golden_w_hd"]

        # Meta-Sample Reweighting
        self.per_sample_grads = weights["per_sample_grads"]
        self.val_grad = weights["val_grad"]
        self.golden_sample_weights = weights["golden_sample_weights"]
        self.golden_reweighted_grad = weights["golden_reweighted_grad"]

        # Learned Optimizer MLP
        self.param_learned = weights["param_learned"]
        self.grad_learned = weights["grad_learned"]
        self.momentum_learned = weights["momentum_learned"]
        self.mlp_w1 = weights["mlp_w1"]
        self.mlp_b1 = weights["mlp_b1"]
        self.mlp_w2 = weights["mlp_w2"]
        self.mlp_b2 = weights["mlp_b2"]
        self.golden_delta_w = weights["golden_delta_w"]
        self.golden_param_learned = weights["golden_param_learned"]
        self.golden_momentum_learned = weights["golden_momentum_learned"]

        # Linear Anchor
        self.anchor = LinearAnchor(8, 4)
        self.anchor.fc.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.fc.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. Hypergradient Descent
        w_hd_out, lr_hd_out = step_hypergradient_descent(
            self.w_hd, self.g_curr, self.g_prev, self.lr_init, self.hyper_lr
        )
        err_lr = float(torch.max(torch.abs(lr_hd_out - self.golden_lr_next)).item())
        err_w_hd = float(torch.max(torch.abs(w_hd_out - self.golden_w_hd)).item())
        err_hd = max(err_lr, err_w_hd)
        assert err_hd < atol, f"HD invariant failed: err={err_hd:.2e} >= {atol:.2e}"
        errors["hd_err"] = err_hd

        # 2. Meta-Sample Reweighting
        weights_out, reweighted_g_out = meta_sample_reweight(
            self.per_sample_grads, self.val_grad
        )
        err_weights = float(torch.max(torch.abs(weights_out - self.golden_sample_weights)).item())
        err_rew_g = float(torch.max(torch.abs(reweighted_g_out - self.golden_reweighted_grad)).item())
        err_reweight = max(err_weights, err_rew_g)
        assert err_reweight < atol, f"Reweight invariant failed: err={err_reweight:.2e} >= {atol:.2e}"
        errors["reweight_err"] = err_reweight

        # 3. Learned Optimizer MLP
        p_learned_out, m_learned_out, delta_w_out = step_learned_optimizer(
            self.param_learned,
            self.grad_learned,
            self.momentum_learned,
            self.mlp_w1,
            self.mlp_b1,
            self.mlp_w2,
            self.mlp_b2,
        )
        err_p_l = float(torch.max(torch.abs(p_learned_out - self.golden_param_learned)).item())
        err_m_l = float(torch.max(torch.abs(m_learned_out - self.golden_momentum_learned)).item())
        err_d_w = float(torch.max(torch.abs(delta_w_out - self.golden_delta_w)).item())
        err_learned = max(err_p_l, err_m_l, err_d_w)
        assert err_learned < atol, f"Learned optimizer invariant failed: err={err_learned:.2e} >= {atol:.2e}"
        errors["learned_err"] = err_learned

        # 4. Linear Anchor
        anchor_out = self.anchor(self.anchor_x)
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())
        assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    from safetensors.torch import save_file

    # Set deterministic seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Dimensions
    D = 8
    B = 4

    # 1. Hypergradient Descent (HD)
    lr_init = torch.tensor([0.01], dtype=torch.float32)
    hyper_lr = torch.tensor([0.001], dtype=torch.float32)
    w_hd = torch.randn(D, dtype=torch.float32)
    g_curr = torch.randn(D, dtype=torch.float32)
    g_prev = torch.randn(D, dtype=torch.float32)

    # HD Step: h_t = g_curr . g_prev, lr_next = lr + hyper_lr * h_t, w_next = w - lr_next * g_curr
    h_t = torch.dot(g_curr, g_prev)
    lr_next = lr_init + hyper_lr * h_t
    w_next_hd = w_hd - lr_next * g_curr

    # 2. Meta-Sample Reweighting
    per_sample_grads = torch.randn(B, D, dtype=torch.float32)
    # Corrupt sample 1 intentionally so its gradient opposes val_grad
    val_grad = torch.randn(D, dtype=torch.float32)
    per_sample_grads[1] = -val_grad * 1.5

    # Alignment: s_i = G_i . val_grad
    alignments = per_sample_grads @ val_grad
    weights_relu = torch.clamp(alignments, min=0.0)
    sum_weights = torch.sum(weights_relu) + 1e-8
    normalized_weights = weights_relu / sum_weights
    reweighted_grad = normalized_weights @ per_sample_grads

    # 3. Learned Coordinate-Wise Optimizer MLP
    param_learned = torch.randn(D, dtype=torch.float32)
    grad_learned = torch.randn(D, dtype=torch.float32)
    momentum_learned = torch.randn(D, dtype=torch.float32)

    mlp_w1 = torch.randn(2, 4, dtype=torch.float32) * 0.5
    mlp_b1 = torch.zeros(4, dtype=torch.float32)
    mlp_w2 = torch.randn(4, 1, dtype=torch.float32) * 0.5
    mlp_b2 = torch.zeros(1, dtype=torch.float32)

    # Coordinate-wise MLP forward
    # Input: [D, 2] stacked (grad, momentum)
    coords_in = torch.stack([grad_learned, momentum_learned], dim=-1) # [D, 2]
    h_mlp = torch.relu(coords_in @ mlp_w1 + mlp_b1) # [D, 4]
    delta_w = (h_mlp @ mlp_w2 + mlp_b2).squeeze(-1) # [D]
    p_learned_next = param_learned - delta_w
    m_learned_next = 0.9 * momentum_learned + 0.1 * grad_learned

    # 4. Standard Linear Anchor
    anchor_x = torch.randn(2, 8, dtype=torch.float32)
    anchor_w = torch.randn(8, 4, dtype=torch.float32)
    anchor_b = torch.randn(4, dtype=torch.float32)
    golden_anchor_out = anchor_x @ anchor_w + anchor_b

    tensors = {
        # Hypergradient Descent
        "w_hd": w_hd.contiguous(),
        "g_curr": g_curr.contiguous(),
        "g_prev": g_prev.contiguous(),
        "lr_init": lr_init.contiguous(),
        "hyper_lr": hyper_lr.contiguous(),
        "golden_lr_next": lr_next.contiguous(),
        "golden_w_hd": w_next_hd.contiguous(),

        # Meta-Sample Reweighting
        "per_sample_grads": per_sample_grads.contiguous(),
        "val_grad": val_grad.contiguous(),
        "golden_sample_weights": normalized_weights.contiguous(),
        "golden_reweighted_grad": reweighted_grad.contiguous(),

        # Learned Optimizer MLP
        "param_learned": param_learned.contiguous(),
        "grad_learned": grad_learned.contiguous(),
        "momentum_learned": momentum_learned.contiguous(),
        "mlp_w1": mlp_w1.contiguous(),
        "mlp_b1": mlp_b1.contiguous(),
        "mlp_w2": mlp_w2.contiguous(),
        "mlp_b2": mlp_b2.contiguous(),
        "golden_delta_w": delta_w.contiguous(),
        "golden_param_learned": p_learned_next.contiguous(),
        "golden_momentum_learned": m_learned_next.contiguous(),

        # Anchor & root standard
        "anchor_x": anchor_x.clone().contiguous(),
        "anchor_w": anchor_w.clone().contiguous(),
        "anchor_b": anchor_b.clone().contiguous(),
        "golden_anchor_out": golden_anchor_out.clone().contiguous(),
        "test_input": anchor_x.clone().contiguous(),
        "weight": anchor_w.clone().contiguous(),
        "bias": anchor_b.clone().contiguous(),
        "test_output": golden_anchor_out.clone().contiguous(),
    }

    out_dir = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting.safetensors"

    save_file(tensors, str(ckpt_path))
    print(f"Saved {len(tensors)} tensors to {ckpt_path}")


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting.safetensors")
    module = HypergradientsUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Invariant Parity Checks Passed: {errs}")
