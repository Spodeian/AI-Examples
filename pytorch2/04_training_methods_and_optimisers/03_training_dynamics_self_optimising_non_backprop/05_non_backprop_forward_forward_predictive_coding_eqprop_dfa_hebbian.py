"""PyTorch 2.x implementation for Category 04 Topic 19:
Non-Backprop: Forward-Forward, Predictive Coding, EqProp, DFA, and Hebbian Learning.

Provides production-grade implementations of:
1. Forward-Forward algorithm: goodness loss and layer normalization (Hinton, 2022).
2. Predictive Coding: representation error minimization step (Rao & Ballard, 1999).
3. Direct Feedback Alignment (DFA): fixed random feedback projection (Nøkland, 2016).
4. Oja's Hebbian Rule: normalized local PCA update (Oja, 1982).
5. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def forward_forward_loss_and_norm(
    h_pos: torch.Tensor,
    h_neg: torch.Tensor,
    theta: float = 2.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-Forward goodness loss and layer normalization (Hinton, 2022).

    h_pos: [B, D], h_neg: [B, D]
    Returns (loss, h_norm).
    """
    g_pos = torch.sum(h_pos ** 2, dim=-1)
    g_neg = torch.sum(h_neg ** 2, dim=-1)
    loss = torch.mean(F.softplus(theta - g_pos) + F.softplus(g_neg - theta))
    h_norm = h_pos / (torch.norm(h_pos, p=2, dim=-1, keepdim=True) + eps)
    return loss, h_norm


def predictive_coding_step(
    x: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    eta: float = 0.1,
    lam: float = 0.01,
) -> torch.Tensor:
    """Predictive Coding representation inference step (Rao & Ballard, 1999).

    x: [D], r: [D], w: [D, D]
    Returns updated representation r_next.
    """
    eps = x - r @ w
    return r + eta * (eps @ w.T - lam * r)


def dfa_project_error(
    top_error: torch.Tensor,
    dfa_feedback: torch.Tensor,
    act_pre: torch.Tensor,
) -> torch.Tensor:
    """Direct Feedback Alignment (DFA) error projection (Nøkland, 2016).

    top_error: [B, C], dfa_feedback: [D, C], act_pre: [B, D]
    Returns backward error vector [B, D].
    """
    relu_mask = (act_pre > 0.0).float()
    return (top_error @ dfa_feedback.T) * relu_mask


def oja_hebbian_step(
    x: torch.Tensor,
    w: torch.Tensor,
    eta: float = 0.05,
) -> torch.Tensor:
    """Oja's local unsupervised Hebbian PCA update (Oja, 1982).

    x: [D], w: [D]
    Returns updated weight vector w_next.
    """
    y = torch.dot(x, w)
    return w + eta * (y * x - (y ** 2) * w)


class LinearAnchor(nn.Module):
    """Linear anchor module for cross-track parity verification."""

    def __init__(self, in_features: int = 8, out_features: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class NonBackpropUnifiedModule(nn.Module):
    """Unified module verifying non-backprop invariants against golden references."""

    def __init__(self, safetensors_path: str | Path) -> None:
        super().__init__()
        weights = load_file(str(safetensors_path))

        # Forward-Forward
        self.h_pos = weights["h_pos"]
        self.h_neg = weights["h_neg"]
        self.golden_ff_loss = weights["golden_ff_loss"]
        self.golden_h_norm = weights["golden_h_norm"]

        # Predictive Coding
        self.pc_x = weights["pc_x"]
        self.pc_r = weights["pc_r"]
        self.pc_w = weights["pc_w"]
        self.golden_pc_r_next = weights["golden_pc_r_next"]

        # DFA
        self.top_error = weights["top_error"]
        self.dfa_feedback = weights["dfa_feedback"]
        self.act_pre = weights["act_pre"]
        self.golden_dfa_grad = weights["golden_dfa_grad"]

        # Oja
        self.oja_x = weights["oja_x"]
        self.oja_w = weights["oja_w"]
        self.golden_oja_w_next = weights["golden_oja_w_next"]

        # Linear Anchor
        self.anchor = LinearAnchor(8, 4)
        self.anchor.fc.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.fc.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. Forward-Forward invariant
        ff_loss, h_norm = forward_forward_loss_and_norm(self.h_pos, self.h_neg, theta=2.0)
        err_loss = float(torch.abs(ff_loss - self.golden_ff_loss[0]).item())
        err_hn = float(torch.max(torch.abs(h_norm - self.golden_h_norm)).item())
        err_ff = max(err_loss, err_hn)
        assert err_ff < atol, f"Forward-Forward invariant failed: {err_ff:.2e}"
        errors["ff_err"] = err_ff

        # 2. Predictive Coding invariant
        pc_r_next = predictive_coding_step(self.pc_x, self.pc_r, self.pc_w, eta=0.1, lam=0.01)
        err_pc = float(torch.max(torch.abs(pc_r_next - self.golden_pc_r_next)).item())
        assert err_pc < atol, f"Predictive Coding invariant failed: {err_pc:.2e}"
        errors["pc_err"] = err_pc

        # 3. DFA invariant
        dfa_grad = dfa_project_error(self.top_error, self.dfa_feedback, self.act_pre)
        err_dfa = float(torch.max(torch.abs(dfa_grad - self.golden_dfa_grad)).item())
        assert err_dfa < atol, f"DFA invariant failed: {err_dfa:.2e}"
        errors["dfa_err"] = err_dfa

        # 4. Oja invariant
        oja_w_next = oja_hebbian_step(self.oja_x, self.oja_w, eta=0.05)
        err_oja = float(torch.max(torch.abs(oja_w_next - self.golden_oja_w_next)).item())
        assert err_oja < atol, f"Oja invariant failed: {err_oja:.2e}"
        errors["oja_err"] = err_oja

        # 5. Linear Anchor invariant
        anchor_out = self.anchor(self.anchor_x)
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian.safetensors")
    module = NonBackpropUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Invariant Parity Checks Passed: {errs}")
