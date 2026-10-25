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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import save_file

    torch.manual_seed(42)
    np.random.seed(42)

    B = 4
    D = 8
    C = 4
    eps = 1e-8

    # 1. Forward-Forward (Hinton, 2022)
    h_pos = torch.randn(B, D, dtype=torch.float32)
    h_neg = torch.randn(B, D, dtype=torch.float32)
    theta_ff = 2.0

    g_pos = torch.sum(h_pos ** 2, dim=-1) # [B]
    g_neg = torch.sum(h_neg ** 2, dim=-1) # [B]
    loss_pos = F.softplus(theta_ff - g_pos)
    loss_neg = F.softplus(g_neg - theta_ff)
    golden_ff_loss = torch.mean(loss_pos + loss_neg)

    # Forward-Forward layer normalization: h / (norm(h) + eps)
    norm_pos = torch.norm(h_pos, p=2, dim=-1, keepdim=True) + eps
    golden_h_norm = h_pos / norm_pos

    # 2. Predictive Coding (Rao & Ballard, 1999)
    pc_x = torch.randn(D, dtype=torch.float32)
    pc_r = torch.randn(D, dtype=torch.float32)
    pc_w = torch.randn(D, D, dtype=torch.float32) * 0.3
    eta_pc = 0.1
    lambda_pc = 0.01

    # Error eps = x - r @ W
    pc_pred = pc_r @ pc_w
    pc_eps = pc_x - pc_pred
    golden_pc_r_next = pc_r + eta_pc * (pc_eps @ pc_w.T - lambda_pc * pc_r)

    # 3. Direct Feedback Alignment (DFA - Nøkland, 2016)
    top_error = torch.randn(B, C, dtype=torch.float32)
    dfa_feedback = torch.randn(D, C, dtype=torch.float32) # Fixed random feedback matrix
    act_pre = torch.randn(B, D, dtype=torch.float32)
    relu_mask = (act_pre > 0.0).float()
    # Error projected via B rather than W_transpose: (top_error @ B^T) * relu_mask
    golden_dfa_grad = (top_error @ dfa_feedback.T) * relu_mask

    # 4. Oja's Hebbian Rule (Oja, 1982)
    oja_x = torch.randn(D, dtype=torch.float32)
    oja_w = torch.randn(D, dtype=torch.float32)
    oja_w = oja_w / torch.norm(oja_w) # Normalized initial weight
    eta_oja = 0.05
    y_oja = torch.dot(oja_x, oja_w)
    golden_oja_w_next = oja_w + eta_oja * (y_oja * oja_x - (y_oja ** 2) * oja_w)

    # 5. Standard Linear Anchor
    anchor_x = torch.randn(2, 8, dtype=torch.float32)
    anchor_w = torch.randn(8, 4, dtype=torch.float32)
    anchor_b = torch.randn(4, dtype=torch.float32)
    golden_anchor_out = anchor_x @ anchor_w + anchor_b

    tensors = {
        # Forward-Forward
        "h_pos": h_pos.contiguous(),
        "h_neg": h_neg.contiguous(),
        "golden_ff_loss": golden_ff_loss.reshape(1).contiguous(),
        "golden_h_norm": golden_h_norm.contiguous(),

        # Predictive Coding
        "pc_x": pc_x.contiguous(),
        "pc_r": pc_r.contiguous(),
        "pc_w": pc_w.contiguous(),
        "golden_pc_r_next": golden_pc_r_next.contiguous(),

        # DFA
        "top_error": top_error.contiguous(),
        "dfa_feedback": dfa_feedback.contiguous(),
        "act_pre": act_pre.contiguous(),
        "golden_dfa_grad": golden_dfa_grad.contiguous(),

        # Oja
        "oja_x": oja_x.contiguous(),
        "oja_w": oja_w.contiguous(),
        "golden_oja_w_next": golden_oja_w_next.contiguous(),

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
    ckpt_path = out_dir / "05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian.safetensors"

    save_file(tensors, str(ckpt_path))
    print(f"Saved {len(tensors)} tensors to {ckpt_path}")


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian.safetensors")
    module = NonBackpropUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Invariant Parity Checks Passed: {errs}")
