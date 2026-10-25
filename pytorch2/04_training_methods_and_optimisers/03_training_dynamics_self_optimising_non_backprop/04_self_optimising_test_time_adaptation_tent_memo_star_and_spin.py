"""PyTorch 2.x implementation for Category 04 Topic 18:
Self-Optimising Test-Time Adaptation: TENT, MEMO, STaR, and SPIN.

Provides production-grade implementations of:
1. TENT: Test-Time Entropy Minimization updating normalization affine parameters (Wang et al., 2021).
2. MEMO: Marginal Entropy Minimization with One test sample across augmentations (Zhang et al., 2022).
3. SPIN: Self-Play Fine-Tuning preference optimization (Chen et al., 2024).
4. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def compute_tent_step(
    logits: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    lr: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TENT forward step updating normalization affine parameters via entropy minimization.

    logits: [B, C], scale: [C], shift: [C]
    Returns (scale_next, shift_next, entropy, grad_scale, grad_shift).
    """
    scale_param = scale.clone().detach().requires_grad_(True)
    shift_param = shift.clone().detach().requires_grad_(True)

    scaled_logits = logits * scale_param + shift_param
    p = F.softmax(scaled_logits, dim=-1)
    log_p = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.mean(torch.sum(p * log_p, dim=-1))

    entropy.backward()

    with torch.no_grad():
        grad_scale = scale_param.grad.clone()
        grad_shift = shift_param.grad.clone()
        scale_next = scale_param - lr * grad_scale
        shift_next = shift_param - lr * grad_shift

    return scale_next, shift_next, entropy.detach(), grad_scale, grad_shift


def compute_memo_marginal_entropy(
    aug_logits: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MEMO marginal entropy minimization across K test augmentations of a single sample (Zhang et al., 2022).

    aug_logits: [K, C]
    Returns (marginal_p, marginal_entropy).
    """
    aug_probs = F.softmax(aug_logits, dim=-1)
    marginal_p = torch.mean(aug_probs, dim=0) # [C]
    marginal_entropy = -torch.sum(marginal_p * torch.log(marginal_p + eps))
    return marginal_p, marginal_entropy


def compute_spin_loss(
    logp_win: torch.Tensor,
    logp_lose: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """SPIN / STaR self-play preference loss (Chen et al., 2024).

    logp_win: [B], logp_lose: [B]
    Returns scalar preference loss.
    """
    return -torch.mean(F.logsigmoid(beta * (logp_win - logp_lose)))


class LinearAnchor(nn.Module):
    """Linear anchor module for cross-track parity verification."""

    def __init__(self, in_features: int = 8, out_features: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestTimeAdaptationUnifiedModule(nn.Module):
    """Unified module verifying test-time adaptation invariants against golden references."""

    def __init__(self, safetensors_path: str | Path) -> None:
        super().__init__()
        weights = load_file(str(safetensors_path))

        # TENT
        self.tent_logits = weights["tent_logits"]
        self.tent_scale = weights["tent_scale"]
        self.tent_shift = weights["tent_shift"]
        self.golden_tent_entropy = weights["golden_tent_entropy"]
        self.golden_scale_grad = weights["golden_scale_grad"]
        self.golden_shift_grad = weights["golden_shift_grad"]
        self.golden_scale_next = weights["golden_scale_next"]
        self.golden_shift_next = weights["golden_shift_next"]

        # MEMO
        self.memo_aug_logits = weights["memo_aug_logits"]
        self.golden_marginal_p = weights["golden_marginal_p"]
        self.golden_memo_entropy = weights["golden_memo_entropy"]

        # SPIN
        self.logp_win = weights["logp_win"]
        self.logp_lose = weights["logp_lose"]
        self.golden_spin_loss = weights["golden_spin_loss"]

        # Linear Anchor
        self.anchor = LinearAnchor(8, 4)
        self.anchor.fc.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.fc.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. TENT invariant
        scale_next, shift_next, entropy, g_scale, g_shift = compute_tent_step(
            self.tent_logits, self.tent_scale, self.tent_shift, lr=0.05
        )
        err_ent = float(torch.abs(entropy - self.golden_tent_entropy[0]).item())
        err_s_next = float(torch.max(torch.abs(scale_next - self.golden_scale_next)).item())
        err_sh_next = float(torch.max(torch.abs(shift_next - self.golden_shift_next)).item())
        err_g_s = float(torch.max(torch.abs(g_scale - self.golden_scale_grad)).item())
        err_g_sh = float(torch.max(torch.abs(g_shift - self.golden_shift_grad)).item())
        err_tent = max(err_ent, err_s_next, err_sh_next, err_g_s, err_g_sh)
        assert err_tent < atol, f"TENT invariant failed: {err_tent:.2e}"
        errors["tent_err"] = err_tent

        # 2. MEMO invariant
        marg_p, memo_ent = compute_memo_marginal_entropy(self.memo_aug_logits)
        err_p = float(torch.max(torch.abs(marg_p - self.golden_marginal_p)).item())
        err_m_ent = float(torch.abs(memo_ent - self.golden_memo_entropy[0]).item())
        err_memo = max(err_p, err_m_ent)
        assert err_memo < atol, f"MEMO invariant failed: {err_memo:.2e}"
        errors["memo_err"] = err_memo

        # 3. SPIN invariant
        spin_loss = compute_spin_loss(self.logp_win, self.logp_lose, beta=0.5)
        err_spin = float(torch.abs(spin_loss - self.golden_spin_loss[0]).item())
        assert err_spin < atol, f"SPIN invariant failed: {err_spin:.2e}"
        errors["spin_err"] = err_spin

        # 4. Linear Anchor invariant
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
    C = 4
    K = 3
    eps = 1e-8

    # 1. TENT: Test-Time Entropy Minimization (Wang et al., 2021)
    tent_logits = torch.randn(B, C, dtype=torch.float32)
    tent_scale = torch.ones(C, dtype=torch.float32, requires_grad=True)
    tent_shift = torch.zeros(C, dtype=torch.float32, requires_grad=True)

    # Affine transformation on logits: logits * scale + shift
    scaled_logits = tent_logits * tent_scale + tent_shift
    p_tent = F.softmax(scaled_logits, dim=-1)
    log_p_tent = F.log_softmax(scaled_logits, dim=-1)
    tent_entropy = -torch.mean(torch.sum(p_tent * log_p_tent, dim=-1))
    tent_entropy.backward()

    golden_scale_grad = tent_scale.grad.clone().detach()
    golden_shift_grad = tent_shift.grad.clone().detach()
    # One step of SGD on affine parameters
    lr_tent = 0.05
    golden_scale_next = (tent_scale.detach() - lr_tent * golden_scale_grad).contiguous()
    golden_shift_next = (tent_shift.detach() - lr_tent * golden_shift_grad).contiguous()

    # 2. MEMO: Marginal Entropy Minimization with One test sample (Zhang et al., 2022)
    memo_aug_logits = torch.randn(K, C, dtype=torch.float32)
    memo_aug_probs = F.softmax(memo_aug_logits, dim=-1)
    golden_marginal_p = torch.mean(memo_aug_probs, dim=0) # [C]
    golden_memo_entropy = -torch.sum(golden_marginal_p * torch.log(golden_marginal_p + eps)) # scalar

    # 3. SPIN: Self-Play Preference Loss (Chen et al., 2024)
    logp_win = torch.randn(B, dtype=torch.float32)
    logp_lose = logp_win - torch.abs(torch.randn(B, dtype=torch.float32)) - 0.5 # win is strictly higher log-prob
    beta_spin = 0.5
    golden_spin_loss = -torch.mean(F.logsigmoid(beta_spin * (logp_win - logp_lose)))

    # 4. Standard Linear Anchor
    anchor_x = torch.randn(2, 8, dtype=torch.float32)
    anchor_w = torch.randn(8, 4, dtype=torch.float32)
    anchor_b = torch.randn(4, dtype=torch.float32)
    golden_anchor_out = anchor_x @ anchor_w + anchor_b

    tensors = {
        # TENT
        "tent_logits": tent_logits.contiguous(),
        "tent_scale": tent_scale.detach().contiguous(),
        "tent_shift": tent_shift.detach().contiguous(),
        "golden_tent_entropy": tent_entropy.detach().reshape(1).contiguous(),
        "golden_scale_grad": golden_scale_grad.contiguous(),
        "golden_shift_grad": golden_shift_grad.contiguous(),
        "golden_scale_next": golden_scale_next.contiguous(),
        "golden_shift_next": golden_shift_next.contiguous(),

        # MEMO
        "memo_aug_logits": memo_aug_logits.contiguous(),
        "golden_marginal_p": golden_marginal_p.contiguous(),
        "golden_memo_entropy": golden_memo_entropy.reshape(1).contiguous(),

        # SPIN
        "logp_win": logp_win.contiguous(),
        "logp_lose": logp_lose.contiguous(),
        "golden_spin_loss": golden_spin_loss.reshape(1).contiguous(),

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
    ckpt_path = out_dir / "04_self_optimising_test_time_adaptation_tent_memo_star_and_spin.safetensors"

    save_file(tensors, str(ckpt_path))
    print(f"Saved {len(tensors)} tensors to {ckpt_path}")


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/04_self_optimising_test_time_adaptation_tent_memo_star_and_spin.safetensors")
    module = TestTimeAdaptationUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Invariant Parity Checks Passed: {errs}")
