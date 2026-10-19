"""PyTorch 2.x implementation for Category 04 Topic 16:
Training Optimisations: Progressive Stacking, Data Pruning, MRL, and Plasticity.

Provides production-grade implementations of:
1. Matryoshka Representation Learning (MRL) multi-scale nested projection.
2. EL2N (Error L2 Norm) early-training sample importance scoring.
3. RHO-Loss (Reducible Holdout Loss) for filtering irreducible noise.
4. ReDo (Recycling Dormant Neurons) activation tracking and dormant mask identification.
5. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file


def mrl_slice_forward(
    z: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    dim_slice: int,
) -> torch.Tensor:
    """Matryoshka Representation Learning (MRL) prefix projection (Kusupati et al., 2022).

    Projects only the first `dim_slice` dimensions of embedding vector z.
    """
    z_sub = z[:, :dim_slice]
    w_sub = w[:dim_slice, :]
    return z_sub @ w_sub + b


def compute_el2n(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Error L2 Norm (EL2N) sample importance score (Paul et al., 2021).

    Measures ||p(x) - y||_2 at early checkpoints to identify hard, learnable samples.
    """
    diff = probs - targets
    return torch.norm(diff, p=2, dim=-1)


def compute_rho_loss(
    loss_model: torch.Tensor,
    loss_ref: torch.Tensor,
) -> torch.Tensor:
    """Reducible Holdout Loss (RHO-Loss) scoring (Mindermann et al., 2022).

    RHO = L_model - L_ref. Filters out unlearnable label noise.
    """
    return loss_model - loss_ref


def detect_dormant_neurons(
    activations: torch.Tensor,
    tau: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detects dormant neurons via mean absolute activation threshold (Sokar et al., 2023).

    Returns (mean_activation, dormant_mask).
    """
    mean_act = torch.mean(torch.abs(activations), dim=0)
    dormant_mask = (mean_act < tau).to(torch.float32)
    return mean_act, dormant_mask


class LinearAnchor(nn.Module):
    """Linear anchor module for cross-track parity verification."""

    def __init__(self, in_features: int = 8, out_features: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TrainingOptimisationsUnifiedModule(nn.Module):
    """Unified module verifying training optimization invariants against safetensors golden references."""

    def __init__(self, safetensors_path: str | Path) -> None:
        super().__init__()
        weights = load_file(str(safetensors_path))

        # MRL
        self.z_emb = weights["z_emb"]
        self.mrl_w = weights["mrl_w"]
        self.mrl_b = weights["mrl_b"]
        self.golden_logits_d2 = weights["golden_logits_d2"]
        self.golden_logits_d4 = weights["golden_logits_d4"]
        self.golden_logits_d8 = weights["golden_logits_d8"]

        # Data Pruning: EL2N & RHO-Loss
        self.p_probs = weights["p_probs"]
        self.y_targets = weights["y_targets"]
        self.golden_el2n = weights["golden_el2n"]
        self.loss_model = weights["loss_model"]
        self.loss_ref = weights["loss_ref"]
        self.golden_rho_loss = weights["golden_rho_loss"]

        # ReDo Plasticity
        self.activations = weights["activations"]
        self.golden_mean_act = weights["mean_act"]
        self.golden_dormant_mask = weights["dormant_mask"]

        # Linear Anchor
        self.anchor = LinearAnchor(8, 4)
        self.anchor.fc.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.fc.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. MRL nested projections
        out_d2 = mrl_slice_forward(self.z_emb, self.mrl_w, self.mrl_b, dim_slice=2)
        out_d4 = mrl_slice_forward(self.z_emb, self.mrl_w, self.mrl_b, dim_slice=4)
        out_d8 = mrl_slice_forward(self.z_emb, self.mrl_w, self.mrl_b, dim_slice=8)

        err_d2 = float(torch.max(torch.abs(out_d2 - self.golden_logits_d2)).item())
        err_d4 = float(torch.max(torch.abs(out_d4 - self.golden_logits_d4)).item())
        err_d8 = float(torch.max(torch.abs(out_d8 - self.golden_logits_d8)).item())
        err_mrl = max(err_d2, err_d4, err_d8)
        assert err_mrl < atol, f"MRL invariant failed: err={err_mrl:.2e} >= {atol:.2e}"
        errors["mrl_err"] = err_mrl

        # 2. EL2N score
        el2n = compute_el2n(self.p_probs, self.y_targets)
        err_el2n = float(torch.max(torch.abs(el2n - self.golden_el2n)).item())
        assert err_el2n < atol, f"EL2N invariant failed: err={err_el2n:.2e} >= {atol:.2e}"
        errors["el2n_err"] = err_el2n

        # 3. RHO-Loss
        rho = compute_rho_loss(self.loss_model, self.loss_ref)
        err_rho = float(torch.max(torch.abs(rho - self.golden_rho_loss)).item())
        assert err_rho < atol, f"RHO-loss invariant failed: err={err_rho:.2e} >= {atol:.2e}"
        errors["rho_loss_err"] = err_rho

        # 4. ReDo Dormant Neurons
        mean_act, mask = detect_dormant_neurons(self.activations, tau=0.15)
        err_mean = float(torch.max(torch.abs(mean_act - self.golden_mean_act)).item())
        err_mask = float(torch.max(torch.abs(mask - self.golden_dormant_mask)).item())
        err_redo = max(err_mean, err_mask)
        assert err_redo < atol, f"ReDo invariant failed: err={err_redo:.2e} >= {atol:.2e}"
        errors["redo_err"] = err_redo

        # 5. Linear Anchor
        anchor_out = self.anchor(self.anchor_x)
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())
        assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/02_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity.safetensors")
    module = TrainingOptimisationsUnifiedModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"PyTorch Invariant Parity Checks Passed: {errs}")
