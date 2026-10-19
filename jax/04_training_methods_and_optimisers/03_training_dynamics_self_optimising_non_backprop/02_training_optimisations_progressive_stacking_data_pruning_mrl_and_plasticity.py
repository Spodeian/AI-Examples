"""JAX / Functional implementation for Category 04 Topic 16:
Training Optimisations: Progressive Stacking, Data Pruning, MRL, and Plasticity.

Provides pure functional implementations of:
1. Matryoshka Representation Learning (MRL) multi-scale nested projection.
2. EL2N (Error L2 Norm) sample importance score.
3. RHO-Loss (Reducible Holdout Loss) unlearnable noise filtering.
4. ReDo (Recycling Dormant Neurons) activation tracking and dormant mask identification.
5. Linear anchor evaluation for transitive multi-track parity.
"""

from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def mrl_slice_forward(
    z: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    dim_slice: int,
) -> np.ndarray:
    """Matryoshka Representation Learning (MRL) prefix projection (Kusupati et al., 2022)."""
    z_sub = z[:, :dim_slice]
    w_sub = w[:dim_slice, :]
    return np.matmul(z_sub, w_sub) + b


def compute_el2n(
    probs: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Error L2 Norm (EL2N) sample importance score (Paul et al., 2021)."""
    diff = probs - targets
    return np.linalg.norm(diff, ord=2, axis=-1)


def compute_rho_loss(
    loss_model: np.ndarray,
    loss_ref: np.ndarray,
) -> np.ndarray:
    """Reducible Holdout Loss (RHO-Loss) scoring (Mindermann et al., 2022)."""
    return loss_model - loss_ref


def detect_dormant_neurons(
    activations: np.ndarray,
    tau: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Detects dormant neurons via mean absolute activation threshold (Sokar et al., 2023)."""
    mean_act = np.mean(np.abs(activations), axis=0)
    dormant_mask = (mean_act < tau).astype(np.float32)
    return mean_act, dormant_mask


class TrainingOptimisationsFunctionalModule:
    """Functional module verifying training optimization invariants against safetensors golden references."""

    def __init__(self, safetensors_path: str | Path) -> None:
        weights = load_file(str(safetensors_path))

        # MRL
        self.z_emb = weights["z_emb"]
        self.mrl_w = weights["mrl_w"]
        self.mrl_b = weights["mrl_b"]
        self.golden_logits_d2 = weights["golden_logits_d2"]
        self.golden_logits_d4 = weights["golden_logits_d4"]
        self.golden_logits_d8 = weights["golden_logits_d8"]

        # Data Pruning
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
        self.anchor_x = weights["anchor_x"]
        self.anchor_w = weights["anchor_w"]
        self.anchor_b = weights["anchor_b"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. MRL nested projections
        out_d2 = mrl_slice_forward(self.z_emb, self.mrl_w, self.mrl_b, dim_slice=2)
        out_d4 = mrl_slice_forward(self.z_emb, self.mrl_w, self.mrl_b, dim_slice=4)
        out_d8 = mrl_slice_forward(self.z_emb, self.mrl_w, self.mrl_b, dim_slice=8)

        err_d2 = float(np.max(np.abs(out_d2 - self.golden_logits_d2)))
        err_d4 = float(np.max(np.abs(out_d4 - self.golden_logits_d4)))
        err_d8 = float(np.max(np.abs(out_d8 - self.golden_logits_d8)))
        err_mrl = max(err_d2, err_d4, err_d8)
        assert err_mrl < atol, f"MRL invariant failed: err={err_mrl:.2e} >= {atol:.2e}"
        errors["mrl_err"] = err_mrl

        # 2. EL2N score
        el2n = compute_el2n(self.p_probs, self.y_targets)
        err_el2n = float(np.max(np.abs(el2n - self.golden_el2n)))
        assert err_el2n < atol, f"EL2N invariant failed: err={err_el2n:.2e} >= {atol:.2e}"
        errors["el2n_err"] = err_el2n

        # 3. RHO-Loss
        rho = compute_rho_loss(self.loss_model, self.loss_ref)
        err_rho = float(np.max(np.abs(rho - self.golden_rho_loss)))
        assert err_rho < atol, f"RHO-loss invariant failed: err={err_rho:.2e} >= {atol:.2e}"
        errors["rho_loss_err"] = err_rho

        # 4. ReDo Dormant Neurons
        mean_act, mask = detect_dormant_neurons(self.activations, tau=0.15)
        err_mean = float(np.max(np.abs(mean_act - self.golden_mean_act)))
        err_mask = float(np.max(np.abs(mask - self.golden_dormant_mask)))
        err_redo = max(err_mean, err_mask)
        assert err_redo < atol, f"ReDo invariant failed: err={err_redo:.2e} >= {atol:.2e}"
        errors["redo_err"] = err_redo

        # 5. Linear Anchor
        anchor_out = np.matmul(self.anchor_x, self.anchor_w) + self.anchor_b
        err_anchor = float(np.max(np.abs(anchor_out - self.golden_anchor_out)))
        assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/02_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity.safetensors")
    module = TrainingOptimisationsFunctionalModule(p)
    errs = module.verify_invariants(1e-5)
    print(f"JAX / Functional Invariant Parity Checks Passed: {errs}")
