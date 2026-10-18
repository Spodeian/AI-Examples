"""Production PyTorch 2.x implementation for Category 01 Topic 08:
08_computation_spaces_latent_reasoning_and_jepa
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from safetensors.torch import load_file
from torch import Tensor


def rms_norm(z: Float[Tensor, "... dim"], eps: float = 1e-5) -> Float[Tensor, "... dim"]:
    """Root Mean Square Normalization along the innermost dimension."""
    variance = torch.mean(z**2, dim=-1, keepdim=True)
    return z * torch.rsqrt(variance + eps)


class ContinuousLatentReasoner(nn.Module):
    """Input-Anchored Recurrent Latent Deliberation Core (Coconut & Universal Transformer).

    Inductive Bias:
        Deliberating in continuous latent space preserves superposition across hypotheses.
        Persistent residual input injection (e_x) anchors the recurrence, preventing semantic drift.
    """

    def __init__(
        self,
        w_proj: Float[Tensor, "d_in d_model"],
        b_proj: Float[Tensor, "d_model"],
        w_z: Float[Tensor, "d_model d_model"],
        w_anc: Float[Tensor, "d_model d_model"],
        b_z: Float[Tensor, "d_model"],
    ) -> None:
        super().__init__()
        self.w_proj = nn.Parameter(w_proj.clone())
        self.b_proj = nn.Parameter(b_proj.clone())
        self.w_z = nn.Parameter(w_z.clone())
        self.w_anc = nn.Parameter(w_anc.clone())
        self.b_z = nn.Parameter(b_z.clone())

    def forward(
        self,
        x: Float[Tensor, "batch d_in"],
        steps: int = 3,
    ) -> tuple[Float[Tensor, "batch k_plus_1 d_model"], Float[Tensor, "batch d_model"]]:
        """Runs K-step input-anchored recurrent deliberation."""
        e_x = F.gelu(x @ self.w_proj + self.b_proj)
        z = e_x.clone()
        trajectories = [z]

        for _ in range(steps):
            delta = F.gelu(z @ self.w_z + e_x @ self.w_anc + self.b_z)
            z = rms_norm(z + delta)
            trajectories.append(z)

        latent_trajectories = torch.stack(trajectories, dim=1)
        return latent_trajectories, z


class PonderNetHalting(nn.Module):
    """Adaptive Computation Halting Hazard Gate (Banino et al. NeurIPS 2021)."""

    def __init__(
        self,
        w_h: Float[Tensor, "d_model 1"],
        b_h: Float[Tensor, "1"],
    ) -> None:
        super().__init__()
        self.w_h = nn.Parameter(w_h.clone())
        self.b_h = nn.Parameter(b_h.clone())

    def forward(
        self,
        trajectories: Float[Tensor, "batch k_plus_1 d_model"],
    ) -> Float[Tensor, "batch k"]:
        """Evaluates halting probabilities p_k for steps k = 1..K."""
        batch = trajectories.shape[0]
        steps = trajectories.shape[1] - 1

        hazards = []
        for k in range(steps):
            z_k = trajectories[:, k + 1]
            lam = torch.sigmoid(z_k @ self.w_h + self.b_h)
            hazards.append(lam)

        hazards_tensor = torch.cat(hazards, dim=-1)
        halting_probs = []
        unhalted = torch.ones(batch, 1, device=trajectories.device, dtype=trajectories.dtype)

        for k in range(steps):
            lam = hazards_tensor[:, k : k + 1]
            p_k = lam * unhalted
            halting_probs.append(p_k)
            unhalted = unhalted * (1.0 - lam)

        return torch.cat(halting_probs, dim=-1)


class JEPAPredictor(nn.Module):
    """Joint-Embedding Predictive Architecture Latent Predictor (LeCun 2022)."""

    def __init__(
        self,
        w_p1: Float[Tensor, "d_model d_model"],
        b_p1: Float[Tensor, "d_model"],
        w_p2: Float[Tensor, "d_model d_model"],
        b_p2: Float[Tensor, "d_model"],
    ) -> None:
        super().__init__()
        self.w_p1 = nn.Parameter(w_p1.clone())
        self.b_p1 = nn.Parameter(b_p1.clone())
        self.w_p2 = nn.Parameter(w_p2.clone())
        self.b_p2 = nn.Parameter(b_p2.clone())

    def forward(
        self,
        s_x: Float[Tensor, "batch d_model"],
    ) -> Float[Tensor, "batch d_model"]:
        """Predicts target latent embedding from context latent state."""
        return F.gelu(s_x @ self.w_p1 + self.b_p1) @ self.w_p2 + self.b_p2

    @staticmethod
    def prediction_error(
        pred: Float[Tensor, "batch d_model"],
        target: Float[Tensor, "batch d_model"],
    ) -> Float[Tensor, "batch"]:
        """Evaluates squared L2 representation prediction loss."""
        return torch.sum((pred - target) ** 2, dim=-1)


class ComputationSpacesModule(nn.Module):
    """Master module coordinating linear anchor, latent reasoning, pondering, and JEPA."""

    def __init__(
        self,
        weight: Float[Tensor, "d_in d_out"],
        bias: Float[Tensor, "d_out"],
        reasoner: ContinuousLatentReasoner,
        ponder: PonderNetHalting,
        jepa: JEPAPredictor,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(bias.clone())
        self.reasoner = reasoner
        self.ponder = ponder
        self.jepa = jepa

    def forward(
        self,
        x: Float[Tensor, "batch d_in"],
    ) -> Float[Tensor, "batch d_out"]:
        """Canonical linear projection matching test_output for cross-track parity."""
        return x @ self.weight + self.bias


def load_from_safetensors(path: str | Path) -> ComputationSpacesModule:
    """Loads weights and constructs a verified ComputationSpacesModule."""
    tensors = load_file(str(path))
    reasoner = ContinuousLatentReasoner(
        w_proj=tensors["layers.w_proj"],
        b_proj=tensors["layers.b_proj"],
        w_z=tensors["layers.w_z"],
        w_anc=tensors["layers.w_anc"],
        b_z=tensors["layers.b_z"],
    )
    ponder = PonderNetHalting(
        w_h=tensors["layers.w_h"],
        b_h=tensors["layers.b_h"],
    )
    jepa = JEPAPredictor(
        w_p1=tensors["layers.w_p1"],
        b_p1=tensors["layers.b_p1"],
        w_p2=tensors["layers.w_p2"],
        b_p2=tensors["layers.b_p2"],
    )
    return ComputationSpacesModule(
        weight=tensors["layers.weight"],
        bias=tensors["layers.bias"],
        reasoner=reasoner,
        ponder=ponder,
        jepa=jepa,
    )


if __name__ == "__main__":
    ckpt_path = Path("weights/01_paradigms_and_dichotomies/08_computation_spaces_latent_reasoning_and_jepa.safetensors")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    tensors = load_file(str(ckpt_path))
    model = load_from_safetensors(ckpt_path)

    x = tensors["test_input"]
    y_gold = tensors["test_output"]
    y_pred = model(x)
    assert torch.allclose(y_pred, y_gold, atol=1e-5), f"Linear anchor mismatch: max diff={(y_pred - y_gold).abs().max()}"

    trajectories, final_thought = model.reasoner(x, steps=3)
    assert torch.allclose(trajectories, tensors["layers.latent_trajectories"], atol=1e-5), "Trajectory mismatch"
    assert torch.allclose(final_thought, tensors["layers.final_thought"], atol=1e-5), "Final thought mismatch"

    halting_probs = model.ponder(trajectories)
    assert torch.allclose(halting_probs, tensors["layers.halting_probs"], atol=1e-5), "Halting probs mismatch"

    jepa_pred = model.jepa(final_thought)
    assert torch.allclose(jepa_pred, tensors["layers.jepa_pred"], atol=1e-5), "JEPA pred mismatch"

    jepa_err = model.jepa.prediction_error(jepa_pred, tensors["layers.target_sy"])
    assert torch.allclose(jepa_err, tensors["layers.jepa_error"], atol=1e-5), "JEPA error mismatch"

    print("PyTorch 2.x Module 08 passed all smoke tests and golden parity assertions!")
