"""Non Contrastive and Masked Ssl Byol Dino Vicreg MAE JEPA — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa
Golden weights: weights/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def byol_prediction_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Normalized cosine prediction loss: 2 - 2 * (p / ||p||) . (z / ||z||)."""
    p_norm = p / (torch.norm(p, dim=-1, keepdim=True) + 1e-12)
    z_norm = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-12)
    cos_sim = (p_norm * z_norm).sum(dim=-1)
    return (2.0 - 2.0 * cos_sim).mean().unsqueeze(0)


def vicreg_loss(
    za: torch.Tensor,
    zb: torch.Tensor,
    sim_coeff: float = 1.0,
    std_coeff: float = 1.0,
    cov_coeff: float = 0.04,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Variance-Invariance-Covariance Regularization loss."""
    b, d = za.shape
    sim_loss = F.mse_loss(za, zb)

    def var_loss(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(torch.var(z, dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(gamma - std))

    std_loss = var_loss(za) + var_loss(zb)

    def cov_loss(z: torch.Tensor) -> torch.Tensor:
        z_c = z - z.mean(dim=0, keepdim=True)
        cov = (z_c.T @ z_c) / (b - 1)
        cov_off = cov - torch.diag(torch.diag(cov))
        return (cov_off ** 2).sum() / d

    cov_loss_val = cov_loss(za) + cov_loss(zb)
    return (sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss_val).unsqueeze(0)


def dino_loss(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    t_center: torch.Tensor,
    tau_s: float = 0.1,
    tau_t: float = 0.05,
) -> torch.Tensor:
    """DINO cross-entropy distillation loss between student and centered sharpened teacher."""
    p_t = F.softmax((t_logits - t_center) / tau_t, dim=-1)
    log_p_s = F.log_softmax(s_logits / tau_s, dim=-1)
    return (-(p_t * log_p_s).sum(dim=-1)).mean().unsqueeze(0)


def koleo_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Kozachenko-Leonenko differential entropy regularization loss."""
    b = z.shape[0]
    z_norm = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-12)
    dist_matrix = torch.cdist(z_norm, z_norm, p=2)
    diag_mask = torch.eye(b, device=z.device) * 1e9
    min_dist, _ = (dist_matrix + diag_mask).min(dim=-1)
    return (-torch.log(min_dist + eps)).mean().unsqueeze(0)


def mae_patch_gather(patches: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather visible patches for Masked Autoencoding."""
    b = patches.shape[0]
    return torch.stack([patches[i, indices[i]] for i in range(b)], dim=0)


def jepa_feature_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """I-JEPA MSE prediction loss in latent representation space."""
    return F.mse_loss(pred, target).unsqueeze(0)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Linear anchor
    test_input = torch.randn(4, 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    test_output = linear_anchor(test_input, weight, bias)

    # 2. BYOL Prediction Loss
    byol_p = torch.randn(4, 8)
    byol_z = torch.randn(4, 8)
    byol_loss_ref = byol_prediction_loss(byol_p, byol_z)

    # 3. VICReg Loss
    vicreg_za = torch.randn(4, 8)
    vicreg_zb = torch.randn(4, 8)
    vicreg_loss_ref = vicreg_loss(vicreg_za, vicreg_zb)

    # 4. DINO Loss
    s_logits = torch.randn(4, 8)
    t_logits = torch.randn(4, 8)
    t_center = torch.randn(1, 8)
    dino_loss_ref = dino_loss(s_logits, t_logits, t_center)

    # 5. KoLeo Loss
    koleo_z = torch.randn(4, 8)
    koleo_loss_ref = koleo_loss(koleo_z)

    # 6. MAE Patch Gather
    mae_patches = torch.randn(2, 6, 8)
    mae_indices = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.int64)
    mae_gathered_ref = mae_patch_gather(mae_patches, mae_indices)

    # 7. JEPA Feature Loss
    jepa_pred = torch.randn(2, 2, 8)
    jepa_target = torch.randn(2, 2, 8)
    jepa_loss_ref = jepa_feature_loss(jepa_pred, jepa_target)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "byol_p": byol_p,
        "byol_z": byol_z,
        "byol_loss_ref": byol_loss_ref,
        "vicreg_za": vicreg_za,
        "vicreg_zb": vicreg_zb,
        "vicreg_loss_ref": vicreg_loss_ref,
        "s_logits": s_logits,
        "t_logits": t_logits,
        "t_center": t_center,
        "dino_loss_ref": dino_loss_ref,
        "koleo_z": koleo_z,
        "koleo_loss_ref": koleo_loss_ref,
        "mae_patches": mae_patches,
        "mae_indices": mae_indices,
        "mae_gathered_ref": mae_gathered_ref,
        "jepa_pred": jepa_pred,
        "jepa_target": jepa_target,
        "jepa_loss_ref": jepa_loss_ref,
    }


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
