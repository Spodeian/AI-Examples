"""Causal Treatment Effects TARNet CFRNet Dragonnet DML and R Learner — PyTorch 2.x reference implementation.

Topic: 09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner
Golden weights: weights/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.safetensors

``python pytorch2/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.py`` verifies against the golden checkpoint;
``python pytorch2/09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def tarnet_forward(
    x: torch.Tensor,
    phi_w: torch.Tensor,
    phi_b: torch.Tensor,
    h0_w: torch.Tensor,
    h0_b: torch.Tensor,
    h1_w: torch.Tensor,
    h1_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Treatment-Agnostic Representation Network (TARNet - Shalit et al. 2017).
    
    phi = ReLU(x @ phi_w + phi_b)
    y0 = phi @ h0_w + h0_b  (Control potential outcome)
    y1 = phi @ h1_w + h1_b  (Treated potential outcome)
    tau = y1 - y0           (Individual treatment effect)
    Returns (phi, y0, y1, tau).
    """
    phi = F.relu(x @ phi_w + phi_b)
    y0 = phi @ h0_w + h0_b
    y1 = phi @ h1_w + h1_b
    tau = y1 - y0
    return phi, y0, y1, tau


def cfrnet_mmd_loss(phi: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Counterfactual Regression Network (CFRNet) linear MMD representation balance penalty.
    
    MMD^2 = || mu_1 - mu_0 ||_2^2
    """
    t_mask = t.squeeze(-1)
    n_treated = t_mask.sum().clamp(min=1.0)
    n_control = (1.0 - t_mask).sum().clamp(min=1.0)

    mu_1 = (phi * t_mask.unsqueeze(-1)).sum(dim=0) / n_treated
    mu_0 = (phi * (1.0 - t_mask).unsqueeze(-1)).sum(dim=0) / n_control

    diff = mu_1 - mu_0
    mmd_sq = torch.sum(diff * diff).unsqueeze(0)
    return mmd_sq


def dragonnet_propensity_loss(
    phi: torch.Tensor,
    t: torch.Tensor,
    w_prop: torch.Tensor,
    b_prop: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dragonnet (Shi et al. 2019) propensity score head and BCE loss.
    
    e = sigmoid(phi @ w_prop + b_prop)
    loss = -1/N sum [t * log(e) + (1-t) * log(1-e)]
    """
    logits = phi @ w_prop + b_prop
    propensity = torch.sigmoid(logits)
    bce = -torch.mean(
        t * torch.log(propensity.clamp(min=eps, max=1.0 - eps))
        + (1.0 - t) * torch.log((1.0 - propensity).clamp(min=eps, max=1.0 - eps))
    ).unsqueeze(0)
    return propensity, bce


def dml_cate_estimate(
    y: torch.Tensor,
    t: torch.Tensor,
    m_hat: torch.Tensor,
    e_hat: torch.Tensor,
) -> torch.Tensor:
    """Double Machine Learning (DML - Chernozhukov et al. 2018) Robinson estimator.
    
    tau_hat = sum_i (T_i - e_hat_i)(Y_i - m_hat_i) / sum_i (T_i - e_hat_i)^2
    """
    res_t = t - e_hat
    res_y = y - m_hat
    numerator = torch.sum(res_t * res_y)
    denominator = torch.sum(res_t * res_t).clamp(min=1e-8)
    tau_dml = (numerator / denominator).unsqueeze(0)
    return tau_dml


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # TARNet weights
    phi_w = torch.randn(4, 4) * 0.2
    phi_b = torch.tensor([0.1, -0.1, 0.2, 0.0], dtype=torch.float32)
    h0_w = torch.randn(4, 1) * 0.3
    h0_b = torch.tensor([0.5], dtype=torch.float32)
    h1_w = torch.randn(4, 1) * 0.3
    h1_b = torch.tensor([1.8], dtype=torch.float32)

    phi_ref, y0_ref, y1_ref, tau_ref = tarnet_forward(
        test_input, phi_w, phi_b, h0_w, h0_b, h1_w, h1_b
    )

    # Binary treatment assignment (4 treated, 4 control)
    t_treatment = torch.tensor([[1.0], [0.0], [1.0], [1.0], [0.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

    # CFRNet MMD loss
    mmd_loss_ref = cfrnet_mmd_loss(phi_ref, t_treatment)

    # Dragonnet propensity head
    w_prop = torch.randn(4, 1) * 0.2
    b_prop = torch.tensor([0.1], dtype=torch.float32)
    propensity_ref, prop_loss_ref = dragonnet_propensity_loss(phi_ref, t_treatment, w_prop, b_prop)

    # DML causal effect inputs
    y_outcome = torch.tensor([[3.4], [1.2], [4.1], [3.9], [0.8], [1.5], [4.8], [1.1]], dtype=torch.float32)
    m_hat = torch.tensor([[2.5], [1.4], [3.0], [2.8], [1.1], [1.3], [3.2], [1.0]], dtype=torch.float32)
    e_hat = torch.tensor([[0.7], [0.3], [0.8], [0.6], [0.4], [0.2], [0.9], [0.3]], dtype=torch.float32)

    dml_tau_ref = dml_cate_estimate(y_outcome, t_treatment, m_hat, e_hat)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "phi_w": phi_w,
        "phi_b": phi_b,
        "h0_w": h0_w,
        "h0_b": h0_b,
        "h1_w": h1_w,
        "h1_b": h1_b,
        "y0_ref": y0_ref,
        "y1_ref": y1_ref,
        "tau_ref": tau_ref,
        "t_treatment": t_treatment,
        "mmd_loss_ref": mmd_loss_ref,
        "w_prop": w_prop,
        "b_prop": b_prop,
        "propensity_ref": propensity_ref,
        "prop_loss_ref": prop_loss_ref,
        "y_outcome": y_outcome,
        "m_hat": m_hat,
        "e_hat": e_hat,
        "dml_tau_ref": dml_tau_ref,
    }
    return tensors


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
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
