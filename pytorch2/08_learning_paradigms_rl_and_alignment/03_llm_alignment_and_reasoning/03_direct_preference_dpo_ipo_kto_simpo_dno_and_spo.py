"""Direct Preference DPO Ipo KTO SimPO DNO and Spo — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo
Golden weights: weights/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def dpo_loss(
    pi_logps_w: torch.Tensor,
    pi_logps_l: torch.Tensor,
    ref_logps_w: torch.Tensor,
    ref_logps_l: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimization (DPO) pairwise loss:
    L_DPO = - E [ ln sigma( beta * (ln(pi_w/ref_w) - ln(pi_l/ref_l)) ) ]
    = E [ softplus( - beta * ((pi_w - ref_w) - (pi_l - ref_l)) ) ]
    """
    pi_diff = pi_logps_w - pi_logps_l
    ref_diff = ref_logps_w - ref_logps_l
    logits = beta * (pi_diff - ref_diff)
    return torch.mean(F.softplus(-logits))


def ipo_loss(
    pi_logps_w: torch.Tensor,
    pi_logps_l: torch.Tensor,
    ref_logps_w: torch.Tensor,
    ref_logps_l: torch.Tensor,
    beta: float = 0.1,
    tau: float = 0.5,
) -> torch.Tensor:
    """Identity Preference Optimization (IPO) non-saturating quadratic loss:
    L_IPO = E [ ( beta * ((pi_w - ref_w) - (pi_l - ref_l)) - 1/(2 * tau) )^2 ]
    """
    pi_diff = pi_logps_w - pi_logps_l
    ref_diff = ref_logps_w - ref_logps_l
    logits = beta * (pi_diff - ref_diff)
    target = 1.0 / (2.0 * tau)
    return torch.mean((logits - target) ** 2)


def simpo_loss(
    pi_logps_w: torch.Tensor,
    pi_logps_l: torch.Tensor,
    len_w: torch.Tensor,
    len_l: torch.Tensor,
    beta: float = 2.0,
    gamma_margin: float = 0.5,
) -> torch.Tensor:
    """Simple Preference Optimization (SimPO) reference-free length-normalized loss:
    L_SimPO = - E [ ln sigma( beta * (pi_w / |y_w| - pi_l / |y_l|) - gamma ) ]
    """
    norm_w = pi_logps_w / len_w
    norm_l = pi_logps_l / len_l
    logits = beta * (norm_w - norm_l) - gamma_margin
    return torch.mean(F.softplus(-logits))


def kto_loss(
    pi_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    desirable_flags: torch.Tensor,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.33,
) -> torch.Tensor:
    """Kahneman-Tversky Optimization (KTO) unpaired prospect theory loss:
    For desirable:   w_D * sigma( - beta * (pi - ref) )
    For undesirable: w_U * sigma( beta * (pi - ref) )
    """
    log_ratio = beta * (pi_logps - ref_logps)
    # sigma(-z) for desirable, sigma(z) for undesirable
    loss_desirable = torch.sigmoid(-log_ratio) * desirable_weight
    loss_undesirable = torch.sigmoid(log_ratio) * undesirable_weight
    per_sample_loss = torch.where(desirable_flags > 0.5, loss_desirable, loss_undesirable)
    return torch.mean(per_sample_loss)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. DPO and IPO inputs
    # B=8 pairs
    pi_logps_w = -torch.rand(8) * 15.0 - 5.0
    pi_logps_l = -torch.rand(8) * 20.0 - 10.0
    ref_logps_w = -torch.rand(8) * 15.0 - 5.0
    ref_logps_l = -torch.rand(8) * 20.0 - 10.0

    dpo_loss_ref = dpo_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, beta=0.1)
    ipo_loss_ref = ipo_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, beta=0.1, tau=0.5)

    # 2. SimPO inputs
    len_w = torch.tensor([12.0, 16.0, 20.0, 8.0, 14.0, 18.0, 22.0, 10.0])
    len_l = torch.tensor([14.0, 15.0, 22.0, 10.0, 16.0, 20.0, 24.0, 12.0])
    simpo_loss_ref = simpo_loss(pi_logps_w, pi_logps_l, len_w, len_l, beta=2.0, gamma_margin=0.5)

    # 3. KTO inputs (unpaired: 4 desirable, 4 undesirable)
    pi_logps_unpaired = -torch.rand(8) * 15.0 - 5.0
    ref_logps_unpaired = -torch.rand(8) * 15.0 - 5.0
    desirable_flags = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    kto_loss_ref = kto_loss(pi_logps_unpaired, ref_logps_unpaired, desirable_flags, beta=0.1)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "pi_logps_w": pi_logps_w,
        "pi_logps_l": pi_logps_l,
        "ref_logps_w": ref_logps_w,
        "ref_logps_l": ref_logps_l,
        "dpo_loss_ref": dpo_loss_ref,
        "ipo_loss_ref": ipo_loss_ref,
        "len_w": len_w,
        "len_l": len_l,
        "simpo_loss_ref": simpo_loss_ref,
        "pi_logps_unpaired": pi_logps_unpaired,
        "ref_logps_unpaired": ref_logps_unpaired,
        "desirable_flags": desirable_flags,
        "kto_loss_ref": kto_loss_ref,
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
