"""Backdoor Trojan and Poisoning Defense Neural Cleanse and Spectral — PyTorch 2.x reference implementation.

Topic: 11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral
Golden weights: weights/11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.safetensors

``python pytorch2/11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.py`` verifies against the golden checkpoint;
``python pytorch2/11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def spectral_signature_scores(representations: torch.Tensor) -> torch.Tensor:
    """Computes spectral signature outlier scores via representation covariance SVD (Tran et al. 2018).
    
    score_i = ( (h_i - mu) . v_1 )^2 where v_1 is the top right-singular vector.
    """
    mu = torch.mean(representations, dim=0, keepdim=True)
    centered = representations - mu
    cov = (centered.t() @ centered) / float(representations.shape[0])
    
    # Deterministic power iteration for top singular vector
    d = cov.shape[0]
    v = torch.ones(d, device=representations.device) / (float(d) ** 0.5)
    for _ in range(30):
        v = cov @ v
        v = v / (torch.norm(v, p=2) + 1e-8)
    
    # Project centered representations onto top singular vector
    proj = centered @ v
    return proj ** 2


def neural_cleanse_anomaly_index(l1_norms: torch.Tensor) -> torch.Tensor:
    """Neural Cleanse MAD outlier detection index for reversed trigger mask L1 norms (Wang et al. 2019).
    
    Anomaly_c = (median(S) - S_c) / (1.4826 * MAD)
    Anomaly > 2.0 indicates an infected backdoor class with an unusually small trigger.
    """
    med = torch.median(l1_norms)
    abs_dev = torch.abs(l1_norms - med)
    mad = torch.median(abs_dev) + 1e-8
    return (med - l1_norms) / (1.4826 * mad)


def strip_entropy(superposed_logits: torch.Tensor) -> torch.Tensor:
    """STRIP runtime input trojan detection via Shannon entropy under perturbation (Gao et al. 2019).
    
    H = - sum_c P(c) * ln(P(c)). Trojan triggers maintain near-zero entropy H ~ 0 across superpositions.
    """
    probs = F.softmax(superposed_logits, dim=-1)
    entropy_per_sample = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    return torch.mean(entropy_per_sample)


def trojan_trigger_apply(
    x: torch.Tensor,
    mask: torch.Tensor,
    pattern: torch.Tensor,
) -> torch.Tensor:
    """Applies BadNets blending trojan trigger: x' = x * (1 - mask) + pattern * mask."""
    return x * (1.0 - mask) + pattern * mask


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # 1. Spectral signatures: N=12, D=6
    reprs = torch.randn(12, 6)
    # Inject a distinct poisoned cluster into samples 0 and 1
    reprs[0] += torch.tensor([4.0, -3.0, 5.0, 1.0, -2.0, 3.0])
    reprs[1] += torch.tensor([3.8, -2.9, 4.8, 1.2, -1.9, 2.9])
    spectral_scores_ref = spectral_signature_scores(reprs)
    
    # 2. Neural Cleanse MAD: C=7 classes (odd length ensures identical median across ecosystems)
    # Class 2 is backdoored with a very small trigger L1 norm (0.4 vs normal ~4.5)
    l1_norms = torch.tensor([4.2, 4.6, 0.4, 4.5, 4.8, 4.3, 4.4])
    mad_anomaly_ref = neural_cleanse_anomaly_index(l1_norms)
    
    # 3. STRIP: K=8 perturbations, C=4
    logits_benign = torch.randn(8, 4)  # High entropy
    entropy_benign_ref = strip_entropy(logits_benign)
    logits_trojan = torch.randn(8, 4) * 0.1
    logits_trojan[:, 2] += 15.0  # Dominant class 2 -> near zero entropy
    entropy_trojan_ref = strip_entropy(logits_trojan)
    
    # 4. Trigger apply: B=4, D=6
    x_input = torch.randn(4, 6)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    pattern = torch.tensor([5.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    x_trojan_ref = trojan_trigger_apply(x_input, mask, pattern)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "representations": reprs,
        "spectral_scores_ref": spectral_scores_ref,
        "l1_norms": l1_norms,
        "mad_anomaly_ref": mad_anomaly_ref,
        "logits_benign": logits_benign,
        "entropy_benign_ref": entropy_benign_ref,
        "logits_trojan": logits_trojan,
        "entropy_trojan_ref": entropy_trojan_ref,
        "x_input": x_input,
        "mask": mask,
        "pattern": pattern,
        "x_trojan_ref": x_trojan_ref,
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
