"""Abliteration Refusal Surgery Circuit Breakers and Sleeper Agents — PyTorch 2.x reference implementation.

Topic: 11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents
Golden weights: weights/11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents.safetensors

``python pytorch2/11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents.py`` verifies against the golden checkpoint;
``python pytorch2/11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def extract_refusal_direction(
    activations_harmful: torch.Tensor,
    activations_harmless: torch.Tensor,
) -> torch.Tensor:
    """Extracts mean refusal direction r_hat = (mu_harmful - mu_harmless) / ||mu_harmful - mu_harmless||_2.
    
    Operates on residual stream or MLP activation representations.
    """
    mu_harmful = torch.mean(activations_harmful, dim=0)
    mu_harmless = torch.mean(activations_harmless, dim=0)
    diff = mu_harmful - mu_harmless
    norm = torch.norm(diff, p=2) + 1e-8
    return diff / norm


def abliterate_weights(
    weight: torch.Tensor,
    refusal_direction: torch.Tensor,
) -> torch.Tensor:
    """Abliterates refusal mechanism by projecting weight matrix orthogonal to refusal direction:
    W' = W - r_hat * (r_hat^T W).
    """
    r = refusal_direction.view(-1, 1)  # [D_in, 1]
    # Component along r: r @ (r.T @ W)
    proj = r @ (r.t() @ weight)
    return weight - proj


def contrastive_activation_addition(
    h: torch.Tensor,
    steering_vector: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Contrastive Activation Addition (CAA / RepE): h' = h + alpha * steering_vector."""
    return h + float(alpha) * steering_vector


def circuit_breaker_loss(
    h_harmful: torch.Tensor,
    h_target_orthogonal: torch.Tensor,
    h_benign: torch.Tensor,
    h_ref_benign: torch.Tensor,
    lambda_cb: float = 1.0,
) -> torch.Tensor:
    """Safety Circuit Breakers loss (Zou et al. 2024 / RepBend).
    
    Penalizes cosine similarity between harmful representations and target benign manifold,
    while penalizing distortion on benign queries.
    """
    # Cosine similarity between harmful representation and target
    dot = torch.sum(h_harmful * h_target_orthogonal, dim=-1)
    norm_h = torch.norm(h_harmful, p=2, dim=-1) + 1e-8
    norm_t = torch.norm(h_target_orthogonal, p=2, dim=-1) + 1e-8
    cos_sim = dot / (norm_h * norm_t)
    l_breaker = torch.mean(cos_sim ** 2)
    
    # Benign representation retention
    l_benign = torch.mean((h_benign - h_ref_benign) ** 2)
    
    return l_breaker + float(lambda_cb) * l_benign


def sleeper_agent_probe(
    h: torch.Tensor,
    probe_w: torch.Tensor,
    probe_b: torch.Tensor,
) -> torch.Tensor:
    """Linear probe for auditing deceptive alignment / backdoor sleeper agents."""
    return h @ probe_w + probe_b


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # Setup dimensions: D=6
    d_model = 6
    act_harmful = torch.randn(10, d_model) + torch.tensor([2.0, -1.0, 1.5, 0.0, -0.5, 1.0])
    act_harmless = torch.randn(10, d_model) + torch.tensor([0.2, 0.5, -0.3, 0.1, 0.2, -0.1])
    
    refusal_dir = extract_refusal_direction(act_harmful, act_harmless)
    
    # Down-projection weight matrix: D_in=6, D_out=4
    w_mlp = torch.randn(6, 4) * 0.5
    w_ablated = abliterate_weights(w_mlp, refusal_dir)
    
    # CAA steering
    h_query = torch.randn(4, d_model)
    h_steered = contrastive_activation_addition(h_query, refusal_dir, alpha=2.0)
    
    # Circuit Breakers
    h_harmful = torch.randn(4, d_model)
    h_target_ortho = torch.randn(4, d_model)
    h_benign = torch.randn(4, d_model)
    h_ref_benign = torch.randn(4, d_model)
    cb_loss_ref = circuit_breaker_loss(h_harmful, h_target_ortho, h_benign, h_ref_benign, lambda_cb=1.5)
    
    # Sleeper probe
    probe_w = torch.randn(d_model, 1) * 0.5
    probe_b = torch.tensor([-0.5])
    probe_scores = sleeper_agent_probe(h_query, probe_w, probe_b)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "act_harmful": act_harmful,
        "act_harmless": act_harmless,
        "refusal_dir_ref": refusal_dir,
        "w_mlp": w_mlp,
        "w_ablated_ref": w_ablated,
        "h_query": h_query,
        "h_steered_ref": h_steered,
        "h_harmful": h_harmful,
        "h_target_ortho": h_target_ortho,
        "h_benign": h_benign,
        "h_ref_benign": h_ref_benign,
        "cb_loss_ref": cb_loss_ref,
        "probe_w": probe_w,
        "probe_b": probe_b,
        "probe_scores_ref": probe_scores,
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
