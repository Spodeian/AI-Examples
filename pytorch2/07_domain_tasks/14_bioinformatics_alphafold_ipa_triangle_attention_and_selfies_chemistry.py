"""Bioinformatics Alphafold Ipa Triangle Attention and Selfies Chemistry — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry
Golden weights: weights/07_domain_tasks/14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry.safetensors

``python pytorch2/07_domain_tasks/14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def triangle_multiplicative_update(
    z: torch.Tensor,
    w_a: torch.Tensor,
    b_a: torch.Tensor,
    w_b: torch.Tensor,
    b_b: torch.Tensor,
    w_g: torch.Tensor,
    b_g: torch.Tensor,
    w_o: torch.Tensor,
    b_o: torch.Tensor,
) -> torch.Tensor:
    """AlphaFold Triangle Multiplicative Update (Outgoing Edges):
    z: [B, N, N, C]
    Returns: [B, N, N, C]
    """
    a = z @ w_a + b_a  # [B, N, N, C]
    b = z @ w_b + b_b  # [B, N, N, C]
    g = torch.sigmoid(z @ w_g + b_g)  # [B, N, N, C]

    # Outgoing sum: S_{i, j, c} = sum_k a_{i, k, c} * b_{j, k, c}
    # einsum: 'bikc, bjkc -> bijc'
    s = torch.einsum("bikc, bjkc -> bijc", a, b)
    out = (g * s) @ w_o + b_o
    return out


def fape_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    r_x: torch.Tensor,
    t_x: torch.Tensor,
    r_y: torch.Tensor,
    t_y: torch.Tensor,
    d_clamp: float = 10.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """AlphaFold Frame Aligned Point Error (FAPE) Loss:
    x, y: [B, N, 3] (predicted and true 3D coordinates)
    r_x, r_y: [B, N, 3, 3] (rotations)
    t_x, t_y: [B, N, 3] (translations)
    Returns: scalar FAPE loss
    """
    # x_diff: x_j - t_{x, i} -> [B, N_i, N_j, 3]
    # r_x: [B, N_i, 3, 3]
    # local_x: R_i^T (x_j - t_i)
    diff_x = x.unsqueeze(1) - t_x.unsqueeze(2)  # [B, N, N, 3]
    local_x = torch.einsum("bijc, biac -> bija", diff_x, r_x)  # [B, N, N, 3]

    diff_y = y.unsqueeze(1) - t_y.unsqueeze(2)  # [B, N, N, 3]
    local_y = torch.einsum("bijc, biac -> bija", diff_y, r_y)  # [B, N, N, 3]

    dist = torch.sqrt(torch.sum((local_x - local_y) ** 2, dim=-1) + eps)
    clamped_dist = torch.clamp(dist, max=d_clamp)
    return clamped_dist.mean()


def ipa_point_distance_penalty(
    q_pts: torch.Tensor,
    k_pts: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """AlphaFold Invariant Point Attention (IPA) 3D Point Distance Penalty:
    q_pts: [B, N, H, P, 3]
    k_pts: [B, N, H, P, 3]
    gamma: [H]
    Returns: [B, N, N, H] penalty matrix = -0.5 * gamma_h * sum_p ||q_{i, h, p} - k_{j, h, p}||^2
    """
    # diff: [B, N_i, 1, H, P, 3] - [B, 1, N_j, H, P, 3] -> [B, N, N, H, P, 3]
    diff = q_pts.unsqueeze(2) - k_pts.unsqueeze(1)
    sq_dist = torch.sum(diff ** 2, dim=(-2, -1))  # sum over P and 3 -> [B, N, N, H]
    penalty = -0.5 * gamma.view(1, 1, 1, -1) * sq_dist
    return penalty


def molecular_valence_loss(
    bond_logits: torch.Tensor,
    max_valence: torch.Tensor,
) -> torch.Tensor:
    """Molecular chemistry valence penalty loss:
    bond_logits: [B, N, N]
    max_valence: [B, N]
    Returns: scalar penalty = mean(sum_i (max(0, valence_i - max_valence_i))^2)
    """
    b, n, _ = bond_logits.shape
    # Symmetric bond probabilities with zero diagonal
    probs = torch.sigmoid(bond_logits)
    mask = 1.0 - torch.eye(n, device=bond_logits.device).unsqueeze(0)
    masked_probs = probs * mask

    valences = torch.sum(masked_probs, dim=-1)  # [B, N]
    excess = F.relu(valences - max_valence)  # [B, N]
    return torch.mean(torch.sum(excess ** 2, dim=-1))


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
    }

    # 1. Triangle Multiplicative Update
    b, n, c = 2, 4, 8
    z = torch.randn(b, n, n, c) * 0.5
    w_a = torch.randn(c, c) * 0.1
    b_a = torch.randn(c) * 0.1
    w_b = torch.randn(c, c) * 0.1
    b_b = torch.randn(c) * 0.1
    w_g = torch.randn(c, c) * 0.1
    b_g = torch.randn(c) * 0.1
    w_o = torch.randn(c, c) * 0.1
    b_o = torch.randn(c) * 0.1

    tri_out_ref = triangle_multiplicative_update(z, w_a, b_a, w_b, b_b, w_g, b_g, w_o, b_o)
    tensors["z"] = z
    tensors["w_a"] = w_a
    tensors["b_a"] = b_a
    tensors["w_b"] = w_b
    tensors["b_b"] = b_b
    tensors["w_g"] = w_g
    tensors["b_g"] = b_g
    tensors["w_o"] = w_o
    tensors["b_o"] = b_o
    tensors["tri_out_ref"] = tri_out_ref

    # 2. FAPE Loss
    x = torch.randn(b, n, 3)
    y = torch.randn(b, n, 3)
    r_x = torch.randn(b, n, 3, 3)
    r_y = torch.randn(b, n, 3, 3)
    t_x = torch.randn(b, n, 3)
    t_y = torch.randn(b, n, 3)
    fape_loss_ref = fape_loss(x, y, r_x, t_x, r_y, t_y, d_clamp=10.0)

    tensors["x"] = x
    tensors["y"] = y
    tensors["r_x"] = r_x
    tensors["r_y"] = r_y
    tensors["t_x"] = t_x
    tensors["t_y"] = t_y
    tensors["fape_loss_ref"] = fape_loss_ref

    # 3. IPA Point Distance Penalty
    h_heads, p_points = 2, 2
    q_pts = torch.randn(b, n, h_heads, p_points, 3)
    k_pts = torch.randn(b, n, h_heads, p_points, 3)
    gamma = torch.tensor([0.5, 0.2])
    ipa_penalty_ref = ipa_point_distance_penalty(q_pts, k_pts, gamma)

    tensors["q_pts"] = q_pts
    tensors["k_pts"] = k_pts
    tensors["gamma"] = gamma
    tensors["ipa_penalty_ref"] = ipa_penalty_ref

    # 4. Molecular Valence Loss
    bond_logits = torch.randn(b, n, n)
    max_valence = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 4.0, 2.0, 1.0]])
    valence_loss_ref = molecular_valence_loss(bond_logits, max_valence)

    tensors["bond_logits"] = bond_logits
    tensors["max_valence"] = max_valence
    tensors["valence_loss_ref"] = valence_loss_ref

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
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
