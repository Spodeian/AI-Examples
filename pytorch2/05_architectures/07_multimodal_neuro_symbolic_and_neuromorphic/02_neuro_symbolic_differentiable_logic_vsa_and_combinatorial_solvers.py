"""Neuro-Symbolic, Differentiable Logic, VSA, and Combinatorial Solvers — PyTorch 2.x reference implementation.

Topic: 05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/02_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers
Golden weights: weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/02_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers.safetensors

``python pytorch2/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/02_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/02_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/02_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def lukasiewicz_logic(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable Łukasiewicz fuzzy logic operators: AND (t-norm), OR (t-conorm), Implication.
    
    Inputs a, b in [0, 1] of shape [B, D].
    Returns (t_norm, t_conorm, implication).
    """
    t_norm = torch.clamp(a + b - 1.0, min=0.0, max=1.0)
    t_conorm = torch.clamp(a + b, min=0.0, max=1.0)
    implication = torch.clamp(1.0 - a + b, min=0.0, max=1.0)
    return t_norm, t_conorm, implication


def vsa_bind_and_bundle(
    key: torch.Tensor,
    val: torch.Tensor,
    other: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vector Symbolic Architecture (VSA / HDC):
    - Binding: Hadamard product key * val
    - Bundling: L2-normalized superposition (bound + other)
    - Associative cleanup: Cosine similarity against codebook.
    
    key, val, other of shape [B, D].
    codebook of shape [M, D].
    """
    bound = key * val
    unnorm_bundled = bound + other
    bundled = F.normalize(unnorm_bundled, p=2.0, dim=-1)
    
    # Cosine cleanup: query bundled against codebook [M, D]
    norm_cb = F.normalize(codebook, p=2.0, dim=-1)
    sim = bundled @ norm_cb.t()
    return bound, bundled, sim


def sinkhorn_knopp_solver(logits: torch.Tensor, num_iters: int = 5, tau: float = 0.5) -> torch.Tensor:
    """Differentiable Sinkhorn-Knopp combinatorial solver:
    Projects unconstrained logits [B, N, N] onto the Birkhoff polytope (doubly stochastic matrices).
    """
    p = torch.exp(logits / tau)
    for _ in range(num_iters):
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
        p = p / (p.sum(dim=-2, keepdim=True) + 1e-8)
    return p


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Logic: B=3, D=5 in [0, 1]
    a = torch.sigmoid(torch.randn(3, 5))
    b = torch.sigmoid(torch.randn(3, 5))
    t_norm, t_conorm, imp = lukasiewicz_logic(a, b)

    # VSA: B=2, D=16, M=4 codebook items
    D = 16
    key = torch.sign(torch.randn(2, D))
    key[key == 0] = 1.0
    val = torch.sign(torch.randn(2, D))
    val[val == 0] = 1.0
    other = torch.randn(2, D) * 0.5
    codebook = torch.randn(4, D)
    bound, bundled, sim = vsa_bind_and_bundle(key, val, other, codebook)

    # Sinkhorn solver: B=2, N=4
    logits = torch.randn(2, 4, 4)
    p_doubly_stochastic = sinkhorn_knopp_solver(logits, num_iters=5, tau=0.5)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "logic_a": a,
        "logic_b": b,
        "t_norm": t_norm,
        "t_conorm": t_conorm,
        "implication": imp,
        "vsa_key": key,
        "vsa_val": val,
        "vsa_other": other,
        "vsa_codebook": codebook,
        "vsa_bound": bound,
        "vsa_bundled": bundled,
        "vsa_sim": sim,
        "sinkhorn_logits": logits,
        "sinkhorn_p": p_doubly_stochastic,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().clone().contiguous().cpu() for k, v in build_golden_tensors().items()}
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
