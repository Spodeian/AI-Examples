"""Expert Parallelism and 3d 4d Parallel Composition — PyTorch 2.x reference implementation.

Topic: 16_distributed_and_scalable/07_expert_parallelism_and_3d_4d_parallel_composition
Golden weights: weights/16_distributed_and_scalable/07_expert_parallelism_and_3d_4d_parallel_composition.safetensors

``python pytorch2/16_distributed_and_scalable/07_expert_parallelism_and_3d_4d_parallel_composition.py`` verifies against the golden checkpoint;
``python pytorch2/16_distributed_and_scalable/07_expert_parallelism_and_3d_4d_parallel_composition.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "16_distributed_and_scalable/07_expert_parallelism_and_3d_4d_parallel_composition"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def compute_3d_mesh_coordinates(dp: int, pp: int, tp: int) -> torch.Tensor:
    """Maps global rank indices to 3D Cartesian coordinates (dp_idx, pp_idx, tp_idx)."""
    world_size = dp * pp * tp
    coords = torch.zeros(world_size, 3, dtype=torch.int64)
    for r in range(world_size):
        dp_idx = r // (pp * tp)
        rem = r % (pp * tp)
        pp_idx = rem // tp
        tp_idx = rem % tp
        coords[r] = torch.tensor([dp_idx, pp_idx, tp_idx], dtype=torch.int64)
    return coords


def expert_parallel_forward(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_experts: torch.Tensor,
    ep_world_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expert Parallelism with Top-2 routing, dispatch, and combine.

    Args:
        x: Input tokens [T, D].
        w_gate: Router gate weights [D, E].
        w_experts: Expert weights [E, D, D].
        ep_world_size: Number of EP ranks (experts partitioned across ranks).

    Returns:
        moe_out: Aggregated output tokens [T, D].
        top2_indices: Chosen expert indices per token [T, 2].
        top2_probs: Softmax routing probabilities [T, 2].
        dispatch_mask: Boolean dispatch indicator per EP rank [ep_world_size, T, 2].
    """
    num_tokens, d_model = x.shape
    num_experts = w_experts.shape[0]
    experts_per_rank = num_experts // ep_world_size

    # Step 1: Routing logits and Top-2 selection
    gate_logits = x @ w_gate  # [T, E]
    top2_logits, top2_indices = torch.topk(gate_logits, k=2, dim=-1)
    top2_probs = F.softmax(top2_logits, dim=-1)  # [T, 2]

    # Step 2: Dispatch All-to-All routing mask
    # dispatch_mask[r, t, k] indicates whether token t's k-th choice is hosted on EP rank r
    dispatch_mask = torch.zeros(ep_world_size, num_tokens, 2, dtype=torch.bool)
    for r in range(ep_world_size):
        r_start = r * experts_per_rank
        r_end = (r + 1) * experts_per_rank
        dispatch_mask[r] = (top2_indices >= r_start) & (top2_indices < r_end)

    # Step 3: Local Expert Computation
    moe_out = torch.zeros_like(x)
    for t in range(num_tokens):
        for k in range(2):
            exp_idx = int(top2_indices[t, k])
            weight_k = top2_probs[t, k]
            exp_out = x[t] @ w_experts[exp_idx]
            moe_out[t] += weight_k * exp_out

    return moe_out, top2_indices, top2_probs, dispatch_mask.to(torch.float32)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    num_tokens = 4
    d_model = 4
    num_experts = 4
    ep_world_size = 2

    x = torch.randn(num_tokens, d_model)
    w_gate = torch.randn(d_model, num_experts) * 0.2
    w_experts = torch.randn(num_experts, d_model, d_model) * 0.25

    moe_out, top2_indices, top2_probs, dispatch_mask = expert_parallel_forward(
        x, w_gate, w_experts, ep_world_size=ep_world_size
    )

    mesh_coords = compute_3d_mesh_coordinates(dp=2, pp=2, tp=2)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x": x,
        "w_gate": w_gate,
        "w_experts": w_experts,
        "moe_out": moe_out,
        "top2_indices": top2_indices.to(torch.int64),
        "top2_probs": top2_probs,
        "dispatch_mask": dispatch_mask,
        "mesh_coords": mesh_coords,
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
