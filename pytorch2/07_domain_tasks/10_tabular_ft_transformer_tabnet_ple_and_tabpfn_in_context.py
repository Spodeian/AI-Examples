"""Tabular Ft Transformer Tabnet Ple and Tabpfn In Context — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context
Golden weights: weights/07_domain_tasks/10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context.safetensors

``python pytorch2/07_domain_tasks/10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ft_transformer_tokenize(
    x_num: torch.Tensor,
    w_num: torch.Tensor,
    b_num: torch.Tensor,
    cat_tokens: torch.Tensor,
    cls_token: torch.Tensor,
) -> torch.Tensor:
    """FT-Transformer Feature Tokenizer:
    x_num: [B, N_num]
    w_num: [N_num, D]
    b_num: [N_num, D]
    cat_tokens: [B, N_cat, D] (pre-gathered categorical embeddings)
    cls_token: [1, 1, D]
    Returns: [B, 1 + N_num + N_cat, D]
    """
    b, n_num = x_num.shape
    # Numerical tokens: x_num[b, j] * w_num[j] + b_num[j]
    num_tokens = x_num.unsqueeze(-1) * w_num.unsqueeze(0) + b_num.unsqueeze(0)  # [B, N_num, D]
    cls_expanded = cls_token.expand(b, 1, -1)  # [B, 1, D]
    return torch.cat([cls_expanded, num_tokens, cat_tokens], dim=1)


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact Sparsemax activation projection onto probability simplex."""
    # z: [B, D]
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    d = z.shape[dim]
    range_d = torch.arange(1, d + 1, dtype=z.dtype, device=z.device)
    for _ in range(z.ndim - 1):
        range_d = range_d.unsqueeze(0)

    cumsum_z = torch.cumsum(z_sorted, dim=dim)
    condition = 1.0 + range_d * z_sorted > cumsum_z
    # Find k(z) = max {k in 1..D : condition}
    k_z = torch.sum(condition.to(z.dtype), dim=dim, keepdim=True)
    # Tau(z) = (sum_{j=1}^k z_(j) - 1) / k
    tau_z = (torch.gather(cumsum_z, dim, (k_z - 1).long()) - 1.0) / k_z
    return torch.clamp(z - tau_z, min=0.0)


def tabnet_attentive_mask(
    p_prior: torch.Tensor,
    h_features: torch.Tensor,
    gamma: float = 1.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TabNet Attentive Transformer feature selection mask and prior update.
    p_prior: [B, D_feat]
    h_features: [B, D_feat]
    Returns: (mask, p_next)
    """
    logits = p_prior * h_features
    mask = sparsemax(logits, dim=-1)
    p_next = p_prior * (gamma - mask)
    return mask, p_next


def ple_task_routing(
    expert_outputs: torch.Tensor,
    gate_logits: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    """Progressive Layered Extraction (PLE) gated task aggregation.
    expert_outputs: [B, Total_Experts, D_out]
    gate_logits: [B, K]
    expert_indices: [K] selection of specific + shared experts
    Returns: routed task representation [B, D_out]
    """
    # Select candidate experts for this task: [B, K, D_out]
    candidates = expert_outputs[:, expert_indices, :]
    weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)  # [B, K, 1]
    return torch.sum(candidates * weights, dim=1)  # [B, D_out]


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

    # 1. FT-Transformer Tokenizer
    b, n_num, n_cat, d = 2, 4, 2, 8
    x_num = torch.randn(b, n_num)
    w_num = torch.randn(n_num, d) * 0.1
    b_num = torch.randn(n_num, d) * 0.1
    cat_tokens = torch.randn(b, n_cat, d) * 0.1
    cls_token = torch.randn(1, 1, d) * 0.1

    tokenized_ref = ft_transformer_tokenize(x_num, w_num, b_num, cat_tokens, cls_token)
    tensors["x_num"] = x_num
    tensors["w_num"] = w_num
    tensors["b_num"] = b_num
    tensors["cat_tokens"] = cat_tokens
    tensors["cls_token"] = cls_token
    tensors["tokenized_ref"] = tokenized_ref

    # 2. Sparsemax
    z = torch.randn(b, 6)
    sparsemax_ref = sparsemax(z)
    tensors["z_sparsemax"] = z
    tensors["sparsemax_ref"] = sparsemax_ref

    # 3. TabNet Attentive Mask
    d_feat = 6
    p_prior = torch.ones(b, d_feat)
    h_feat = torch.randn(b, d_feat)
    mask_ref, p_next_ref = tabnet_attentive_mask(p_prior, h_feat, gamma=1.3)
    tensors["p_prior"] = p_prior
    tensors["h_feat"] = h_feat
    tensors["tabnet_mask_ref"] = mask_ref
    tensors["tabnet_p_next_ref"] = p_next_ref

    # 4. PLE Task Routing
    total_experts, d_out = 6, 8
    expert_outputs = torch.randn(b, total_experts, d_out)
    gate_logits = torch.randn(b, 4)
    expert_indices = torch.tensor([0, 1, 4, 5], dtype=torch.long)  # 2 specific + 2 shared
    ple_routed_ref = ple_task_routing(expert_outputs, gate_logits, expert_indices)

    tensors["expert_outputs"] = expert_outputs
    tensors["gate_logits"] = gate_logits
    tensors["expert_indices"] = expert_indices
    tensors["ple_routed_ref"] = ple_routed_ref

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
