"""Tree Speculative Medusa Eagle GoT and Jacobi — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/04_tree_speculative_medusa_eagle_got_and_jacobi
Golden weights: weights/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/04_tree_speculative_medusa_eagle_got_and_jacobi.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/04_tree_speculative_medusa_eagle_got_and_jacobi.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/04_tree_speculative_medusa_eagle_got_and_jacobi.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/04_tree_speculative_medusa_eagle_got_and_jacobi"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def medusa_heads_predict(
    hidden: torch.Tensor,
    head_weights: torch.Tensor,
    head_biases: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Medusa multi-head forward prediction.

    Evaluates K auxiliary heads attached to final transformer hidden state to predict
    candidate token distributions at offsets t+1, t+2, ..., t+K.

    Args:
        hidden: Hidden state representation of shape [batch, dim].
        head_weights: Weight tensors for K heads of shape [num_heads, dim, vocab].
        head_biases: Bias tensors for K heads of shape [num_heads, vocab].

    Returns:
        logits: Multi-head candidate logits of shape [batch, num_heads, vocab].
        top_tokens: Top-1 predicted token per head of shape [batch, num_heads].
    """
    batch, dim = hidden.shape
    num_heads, _, vocab = head_weights.shape

    logits_list = []
    top_tokens_list = []

    for k in range(num_heads):
        head_logits = hidden @ head_weights[k] + head_biases[k]
        top_tok = torch.argmax(head_logits, dim=-1)
        logits_list.append(head_logits)
        top_tokens_list.append(top_tok)

    logits = torch.stack(logits_list, dim=1)  # [batch, num_heads, vocab]
    top_tokens = torch.stack(top_tokens_list, dim=1)  # [batch, num_heads]
    return logits, top_tokens


def build_tree_attention_mask(
    parent_indices: torch.Tensor,
) -> torch.Tensor:
    """Construct 2D causal tree attention mask from tree node ancestry.

    Given a tree with N nodes where parent_indices[i] indicates the parent node of i
    (with parent_indices[0] == 0 for root), builds mask M of shape [N, N] where:
        M[i, j] = 0.0 if j is in Ancestors(i) or j == i
        M[i, j] = -1e9 otherwise.

    Args:
        parent_indices: 1D tensor of shape [num_nodes] containing parent node indices.

    Returns:
        mask: 2D attention mask of shape [num_nodes, num_nodes].
    """
    num_nodes = parent_indices.shape[0]
    mask = torch.full((num_nodes, num_nodes), -1e9, dtype=torch.float32, device=parent_indices.device)

    for i in range(num_nodes):
        # Walk up ancestor tree from node i
        curr = i
        visited = set()
        while curr not in visited:
            mask[i, curr] = 0.0
            visited.add(curr)
            if curr == 0:
                break
            curr = parent_indices[curr].item()

    return mask


def tree_longest_accepted_path(
    candidate_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    parent_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the longest continuous matching path in speculative candidate tree.

    Args:
        candidate_tokens: Tree node token IDs of shape [num_nodes].
        target_tokens: Ground-truth target verification tokens for each node, shape [num_nodes].
        parent_indices: Tree hierarchy parent mapping of shape [num_nodes].

    Returns:
        accepted_nodes: Binary indicator of shape [num_nodes] for nodes on accepted longest path.
        path_length: Scalar tensor representing total accepted tokens on the path.
    """
    num_nodes = candidate_tokens.shape[0]
    is_match = (candidate_tokens == target_tokens)

    # Path depth from root along matching edges
    accepted = [False] * num_nodes
    # Root (node 0) is prompt / verified anchor
    accepted[0] = True

    # Breadth-first / topological progression to mark valid connected matches
    for i in range(1, num_nodes):
        parent = parent_indices[i].item()
        if accepted[parent] and is_match[i].item():
            accepted[i] = True

    # Identify deepest node
    best_len = 0
    best_node = 0
    depths = [0] * num_nodes
    for i in range(num_nodes):
        if accepted[i]:
            if i == 0:
                depths[i] = 1
            else:
                depths[i] = depths[parent_indices[i].item()] + 1
            if depths[i] > best_len:
                best_len = depths[i]
                best_node = i

    # Trace back longest path
    path_mask = torch.zeros(num_nodes, dtype=torch.float32, device=candidate_tokens.device)
    curr = best_node
    while True:
        path_mask[curr] = 1.0
        if curr == 0:
            break
        curr = parent_indices[curr].item()

    return path_mask, torch.tensor(float(best_len), dtype=torch.float32, device=candidate_tokens.device)


def jacobi_lookahead_step(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jacobi fixed-point non-autoregressive iteration step.

    Computes parallel point update x_{new} = tanh(x W + b) and returns convergence residual.

    Args:
        x: Token representations of shape [batch, seq_len, dim].
        weight: Transition matrix of shape [dim, dim].
        bias: Bias of shape [dim].

    Returns:
        x_next: Updated token representations of shape [batch, seq_len, dim].
        residual: Maximum absolute delta ||x_next - x||_inf of shape [batch].
    """
    x_next = torch.tanh(x @ weight + bias)
    residual, _ = torch.max(torch.abs(x_next - x).flatten(start_dim=1), dim=-1)
    return x_next, residual


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Medusa Heads
    batch = 2
    dim = 8
    vocab = 12
    num_heads = 3
    hidden = torch.randn(batch, dim)
    head_weights = torch.randn(num_heads, dim, vocab) * 0.2
    head_biases = torch.randn(num_heads, vocab) * 0.1
    medusa_logits, top_tokens = medusa_heads_predict(hidden, head_weights, head_biases)

    # 2. Tree Attention Mask
    # 6 nodes: 0 (root), 1 (child of 0), 2 (child of 0), 3 (child of 1), 4 (child of 1), 5 (child of 2)
    parent_indices = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64)
    tree_mask = build_tree_attention_mask(parent_indices)

    # 3. Tree Longest Accepted Path
    candidate_tokens = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int64)
    # Target matches node 1 and node 4 (path 0 -> 1 -> 4 of length 3), while node 2 matches but 5 fails
    target_tokens = torch.tensor([10, 20, 30, 99, 50, 99], dtype=torch.int64)
    path_mask, path_length = tree_longest_accepted_path(candidate_tokens, target_tokens, parent_indices)

    # 4. Jacobi Iteration
    x_jacobi = torch.randn(batch, 4, dim) * 0.5
    w_jacobi = torch.randn(dim, dim) * 0.2
    b_jacobi = torch.randn(dim) * 0.1
    x_next, jacobi_res = jacobi_lookahead_step(x_jacobi, w_jacobi, b_jacobi)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "hidden": hidden,
        "head_weights": head_weights,
        "head_biases": head_biases,
        "medusa_logits": medusa_logits,
        "top_tokens": top_tokens.to(torch.float32),
        "parent_indices": parent_indices.to(torch.float32),
        "tree_mask": tree_mask,
        "candidate_tokens": candidate_tokens.to(torch.float32),
        "target_tokens": target_tokens.to(torch.float32),
        "path_mask": path_mask,
        "path_length": path_length,
        "x_jacobi": x_jacobi,
        "w_jacobi": w_jacobi,
        "b_jacobi": b_jacobi,
        "x_next": x_next,
        "jacobi_res": jacobi_res,
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
        assert err < 1e-5, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
