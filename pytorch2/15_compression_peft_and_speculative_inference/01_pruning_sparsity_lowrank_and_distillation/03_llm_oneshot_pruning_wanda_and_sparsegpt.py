"""LLM Oneshot Pruning Wanda and Sparsegpt — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/03_llm_oneshot_pruning_wanda_and_sparsegpt
Golden weights: weights/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/03_llm_oneshot_pruning_wanda_and_sparsegpt.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/03_llm_oneshot_pruning_wanda_and_sparsegpt.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/03_llm_oneshot_pruning_wanda_and_sparsegpt.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/03_llm_oneshot_pruning_wanda_and_sparsegpt"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def compute_activation_norms(x: torch.Tensor) -> torch.Tensor:
    """Computes column-wise L2 norm of calibration activations X."""
    return torch.norm(x, p=2, dim=0)


def wanda_prune(
    w: torch.Tensor, act_norms: torch.Tensor, sparsity: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wanda (Pruning by Weights and Activations) one-shot row-wise pruning.

    Score: S_ij = |W_ij| * ||X_:,j||_2.
    In each row, drops the smallest fraction `sparsity` of scores.

    Args:
        w: Weight tensor of shape [d_out, d_in].
        act_norms: Activation L2 norms of shape [d_in].
        sparsity: Fraction of weights to prune per row.

    Returns:
        w_pruned: Pruned weight tensor.
        mask: Binary indicator tensor (1 = kept, 0 = pruned).
    """
    d_out, d_in = w.shape
    scores = torch.abs(w) * act_norms.unsqueeze(0)
    k = max(1, int(sparsity * d_in))

    # Row-wise thresholding
    tau = torch.kthvalue(scores, k, dim=-1, keepdim=True).values
    mask = (scores >= tau).to(torch.float32)
    w_pruned = w * mask
    return w_pruned, mask


def sparsegpt_prune(
    w: torch.Tensor, h_inv: torch.Tensor, sparsity: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """SparseGPT Optimal Brain Surgeon (OBS) one-shot pruning with column compensation.

    Args:
        w: Weight tensor of shape [d_out, d_in].
        h_inv: Inverse Hessian matrix (X^T X + lambda I)^-1 of shape [d_in, d_in].
        sparsity: Fraction of weights to prune.

    Returns:
        w_pruned: Compensated pruned weight tensor.
        mask: Binary indicator tensor.
    """
    d_out, d_in = w.shape
    w_work = w.clone()
    diag = torch.diag(h_inv)

    # Column OBS error metric: score_j = ||w_:,j||_2 / sqrt([H^-1]_jj)
    col_scores = torch.norm(w_work, p=2, dim=0) / torch.sqrt(torch.clamp(diag, min=1e-8))
    k = max(1, int(sparsity * d_in))
    prune_cols = torch.topk(col_scores, k, largest=False).indices.sort().values

    mask = torch.ones_like(w)
    for j in prune_cols:
        col_w = w_work[:, j].clone()
        denom = float(h_inv[j, j])
        if denom < 1e-8:
            denom = 1e-8
        # Error feedback compensation to unpruned remaining columns j' > j
        w_work[:, j] = 0.0
        mask[:, j] = 0.0
        if j + 1 < d_in:
            h_slice = h_inv[j, (j + 1) :]
            w_work[:, (j + 1) :] -= (col_w.unsqueeze(1) / denom) * h_slice.unsqueeze(0)

    return w_work, mask


def compute_reconstruction_error(
    x: torch.Tensor, w_orig: torch.Tensor, w_pruned: torch.Tensor
) -> float:
    """Computes relative Frobenius reconstruction error ||X W_orig^T - X W_pruned^T||_F^2 / ||X W_orig^T||_F^2."""
    y_orig = x @ w_orig.T
    y_pruned = x @ w_pruned.T
    num = torch.sum((y_orig - y_pruned) ** 2)
    den = torch.sum(y_orig**2) + 1e-8
    return float(num / den)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Calibration activations X: [32 tokens, 16 features]
    x = torch.randn(32, 16)
    # Weight matrix: [8 out_features, 16 in_features]
    w = torch.randn(8, 16) * 0.2

    # 1. Activation norms
    act_norms = compute_activation_norms(x)

    # 2. Inverse Hessian: H = X^T X + lambda I
    h = x.T @ x + 0.1 * torch.eye(16)
    h_inv = torch.linalg.inv(h)

    # 3. Wanda Pruning (50% sparsity)
    w_wanda, mask_wanda = wanda_prune(w, act_norms, sparsity=0.5)

    # 4. SparseGPT Pruning (50% sparsity)
    w_sparsegpt, mask_sparsegpt = sparsegpt_prune(w, h_inv, sparsity=0.5)

    # 5. Relative reconstruction errors
    err_wanda = compute_reconstruction_error(x, w, w_wanda)
    err_sparsegpt = compute_reconstruction_error(x, w, w_sparsegpt)
    err_metrics = torch.tensor([err_wanda, err_sparsegpt], dtype=torch.float32)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x": x,
        "w": w,
        "act_norms": act_norms,
        "h_inv": h_inv,
        "w_wanda": w_wanda,
        "mask_wanda": mask_wanda,
        "w_sparsegpt": w_sparsegpt,
        "mask_sparsegpt": mask_sparsegpt,
        "err_metrics": err_metrics,
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
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
