"""Mechanistic Interpretability Probing Lens Patching and SAE — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae
Golden weights: weights/10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def logit_lens(
    hidden_states: torch.Tensor,
    unembed_weight: torch.Tensor,
    unembed_bias: torch.Tensor,
) -> torch.Tensor:
    """Logit Lens: Project intermediate residual stream hidden states into vocabulary logits.
    
    logits_l = h_l @ W_U + b_U
    """
    return hidden_states @ unembed_weight + unembed_bias


def attribution_patching(
    clean_acts: torch.Tensor,
    corrupt_acts: torch.Tensor,
    clean_grads: torch.Tensor,
) -> torch.Tensor:
    """First-order attribution patching (Syed et al., 2023) approximating activation patching.
    
    AtP(a) = (a_corrupt - a_clean) * grad_a L(clean)
    """
    diff = corrupt_acts - clean_acts
    atp_per_dim = diff * clean_grads
    return atp_per_dim.sum(dim=-1)


def topk_sparse_autoencoder(
    x: torch.Tensor,
    w_enc: torch.Tensor,
    b_enc: torch.Tensor,
    w_dec: torch.Tensor,
    b_dec: torch.Tensor,
    k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-K Sparse Autoencoder (Gao et al., 2024).
    
    z = (x - b_dec) @ W_enc + b_enc
    f_z = TopK(ReLU(z), k)
    x_hat = f_z @ W_dec + b_dec
    """
    x_cent = x - b_dec
    pre_act = x_cent @ w_enc + b_enc
    relu_act = torch.relu(pre_act)
    
    # Top-K selection
    val, idx = torch.topk(relu_act, k=k, dim=-1)
    f_z = torch.zeros_like(relu_act)
    f_z.scatter_(dim=-1, index=idx, src=val)
    
    x_hat = f_z @ w_dec + b_dec
    return f_z, x_hat


def jumprelu_activation(
    z: torch.Tensor,
    threshold: torch.Tensor,
) -> torch.Tensor:
    """JumpReLU hard thresholding without L1 shrinkage bias (Rajamanoharan et al., 2024).
    
    JumpReLU(z, theta) = z * I(z > theta)
    """
    mask = torch.where(z > threshold, 1.0, 0.0)
    return z * mask


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. Logit Lens setup: N=4 hidden states, D=4 model dim, V=5 vocabulary size
    ll_h = torch.tensor([
        [1.2, -0.5, 0.8, 0.3],
        [-0.4, 1.6, 0.1, -0.8],
        [0.7, 0.2, 1.4, 0.5],
        [-0.1, -0.9, 0.6, 1.1],
    ], dtype=torch.float32)
    ll_w = torch.tensor([
        [0.4, -0.2, 0.5, 0.1, 0.3],
        [-0.1, 0.6, 0.2, -0.4, 0.0],
        [0.5, 0.1, -0.3, 0.7, 0.2],
        [0.2, -0.5, 0.4, 0.1, -0.2],
    ], dtype=torch.float32)
    ll_b = torch.tensor([0.1, -0.1, 0.2, 0.0, -0.2], dtype=torch.float32)
    ll_logits_ref = logit_lens(ll_h, ll_w, ll_b)

    # 2. Attribution Patching setup: N=4 samples, D=4 activation dimensions
    atp_clean = torch.tensor([
        [0.8, 1.2, -0.4, 0.5],
        [0.2, -0.6, 1.5, 0.1],
        [-0.5, 0.9, 0.3, 1.4],
        [1.1, 0.0, -0.8, 0.7],
    ], dtype=torch.float32)
    atp_corrupt = torch.tensor([
        [0.1, 1.0, 0.2, -0.3],
        [-0.4, -0.1, 0.8, 0.6],
        [0.2, 0.4, -0.5, 0.9],
        [0.5, -0.7, 0.1, 0.2],
    ], dtype=torch.float32)
    atp_grads = torch.tensor([
        [0.5, -0.2, 0.8, 0.4],
        [-0.3, 0.7, 0.1, -0.5],
        [0.6, 0.1, -0.4, 0.3],
        [-0.1, 0.4, 0.5, -0.2],
    ], dtype=torch.float32)
    atp_scores_ref = attribution_patching(atp_clean, atp_corrupt, atp_grads)

    # 3. Top-K Sparse Autoencoder setup: N=3 samples, D=4 model dim, D_sae=6 dictionary dim, K=2
    sae_x = torch.tensor([
        [1.5, -0.8, 2.0, 0.4],
        [-0.5, 1.2, 0.3, -1.0],
        [0.8, 0.1, -0.6, 1.4],
    ], dtype=torch.float32)
    sae_w_enc = torch.tensor([
        [0.4, -0.2, 0.5, 0.1, -0.3, 0.6],
        [-0.1, 0.5, -0.4, 0.3, 0.2, -0.1],
        [0.6, 0.2, 0.1, -0.5, 0.4, 0.3],
        [-0.2, 0.3, -0.1, 0.4, -0.5, 0.2],
    ], dtype=torch.float32)
    sae_b_enc = torch.tensor([0.1, -0.1, 0.2, 0.0, -0.2, 0.3], dtype=torch.float32)
    sae_w_dec = torch.tensor([
        [0.3, -0.1, 0.5, -0.2],
        [-0.2, 0.4, 0.1, 0.3],
        [0.4, -0.3, 0.2, -0.1],
        [0.1, 0.2, -0.4, 0.5],
        [-0.3, 0.1, 0.3, -0.4],
        [0.5, -0.2, 0.2, 0.1],
    ], dtype=torch.float32)
    sae_b_dec = torch.tensor([0.2, -0.1, 0.1, 0.0], dtype=torch.float32)
    sae_feat_ref, sae_xhat_ref = topk_sparse_autoencoder(sae_x, sae_w_enc, sae_b_enc, sae_w_dec, sae_b_dec, k=2)

    # 4. JumpReLU setup: 6 feature activations, scalar threshold
    jump_z = torch.tensor([1.4, 0.3, -0.5, 2.1, 0.7, 0.0], dtype=torch.float32)
    jump_thresh = torch.tensor(0.5, dtype=torch.float32)
    jump_out_ref = jumprelu_activation(jump_z, jump_thresh)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "ll_h": ll_h,
        "ll_w": ll_w,
        "ll_b": ll_b,
        "ll_logits_ref": ll_logits_ref,
        "atp_clean": atp_clean,
        "atp_corrupt": atp_corrupt,
        "atp_grads": atp_grads,
        "atp_scores_ref": atp_scores_ref,
        "sae_x": sae_x,
        "sae_w_enc": sae_w_enc,
        "sae_b_enc": sae_b_enc,
        "sae_w_dec": sae_w_dec,
        "sae_b_dec": sae_b_dec,
        "sae_feat_ref": sae_feat_ref,
        "sae_xhat_ref": sae_xhat_ref,
        "jump_z": jump_z,
        "jump_thresh": jump_thresh,
        "jump_out_ref": jump_out_ref,
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
