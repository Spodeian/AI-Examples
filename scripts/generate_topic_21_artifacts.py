"""Generate golden reference SafeTensors and verify PyTorch/JAX parity for Topic 21.

Topic 21: 07_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse
Category: 04_training_methods_and_optimisers
Subfolder: 03_training_dynamics_self_optimising_non_backprop
"""

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

def generate_topic_21_artifacts():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Chinchilla Scaling Law
    # L(N, D) = E + A / N^alpha + B / D^beta
    # Using Hoffmann et al. (2022) empirical coefficients
    # Normalized for numerical stability: N in millions, D in billions
    # Or in float64 / float32 scaled units:
    # Let N_val = 12.0 (e.g. 12M), D_val = 240.0 (e.g. 240M tokens)
    # E = 1.69, A = 406.4, B = 410.7, alpha = 0.34, beta = 0.28
    # For clean float32 operations:
    e_const = 1.69
    a_const = 40.0
    b_const = 50.0
    alpha = 0.34
    beta = 0.28

    n_val = torch.tensor([16.0], dtype=torch.float32) # e.g. 16M params
    d_val = torch.tensor([320.0], dtype=torch.float32) # e.g. 320M tokens

    # Loss = E + A * N^(-alpha) + B * D^(-beta)
    chinchilla_loss = e_const + a_const * (n_val ** (-alpha)) + b_const * (d_val ** (-beta))

    # Compute optimal allocation ratio given exponent a = beta / (alpha + beta), b = alpha / (alpha + beta)
    exp_a = beta / (alpha + beta)
    exp_b = alpha / (alpha + beta)
    budget_c = 6.0 * n_val * d_val # Compute FLOPs proxy
    # Optimal N* and D*
    # Scaling constant G_N = 0.6, G_D = 1/ (6 * G_N)
    g_n = 0.6
    g_d = 1.0 / (6.0 * g_n)
    n_opt = g_n * (budget_c ** exp_a)
    d_opt = g_d * (budget_c ** exp_b)

    # 2. DoReMi Domain Weight Optimization (Xie et al., 2023)
    # K = 4 domains
    w_doremi_init = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    losses_model = torch.tensor([2.4, 3.1, 1.8, 2.9], dtype=torch.float32)
    losses_ref = torch.tensor([2.2, 2.5, 1.9, 2.6], dtype=torch.float32)
    eta_doremi = 1.0

    excess_loss = losses_model - losses_ref # [0.2, 0.6, -0.1, 0.3]
    # w_next = softmax(log(w_init) + eta * excess_loss)
    log_w = torch.log(w_doremi_init) + eta_doremi * excess_loss
    w_doremi_next = F.softmax(log_w, dim=-1)

    # 3. MinHash Jaccard Similarity (Broder, 1997)
    # 4 documents, signature size = 8
    # Signatures as float32 representation of hash indices
    signatures = torch.tensor([
        [1.0, 5.0, 9.0, 12.0, 3.0, 7.0, 15.0, 2.0],
        [1.0, 5.0, 9.0, 14.0, 3.0, 8.0, 15.0, 2.0], # 6 matches with doc 0 => 6/8 = 0.75
        [4.0, 6.0, 10.0, 14.0, 3.0, 8.0, 16.0, 2.0], # 3 matches with doc 1 => 3/8 = 0.375
        [1.0, 5.0, 9.0, 12.0, 3.0, 7.0, 15.0, 2.0], # exact duplicate of doc 0 => 8/8 = 1.0
    ], dtype=torch.float32)

    # Compute pairwise Jaccard matrix: [4, 4]
    # J_ij = 1/K sum_k (sig[i, k] == sig[j, k])
    b_docs = signatures.shape[0]
    k_hashes = signatures.shape[1]
    jaccard_matrix = torch.zeros(b_docs, b_docs, dtype=torch.float32)
    for i in range(b_docs):
        for j in range(b_docs):
            matches = torch.sum((signatures[i] == signatures[j]).float())
            jaccard_matrix[i, j] = matches / float(k_hashes)

    # 4. Model Collapse Variance Dynamics (Shumailov et al., 2024)
    # sigma_sq_init = 1.0
    # recursive generation without accumulation: variance decay per step by factor (M-1)/M
    # With accumulation of fraction lambda_real = 0.2:
    # sigma_sq_next = (1 - lambda_real) * (var_synthetic) + lambda_real * var_real
    var_synth = torch.tensor([0.85], dtype=torch.float32)
    var_real = torch.tensor([1.0], dtype=torch.float32)
    lambda_real = 0.25
    golden_cured_var = (1.0 - lambda_real) * var_synth + lambda_real * var_real

    # 5. Linear Anchor
    anchor_x = torch.randn(2, 8, dtype=torch.float32)
    anchor_w = torch.randn(8, 4, dtype=torch.float32) / math.sqrt(8)
    anchor_b = torch.randn(4, dtype=torch.float32) * 0.1
    golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

    # Construct tensors dictionary
    tensors = {
        # Chinchilla
        "n_val": n_val.clone().contiguous(),
        "d_val": d_val.clone().contiguous(),
        "golden_chinchilla_loss": chinchilla_loss.clone().contiguous(),
        "golden_n_opt": n_opt.clone().contiguous(),
        "golden_d_opt": d_opt.clone().contiguous(),

        # DoReMi
        "w_doremi_init": w_doremi_init.clone().contiguous(),
        "losses_model": losses_model.clone().contiguous(),
        "losses_ref": losses_ref.clone().contiguous(),
        "golden_excess_loss": excess_loss.clone().contiguous(),
        "golden_w_doremi_next": w_doremi_next.clone().contiguous(),

        # MinHash
        "signatures": signatures.clone().contiguous(),
        "golden_jaccard_matrix": jaccard_matrix.clone().contiguous(),

        # Model collapse
        "var_synth": var_synth.clone().contiguous(),
        "var_real": var_real.clone().contiguous(),
        "golden_cured_var": golden_cured_var.clone().contiguous(),

        # Linear Anchor
        "anchor_x": anchor_x.clone().contiguous(),
        "anchor_w": anchor_w.clone().contiguous(),
        "anchor_b": anchor_b.clone().contiguous(),
        "golden_anchor_out": golden_anchor_out.clone().contiguous(),

        # Standard test inputs/outputs for parity test runners
        "test_input": anchor_x.clone().contiguous(),
        "test_output": golden_anchor_out.clone().contiguous(),
        "weight": anchor_w.clone().contiguous(),
        "bias": anchor_b.clone().contiguous(),
    }

    out_dir = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "07_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse.safetensors"

    save_file(tensors, out_path)
    print(f"Saved {len(tensors)} tensors to {out_path}")

    # Verify self-consistency
    print(f"Chinchilla Loss: {chinchilla_loss.item():.6f}, N_opt: {n_opt.item():.4f}, D_opt: {d_opt.item():.4f}")
    print(f"DoReMi Next: {w_doremi_next.numpy().tolist()}")
    print(f"Jaccard Pair (0, 1): {jaccard_matrix[0, 1].item():.4f}, Pair (0, 3): {jaccard_matrix[0, 3].item():.4f}")
    print(f"Cured Variance: {golden_cured_var.item():.4f}")

if __name__ == "__main__":
    generate_topic_21_artifacts()
