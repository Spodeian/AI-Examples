"""Generate golden weights and reference I/O tensors for Topic 05:
05_parametric_vs_nonparametric_and_neural_processes.

Pillars:
1. Parametric Model: Fixed-weight MLP with O(1) memory capacity.
2. Non-Parametric Model: Exact Gaussian Process with RBF Kernel and O(C^3) Cholesky solve.
3. Semi-Parametric Attentive Neural Process (ANP): O(C * T) cross-attention representation.
4. Semi-Parametric Conditional Neural Process (CNP): Permutation-invariant uniform mean pooling.
5. Canonical Linear Anchor for cross-track parity harness.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

import core


def generate_topic_05_cat01_weights(output_path: Path) -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    num_context = 4
    num_target = 4
    dim_x = 2
    dim_y = 1
    dim_h = 4

    # 1. Input batches: context set (x_c, y_c) and target query points x_t
    x_context = torch.randn(batch_size, num_context, dim_x, generator=gen, dtype=torch.float32)
    y_context = torch.randn(batch_size, num_context, dim_y, generator=gen, dtype=torch.float32)
    x_target = torch.randn(batch_size, num_target, dim_x, generator=gen, dtype=torch.float32)

    # 2. Canonical linear reference anchor (using x_target[:, 0, :])
    test_input = x_target[:, 0, :]  # [B, 2]
    weight = torch.randn(dim_x, dim_y, generator=gen, dtype=torch.float32)
    bias = torch.randn(dim_y, generator=gen, dtype=torch.float32)
    test_output = torch.matmul(test_input, weight) + bias

    # 3. Parametric Model: 2-layer MLP on target coordinates x_t
    w_mlp1 = torch.randn(dim_x, dim_h, generator=gen, dtype=torch.float32)
    b_mlp1 = torch.randn(dim_h, generator=gen, dtype=torch.float32)
    w_mlp2 = torch.randn(dim_h, dim_y, generator=gen, dtype=torch.float32)
    b_mlp2 = torch.randn(dim_y, generator=gen, dtype=torch.float32)

    h_mlp = F.relu(torch.matmul(x_target, w_mlp1) + b_mlp1)
    y_parametric = torch.matmul(h_mlp, w_mlp2) + b_mlp2  # [B, N_t, 1]

    # 4. Exact Non-Parametric GP Model (RBF Kernel with lengthscale=1.0, noise_var=0.1)
    gp_lengthscale = torch.tensor([1.0], dtype=torch.float32)
    gp_noise_var = torch.tensor([0.1], dtype=torch.float32)

    # Compute pairwise squared Euclidean distances:
    # dist_sq_cc: [B, N_c, N_c]
    x_c_norm = torch.sum(x_context**2, dim=-1, keepdim=True)  # [B, N_c, 1]
    dist_cc = x_c_norm + x_c_norm.transpose(-2, -1) - 2.0 * torch.matmul(x_context, x_context.transpose(-2, -1))
    dist_cc = torch.clamp(dist_cc, min=0.0)
    k_cc = torch.exp(-0.5 * dist_cc / (gp_lengthscale**2))  # [B, N_c, N_c]

    # Regularized kernel matrix: K_cc + sigma^2 * I
    eye_c = torch.eye(num_context, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1)
    k_cc_reg = k_cc + gp_noise_var * eye_c  # [B, N_c, N_c]

    # Solve dual weights: alpha = (K_cc_reg)^{-1} Y_c
    alpha = torch.linalg.solve(k_cc_reg, y_context)  # [B, N_c, 1]

    # Cross kernel K_tc between target x_t and context x_c: [B, N_t, N_c]
    x_t_norm = torch.sum(x_target**2, dim=-1, keepdim=True)  # [B, N_t, 1]
    dist_tc = x_t_norm + x_c_norm.transpose(-2, -1) - 2.0 * torch.matmul(x_target, x_context.transpose(-2, -1))
    dist_tc = torch.clamp(dist_tc, min=0.0)
    k_tc = torch.exp(-0.5 * dist_tc / (gp_lengthscale**2))  # [B, N_t, N_c]

    y_gp = torch.matmul(k_tc, alpha)  # [B, N_t, 1]

    # 5. Semi-Parametric Attentive Neural Process (ANP)
    # Context encoder: [x_c, y_c] -> dim_x + dim_y = 3 -> dim_h = 4
    w_enc = torch.randn(dim_x + dim_y, dim_h, generator=gen, dtype=torch.float32)
    b_enc = torch.randn(dim_h, generator=gen, dtype=torch.float32)

    xy_c = torch.cat([x_context, y_context], dim=-1)  # [B, N_c, 3]
    v_c = F.relu(torch.matmul(xy_c, w_enc) + b_enc)  # [B, N_c, 4]

    # Cross-Attention: Q from x_target, K from x_context, V from v_c
    w_q = torch.randn(dim_x, dim_h, generator=gen, dtype=torch.float32)
    w_k = torch.randn(dim_x, dim_h, generator=gen, dtype=torch.float32)
    w_v = torch.randn(dim_h, dim_h, generator=gen, dtype=torch.float32)

    q = torch.matmul(x_target, w_q)  # [B, N_t, dim_h]
    k = torch.matmul(x_context, w_k)  # [B, N_c, dim_h]
    v = torch.matmul(v_c, w_v)  # [B, N_c, dim_h]

    scale = 1.0 / (dim_h**0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, N_t, N_c]
    attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N_t, N_c]
    r_anp = torch.matmul(attn_weights, v)  # [B, N_t, dim_h]

    # Decoder: [x_t, r_anp] -> dim_x + dim_h = 6 -> dim_y = 1
    w_dec = torch.randn(dim_x + dim_h, dim_y, generator=gen, dtype=torch.float32)
    b_dec = torch.randn(dim_y, generator=gen, dtype=torch.float32)

    xt_r_anp = torch.cat([x_target, r_anp], dim=-1)  # [B, N_t, 6]
    y_anp = torch.matmul(xt_r_anp, w_dec) + b_dec  # [B, N_t, 1]

    # 6. Vanilla CNP (Uniform Mean Pooling of v_c)
    r_cnp_pooled = torch.mean(v_c, dim=1, keepdim=True)  # [B, 1, dim_h]
    r_cnp = r_cnp_pooled.expand(-1, num_target, -1)  # [B, N_t, dim_h]
    xt_r_cnp = torch.cat([x_target, r_cnp], dim=-1)  # [B, N_t, 6]
    y_cnp = torch.matmul(xt_r_cnp, w_dec) + b_dec  # [B, N_t, 1]

    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "x_context": x_context.detach(),
        "y_context": y_context.detach(),
        "x_target": x_target.detach(),
        "w_mlp1": w_mlp1.detach(),
        "b_mlp1": b_mlp1.detach(),
        "w_mlp2": w_mlp2.detach(),
        "b_mlp2": b_mlp2.detach(),
        "y_parametric": y_parametric.detach(),
        "gp_lengthscale": gp_lengthscale.detach(),
        "gp_noise_var": gp_noise_var.detach(),
        "alpha": alpha.detach(),
        "y_gp": y_gp.detach(),
        "w_enc": w_enc.detach(),
        "b_enc": b_enc.detach(),
        "w_q": w_q.detach(),
        "w_k": w_k.detach(),
        "w_v": w_v.detach(),
        "w_dec": w_dec.detach(),
        "b_dec": b_dec.detach(),
        "attn_weights": attn_weights.detach(),
        "y_anp": y_anp.detach(),
        "y_cnp": y_cnp.detach(),
    }

    core.parity.save_reference_checkpoint(
        path=output_path,
        weights_dict=weights_dict,
        input_tensor=test_input,
        output_tensor=test_output,
        metadata={
            "topic": "05_parametric_vs_nonparametric_and_neural_processes",
            "category": "01_paradigms_and_dichotomies",
            "batch_size": str(batch_size),
            "num_context": str(num_context),
            "num_target": str(num_target),
            "dim_x": str(dim_x),
            "dim_y": str(dim_y),
            "dim_h": str(dim_h),
        },
    )

    print(f"Generated Topic 05 Category 01 weights at: {output_path}")
    print(f"x_context shape: {x_context.shape}")
    print(f"x_target shape: {x_target.shape}")
    print(f"y_parametric shape: {y_parametric.shape}")
    print(f"y_gp shape: {y_gp.shape}")
    print(f"y_anp shape: {y_anp.shape}")
    print(f"y_cnp shape: {y_cnp.shape}")


if __name__ == "__main__":
    out_file = (
        Path(__file__).resolve().parent.parent
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "05_parametric_vs_nonparametric_and_neural_processes.safetensors"
    )
    generate_topic_05_cat01_weights(out_file)
