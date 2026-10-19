"""Generate reference weights and golden outputs for Category 04, Topic 05.

Topic 05: Second-Order and Curvature Optimisers (L-BFGS, Sophia, AdaHessian, Levenberg-Marquardt)
Enforces transitive numerical parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

from pathlib import Path
import torch
from safetensors.torch import save_file

def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    d = 8
    m_hist = 4
    n_res = 8

    # Parameters and states
    theta = torch.randn(d, dtype=torch.float32)
    grad = torch.randn(d, dtype=torch.float32)
    momentum = torch.randn(d, dtype=torch.float32) * 0.1
    hess_diag = torch.abs(torch.randn(d, dtype=torch.float32)) + 0.5
    hess_diag_est = torch.abs(torch.randn(d, dtype=torch.float32)) + 0.5

    ada_v = torch.abs(torch.randn(d, dtype=torch.float32)) * 0.1 + 0.01
    ada_d = torch.abs(torch.randn(d, dtype=torch.float32)) + 0.2

    # L-BFGS history: ensure s_i^T y_i > 0 for curvature condition
    lbfgs_s = torch.randn(m_hist, d, dtype=torch.float32) * 0.5
    lbfgs_y = torch.randn(m_hist, d, dtype=torch.float32) * 0.5
    for i in range(m_hist):
        dot = torch.dot(lbfgs_s[i], lbfgs_y[i])
        if dot <= 0.1:
            lbfgs_y[i] = lbfgs_y[i] + lbfgs_s[i] * (0.5 - dot.item() / (torch.norm(lbfgs_s[i])**2 + 1e-6))

    # Levenberg-Marquardt Jacobian and Residual
    lm_jacobian = torch.randn(n_res, d, dtype=torch.float32)
    lm_residual = torch.randn(n_res, dtype=torch.float32)
    damping = 0.1

    # Linear Anchor
    anchor_x = torch.randn(2, d, dtype=torch.float32)
    anchor_w = torch.randn(d, 4, dtype=torch.float32)
    anchor_b = torch.randn(4, dtype=torch.float32)
    golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

    # 1. L-BFGS two-loop golden computation
    def lbfgs_two_loop(s_hist, y_hist, g, lr=0.1):
        m, d_dim = s_hist.shape
        q = g.clone()
        alphas = torch.zeros(m, dtype=torch.float32)
        rhos = torch.zeros(m, dtype=torch.float32)

        for i in reversed(range(m)):
            rho_i = 1.0 / torch.dot(y_hist[i], s_hist[i])
            rhos[i] = rho_i
            alpha_i = rho_i * torch.dot(s_hist[i], q)
            alphas[i] = alpha_i
            q = q - alpha_i * y_hist[i]

        gamma = torch.dot(s_hist[-1], y_hist[-1]) / torch.dot(y_hist[-1], y_hist[-1])
        r = gamma * q

        for i in range(m):
            beta = rhos[i] * torch.dot(y_hist[i], r)
            r = r + s_hist[i] * (alphas[i] - beta)

        direction = r
        return direction

    lbfgs_dir = lbfgs_two_loop(lbfgs_s, lbfgs_y, grad, lr=0.1)
    golden_lbfgs_theta = theta - 0.1 * lbfgs_dir

    # 2. Sophia-G golden computation
    sophia_lr = 0.01
    sophia_beta1 = 0.96
    sophia_beta2 = 0.99
    sophia_wd = 0.01
    sophia_rho = 0.04
    sophia_eps = 1e-6

    golden_sophia_m = sophia_beta1 * momentum + (1.0 - sophia_beta1) * grad
    golden_sophia_h = sophia_beta2 * hess_diag + (1.0 - sophia_beta2) * hess_diag_est
    ratio = golden_sophia_m / torch.clamp(golden_sophia_h, min=sophia_eps)
    clipped_ratio = torch.clamp(ratio, -sophia_rho, sophia_rho)
    golden_sophia_theta = theta * (1.0 - sophia_lr * sophia_wd) - sophia_lr * clipped_ratio

    # 3. AdaHessian golden computation
    ada_lr = 0.01
    ada_beta1 = 0.9
    ada_beta2 = 0.999
    ada_wd = 0.01
    step = 1
    ada_eps = 1e-4

    golden_adahessian_m = ada_beta1 * momentum + (1.0 - ada_beta1) * grad
    golden_adahessian_v = ada_beta2 * ada_v + (1.0 - ada_beta2) * (ada_d ** 2)

    # Bias correction (computed in double for numerical precision)
    bias_m = float(1.0 - ada_beta1 ** step)
    bias_v = float(1.0 - ada_beta2 ** step)
    m_hat = golden_adahessian_m / bias_m
    v_hat = golden_adahessian_v / bias_v

    ada_update = m_hat / (torch.sqrt(v_hat) + ada_eps)
    golden_adahessian_theta = theta * (1.0 - ada_lr * ada_wd) - ada_lr * ada_update

    # 4. Levenberg-Marquardt golden computation
    # (J^T J + lambda I) delta = - J^T R
    jt = lm_jacobian.T
    jt_j = torch.matmul(jt, lm_jacobian)
    a_mat = jt_j + damping * torch.eye(d, dtype=torch.float32)
    b_vec = -torch.matmul(jt, lm_residual)
    delta_lm = torch.linalg.solve(a_mat, b_vec)
    golden_lm_theta = theta + delta_lm

    tensors = {
        "theta": theta,
        "grad": grad,
        "momentum": momentum,
        "hess_diag": hess_diag,
        "hess_diag_est": hess_diag_est,
        "ada_v": ada_v,
        "ada_d": ada_d,
        "lbfgs_s": lbfgs_s,
        "lbfgs_y": lbfgs_y,
        "lm_jacobian": lm_jacobian,
        "lm_residual": lm_residual,
        "anchor_x": anchor_x,
        "anchor_w": anchor_w,
        "anchor_b": anchor_b,
        "golden_anchor_out": golden_anchor_out,
        "weight": anchor_w.clone(),
        "bias": anchor_b.clone(),
        "test_input": anchor_x.clone(),
        "test_output": golden_anchor_out.clone(),
        "golden_lbfgs_theta": golden_lbfgs_theta,
        "golden_sophia_theta": golden_sophia_theta,
        "golden_sophia_m": golden_sophia_m,
        "golden_sophia_h": golden_sophia_h,
        "golden_adahessian_theta": golden_adahessian_theta,
        "golden_adahessian_m": golden_adahessian_m,
        "golden_adahessian_v": golden_adahessian_v,
        "golden_lm_theta": golden_lm_theta,
    }

    out_dir = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "05_second_order_and_curvature_optimisers.safetensors"
    save_file(tensors, str(out_file))
    print(f"Saved {len(tensors)} tensors to {out_file}")

if __name__ == "__main__":
    main()
