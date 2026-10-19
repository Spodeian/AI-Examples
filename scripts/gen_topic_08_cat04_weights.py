"""Generate reference weights and golden outputs for Category 04, Topic 08.

Topic 08: Constrained, Proximal, Frank-Wolfe, ADMM, and Mirror Descent
(04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/01_constrained_proximal_frank_wolfe_admm_and_mirror_descent)
Enforces transitive numerical parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

from pathlib import Path
import torch
from safetensors.torch import save_file


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")
    d = 8

    # State parameters
    # 1. PGD: L2 ball projection (radius R = 1.0)
    param_pgd = torch.randn(d, dtype=torch.float32)
    grad_pgd = torch.randn(d, dtype=torch.float32) * 2.0
    lr_pgd = 0.1
    R_pgd = 1.0
    v_pgd = param_pgd - lr_pgd * grad_pgd
    norm_v_pgd = torch.norm(v_pgd)
    golden_param_pgd = v_pgd if norm_v_pgd <= R_pgd else v_pgd * (R_pgd / norm_v_pgd)

    # 2. Frank-Wolfe: L1 ball (radius tau = 1.0)
    # Linear subproblem: s = argmin_{||s||_1 <= tau} <s, grad>
    # -> s = -tau * sign(grad_i) * e_i where i = argmax |grad_i|
    param_fw = torch.randn(d, dtype=torch.float32)
    param_fw = param_fw / torch.norm(param_fw, p=1) * 0.5  # start feasible inside L1 ball
    grad_fw = torch.randn(d, dtype=torch.float32)
    tau_fw = 1.0
    gamma_fw = 0.2  # step size in [0, 1]
    abs_g = torch.abs(grad_fw)
    idx_max = torch.argmax(abs_g)
    s_fw = torch.zeros(d, dtype=torch.float32)
    s_fw[idx_max] = -tau_fw * torch.sign(grad_fw[idx_max])
    golden_param_fw = (1.0 - gamma_fw) * param_fw + gamma_fw * s_fw

    # 3. Proximal ISTA: Soft-Thresholding for L1 regularized loss f(x) + lambda * ||x||_1
    param_ista = torch.randn(d, dtype=torch.float32)
    grad_ista = torch.randn(d, dtype=torch.float32)
    lr_ista = 0.05
    lambda_ista = 0.1
    thresh = lr_ista * lambda_ista
    v_ista = param_ista - lr_ista * grad_ista
    golden_param_ista = torch.sign(v_ista) * torch.clamp(torch.abs(v_ista) - thresh, min=0.0)

    # 4. Entropic Mirror Descent on Simplex Delta^d
    # p_{t+1} = p_t * exp(-lr * grad) / sum(p_t * exp(-lr * grad))
    param_md = torch.softmax(torch.randn(d, dtype=torch.float32), dim=0)
    grad_md = torch.randn(d, dtype=torch.float32)
    lr_md = 0.1
    unnorm_md = param_md * torch.exp(-lr_md * grad_md)
    golden_param_md = unnorm_md / torch.sum(unnorm_md)

    # 5. Linear Anchor
    anchor_x = torch.randn(2, d, dtype=torch.float32)
    anchor_w = torch.randn(d, 4, dtype=torch.float32)
    anchor_b = torch.randn(4, dtype=torch.float32)
    golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

    tensors = {
        # PGD
        "param_pgd": param_pgd,
        "grad_pgd": grad_pgd,
        "golden_param_pgd": golden_param_pgd,
        # Frank-Wolfe
        "param_fw": param_fw,
        "grad_fw": grad_fw,
        "golden_param_fw": golden_param_fw,
        # ISTA
        "param_ista": param_ista,
        "grad_ista": grad_ista,
        "golden_param_ista": golden_param_ista,
        # Mirror Descent
        "param_md": param_md,
        "grad_md": grad_md,
        "golden_param_md": golden_param_md,
        # Linear Anchor
        "anchor_x": anchor_x,
        "anchor_w": anchor_w,
        "anchor_b": anchor_b,
        "golden_anchor_out": golden_anchor_out,
        # Standard Parity keys
        "weight": anchor_w.clone(),
        "bias": anchor_b.clone(),
        "test_input": anchor_x.clone(),
        "test_output": golden_anchor_out.clone(),
    }

    out_dir = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.safetensors"
    save_file(tensors, str(out_file))
    print(f"Saved {len(tensors)} tensors to {out_file}")


if __name__ == "__main__":
    main()
