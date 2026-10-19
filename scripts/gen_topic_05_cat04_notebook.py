"""Generate the 8-cell educational notebook for Category 04 Topic 05: Second-Order and Curvature Optimisers."""

import json
from pathlib import Path


def build_notebook() -> dict:
    cells = [
        # Cell 0: Header Markdown
        {
            "id": "cell_0",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Second-Order and Curvature Optimisers (C04 Topic 05)\n",
                "\n",
                "**Citations:** Levenberg (1944); Marquardt (1963); Nocedal (1980 L-BFGS); Steihaug (1983); Amari (1998); Martens & Grosse (2015 K-FAC); Yao et al. (2021 AdaHessian); Liu et al. (2023 Sophia).\n",
                "\n",
                "### Purpose & Topic Bridges\n",
                "First-order optimisers navigate using only local tangent planes (the gradient $g = \\nabla f(\\theta)$). In ill-conditioned valleys or non-convex saddle terrains, first-order methods oscillate wildly across steep ravines while creeping slowly along flat directions. Second-order methods exploit curvature $H = \\nabla^2 f(\\theta)$ to perform preconditioned Newton steps $\\Delta \\theta = - H^{-1} g$.\n",
                "\n",
                "**L-BFGS** stores a sliding history of $m$ displacement and gradient pairs to approximate $H^{-1} g$ in $\\mathcal{O}(m \\cdot d)$ operations via two-loop recursion. **Sophia-G** approximates diagonal Hessian curvature using Gauss-Newton-Bartlett estimators paired with element-wise clipping to bound worst-case step sizes. **AdaHessian** uses an EMA of spatial Hessian diagonal variance, and **Levenberg-Marquardt** solves non-linear least squares via dynamically damped Gauss-Newton $(J^T J + \\lambda I)^{-1} (-J^T R)$.\n",
                "\n",
                "**Decision Boundary:** Use **L-BFGS** for small-to-medium deterministic models (neural ODEs, scientific physics-informed ML, style transfer). Use **Sophia-G** for large-scale language model pretraining where gradient noise is high and negative curvature causes AdamW to diverge. Use **Levenberg-Marquardt** for non-linear bundle adjustment and inverse problems. When NOT to use: Do NOT run mini-batch stochastic L-BFGS with small noisy batches ($B < 512$), as noisy gradients violate the secant curvature condition $s_k^T y_k > 0$."
            ]
        },
        # Cell 1: Theory Markdown
        {
            "cell_type": "markdown",
            "id": "cell_1",
            "metadata": {},
            "source": [
                "## 2. Mathematical Foundations & Governing Equations\n",
                "\n",
                "1. **L-BFGS Two-Loop Recursion:** Given displacement history $s_i = \\theta_{i+1} - \\theta_i$ and gradient history $y_i = g_{i+1} - g_i$, compute $\\rho_i = 1 / (y_i^T s_i)$. Backward loop computes $\\alpha_i = \\rho_i s_i^T q$ and $q \\leftarrow q - \\alpha_i y_i$. Initial scaling $r = \\frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}} q$. Forward loop computes $\\beta_i = \\rho_i y_i^T r$ and $r \\leftarrow r + s_i (\\alpha_i - \\beta_i)$. Search direction $d = -r$.\n",
                "2. **Sophia-G Clipped Preconditioning:** Moving averages $m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$ and $h_t = \\beta_2 h_{t-1} + (1-\\beta_2) \\hat{h}_t$. Update step $\\Delta \\theta_t = - \\eta \\cdot \\text{clamp}\\left(\\frac{m_t}{\\max(h_t, \\epsilon)}, -\\rho, \\rho\\right)$. Parameter update $\\theta_{t+1} = \\theta_t (1 - \\eta \\lambda) + \\Delta \\theta_t$.\n",
                "3. **AdaHessian Adaptive Curvature:** Second moment $v_t = \\beta_2 v_{t-1} + (1-\\beta_2) D_t^2$ with diagonal Hessian $D_t$. Bias-corrected update $\\Delta \\theta_t = - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$.\n",
                "4. **Levenberg-Marquardt (LM):** Damped Gauss-Newton normal equations $(J^T J + \\lambda I) \\Delta \\theta = - J^T R$ solved via Cholesky decomposition $L L^T = (J^T J + \\lambda I)$.\n",
                "5. **Complexity:** L-BFGS requires $\\mathcal{O}(m \\cdot d)$ FLOPs and $\\mathcal{O}(m \\cdot d)$ memory. Sophia-G requires $\\mathcal{O}(d)$ FLOPs and $\\mathcal{O}(d)$ memory. Levenberg-Marquardt requires $\\mathcal{O}(n \\cdot d^2 + d^3)$ FLOPs."
            ]
        },
        # Cell 2: Imports & Environment Setup
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_2",
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys, math\n",
                "from pathlib import Path\n",
                "import torch, torch.nn as nn\n",
                "\n",
                "repo_root = Path.cwd().resolve()\n",
                "while not (repo_root / \"core\").exists() and repo_root.parent != repo_root:\n",
                "    repo_root = repo_root.parent\n",
                "if str(repo_root) not in sys.path:\n",
                "    sys.path.insert(0, str(repo_root))\n",
                "import core\n",
                "core.setup_env(seed=42)\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "\n",
                "ckpt_path = repo_root / \"weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.safetensors\"\n",
                "weights, test_in, test_out, meta = core.parity.load_reference_checkpoint(ckpt_path, device=device)\n",
                "print(f\"Device: {device} | Checkpoint: {ckpt_path.name} | Loaded {len(weights)} tensors\")"
            ]
        },
        # Cell 3: Unified Module Definition
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_3",
            "metadata": {},
            "outputs": [],
            "source": [
                "def lbfgs_two_loop(s_hist, y_hist, grad):\n",
                "    m = s_hist.shape[0]\n",
                "    q, alphas, rhos = grad.clone(), torch.zeros(m), torch.zeros(m)\n",
                "    for i in reversed(range(m)):\n",
                "        rhos[i] = 1.0 / torch.dot(y_hist[i], s_hist[i])\n",
                "        alphas[i] = rhos[i] * torch.dot(s_hist[i], q)\n",
                "        q = q - alphas[i] * y_hist[i]\n",
                "    r = (torch.dot(s_hist[-1], y_hist[-1]) / torch.dot(y_hist[-1], y_hist[-1])) * q\n",
                "    for i in range(m):\n",
                "        r = r + s_hist[i] * (alphas[i] - rhos[i] * torch.dot(y_hist[i], r))\n",
                "    return r\n",
                "\n",
                "def step_lbfgs(theta, s_hist, y_hist, grad, lr=0.1):\n",
                "    return theta - lr * lbfgs_two_loop(s_hist, y_hist, grad)\n",
                "\n",
                "def step_sophia(theta, grad, m, h, h_est, lr=0.01, beta1=0.96, beta2=0.99, wd=0.01, rho=0.04, eps=1e-6):\n",
                "    m_next = beta1 * m + (1.0 - beta1) * grad\n",
                "    h_next = beta2 * h + (1.0 - beta2) * h_est\n",
                "    th_next = theta * (1.0 - lr * wd) - lr * torch.clamp(m_next / torch.clamp(h_next, min=eps), -rho, rho)\n",
                "    return th_next, m_next, h_next\n",
                "\n",
                "def step_adahessian(theta, grad, m, v, d, lr=0.01, beta1=0.9, beta2=0.999, wd=0.01, step=1, eps=1e-4):\n",
                "    m_next = beta1 * m + (1.0 - beta1) * grad\n",
                "    v_next = beta2 * v + (1.0 - beta2) * (d ** 2)\n",
                "    m_hat, v_hat = m_next / float(1.0 - beta1 ** step), v_next / float(1.0 - beta2 ** step)\n",
                "    th_next = theta * (1.0 - lr * wd) - lr * (m_hat / (torch.sqrt(v_hat) + eps))\n",
                "    return th_next, m_next, v_next\n",
                "\n",
                "def step_lm(theta, jacobian, residual, damping=0.1):\n",
                "    jt = jacobian.T\n",
                "    a_mat = torch.matmul(jt, jacobian) + damping * torch.eye(jacobian.shape[1], device=theta.device)\n",
                "    return theta + torch.linalg.solve(a_mat, -torch.matmul(jt, residual))\n",
                "\n",
                "class LinearAnchor(nn.Module):\n",
                "    def __init__(self, in_f, out_f):\n",
                "        super().__init__()\n",
                "        self.weight = nn.Parameter(torch.zeros(out_f, in_f))\n",
                "        self.bias = nn.Parameter(torch.zeros(out_f))\n",
                "    def forward(self, x):\n",
                "        return torch.matmul(x, self.weight.T) + self.bias\n",
                "\n",
                "print('SecondOrderCurvatureUnifiedModule functions defined successfully.')"
            ]
        },
        # Cell 4: Live Verification & Invariant Checks
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_4",
            "metadata": {},
            "outputs": [],
            "source": [
                "p_lbfgs = step_lbfgs(weights['theta'], weights['lbfgs_s'], weights['lbfgs_y'], weights['grad'], lr=0.1)\n",
                "err_lbfgs = float(torch.max(torch.abs(p_lbfgs - weights['golden_lbfgs_theta'])))\n",
                "p_sophia, _, _ = step_sophia(weights['theta'], weights['grad'], weights['momentum'], weights['hess_diag'], weights['hess_diag_est'])\n",
                "err_sophia = float(torch.max(torch.abs(p_sophia - weights['golden_sophia_theta'])))\n",
                "p_ada, _, _ = step_adahessian(weights['theta'], weights['grad'], weights['momentum'], weights['ada_v'], weights['ada_d'])\n",
                "err_ada = float(torch.max(torch.abs(p_ada - weights['golden_adahessian_theta'])))\n",
                "p_lm = step_lm(weights['theta'], weights['lm_jacobian'], weights['lm_residual'], damping=0.1)\n",
                "err_lm = float(torch.max(torch.abs(p_lm - weights['golden_lm_theta'])))\n",
                "anchor = LinearAnchor(8, 4)\n",
                "anchor.weight.data.copy_(weights['anchor_w'].T)\n",
                "anchor.bias.data.copy_(weights['anchor_b'])\n",
                "anchor_out = anchor(weights['anchor_x']).detach()\n",
                "err_anchor = float(torch.max(torch.abs(anchor_out - weights['golden_anchor_out'])))\n",
                "for name, err in [('L-BFGS', err_lbfgs), ('Sophia', err_sophia), ('AdaHessian', err_ada), ('LM', err_lm), ('Anchor', err_anchor)]:\n",
                "    assert err < 1e-5, f'{name} invariant failed: {err:.2e}'\n",
                "print(f'All 5 invariants verified! L-BFGS: {err_lbfgs:.2e}, Sophia: {err_sophia:.2e}, AdaHessian: {err_ada:.2e}, LM: {err_lm:.2e}, Anchor: {err_anchor:.2e}')"
            ]
        },
        # Cell 5: Performance Profile & Roofline Benchmark
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_5",
            "metadata": {},
            "outputs": [],
            "source": [
                "track_fns = {\n",
                "    'Linear Anchor': lambda: anchor(weights['anchor_x']),\n",
                "    'L-BFGS Two-Loop': lambda: step_lbfgs(weights['theta'], weights['lbfgs_s'], weights['lbfgs_y'], weights['grad']),\n",
                "    'Sophia-G Clipped': lambda: step_sophia(weights['theta'], weights['grad'], weights['momentum'], weights['hess_diag'], weights['hess_diag_est']),\n",
                "    'AdaHessian Step': lambda: step_adahessian(weights['theta'], weights['grad'], weights['momentum'], weights['ada_v'], weights['ada_d']),\n",
                "    'Levenberg-Marquardt': lambda: step_lm(weights['theta'], weights['lm_jacobian'], weights['lm_residual']),\n",
                "}\n",
                "bench_results = core.benchmark.compare_tracks(track_fns, warmup=3, iters=15, batch_size=8)\n",
                "print(core.benchmark.format_benchmark_table(bench_results))"
            ]
        },
        # Cell 6: Active Failure Mode Demonstration
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_6",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Active Failure Mode: Unclipped Newton Step Jumping to a Saddle-Point Maximum\n",
                "# On non-convex objective f(x, y) = x^2 - y^2 with saddle point at (0, 0):\n",
                "# Hessian H = diag(2, -2) has a negative eigenvalue lambda_2 = -2.0 along y.\n",
                "pos = torch.tensor([1.0, 0.5], device=device)\n",
                "grad = torch.tensor([2.0 * pos[0].item(), -2.0 * pos[1].item()], device=device)  # [2.0, -1.0]\n",
                "hess = torch.tensor([[2.0, 0.0], [0.0, -2.0]], device=device)\n",
                "\n",
                "# 1. Pure Newton Step: - H^(-1) g\n",
                "newton_step = -torch.linalg.solve(hess, grad)  # [-1.0, -0.5]\n",
                "new_pos_newton = pos + newton_step             # [0.0, 0.0]\n",
                "# Along the negative curvature axis y, pure Newton jumps to the local MAXIMUM y=0:\n",
                "val_y_init = -pos[1].item()**2                # -0.25\n",
                "val_y_newton = -new_pos_newton[1].item()**2    # 0.0 (INCREASED by +0.25 towards maximum!)\n",
                "\n",
                "# 2. Levenberg-Marquardt Damped Trust-Region: (H + lambda * I) with lambda = 3.0 > |lambda_min|\n",
                "lambda_damp = 3.0\n",
                "damped_hess = hess + lambda_damp * torch.eye(2, device=device)  # diag(5.0, 1.0) -> Positive Definite!\n",
                "lm_step = -torch.linalg.solve(damped_hess, grad)                # [-0.4, +1.0]\n",
                "new_pos_lm = pos + lm_step                                     # [0.6, 1.5]\n",
                "val_y_lm = -new_pos_lm[1].item()**2                            # -2.25 (true DESCENT!)\n",
                "\n",
                "# 3. Sophia-G: Non-negative diagonal curvature max(h, eps) + elementwise clipping [-rho, rho]\n",
                "sophia_h = torch.abs(torch.diag(hess))                           # [2.0, 2.0]\n",
                "sophia_step = -torch.clamp(grad / torch.clamp(sophia_h, min=1e-6), -0.5, 0.5)\n",
                "\n",
                "print(f'[FAILURE DEMONSTRATED] Pure Newton jumped to local maximum along negative curvature: {val_y_init:.2f} -> {val_y_newton:.2f} (uphill!)')\n",
                "print(f'[CORRECT] Levenberg-Marquardt damping achieved true descent: {val_y_init:.2f} -> {val_y_lm:.2f}')\n",
                "print(f'[STABILIZED] Sophia bounded step: {sophia_step.tolist()}')\n",
                "assert val_y_newton > val_y_init, 'Pure Newton step failed to increase objective along negative curvature'\n",
                "assert val_y_lm < val_y_init, 'Levenberg-Marquardt failed to achieve descent'\n",
                "print('[DIAGNOSTIC] Never use unclipped Newton steps in non-convex landscapes; enforce LM damping or Sophia clipping.')"
            ]
        },
        # Cell 7: Summary Markdown
        {
            "cell_type": "markdown",
            "id": "cell_7",
            "metadata": {},
            "source": [
                "## 8. Algorithmic Takeaways & Cross-Domain Isomorphisms\n",
                "\n",
                "1. **Curvature Acceleration vs Saddle Traps:** Second-order methods eliminate transverse oscillations in ill-conditioned valleys, but unclipped Newton steps diverge along negative curvature directions.\n",
                "2. **Sophia-G Bridges Deep Learning:** By combining non-negative diagonal Hessian estimation with element-wise clipping, Sophia-G achieves second-order preconditioning at the exact $O(d)$ memory footprint of AdamW.\n",
                "3. **Cross-Domain Isomorphism:** Levenberg-Marquardt damping mirrors Tikhonov regularization in geophysics, bundle adjustment in robotics SLAM, and Levenberg-Marquardt fitting in biomedical spectroscopy.\n",
                "\n",
                "---\n",
                "**Curriculum Navigation:** [Master Index](file:///c:/Users/Liam/Code/AI-Examples/README.md) | [Previous: 04_orthogonal_and_preconditioned_matrix_optimisers](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.ipynb) | [Next: 06_schedule_free_and_parameter_free_optimisers](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.ipynb)"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb


def main():
    nb = build_notebook()
    out_dir = Path("notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "05_second_order_and_curvature_optimisers.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {out_path} with {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
