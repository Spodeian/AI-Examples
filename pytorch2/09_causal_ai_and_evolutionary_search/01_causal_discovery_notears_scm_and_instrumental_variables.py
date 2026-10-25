"""Causal Discovery NOTEARS SCM and Instrumental Variables — PyTorch 2.x reference implementation.

Topic: 09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables
Golden weights: weights/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.safetensors

``python pytorch2/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.py`` verifies against the golden checkpoint;
``python pytorch2/09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def notears_acyclicity_constraint(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Zheng et al. (2018) NOTEARS continuous characterization of acyclicity.
    
    h(W) = Tr( exp(W \odot W) ) - d
    grad_h(W) = 2.0 * (exp(W \odot W))^T \odot W
    Returns (h(W), grad_h(W)).
    """
    d = w.shape[0]
    m = w * w
    exp_m = torch.matrix_exp(m)
    h = (torch.trace(exp_m) - float(d)).unsqueeze(0)
    grad_h = 2.0 * (exp_m.t() * w)
    return h, grad_h


def scm_generate_and_counterfactual(
    scm_w: torch.Tensor,
    scm_u: torch.Tensor,
    do_var_idx: int = 1,
    do_val: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pearl's Structural Causal Model forward computation and 3-step counterfactual.
    
    1. Factual forward: x = u @ (I - W)^{-1}
    2. Abduction: Recover exogenous noise u = x @ (I - W)
    3. Action: Sever incoming edges to intervened variable (col = 0) and set u[do_var_idx] = do_val
    4. Prediction: x_cf = u_intervened @ (I - W_intervened)^{-1}
    """
    d = scm_w.shape[0]
    eye = torch.eye(d, dtype=scm_w.dtype, device=scm_w.device)
    inv_w = torch.inverse(eye - scm_w)
    x_factual = scm_u @ inv_w

    # Intervene
    w_intervened = scm_w.clone()
    w_intervened[:, do_var_idx] = 0.0
    u_intervened = scm_u.clone()
    u_intervened[0, do_var_idx] = do_val
    inv_w_int = torch.inverse(eye - w_intervened)
    x_counterfactual = u_intervened @ inv_w_int

    return x_factual, x_counterfactual


def two_stage_least_squares(
    z_inst: torch.Tensor,
    t_treat: torch.Tensor,
    y_outcome: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-Stage Least Squares (2SLS) Instrumental Variable Estimator.
    
    Stage 1: T_hat = Z @ (Z^T Z)^{-1} Z^T @ T
    Stage 2: beta_2SLS = (T_hat^T T_hat)^{-1} T_hat^T @ Y
    """
    # Stage 1: Projection matrix P_Z = Z (Z^T Z)^{-1} Z^T
    zt_z = z_inst.t() @ z_inst
    zt_t = z_inst.t() @ t_treat
    gamma = torch.inverse(zt_z) @ zt_t
    t_hat = z_inst @ gamma

    # Stage 2: Regress Y on predicted treatment T_hat
    th_th = t_hat.t() @ t_hat
    th_y = t_hat.t() @ y_outcome
    beta = torch.inverse(th_th) @ th_y

    return t_hat, beta


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. NOTEARS DAG constraint
    w_dag = torch.tensor([
        [0.00, 0.40, 0.00, 0.25],
        [0.00, 0.00, 0.50, 0.00],
        [0.00, 0.00, 0.00, 0.35],
        [0.00, 0.00, 0.00, 0.00],
    ], dtype=torch.float32)
    h_dag_ref, grad_h_ref = notears_acyclicity_constraint(w_dag)

    # 2. Structural Causal Model & Counterfactual
    scm_w = torch.tensor([
        [0.00, 0.00, 0.00, 0.00],
        [0.60, 0.00, 0.00, 0.00],
        [0.00, 0.45, 0.00, 0.00],
        [0.30, 0.00, 0.50, 0.00],
    ], dtype=torch.float32)
    scm_u = torch.tensor([[0.5, -0.2, 0.8, 0.1]], dtype=torch.float32)
    scm_x_factual_ref, counterfactual_x_ref = scm_generate_and_counterfactual(scm_w, scm_u, do_var_idx=1, do_val=3.0)

    # 3. Two-Stage Least Squares (2SLS) Instrumental Variables
    z_inst = torch.tensor([
        [1.2, 0.5],
        [-0.8, 1.1],
        [0.3, -1.4],
        [1.5, 0.2],
        [-1.1, -0.7],
        [0.9, 0.8],
        [-0.4, -1.2],
        [0.6, 1.3],
    ], dtype=torch.float32)
    t_treat = torch.tensor([[1.8], [-0.5], [0.1], [2.3], [-1.4], [1.6], [-0.9], [1.2]], dtype=torch.float32)
    y_outcome = torch.tensor([[4.6], [-1.1], [0.3], [5.8], [-3.4], [4.1], [-2.2], [3.1]], dtype=torch.float32)

    t_hat_ref, beta_2sls_ref = two_stage_least_squares(z_inst, t_treat, y_outcome)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "w_dag": w_dag,
        "h_dag_ref": h_dag_ref,
        "grad_h_ref": grad_h_ref,
        "scm_w": scm_w,
        "scm_u": scm_u,
        "scm_x_factual_ref": scm_x_factual_ref,
        "counterfactual_x_ref": counterfactual_x_ref,
        "z_inst": z_inst,
        "t_treat": t_treat,
        "y_outcome": y_outcome,
        "t_hat_ref": t_hat_ref,
        "beta_2sls_ref": beta_2sls_ref,
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
