"""Optimisation as Architecture LISTA ADMM-Net CRATE and Mesa-Optimizers — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers
Golden weights: weights/06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def soft_threshold(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    r"""Elementwise soft thresholding: sign(x) * max(|x| - theta, 0)."""
    return torch.sign(x) * torch.clamp(torch.abs(x) - theta, min=0.0)


def lista_unroll(
    y: torch.Tensor,
    w_e: torch.Tensor,
    w_s: torch.Tensor,
    theta: torch.Tensor,
    num_layers: int = 3,
) -> torch.Tensor:
    r"""Learned ISTA (LISTA) layer unrolling:
    z_{k+1} = \eta_{\theta_k}( z_k W_s + y W_e )
    y: [B, M] (measurements)
    w_e: [M, N]
    w_s: [N, N]
    theta: [N]
    z_0 = 0
    """
    B = y.shape[0]
    N = w_e.shape[1]
    z = torch.zeros(B, N, dtype=y.dtype, device=y.device)
    for _ in range(num_layers):
        lin = z @ w_s + y @ w_e
        z = soft_threshold(lin, theta)
    return z


def admm_net_step(
    y: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    u: torch.Tensor,
    w_inv: torch.Tensor,
    rho: float = 1.0,
    theta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""ADMM-Net unrolled iteration step:
    min_{x, z} (1/2) ||A x - y||_2^2 + g(z)  s.t.  x - z = 0
    1. x_{k+1} = (A^T A + rho I)^{-1} (A^T y + rho (z_k - u_k))
       Parameterize (A^T A + rho I)^{-1} directly as learned W_inv
    2. z_{k+1} = soft_threshold( x_{k+1} + u_k, theta / rho )
    3. u_{k+1} = u_k + x_{k+1} - z_{k+1} (dual multiplier update)
    """
    rhs = y + rho * (z - u)
    x_next = rhs @ w_inv
    z_next = soft_threshold(x_next + u, torch.tensor(theta / rho))
    u_next = u + x_next - z_next
    return x_next, z_next, u_next


def crate_step(
    z: torch.Tensor,
    d_dict: torch.Tensor,
    step_size: float = 0.1,
    lambda_reg: float = 0.05,
) -> torch.Tensor:
    r"""CRATE (Coding Rate Reduction Transform Architecture) Step:
    Transforms representation by maximizing coding rate reduction / sparse coding:
    z_{k+1} = \operatorname{Prox}_{\lambda} ( z_k - \eta \nabla \mathcal{R}(z_k) )
    Gradient step with dictionary D:
    grad = z_k (D^T D - I)
    z_{k+1} = soft_threshold( z_k - step_size * grad, lambda_reg )
    """
    dtd = d_dict.t() @ d_dict
    gram_eye = dtd - torch.eye(d_dict.shape[1], dtype=z.dtype, device=z.device)
    grad = z @ gram_eye
    z_step = z - step_size * grad
    return soft_threshold(z_step, torch.tensor(lambda_reg))


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B, M, N = 4, 3, 5
    
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # 1. LISTA parameters
    y_meas = torch.randn(B, M) * 0.8
    w_e = torch.randn(M, N) * 0.4
    raw_ws = torch.randn(N, N) * 0.2
    # Contractive weight matrix
    w_s = raw_ws / (torch.linalg.matrix_norm(raw_ws, ord=2) * 1.5)
    theta_lista = torch.abs(torch.randn(N)) * 0.05 + 0.02
    
    z_lista = lista_unroll(y_meas, w_e, w_s, theta_lista, num_layers=3)
    
    # 2. ADMM-Net parameters
    # Measurement D=N, M=N for direct square unroll
    y_admm = torch.randn(B, N) * 0.6
    x0 = torch.zeros(B, N)
    z0 = torch.zeros(B, N)
    u0 = torch.zeros(B, N)
    raw_winv = torch.randn(N, N) * 0.3
    w_inv = raw_winv.t() @ raw_winv + 0.5 * torch.eye(N)
    
    x1, z1, u1 = admm_net_step(y_admm, x0, z0, u0, w_inv, rho=1.0, theta=0.08)
    
    # 3. CRATE step
    d_dict = torch.randn(M, N) * 0.5
    z_crate_in = torch.randn(B, N) * 0.5
    z_crate_out = crate_step(z_crate_in, d_dict, step_size=0.1, lambda_reg=0.04)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        
        "y_meas": y_meas,
        "w_e": w_e,
        "w_s": w_s,
        "theta_lista": theta_lista,
        "z_lista": z_lista,
        
        "y_admm": y_admm,
        "w_inv": w_inv,
        "x1": x1,
        "z1": z1,
        "u1": u1,
        
        "d_dict": d_dict,
        "z_crate_in": z_crate_in,
        "z_crate_out": z_crate_out,
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
