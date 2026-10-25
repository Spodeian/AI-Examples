"""Neural Operators FNO DeepONet and Meshgraphnets — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets
Golden weights: weights/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def fno_spectral_conv_1d(
    v: torch.Tensor,
    r_real: torch.Tensor,
    r_imag: torch.Tensor,
    s_len: int,
    k_modes: int,
) -> torch.Tensor:
    """Performs 1D discrete spectral convolution with K truncated Fourier modes."""
    k = torch.arange(k_modes, device=v.device, dtype=v.dtype).unsqueeze(1)
    n = torch.arange(s_len, device=v.device, dtype=v.dtype).unsqueeze(0)
    angle = 2.0 * math.pi * k * n / s_len

    c_mat = torch.cos(angle)
    s_mat = -torch.sin(angle)

    v_r = torch.matmul(v, c_mat.t())
    v_i = torch.matmul(v, s_mat.t())

    # Complex spectral multiplication: u = (v_r + i v_i)(r_real + i r_imag)
    # out_r = v_r @ r_real^T - v_i @ r_imag^T
    # out_i = v_r @ r_imag^T + v_i @ r_real^T
    u_r = torch.einsum("bck, koc -> bok", v_r, r_real) - torch.einsum("bck, koc -> bok", v_i, r_imag)
    u_i = torch.einsum("bck, koc -> bok", v_r, r_imag) + torch.einsum("bck, koc -> bok", v_i, r_real)

    # Inverse discrete synthesis
    weights_k = torch.tensor([1.0] + [2.0] * (k_modes - 1), device=v.device, dtype=v.dtype).unsqueeze(1)
    m_cos = torch.cos(angle) * weights_k
    m_sin = torch.sin(angle) * weights_k

    k_v = (1.0 / s_len) * (torch.matmul(u_r, m_cos) - torch.matmul(u_i, m_sin))
    return k_v


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
    }

    # 1. Fourier Neural Operator (FNO-1D)
    # B=2, d_in=1, d_v=4, S=8, K=4 modes
    b_dim, d_v, s_len, k_modes = 2, 4, 8, 4
    fno_input = torch.randn(b_dim, 1, s_len)
    w_p = torch.randn(d_v, 1) * 0.5
    b_p = torch.randn(d_v, 1) * 0.1

    # Lifting: P(a) = GELU(W_p a + b_p)
    v0 = F.gelu(w_p @ fno_input + b_p, approximate="none")

    # Spectral Convolution + Local Bypass
    r_real = torch.randn(k_modes, d_v, d_v) * 0.2
    r_imag = torch.randn(k_modes, d_v, d_v) * 0.2
    k_v0 = fno_spectral_conv_1d(v0, r_real, r_imag, s_len, k_modes)

    w_bypass = torch.randn(d_v, d_v) * 0.2
    bypass = torch.einsum("oc, bcs -> bos", w_bypass, v0)
    v1 = F.gelu(k_v0 + bypass, approximate="none")

    # Projection: Q(v1) = W_q v1 + b_q
    w_q = torch.randn(1, d_v) * 0.5
    b_q = torch.randn(1, 1) * 0.1
    fno_output = w_q @ v1 + b_q

    # 2. DeepONet (Branch-Trunk Operator)
    m_sensors, p_latent = 8, 8
    u_sensor = torch.randn(b_dim, m_sensors)
    y_query = torch.linspace(0.0, 1.0, 4).unsqueeze(1)
    w_b = torch.randn(m_sensors, p_latent) * 0.3
    b_b = torch.randn(p_latent) * 0.1
    w_t = torch.randn(1, p_latent) * 0.3
    b_t = torch.randn(p_latent) * 0.1
    b_out0 = torch.randn(1) * 0.1

    branch_feat = F.gelu(u_sensor @ w_b + b_b, approximate="none")
    trunk_feat = F.gelu(y_query @ w_t + b_t, approximate="none")
    deeponet_output = branch_feat @ trunk_feat.t() + b_out0

    # 3. MeshGraphNets (Encode-Process-Decode on Unstructured Mesh)
    n_nodes = 4
    v_nodes = torch.randn(n_nodes, 4)
    src_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    dst_indices = torch.tensor([1, 2, 3, 0], dtype=torch.int64)
    e_feat = torch.randn(n_nodes, 2)

    w_e = torch.randn(10, 4) * 0.3
    b_e = torch.randn(4) * 0.1
    w_v = torch.randn(8, 4) * 0.3
    b_v = torch.randn(4) * 0.1

    e_in = torch.cat([v_nodes[src_indices], v_nodes[dst_indices], e_feat], dim=-1)
    e_up = F.gelu(e_in @ w_e + b_e, approximate="none")

    e_agg = torch.zeros(n_nodes, 4)
    e_agg.index_add_(0, dst_indices, e_up)

    v_in = torch.cat([v_nodes, e_agg], dim=-1)
    mgn_output = F.gelu(v_in @ w_v + b_v, approximate="none")

    tensors.update({
        "fno_input": fno_input,
        "w_p": w_p,
        "b_p": b_p,
        "v0": v0,
        "r_real": r_real,
        "r_imag": r_imag,
        "k_v0": k_v0,
        "w_bypass": w_bypass,
        "v1": v1,
        "w_q": w_q,
        "b_q": b_q,
        "fno_output": fno_output,
        "u_sensor": u_sensor,
        "y_query": y_query,
        "w_b": w_b,
        "b_b": b_b,
        "w_t": w_t,
        "b_t": b_t,
        "b_out0": b_out0,
        "branch_feat": branch_feat,
        "trunk_feat": trunk_feat,
        "deeponet_output": deeponet_output,
        "v_nodes": v_nodes,
        "src_indices": src_indices.to(torch.int32),
        "dst_indices": dst_indices.to(torch.int32),
        "e_feat": e_feat,
        "w_e": w_e,
        "b_e": b_e,
        "w_v": w_v,
        "b_v": b_v,
        "mgn_output": mgn_output,
    })
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
