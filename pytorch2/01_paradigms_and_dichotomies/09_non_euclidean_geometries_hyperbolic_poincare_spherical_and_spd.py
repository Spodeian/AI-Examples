"""Production PyTorch 2.x implementation for Category 01 Topic 09:
09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float
from safetensors.torch import load_file
from torch import Tensor


class PoincareBall:
    """Operations on the Poincaré Ball Model of Hyperbolic Space (D_c^d)."""

    @staticmethod
    def project(x: Float[Tensor, "... d"], c: float = 1.0, eps: float = 1e-5) -> Float[Tensor, "... d"]:
        """Projects points strictly inside the Poincaré ball boundary ||x|| < 1/sqrt(c)."""
        max_norm = (1.0 - eps) / (c ** 0.5)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)

    @staticmethod
    def exp_map_zero(v: Float[Tensor, "... d"], c: float = 1.0) -> Float[Tensor, "... d"]:
        """Exponential map from tangent space at origin to the Poincaré ball."""
        sqrt_c = c ** 0.5
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        tanh_term = torch.tanh(sqrt_c * v_norm / 2.0)
        return (tanh_term / (sqrt_c * v_norm)) * v

    @staticmethod
    def log_map_zero(x: Float[Tensor, "... d"], c: float = 1.0) -> Float[Tensor, "... d"]:
        """Logarithmic map from the Poincaré ball to tangent space at origin."""
        sqrt_c = c ** 0.5
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        clamped_norm = torch.clamp(sqrt_c * x_norm, max=0.9999)
        atanh_term = torch.atanh(clamped_norm)
        return (2.0 * atanh_term / (sqrt_c * x_norm)) * x

    @staticmethod
    def mobius_add(x: Float[Tensor, "... d"], y: Float[Tensor, "... d"], c: float = 1.0) -> Float[Tensor, "... d"]:
        """Möbius addition x (+) y in the Poincaré ball."""
        xy_inner = torch.sum(x * y, dim=-1, keepdim=True)
        x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_sq = torch.sum(y ** 2, dim=-1, keepdim=True)

        num = (1.0 + 2.0 * c * xy_inner + c * y_sq) * x + (1.0 - c * x_sq) * y
        denom = 1.0 + 2.0 * c * xy_inner + (c ** 2) * x_sq * y_sq
        return num / denom.clamp_min(1e-15)

    @classmethod
    def distance(cls, x: Float[Tensor, "... d"], y: Float[Tensor, "... d"], c: float = 1.0) -> Float[Tensor, "..."]:
        """Geodesic distance between x and y in the Poincaré ball."""
        neg_x = -x
        diff = cls.mobius_add(neg_x, y, c=c)
        diff_norm = torch.norm(diff, p=2, dim=-1).clamp(max=0.9999)
        return (2.0 / (c ** 0.5)) * torch.atanh(diff_norm)


class SphericalManifold:
    """Operations on the Unit Hypersphere Manifold S^(d-1)."""

    @staticmethod
    def project(x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        """Normalizes vectors onto the unit hypersphere."""
        return x / torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-15)

    @staticmethod
    def distance(p: Float[Tensor, "... d"], q: Float[Tensor, "... d"]) -> Float[Tensor, "..."]:
        """Geodesic angular distance on the hypersphere: arccos(<p, q>)."""
        dot = torch.sum(p * q, dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(dot)

    @staticmethod
    def alignment_loss(p: Float[Tensor, "... d"], q: Float[Tensor, "... d"]) -> Float[Tensor, "..."]:
        """Contrastive alignment loss on the sphere: ||p - q||_2^2."""
        return torch.sum((p - q) ** 2, dim=-1)


class SPDManifold:
    """Operations on the Symmetric Positive Definite Manifold S++^d via Log-Euclidean Metric."""

    @staticmethod
    def matrix_log(P: Float[Tensor, "... d d"]) -> Float[Tensor, "... d d"]:
        """Computes matrix logarithm for symmetric positive-definite matrices."""
        L, Q_mat = torch.linalg.eigh(P)
        log_L = torch.diag_embed(torch.log(L.clamp_min(1e-6)))
        return torch.matmul(Q_mat, torch.matmul(log_L, Q_mat.transpose(-1, -2)))

    @classmethod
    def log_euclidean_distance(cls, P: Float[Tensor, "... d d"], Q: Float[Tensor, "... d d"]) -> Float[Tensor, "..."]:
        """Log-Euclidean Riemannian metric: ||log(P) - log(Q)||_F."""
        log_P = cls.matrix_log(P)
        log_Q = cls.matrix_log(Q)
        return torch.norm(log_P - log_Q, p="fro", dim=(-2, -1))


class NonEuclideanModule(nn.Module):
    """Master module coordinating linear anchor and non-Euclidean manifold operations."""

    def __init__(
        self,
        weight: Float[Tensor, "d_in d_out"],
        bias: Float[Tensor, "d_out"],
        c: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(bias.clone())
        self.c = float(c)

    def forward(self, x: Float[Tensor, "batch d_in"]) -> Float[Tensor, "batch d_out"]:
        """Canonical linear projection matching test_output for cross-track parity."""
        return x @ self.weight + self.bias


def load_from_safetensors(path: str | Path) -> NonEuclideanModule:
    """Loads weights and returns an initialized NonEuclideanModule."""
    tensors = load_file(str(path))
    c_val = float(tensors["layers.c"].item())
    return NonEuclideanModule(
        weight=tensors["layers.weight"],
        bias=tensors["layers.bias"],
        c=c_val,
    )


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/01_paradigms_and_dichotomies/09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "01_paradigms_and_dichotomies/09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    ckpt_path = Path("weights/01_paradigms_and_dichotomies/09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd.safetensors")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    tensors = load_file(str(ckpt_path))
    model = load_from_safetensors(ckpt_path)

    x = tensors["test_input"]
    y_gold = tensors["test_output"]
    y_pred = model(x)
    assert torch.allclose(y_pred, y_gold, atol=1e-5), f"Linear anchor mismatch: max diff={(y_pred - y_gold).abs().max()}"

    # 1. Hyperbolic Poincare Ball parity
    c_val = model.c
    v = tensors["layers.euclidean_v"]
    exp_calc = PoincareBall.exp_map_zero(v, c=c_val)
    assert torch.allclose(exp_calc, tensors["layers.exp_map_0"], atol=1e-5), "Exp map mismatch"

    x_hyp = tensors["layers.poincare_x"]
    log_calc = PoincareBall.log_map_zero(x_hyp, c=c_val)
    assert torch.allclose(log_calc, tensors["layers.log_map_0"], atol=1e-5), "Log map mismatch"

    y_hyp = tensors["layers.poincare_y"]
    mobius_calc = PoincareBall.mobius_add(x_hyp, y_hyp, c=c_val)
    assert torch.allclose(mobius_calc, tensors["layers.mobius_add"], atol=1e-5), "Mobius add mismatch"

    dist_calc = PoincareBall.distance(x_hyp, y_hyp, c=c_val)
    assert torch.allclose(dist_calc, tensors["layers.poincare_dist"], atol=1e-5), "Poincare dist mismatch"

    # 2. Spherical Geometry parity
    p = tensors["layers.sphere_p"]
    q = tensors["layers.sphere_q"]
    s_dist_calc = SphericalManifold.distance(p, q)
    assert torch.allclose(s_dist_calc, tensors["layers.sphere_dist"], atol=1e-5), "Sphere dist mismatch"

    align_calc = SphericalManifold.alignment_loss(p, q)
    assert torch.allclose(align_calc, tensors["layers.alignment_loss"], atol=1e-5), "Sphere align mismatch"

    # 3. SPD Log-Euclidean distance parity
    P = tensors["layers.spd_P"]
    Q = tensors["layers.spd_Q"]
    spd_dist_calc = SPDManifold.log_euclidean_distance(P, Q)
    assert torch.allclose(spd_dist_calc, tensors["layers.log_euclidean_dist"], atol=1e-5), "SPD dist mismatch"

    print("PyTorch 2.x Module 09 passed all smoke tests and golden parity assertions!")
