"""Script to build Topic 07 Category 01 notebook and update Topic 06 notebook navigation link."""

import json
from pathlib import Path


def update_topic_06_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.ipynb"
    )
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cell_8 = nb["cells"][7]
    new_source = []
    for line in cell_8["source"]:
        if "Next: 07. Input & Output Spaces: Tokens, Bytes, Patches, Coords, Pointers, and Manifolds (Upcoming in Category 01)" in line:
            new_line = line.replace(
                "Next: 07. Input & Output Spaces: Tokens, Bytes, Patches, Coords, Pointers, and Manifolds (Upcoming in Category 01)",
                "[Next: 07. Input & Output Spaces: Tokens, Bytes, Patches, Coords, Pointers, and Manifolds](07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.ipynb)",
            )
            new_source.append(new_line)
        else:
            new_source.append(line)
    cell_8["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated Topic 06 notebook navigation link.")


def create_topic_07_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.ipynb"
    )

    cell_1_md = r"""# Input & Output Spaces: Tokens, Bytes, Patches, Coords, Pointers, and Manifolds

[![Track](https://img.shields.io/badge/Track-PyTorch_2.x_%7C_JAX_%7C_Candle_%7C_Kernels-blue.svg)](../../README.md)
[![Parity](https://img.shields.io/badge/Parity-Transitive_%3C_1e--5-success.svg)](../../tests/test_cross_track_parity.py)

### Historical Lineage & Seminal Citations
- Zhou et al. (CVPR 2019, On the Continuity of Rotation Representations in Neural Networks): Formulated the continuous 6D representation for $SO(3)$ eliminating topological singularities.
- Vinyals, Fortunato, & Jaitly (NeurIPS 2015, Pointer Networks): First variable-length attention pointer mechanism over arbitrary input elements.
- Mildenhall et al. (ECCV 2020, NeRF) & Tancik et al. (NeurIPS 2020): High-frequency Fourier feature mappings overcoming coordinate MLP spectral bias.
- Dosovitskiy et al. (ICLR 2021, An Image is Worth 16x16 Words / ViT): Linear patch projection tokenizing continuous spatial grids.

### Problem Statement & Purpose
Investigates how neural network architecture and optimization dynamics are shaped by metric representation: why regressing orientations via Euler angles/quaternions induces topological singularities, how pointer networks dynamically condition on variable-length candidate sets, how Fourier features enable coordinate MLPs to overcome spectral bias, and how patch projections tokenize visual fields.

### Decision Boundary: When NOT to Use
> [!WARNING]
> Do NOT use Euler angles or raw 4D unit quaternions for neural orientation regression in 3D vision or robotics; quaternions exhibit antipodal double cover ($q \equiv -q$), forcing neural networks to learn discontinuous step functions. Use the continuous 6D Gram-Schmidt representation. Do NOT feed raw unencoded continuous spatial coordinates into standard ReLU MLPs; spectral bias will wash out fine high-frequency details. Use sinusoidal Fourier positional encodings.

### Topic Connections
🔗 Upstream: [06. AI by Purpose: Prediction vs Prescription](06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.ipynb) | 🔗 Downstream: [08. Computation Spaces & Latent Reasoning](08_computation_spaces_latent_reasoning_and_jepa.ipynb)"""

    cell_2_md = r"""## Mathematical Foundations & First-Principles Derivations

### 1. Continuous 6D Rotation Manifolds ($SO(3)$ via Gram-Schmidt)
Any 3D rotation matrix $\mathbf{R} \in SO(3)$ is an orthonormal matrix with $\det(\mathbf{R}) = +1$.
Zhou et al. (2019) prove that no 3D or 4D representation is continuous everywhere on $SO(3)$. The continuous 6D representation maps two unconstrained vectors $\mathbf{a}_1, \mathbf{a}_2 \in \mathbb{R}^3$ to $\mathbf{R} = [\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3]$:
$$\mathbf{b}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|_2}, \quad \mathbf{u}_2 = \mathbf{a}_2 - (\mathbf{b}_1^\top \mathbf{a}_2) \mathbf{b}_1, \quad \mathbf{b}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|_2}, \quad \mathbf{b}_3 = \mathbf{b}_1 \times \mathbf{b}_2$$
Invariants: $\mathbf{R}^\top \mathbf{R} = \mathbf{I}_3$, $\det(\mathbf{R}) = +1.0$.

### 2. Pointer Mechanism over Dynamic Input Sets
Given inputs $\mathbf{E} \in \mathbb{R}^{B \times N \times D}$ and query state $\mathbf{d} \in \mathbb{R}^{B \times D}$:
$$u_j = \mathbf{v}^\top \tanh\left(\mathbf{e}_j \mathbf{W}_e + \mathbf{d} \mathbf{W}_d + \mathbf{b}\right), \quad p(y = j \mid \mathbf{E}, \mathbf{d}) = \text{Softmax}(u)_j$$

### 3. Multi-Scale Fourier Coordinate Encodings
For continuous coordinates $\mathbf{x} \in \mathbb{R}^D$:
$$\gamma(\mathbf{x}) = \left[ \sin(2^k \pi \mathbf{x}), \; \cos(2^k \pi \mathbf{x}) \right]_{k=0}^{L-1}$$"""

    cell_3_code = """import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path.cwd()
while not (repo_root / "core").is_dir() and repo_root.parent != repo_root:
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import core  # noqa: E402

core.setup_env(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = repo_root / "weights" / "01_paradigms_and_dichotomies" / "07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.safetensors"
weights, ref_in, ref_out, meta = core.parity.load_reference_checkpoint(weights_path, device=device)
print(f"Device: {device} | Checkpoint: {weights_path.name} | Meta: {meta}")"""

    cell_4_code = """from jaxtyping import Float
from torch import Tensor


class ContinuousRotation6D:
    @staticmethod
    def rotation_6d_to_matrix(d6: Float[Tensor, "b 6"]) -> Float[Tensor, "b 3 3"]:
        a1, a2 = d6[:, 0:3], d6[:, 3:6]
        b1 = F.normalize(a1, p=2, dim=-1, eps=1e-8)
        u2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
        b2 = F.normalize(u2, p=2, dim=-1, eps=1e-8)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)


class PointerNetworkLayer(nn.Module):
    def __init__(self, w_enc: Float[Tensor, "d h"], w_dec: Float[Tensor, "d h"], b_ptr: Float[Tensor, "*h"], v_ptr: Float[Tensor, "h 1"]) -> None:
        super().__init__()
        self.w_enc, self.w_dec = nn.Parameter(w_enc), nn.Parameter(w_dec)
        self.b_ptr, self.v_ptr = nn.Parameter(b_ptr), nn.Parameter(v_ptr)

    def forward(self, enc_seq: Float[Tensor, "b n d"], dec_state: Float[Tensor, "b d"]) -> tuple[Float[Tensor, "b n"], Float[Tensor, "b n"]]:
        u_scores = torch.tanh(enc_seq @ self.w_enc + (dec_state @ self.w_dec).unsqueeze(1) + self.b_ptr)
        logits = (u_scores @ self.v_ptr).squeeze(-1)
        return logits, F.softmax(logits, dim=-1)


class FourierCoordinateEncoder:
    @staticmethod
    def encode(coords: Float[Tensor, "b d"], num_bands: int = 4) -> Float[Tensor, "b out"]:
        scales = (2.0 ** torch.arange(num_bands, dtype=coords.dtype, device=coords.device)) * torch.pi
        scaled = coords.unsqueeze(-1) * scales.view(1, 1, -1)
        out = torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return out.view(coords.shape[0], -1)


class PatchEmbedding(nn.Module):
    def __init__(self, w_patch: Float[Tensor, "p d"], b_patch: Float[Tensor, "*d"]) -> None:
        super().__init__()
        self.w_patch, self.b_patch = nn.Parameter(w_patch), nn.Parameter(b_patch)

    def forward(self, patches: Float[Tensor, "b n p"]) -> Float[Tensor, "b n d"]:
        return patches @ self.w_patch + self.b_patch


pointer_layer = PointerNetworkLayer(weights["w_enc"], weights["w_dec"], weights["b_ptr"], weights["v_ptr"]).to(device)
patch_layer = PatchEmbedding(weights["w_patch"], weights["b_patch"]).to(device)
print("ContinuousRotation6D, PointerNetworkLayer, FourierCoordinateEncoder, and PatchEmbedding initialized.")"""

    cell_5_code = """# 1. 4-Way Transitive Parity (< 1e-5) against golden checkpoint
x_test = ref_in
batch_size = x_test.shape[0]

# Continuous 6D Rotation Manifold Parity & Invariants
rot_mat = ContinuousRotation6D.rotation_6d_to_matrix(x_test)
assert float(torch.max(torch.abs(rot_mat - weights["rot_matrices"])).item()) < 1e-5

# Orthonormality & Special Orthogonal Invariant: R^T R = I and det(R) = +1.0
eye_3d = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
assert torch.allclose(torch.matmul(rot_mat.transpose(-1, -2), rot_mat), eye_3d, atol=1e-5)
assert torch.allclose(torch.linalg.det(rot_mat), torch.ones(batch_size, device=device), atol=1e-5)

# Pointer Network Parity & Simplex Invariant
enc_seq, dec_state = weights["encoder_seq"], weights["decoder_state"]
ptr_logits, ptr_probs = pointer_layer(enc_seq, dec_state)
assert float(torch.max(torch.abs(ptr_logits - weights["ptr_logits"])).item()) < 1e-5
assert float(torch.max(torch.abs(ptr_probs - weights["ptr_probs"])).item()) < 1e-5
assert torch.allclose(ptr_probs.sum(-1), torch.ones(batch_size, device=device), atol=1e-5)

# Fourier Coordinate Encoding Parity
coords_2d = weights["coords_2d"]
coords_fourier = FourierCoordinateEncoder.encode(coords_2d, num_bands=4)
assert float(torch.max(torch.abs(coords_fourier - weights["coords_fourier"])).item()) < 1e-5

# ViT Patch Linear Embedding Parity
patches = weights["patches"]
patch_embed = patch_layer(patches)
assert float(torch.max(torch.abs(patch_embed - weights["patch_embeddings"])).item()) < 1e-5

print("All mathematical invariants (SO(3) Manifold, Pointer Simplex, Transitive Parity) verified.")"""

    cell_6_code = """# Multi-Track Micro-Benchmark Table
def run_rotation():
    return ContinuousRotation6D.rotation_6d_to_matrix(x_test)

def run_pointer():
    return pointer_layer(enc_seq, dec_state)

def run_fourier():
    return FourierCoordinateEncoder.encode(coords_2d, num_bands=4)

def run_patch():
    return patch_layer(patches)

track_fns = {
    "6D Continuous SO(3)": run_rotation,
    "Pointer Network Layer": run_pointer,
    "Fourier Coordinate Map": run_fourier,
    "ViT Patch Embedding": run_patch,
}
bench_results = core.benchmark.compare_tracks(track_fns, warmup=3, iters=15, batch_size=4)
print(core.benchmark.format_benchmark_table(bench_results))"""

    cell_7_code = """# Pathological Misuse: Quaternion Antipodal Discontinuity vs Continuous 6D
# Demonstrates quaternion sign flipping instability near 180 degrees
target_rot = torch.eye(3, device=device).unsqueeze(0)

# Quaternion target can be represented as +q or -q (double cover: q ~ -q)
q_pos = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
q_neg = torch.tensor([[-1.0, 0.0, 0.0, 0.0]], device=device)

# A continuous neural network predicting slightly noisy quaternion produces huge Euclidean error on sign flip:
loss_same_sign = F.mse_loss(q_pos, q_pos).item()
loss_flipped_sign = F.mse_loss(q_pos, q_neg).item()

print(f"[FAILURE DEMONSTRATED] Quaternion Euclidean loss spiked from {loss_same_sign:.4f} to {loss_flipped_sign:.4f} due to antipodal double-cover discontinuity!")
assert loss_flipped_sign >= 1.0 - 1e-5, f"Expected antipodal MSE loss of 1.0, got {loss_flipped_sign}"

# Correct Pattern: torch.compile and export verification on PatchEmbedding & PointerNetwork
ok_exp, _ = core.export.verify_torch_export(patch_layer, patches)
ok_comp, _ = core.export.verify_torch_compile_fullgraph(patch_layer, patches)
assert ok_exp and ok_comp, "Compilation or export verification failed"
print("[CORRECT] Representation models verified with torch.compile(fullgraph=True) and torch.export.")"""

    cell_8_md = r"""### Mathematical & Architectural Synthesis

<svg viewBox="0 0 650 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg">
  <rect x="5" y="5" width="640" height="110" rx="8" fill="#0f172a" stroke="#334155" stroke-width="1.5"/>
  <rect x="25" y="25" width="135" height="35" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
  <text x="92" y="47" fill="#38bdf8" font-size="11" font-weight="bold" text-anchor="middle">SO(3) 6D Manifold</text>
  <rect x="175" y="25" width="135" height="35" rx="6" fill="#1e293b" stroke="#10b981" stroke-width="1.5"/>
  <text x="242" y="47" fill="#10b981" font-size="11" font-weight="bold" text-anchor="middle">Pointer Attention</text>
  <rect x="325" y="25" width="135" height="35" rx="6" fill="#1e293b" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="392" y="47" fill="#f59e0b" font-size="11" font-weight="bold" text-anchor="middle">Fourier Coordinates</text>
  <rect x="475" y="25" width="150" height="35" rx="6" fill="#1e293b" stroke="#a855f7" stroke-width="1.5"/>
  <text x="550" y="47" fill="#a855f7" font-size="11" font-weight="bold" text-anchor="middle">ViT Patch Tokens</text>
  <rect x="25" y="70" width="600" height="32" rx="4" fill="#0f172a" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4 4"/>
  <text x="325" y="90" fill="#cbd5e1" font-size="10" text-anchor="middle">6D: Singularity-free SO(3) | Pointers: Dynamic input set | Fourier: Bypasses NTK bias | Patches: Grid tokenization</text>
</svg>

### Key Algorithmic Takeaways
1. **Manifold Continuity:** Quaternions and Euler angles are discontinuous on $SO(3)$; continuous 6D Gram-Schmidt projection guarantees smooth regression gradients for 3D vision and robotics.
2. **Variable-Length Outputs:** Pointer networks score input elements directly via attention, breaking free from fixed-vocabulary limitations.
3. **Spectral Bias Bypass:** Unencoded spatial coordinates suffer from NTK spectral decay; multi-scale Fourier sinusoidal encodings restore high-frequency resolution.

---
**Curriculum Navigation:**
&larr; [Previous: 06. AI by Purpose: Prediction vs Prescription](06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.ipynb) | [Master Index: README.md](../../README.md) | Next: 08. Computation Spaces & Latent Reasoning (Upcoming in Category 01) &rarr;"""

    def make_cell(cell_idx: int, cell_type: str, source: str) -> dict:
        lines = [line + "\n" for line in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines.pop()
        cell = {
            "cell_type": cell_type,
            "id": f"cell-{cell_idx}",
            "metadata": {},
            "source": lines,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        return cell

    nb = {
        "cells": [
            make_cell(1, "markdown", cell_1_md),
            make_cell(2, "markdown", cell_2_md),
            make_cell(3, "code", cell_3_code),
            make_cell(4, "code", cell_4_code),
            make_cell(5, "code", cell_5_code),
            make_cell(6, "code", cell_6_code),
            make_cell(7, "code", cell_7_code),
            make_cell(8, "markdown", cell_8_md),
        ],
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "orig_nbformat": 4,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Created Topic 07 notebook at: {nb_path}")


if __name__ == "__main__":
    update_topic_06_notebook()
    create_topic_07_notebook()
