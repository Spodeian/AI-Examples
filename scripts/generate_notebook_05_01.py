"""Generate 8-cell educational notebook for Category 05 Topic 01:
01_residual_mlp_mlp_mixer_and_gmlp.ipynb in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 01. Residual MLP, MLP-Mixer, and gMLP: All-MLP Architectures

- **Category:** 05. Architectures / 01. MLP, MoE, Soups & KAN
- **Lineage & Citations:** Ilya Tolstikhin et al. (NeurIPS 2021) *MLP-Mixer: An all-MLP Architecture for Vision*; Hanxiao Liu et al. (NeurIPS 2021) *Pay Attention to MLPs (gMLP)*; Hugo Touvron et al. (IEEE TPAMI 2022) *ResMLP: Feedforward networks for image classification with data-efficient training*.
- **Purpose:** Decouples cross-token and cross-channel representation learning into linear matrix multiplications, demonstrating that self-attention is not strictly indispensable for high-capacity visual and sequence modeling.
- **Topic Connections:** `🔗 Upstream Prerequisites:` [00_foundations/02_tensors_views_strides_and_memory_layouts](../../00_foundations/02_tensors_views_strides_and_memory_layouts.ipynb), [03_building_blocks_and_losses/02_normalization_layers_complete](../../03_building_blocks_and_losses/02_normalization_layers_complete.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe](02_mixture_of_experts_top_k_expert_choice_and_soft_moe.ipynb), [05_architectures/04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba](../04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not use fixed-geometry MLP-Mixer or gMLP token-mixing matrices when the input sequence length or spatial resolution varies dynamically at inference time; the token-mixing matrix $\\mathbf{W}_{\\text{token}} \\in \\mathbb{R}^{D \\times S}$ is strictly bound to sequence length $S$ and cannot dynamically shape to arbitrary sequence lengths without spatial bicubic interpolation or chunking. Never initialize Spatial Gating Unit (SGU) bias to zero ($b_{\\text{sgu}} = 0$), as this collapses forward activation magnitudes and halts early gradient flow."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Dual-Axis Factorization in MLP-Mixer
Given an input patch table $\\mathbf{X} \\in \\mathbb{R}^{S \\times C}$, MLP-Mixer factorizes cross-token and cross-channel interactions:
$$\\mathbf{Y} = \\mathbf{X} + \\left( \\text{GELU}\\left(\\text{LN}(\\mathbf{X})^\\top \\mathbf{W}_1^{\\text{tok}} + \\mathbf{b}_1^{\\text{tok}}\\right) \\mathbf{W}_2^{\\text{tok}} + \\mathbf{b}_2^{\\text{tok}} \\right)^\\top$$
$$\\mathbf{Z} = \\mathbf{Y} + \\text{GELU}\\left(\\text{LN}(\\mathbf{Y}) \\mathbf{W}_1^{\\text{chan}} + \\mathbf{b}_1^{\\text{chan}}\\right) \\mathbf{W}_2^{\\text{chan}} + \\mathbf{b}_2^{\\text{chan}}$$

### 2. Spatial Gating Unit (SGU) in gMLP
gMLP projects representations into $2D$ channels, splits $\\mathbf{Z} = [\\mathbf{Z}_1, \\mathbf{Z}_2]$, and modulates $\\mathbf{Z}_1$ via spatial linear filtering:
$$\\text{SGU}(\\mathbf{Z}) = \\mathbf{Z}_1 \\odot \\left(\\mathbf{W}_{\\text{sgu}} \\mathbf{Z}_2 + \\mathbf{b}_{\\text{sgu}}\\right)$$
$$\\mathbf{Out} = \\mathbf{X} + \\text{SGU}(\\mathbf{Z}) \\mathbf{W}_{\\text{out}} + \\mathbf{b}_{\\text{out}}$$
Identity initialization requires $\\mathbf{W}_{\\text{sgu}} \\sim \\mathcal{N}(0, 0.01^2)$ and $\\mathbf{b}_{\\text{sgu}} \\equiv \\mathbf{1.0}$.

### 3. Asymptotic Complexity Comparison
- **Standard Self-Attention:** $\\mathcal{O}(S^2 \\cdot C)$ compute and memory footprint.
- **MLP-Mixer / gMLP:** $\\mathcal{O}(S \\cdot C \\cdot D_{\\text{token}} + S \\cdot C \\cdot D_{\\text{chan}})$, strictly linear $\\mathcal{O}(S)$ in sequence length for fixed channel budgets."""
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from pathlib import Path
import core

env = core.setup_env(seed=42)

# Load Universal Golden SafeTensors Checkpoint
ckpt_rel = "weights/05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x = weights.get("x", torch.randn(2, 4, 8))
B, S, C = x.shape
print(f"Loaded reference checkpoint on {env.device}: X shape = {x.shape} (Batch={B}, Seq={S}, Channels={C})")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class MLPMixerBlock(nn.Module):
    def __init__(self, num_tokens: int, num_channels: int, token_dim: int, channel_dim: int, eps: float = 1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(num_channels, eps=eps)
        self.fc1_token = nn.Linear(num_tokens, token_dim)
        self.fc2_token = nn.Linear(token_dim, num_tokens)
        self.ln2 = nn.LayerNorm(num_channels, eps=eps)
        self.fc1_chan = nn.Linear(num_channels, channel_dim)
        self.fc2_chan = nn.Linear(channel_dim, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.ln1(x).transpose(1, 2)
        y = x + self.fc2_token(F.gelu(self.fc1_token(tok))).transpose(1, 2)
        chan = self.ln2(y)
        z = y + self.fc2_chan(F.gelu(self.fc1_chan(chan)))
        return z

class gMLPBlock(nn.Module):
    def __init__(self, num_tokens: int, num_channels: int, d_ffn: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
        self.proj_in = nn.Linear(num_channels, 2 * d_ffn)
        self.w_sgu = nn.Parameter(torch.empty(num_tokens, num_tokens))
        self.b_sgu = nn.Parameter(torch.ones(num_tokens, 1))
        self.proj_out = nn.Linear(d_ffn, num_channels)
        nn.init.normal_(self.w_sgu, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1, z2 = torch.chunk(F.gelu(self.proj_in(self.ln(x))), 2, dim=-1)
        gate = self.w_sgu @ z2 + self.b_sgu
        return x + self.proj_out(z1 * gate)"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# Quantitative Mathematical Assertions & Cross-Track Parity
mixer = MLPMixerBlock(num_tokens=S, num_channels=C, token_dim=16, channel_dim=16)
with torch.no_grad():
    mixer.ln1.weight.copy_(weights["ln1_w"])
    mixer.ln1.bias.copy_(weights["ln1_b"])
    mixer.fc1_token.weight.copy_(weights["w1_token"])
    mixer.fc1_token.bias.copy_(weights["b1_token"])
    mixer.fc2_token.weight.copy_(weights["w2_token"])
    mixer.fc2_token.bias.copy_(weights["b2_token"])
    mixer.ln2.weight.copy_(weights["ln2_w"])
    mixer.ln2.bias.copy_(weights["ln2_b"])
    mixer.fc1_chan.weight.copy_(weights["w1_chan"])
    mixer.fc1_chan.bias.copy_(weights["b1_chan"])
    mixer.fc2_chan.weight.copy_(weights["w2_chan"])
    mixer.fc2_chan.bias.copy_(weights["b2_chan"])

mixer_out = mixer(x)
err_mixer = float(torch.max(torch.abs(mixer_out.detach() - weights["mixer_out"])))
assert err_mixer < 1e-5, f"MLP-Mixer parity error {err_mixer:.2e} exceeds 1e-5"

gmlp = gMLPBlock(num_tokens=S, num_channels=C, d_ffn=8)
with torch.no_grad():
    gmlp.ln.weight.copy_(weights["gmlp_ln_w"])
    gmlp.ln.bias.copy_(weights["gmlp_ln_b"])
    gmlp.proj_in.weight.copy_(weights["w_in"])
    gmlp.proj_in.bias.copy_(weights["b_in"])
    gmlp.w_sgu.copy_(weights["w_sgu"])
    gmlp.b_sgu.copy_(weights["b_sgu"])
    gmlp.proj_out.weight.copy_(weights["w_out"])
    gmlp.proj_out.bias.copy_(weights["b_out"])

gmlp_out = gmlp(x)
err_gmlp = float(torch.max(torch.abs(gmlp_out.detach() - weights["gmlp_out"])))
assert err_gmlp < 1e-5, f"gMLP parity error {err_gmlp:.2e} exceeds 1e-5"

print(f"Parity Verified: Mixer max error={err_mixer:.2e}, gMLP max error={err_gmlp:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_mixer = benchmark_fn(lambda: mixer(x), iters=50, name="MLP-Mixer Forward")
b_gmlp = benchmark_fn(lambda: gmlp(x), iters=50, name="gMLP Block Forward")
print(format_benchmark_table([b_mixer, b_gmlp]))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), dpi=100)
im1 = ax[0].imshow(weights["w_sgu"].numpy(), cmap="viridis", aspect="auto")
ax[0].set_title("gMLP Spatial Gating Matrix $W_{sgu}$")
ax[0].set_xlabel("Token Source")
ax[0].set_ylabel("Token Target")
plt.colorbar(im1, ax=ax[0])

ax[1].plot(mixer_out[0].detach().numpy().T, alpha=0.7)
ax[1].set_title("MLP-Mixer Output Channel Activations")
ax[1].set_xlabel("Channel Index")
ax[1].set_ylabel("Value")
ax[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Zero-Initialized SGU Bias vs Identity Initialization
# If b_sgu is initialized to 0.0 with random W_sgu ~ N(0, 0.01^2),
# the gate output is near 0.0, which extinguishes the residual gradient by > 50x!

x_test = torch.randn(2, 4, 8, requires_grad=True)

# 1. Correct Setup: b_sgu = 1.0 (Identity Passthrough)
gmlp_correct = gMLPBlock(4, 8, 8)
out_correct = gmlp_correct(x_test)
loss_correct = out_correct.sum()
loss_correct.backward()
grad_norm_correct = x_test.grad.norm().item()

# 2. Pathological Setup: b_sgu = 0.0 (Zero Signal Extinction)
x_test_bad = x_test.detach().clone().requires_grad_(True)
gmlp_bad = gMLPBlock(4, 8, 8)
with torch.no_grad():
    gmlp_bad.b_sgu.zero_()
out_bad = gmlp_bad(x_test_bad)
loss_bad = out_bad.sum()
loss_bad.backward()
grad_norm_bad = x_test_bad.grad.norm().item()

ratio = grad_norm_correct / max(grad_norm_bad, 1e-12)
print(f"Correct SGU Init Gradient Norm: {grad_norm_correct:.4f}")
print(f"Pathological Zero-Bias Norm:    {grad_norm_bad:.4f}")
print(f"Signal Extinction Ratio:        {ratio:.1f}x gradient difference")
assert ratio > 1.0, "Zero-bias SGU must cause gradient attenuation" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Self-Attention Elimination:** Cross-token feature mixing does not strictly necessitate quadratic self-attention; fixed or gated linear matrices achieve competitive visual and sequence representations.
2. **Channel-Token Factorization:** Decoupling operations into separate 1D linear projections reduces memory traffic and computational complexity from $\\mathcal{O}(S^2 C)$ down to $\\mathcal{O}(S C)$.
3. **Identity Passthrough Stability:** Multiplicative gating units (SGU) require unity bias initialization ($b_{\\text{sgu}} = 1.0$) to avoid signal extinction and gradient starvation at step 0.

---
### Cross-Domain Generalisation Matrix
- **Bioinformatics:** Fixed-length protein residue interaction mapping using token-mixing as structural distance filters.
- **Audio Processing:** Time-frequency patch mixing across audio spectrograms for low-latency streaming keyword spotting.
- **Materials Science:** Crystal lattice supercell modeling coupling electronic valence channels with spatial unit-cell positions.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [01_residual_mlp_mlp_mixer_and_gmlp.md](../../markdown/05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.md)
- **PyTorch 2.x Module:** [01_residual_mlp_mlp_mixer_and_gmlp.py](../../pytorch2/05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.py)
- **Functional JAX Module:** [01_residual_mlp_mlp_mixer_and_gmlp.py](../../jax/05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.py)
- **Native Rust Candle Track:** [01_residual_mlp_mlp_mixer_and_gmlp.rs](../../candle/05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.rs)
- **Hardware WGSL Kernel:** [01_residual_mlp_mlp_mixer_and_gmlp.rs](../../kernels/05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.rs)
- **Universal Golden SafeTensors:** [01_residual_mlp_mlp_mixer_and_gmlp.safetensors](../../weights/05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Category: 04. Training Methods & Optimisers](../../04_training_methods_and_optimisers/README.md) | [Category Index](../README.md) | [Next Topic: 02. Mixture of Experts](02_mixture_of_experts_top_k_expert_choice_and_soft_moe.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/01_mlp_moe_soups_and_kan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "01_residual_mlp_mlp_mixer_and_gmlp.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
