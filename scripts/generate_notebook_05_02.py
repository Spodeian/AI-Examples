"""Generate 8-cell educational notebook for Category 05 Topic 02:
02_mixture_of_experts_top_k_expert_choice_and_soft_moe.ipynb in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 02. Mixture of Experts: Sparse Top-K, Expert Choice, and Soft MoE

- **Category:** 05. Architectures / 01. MLP, MoE, Soups & KAN
- **Lineage & Citations:** Robert A. Jacobs et al. (1991) *Adaptive Mixtures of Local Experts*; Noam Shazeer et al. (ICLR 2017) *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer*; William Fedus et al. (JMLR 2022) *Switch Transformers*; Yanqi Zhou et al. (NeurIPS 2022) *Expert Choice Routing*; Joan Puigcerver et al. (ICLR 2024) *From Sparse to Soft Mixtures of Experts*.
- **Purpose:** Decouples parameter capacity from per-token FLOPs via conditional sparsity, allowing trillion-parameter scale with constant inference latency.
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp](01_residual_mlp_mlp_mixer_and_gmlp.ipynb), [03_building_blocks_and_losses/12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting](../../03_building_blocks_and_losses/12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths](03_deepseek_moe_aux_loss_free_and_mixture_of_depths.ipynb), [05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling](04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not omit the auxiliary load-balancing loss $\\mathcal{L}_{\\text{aux}} = \\alpha E \\sum f_i P_i$ in Top-$K$ MoE; without it, the gating router collapses into a degenerate state where a single expert receives 100% of tokens while other experts permanently starve. Never apply standard Expert-Choice or Soft MoE across sequence tokens in causal autoregressive generation without future-token masking, as dispatch normalization across the entire sequence leaks future tokens into earlier slots."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Sparse Top-$K$ MoE & Load Balancing
Given input tokens $\\mathbf{X} \\in \\mathbb{R}^{T \\times D}$ and $E$ feedforward experts $f_1, \\dots, f_E$:
$$\\mathbf{H} = \\mathbf{X} \\mathbf{W}_g + \\mathbf{b}_g \\quad \\in \\mathbb{R}^{T \\times E}, \\quad \\mathcal{K}_t = \\text{TopK}(\\mathbf{H}_t, k), \\quad g_{t, i} = \\frac{\\exp(H_{t, i})}{\\sum_{j \\in \\mathcal{K}_t} \\exp(H_{t, j})}$$
$$\\mathbf{y}_t = \\sum_{i \\in \\mathcal{K}_t} g_{t, i} f_i(\\mathbf{x}_t), \\quad \\mathcal{L}_{\\text{aux}} = \\alpha E \\sum_{i=1}^E f_i P_i, \\quad \\mathcal{L}_z = \\frac{\\beta}{T} \\sum_{t=1}^T \\left(\\log \\sum_{i=1}^E e^{H_{t, i}}\\right)^2$$

### 2. Expert Choice (Inverted Routing) & Soft MoE
- **Expert Choice:** Transpose router scores $\\mathbf{G} = \\text{softmax}(\\mathbf{H})^\\top \\in \\mathbb{R}^{E \\times T}$. Expert $e$ selects top-$C$ tokens: $\\mathbf{y}_t = \\sum_{e: t \\in \\mathcal{T}_e} G_{e, t} f_e(\\mathbf{x}_t)$, guaranteeing exact $C$ workload per expert.
- **Soft MoE:** Replaces discrete dispatch with continuous slot mixing:
$$\\mathbf{\\Phi} = \\text{softmax}(\\mathbf{X} \\mathbf{W}_{\\text{disp}}, \\text{dim}=0), \\quad \\tilde{\\mathbf{X}} = \\mathbf{\\Phi}^\\top \\mathbf{X}, \\quad \\tilde{\\mathbf{Y}}_j = f_e(\\tilde{\\mathbf{X}}_j), \\quad \\mathbf{\\Psi} = \\text{softmax}(\\mathbf{X} \\mathbf{W}_{\\text{comb}}, \\text{dim}=1), \\quad \\mathbf{Y} = \\mathbf{\\Psi} \\tilde{\\mathbf{Y}}$$

### 3. Asymptotic Time & Memory Complexity
- **Dense FFN:** $\\mathcal{O}(T \\cdot D \\cdot D_{\\text{ff}})$ FLOPs and activations.
- **Top-$K$ MoE:** $\\mathcal{O}(T \\cdot D \\cdot E + T \\cdot k \\cdot D \\cdot D_{\\text{ff}})$ compute, decoupling parameter capacity $\\mathcal{O}(E \\cdot D \\cdot D_{\\text{ff}})$ from per-token compute $\\mathcal{O}(k \\cdot D \\cdot D_{\\text{ff}})$.
- **Soft MoE:** $\\mathcal{O}(T \\cdot M \\cdot D + M \\cdot D \\cdot D_{\\text{ff}})$ compute for $M = E \\cdot p$ total slots."""
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
ckpt_rel = "weights/05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x = weights["input"]
B, S, D = x.shape
num_experts, top_k = 4, 2
print(f"Loaded MoE reference checkpoint on {env.device}: X={x.shape}, Experts={num_experts}, Top-K={top_k}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

class TopKMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.d_model, self.num_experts, self.top_k = d_model, num_experts, top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([ExpertMLP(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        x_flat = x.reshape(-1, self.d_model)
        T = x_flat.shape[0]
        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)

        out = torch.zeros_like(x_flat)
        f_count = torch.zeros(self.num_experts, device=x.device)
        for t in range(T):
            for k in range(self.top_k):
                e = topk_idx[t, k].item()
                f_count[e] += 1.0
                out[t] += topk_weights[t, k] * self.experts[e](x_flat[t:t+1]).squeeze(0)

        aux_loss = self.num_experts * torch.sum((f_count / T) * probs.mean(dim=0))
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1)**2)
        return out.view_as(x), aux_loss.view(1), z_loss.view(1)"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# Quantitative Mathematical Assertions & Cross-Track Parity
topk_moe = TopKMoE(d_model=D, d_ff=32, num_experts=num_experts, top_k=top_k)
with torch.no_grad():
    topk_moe.router.weight.copy_(weights["router_weight"].t())
    topk_moe.router.bias.copy_(weights["router_bias"])
    for e in range(num_experts):
        topk_moe.experts[e].fc1.weight.copy_(weights["experts_w1"][e].t())
        topk_moe.experts[e].fc1.bias.copy_(weights["experts_b1"][e])
        topk_moe.experts[e].fc2.weight.copy_(weights["experts_w2"][e].t())
        topk_moe.experts[e].fc2.bias.copy_(weights["experts_b2"][e])

out_topk, aux_loss, z_loss = topk_moe(x)
err_topk = float(torch.max(torch.abs(out_topk.detach() - weights["topk_output"])))
err_aux = float(torch.abs(aux_loss.detach() - weights["aux_loss"]))
err_z = float(torch.abs(z_loss.detach() - weights["z_loss"]))

assert err_topk < 1e-5, f"Top-K MoE error {err_topk:.2e} exceeds 1e-5"
assert err_aux < 1e-5, f"Aux loss error {err_aux:.2e} exceeds 1e-5"
assert err_z < 1e-5, f"Z-loss error {err_z:.2e} exceeds 1e-5"

# Soft MoE Verification
phi = F.softmax(x.reshape(-1, D) @ weights["soft_disp_weight"] + weights["soft_disp_bias"], dim=0)
psi = F.softmax(x.reshape(-1, D) @ weights["soft_comb_weight"] + weights["soft_comb_bias"], dim=1)
slot_in = phi.t() @ x.reshape(-1, D)
slot_out = torch.stack([topk_moe.experts[j](slot_in[j:j+1]).squeeze(0) for j in range(4)])
out_soft = (psi @ slot_out).view_as(x)
err_soft = float(torch.max(torch.abs(out_soft.detach() - weights["soft_moe_output"])))
assert err_soft < 1e-5, f"Soft MoE error {err_soft:.2e} exceeds 1e-5"

print(f"Parity Verified: TopK={err_topk:.2e}, Aux={err_aux:.2e}, Z={err_z:.2e}, Soft={err_soft:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_topk = benchmark_fn(lambda: topk_moe(x), iters=50, name="Top-K MoE Forward")
print(format_benchmark_table([b_topk]))

logits = x.reshape(-1, D) @ weights["router_weight"] + weights["router_bias"]
probs = F.softmax(logits, dim=-1).detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), dpi=100)
im1 = ax[0].imshow(probs, cmap="Blues", aspect="auto")
ax[0].set_title("Router Token-to-Expert Probabilities")
ax[0].set_xlabel("Expert Index")
ax[0].set_ylabel("Token Index")
plt.colorbar(im1, ax=ax[0])

ax[1].bar(range(num_experts), probs.sum(axis=0), color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
ax[1].set_title("Total Expert Load (Token Distribution)")
ax[1].set_xlabel("Expert Index")
ax[1].set_ylabel("Accumulated Probability")
ax[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Routing Collapse Without Auxiliary Load-Balancing Loss
# Training router without aux loss quickly drives load to a single expert!

router_bad = nn.Linear(D, num_experts)
opt_bad = torch.optim.SGD(router_bad.parameters(), lr=0.1)

# Synthetic training loop favoring arbitrary initial bias
for step in range(40):
    opt_bad.zero_grad()
    h = router_bad(x.reshape(-1, D))
    # Loss only maximizes confidence of chosen top-1 expert without balancing
    loss_unbalanced = -torch.mean(torch.max(F.softmax(h, dim=-1), dim=-1).values)
    loss_unbalanced.backward()
    opt_bad.step()

with torch.no_grad():
    collapsed_probs = F.softmax(router_bad(x.reshape(-1, D)), dim=-1).mean(dim=0)
max_share = collapsed_probs.max().item()
print(f"Top Expert Routing Share without Aux Loss: {max_share*100:.1f}%")
print(f"All Expert Probabilities: {collapsed_probs.tolist()}")
assert max_share > 0.45, "Router must concentrate into dominant expert without aux loss" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Decoupled Parameter Scaling:** MoE layers decouple total parameter capacity from FLOPs per token, enabling trillion-parameter models with compute footprints matching small dense baselines.
2. **Auxiliary Loss Necessity:** Discrete top-$k$ routing suffers from positive feedback routing collapse; auxiliary load-balancing $\\mathcal{L}_{\\text{aux}}$ and router Z-loss $\\mathcal{L}_z$ are non-negotiable stabilizers.
3. **Continuous Soft MoE Alternative:** Soft MoE achieves full differentiability and zero dropped tokens by projecting tokens into expert slots and softly synthesizing tokens from slot outputs.

---
### Cross-Domain Generalisation Matrix
- **Vision Transformers (V-MoE):** Routing image patches to specialized experts based on visual frequency and spatial semantics.
- **Multiphase Fluid Simulation:** Routing fluid grid cells to laminar, turbulent boundary, or shock-wave physics solvers.
- **Genomic Sequence Modeling:** Specializing distinct experts into non-coding intronic DNA, exonic codons, and splice junctions.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [02_mixture_of_experts_top_k_expert_choice_and_soft_moe.md](../../markdown/05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.md)
- **PyTorch 2.x Module:** [02_mixture_of_experts_top_k_expert_choice_and_soft_moe.py](../../pytorch2/05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.py)
- **Functional JAX Module:** [02_mixture_of_experts_top_k_expert_choice_and_soft_moe.py](../../jax/05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.py)
- **Native Rust Candle Track:** [02_mixture_of_experts_top_k_expert_choice_and_soft_moe.rs](../../candle/05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.rs)
- **Hardware WGSL Kernel:** [02_mixture_of_experts_top_k_expert_choice_and_soft_moe.rs](../../kernels/05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.rs)
- **Universal Golden SafeTensors:** [02_mixture_of_experts_top_k_expert_choice_and_soft_moe.safetensors](../../weights/05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Topic: 01. Residual MLP, MLP-Mixer & gMLP](01_residual_mlp_mlp_mixer_and_gmlp.ipynb) | [Category Index](../README.md) | [Next Topic: 03. DeepSeekMoE & Mixture of Depths](03_deepseek_moe_aux_loss_free_and_mixture_of_depths.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/01_mlp_moe_soups_and_kan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "02_mixture_of_experts_top_k_expert_choice_and_soft_moe.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
