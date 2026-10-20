"""Generate 8-cell educational notebook for Category 05 Topic 03:
03_deepseek_moe_aux_loss_free_and_mixture_of_depths.ipynb in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 03. DeepSeekMoE: Fine-Grained Experts, Aux-Loss-Free Balancing & Mixture of Depths

- **Category:** 05. Architectures / 01. MLP, MoE, Soups & KAN
- **Lineage & Citations:** Damai Dai et al. (arXiv 2024) *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*; David Raposo et al. (DeepMind 2024) *Mixture-of-Depths: Dynamically allocating compute in language models*; DeepSeek-AI (2024) *DeepSeek-V3 Technical Report: Auxiliary-Loss-Free Load Balancing*.
- **Purpose:** Segments intermediate capacity into fine-grained slices with dedicated shared experts, replaces gradient-distorting auxiliary losses with dynamic bias tracking, and routes compute across depth via fixed-capacity token skipping.
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe](02_mixture_of_experts_top_k_expert_choice_and_soft_moe.ipynb), [03_building_blocks_and_losses/12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting](../../03_building_blocks_and_losses/12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling](04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.ipynb), [05_architectures/04_transformers_and_heads/01_multihead_grouped_query_and_multi_query_attention](../04_transformers_and_heads/01_multihead_grouped_query_and_multi_query_attention.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Never inject the dynamic balancing bias $b_i$ into the softmax gating weights applied to expert outputs; $b_i$ must *strictly* participate in `topk(s + b, k)` index selection, while gating weights must normalize over unbiased raw logits to avoid representation distortion and gradient corruption. Never perform unmasked sequence-wide top-$C$ token selection in Mixture of Depths (MoD) during autoregressive causal generation, as sorting across the entire sequence dimension leaks future token representations into earlier positions."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. DeepSeekMoE: Fine-Grained Segmentation & Dedicated Shared Experts
Given an input token representation $\\mathbf{x} \\in \\mathbb{R}^{D}$:
$$\\mathbf{y}_t = \\sum_{i=1}^{N_s} f_i^{\\text{shared}}(\\mathbf{x}_t) + \\sum_{j \\in \\mathcal{K}_t} g_{t, j} f_j^{\\text{routed}}(\\mathbf{x}_t)$$
Where intermediate dimension $D_{\\text{ff}}^{\\text{fine}} = D_{\\text{ff}} / m$ creates $N_r = m \\cdot E$ fine-grained experts, yielding $\\binom{N_r}{k}$ combinatoric specializations for the same FLOP budget.

### 2. Auxiliary-Loss-Free Load Balancing (DeepSeek-V3)
Rather than adding an auxiliary loss $\\mathcal{L}_{\\text{aux}}$ that degrades primary language modeling perplexity, maintain a dynamic bias $\\mathbf{b}_{\\text{dyn}} \\in \\mathbb{R}^{N_r}$ updated outside autograd:
$$\\mathcal{K}_t = \\text{TopK}(\\mathbf{s}_t + \\mathbf{b}_{\\text{dyn}}, k), \\quad g_{t, j} = \\frac{\\exp(s_{t, j})}{\\sum_{l \\in \\mathcal{K}_t} \\exp(s_{t, l})}$$
$$b_{\\text{dyn}, i} \\leftarrow b_{\\text{dyn}, i} + \\gamma \\cdot \\text{sign}(\\bar{c} - c_i)$$

### 3. Mixture of Depths (MoD) & Asymptotic Complexity
- **MoD Routing:** Select top-$C$ tokens ($C = \\lfloor \\rho \\cdot S \\rfloor$) based on router scores $r_t = \\sigma(\\mathbf{x}_t \\mathbf{w}_{\\text{mod}} + b_{\\text{mod}})$:
$$\\mathbf{y}_t = \\begin{cases} \\mathbf{x}_t + r_t \\cdot \\text{Block}(\\mathbf{x}_t), & t \\in \\text{TopK}(r, C) \\\\ \\mathbf{x}_t, & \\text{otherwise} \\end{cases}$$
- **Asymptotic Complexity:**
  - Standard Dense Layer: $\\mathcal{O}(S \\cdot D \\cdot D_{\\text{ff}})$
  - Mixture of Depths (MoD): $\\mathcal{O}(S \\cdot D + C \\cdot D \\cdot D_{\\text{ff}})$, achieving a factor of $C / S = \\rho$ FLOP reduction while preserving deterministic tensor shapes."""
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
ckpt_rel = "weights/05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x = weights["input"]
B, S, D = x.shape
num_shared, num_routed, top_k = 1, 4, 2
print(f"Loaded DeepSeekMoE checkpoint on {env.device}: X={x.shape}, Shared={num_shared}, Routed={num_routed}, Top-K={top_k}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

class DeepSeekMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_shared: int = 1, num_routed: int = 4, top_k: int = 2):
        super().__init__()
        self.d_model, self.num_shared, self.num_routed, self.top_k = d_model, num_shared, num_routed, top_k
        self.shared_experts = nn.ModuleList([ExpertMLP(d_model, d_ff) for _ in range(num_shared)])
        self.routed_experts = nn.ModuleList([ExpertMLP(d_model, d_ff) for _ in range(num_routed)])
        self.router = nn.Linear(d_model, num_routed)
        self.register_buffer("b_dyn", torch.zeros(num_routed))

    def forward(self, x: torch.Tensor):
        x_flat = x.reshape(-1, self.d_model)
        T = x_flat.shape[0]

        # 1. Evaluate all shared experts unconditionally
        out = sum(shared(x_flat) for shared in self.shared_experts)

        # 2. Router logits and biased top-k selection
        raw_logits = self.router(x_flat)
        biased_logits = raw_logits + self.b_dyn
        _, topk_idx = torch.topk(biased_logits, k=self.top_k, dim=-1)

        # Invariant: Gating weights computed on UNBIASED raw logits!
        raw_topk_logits = torch.gather(raw_logits, dim=-1, index=topk_idx)
        topk_weights = F.softmax(raw_topk_logits, dim=-1)

        counts = torch.zeros(self.num_routed, device=x.device)
        for t in range(T):
            for k in range(self.top_k):
                e = topk_idx[t, k].item()
                counts[e] += 1.0
                out[t] += topk_weights[t, k] * self.routed_experts[e](x_flat[t:t+1]).squeeze(0)

        return out.view_as(x), counts

    def update_bias(self, counts: torch.Tensor, gamma: float = 0.1):
        with torch.no_grad():
            self.b_dyn.add_(gamma * torch.sign(counts.mean() - counts))
        return self.b_dyn

class MixtureOfDepths(nn.Module):
    def __init__(self, d_model: int, block: nn.Module, capacity: int = 2):
        super().__init__()
        self.d_model, self.block, self.capacity = d_model, block, capacity
        self.router = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        b_sz, s_len, _ = x.shape
        r = torch.sigmoid(self.router(x).squeeze(-1))
        _, top_idx = torch.topk(r, k=self.capacity, dim=-1)
        out = x.clone()
        for b in range(b_sz):
            for c in range(self.capacity):
                s = top_idx[b, c].item()
                tok = x[b, s:s+1]
                out[b, s] = tok.squeeze(0) + r[b, s] * self.block(tok).squeeze(0)
        return out"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# Quantitative Mathematical Assertions & Cross-Track Parity
ds_moe = DeepSeekMoE(d_model=D, d_ff=16, num_shared=num_shared, num_routed=num_routed, top_k=top_k)
with torch.no_grad():
    ds_moe.b_dyn.copy_(weights["b_dyn"])
    ds_moe.router.weight.copy_(weights["router_weight"].t())
    ds_moe.router.bias.copy_(weights["router_bias"])
    for s in range(num_shared):
        ds_moe.shared_experts[s].fc1.weight.copy_(weights["shared_w1"][s].t())
        ds_moe.shared_experts[s].fc1.bias.copy_(weights["shared_b1"][s])
        ds_moe.shared_experts[s].fc2.weight.copy_(weights["shared_w2"][s].t())
        ds_moe.shared_experts[s].fc2.bias.copy_(weights["shared_b2"][s])
    for r in range(num_routed):
        ds_moe.routed_experts[r].fc1.weight.copy_(weights["routed_w1"][r].t())
        ds_moe.routed_experts[r].fc1.bias.copy_(weights["routed_b1"][r])
        ds_moe.routed_experts[r].fc2.weight.copy_(weights["routed_w2"][r].t())
        ds_moe.routed_experts[r].fc2.bias.copy_(weights["routed_b2"][r])

out_ds, counts = ds_moe(x)
err_ds = float(torch.max(torch.abs(out_ds.detach() - weights["deepseek_moe_output"])))
assert err_ds < 1e-5, f"DeepSeekMoE parity error {err_ds:.2e} exceeds 1e-5"

b_updated = ds_moe.update_bias(counts, gamma=0.1)
err_b = float(torch.max(torch.abs(b_updated.detach() - weights["b_dyn_updated"])))
assert err_b < 1e-5, f"Dynamic bias parity error {err_b:.2e} exceeds 1e-5"

# MoD Verification
mod = MixtureOfDepths(d_model=D, block=ds_moe.shared_experts[0], capacity=2)
with torch.no_grad():
    mod.router.weight.copy_(weights["mod_weight"].t())
    mod.router.bias.copy_(weights["mod_bias"])

out_mod = mod(x)
err_mod = float(torch.max(torch.abs(out_mod.detach() - weights["mod_output"])))
assert err_mod < 1e-5, f"Mixture of Depths error {err_mod:.2e} exceeds 1e-5"

print(f"Parity Verified: DeepSeekMoE={err_ds:.2e}, DynamicBias={err_b:.2e}, MoD={err_mod:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_ds = benchmark_fn(lambda: ds_moe(x), iters=50, name="DeepSeekMoE Forward")
b_mod = benchmark_fn(lambda: mod(x), iters=50, name="Mixture of Depths Forward")
print(format_benchmark_table([b_ds, b_mod]))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), dpi=100)
ax[0].bar(range(num_routed), counts.cpu().numpy(), color="#2ca02c", alpha=0.8)
ax[0].axhline(counts.mean().item(), color="red", linestyle="--", label="Target Average Load")
ax[0].set_title("Expert Token Load Distribution")
ax[0].set_xlabel("Routed Expert Index")
ax[0].set_ylabel("Tokens Routed")
ax[0].legend()
ax[0].grid(True, linestyle="--", alpha=0.5)

b_vals = b_updated.cpu().numpy()
ax[1].stem(range(num_routed), b_vals)
ax[1].set_title("Updated Dynamic Router Bias $b_{dyn}$")
ax[1].set_xlabel("Routed Expert Index")
ax[1].set_ylabel("Bias Delta")
ax[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Leaking Dynamic Bias into Output Gating Multipliers
# If b_dyn is added into softmax gating weights, unpopular experts receive artificially
# magnified gradients, corrupting the feature representation manifold!

raw_logits = ds_moe.router(x.reshape(-1, D))
b_bad = torch.tensor([5.0, -5.0, 0.0, 0.0])  # extreme bias

# 1. Correct (DeepSeek-V3): b_dyn strictly inside top-k selection, UNBIASED weights
_, idx = torch.topk(raw_logits + b_bad, k=2, dim=-1)
w_correct = F.softmax(torch.gather(raw_logits, -1, idx), dim=-1)

# 2. Pathological: b_dyn included in gating weights
w_flawed = F.softmax(torch.gather(raw_logits + b_bad, -1, idx), dim=-1)

divergence = torch.max(torch.abs(w_correct - w_flawed)).item()
print(f"Unbiased Weight Sample (Correct): {w_correct[0].detach().tolist()}")
print(f"Biased Weight Sample (Flawed):   {w_flawed[0].detach().tolist()}")
print(f"Weight Distortion Magnitude:     {divergence:.4f}")
assert divergence > 0.4, "Leaking dynamic bias into gating must severely distort weights!" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Fine-Grained Specialization:** Dividing total FFN capacity into smaller slices with dedicated shared experts creates orders of magnitude more combinatoric pathways while anchoring universal knowledge.
2. **Auxiliary-Loss-Free Stability:** Updating dynamic routing bias outside autograd cleanly achieves load balance without degrading primary task perplexity.
3. **Depth-Wise Compute Conditioning:** Mixture of Depths dynamically allocates compute along the depth axis with fixed tensor capacities, reducing FLOPs with zero kernel reallocation overhead.

---
### Cross-Domain Generalisation Matrix
- **Large Language Models (DeepSeek-V3):** Shared experts model core syntax, grammar, and reasoning priors, while routed experts specialize into programming, mathematics, and domains.
- **Audio Encoders & Speech Processing:** Shared experts process harmonic vowel formants, while routed experts handle burst transients and background soundscapes.
- **Biomolecular Graph Transformers:** Shared experts model peptide covalent backbones, while routed experts specialize in ligand docking cavities and hydrophobic pockets.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [03_deepseek_moe_aux_loss_free_and_mixture_of_depths.md](../../markdown/05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.md)
- **PyTorch 2.x Module:** [03_deepseek_moe_aux_loss_free_and_mixture_of_depths.py](../../pytorch2/05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.py)
- **Functional JAX Module:** [03_deepseek_moe_aux_loss_free_and_mixture_of_depths.py](../../jax/05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.py)
- **Native Rust Candle Track:** [03_deepseek_moe_aux_loss_free_and_mixture_of_depths.rs](../../candle/05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.rs)
- **Hardware WGSL Kernel:** [03_deepseek_moe_aux_loss_free_and_mixture_of_depths.rs](../../kernels/05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.rs)
- **Universal Golden SafeTensors:** [03_deepseek_moe_aux_loss_free_and_mixture_of_depths.safetensors](../../weights/05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Topic: 02. Mixture of Experts](02_mixture_of_experts_top_k_expert_choice_and_soft_moe.ipynb) | [Category Index](../README.md) | [Next Topic: 04. Soup of Experts & SMEAR](04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/01_mlp_moe_soups_and_kan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "03_deepseek_moe_aux_loss_free_and_mixture_of_depths.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
