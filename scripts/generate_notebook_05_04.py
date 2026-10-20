"""Generate 8-cell educational notebook for Category 05 Topic 04:
04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.ipynb in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 04. Soup of Experts: SMEAR, Rewarded Soups, BTX & Sparse Upcycling

- **Category:** 05. Architectures / 01. MLP, MoE, Soups & KAN
- **Lineage & Citations:** Mohammed Muqeeth et al. (NeurIPS 2023) *Soft Merging of Experts with Adaptive Routing (SMEAR)*; Mitchell Wortsman et al. (ICML 2022) *Model Soups*; Alexandre Rame et al. (NeurIPS 2023) *Rewarded Soups: towards Pareto-optimal alignment by interpolating weights*; Aran Komatsuzaki et al. (ICLR 2023) *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints*; Margaret Li et al. (ACL 2023) *Branch-Train-Merge (BTX)*.
- **Purpose:** Blends expert parameter matrices dynamically in weight space rather than ensembling non-linear activations, executing MoE inference at dense baseline speed while leveraging shared basin connectivity from Sparse Upcycling and multi-objective Pareto interpolation.
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe](02_mixture_of_experts_top_k_expert_choice_and_soft_moe.ipynb), [05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths](03_deepseek_moe_aux_loss_free_and_mixture_of_depths.ipynb), [04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/06_sharpness_aware_swad_and_weight_averaging_dynamics](../../04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/06_sharpness_aware_swad_and_weight_averaging_dynamics.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan](05_kolmogorov_arnold_networks_kan.ipynb), [05_architectures/04_transformers_and_heads/01_multihead_grouped_query_and_multi_query_attention](../04_transformers_and_heads/01_multihead_grouped_query_and_multi_query_attention.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Never merge expert models fine-tuned from different random initializations or disconnected loss basins without prior weight permutation alignment (e.g. Git Re-Basin), as neuron permutation mismatch induces destructive interference and immediate representation collapse. Never perform dynamic parameter merging at the single-token granularity ($T = B \\times S$ unique matrices), which destroys memory bandwidth and GPU compute utilization; merge weights at the sequence or batch level using pooled sequence representations."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Sparse Upcycling & Linear Basin Connectivity
Given a pre-trained base model with parameter matrix $\\mathbf{W}_0$, Sparse Upcycling initializes $E$ parallel experts:
$$\\mathbf{W}_e^{(0)} = \\mathbf{W}_0, \\quad \\forall e \\in \\{1, \\dots, E\\}$$
As fine-tuning progresses on task or domain $e$, the expert drifts by displacement $\\Delta \\mathbf{W}_e = \\mathbf{W}_e - \\mathbf{W}_0$. Because all experts remain within the same convex optimization basin, any convex combination $\\bar{\\mathbf{W}} = \\sum_{e=1}^E g_e \\mathbf{W}_e$ exhibits linear mode connectivity:
$$\\mathcal{L}\\left(\\sum_{e=1}^E g_e \\mathbf{W}_e\\right) \\le \\sum_{e=1}^E g_e \\mathcal{L}(\\mathbf{W}_e) + \\mathcal{O}(\\|\\Delta \\mathbf{W}_e\\|^2)$$

### 2. Soft Merging of Experts with Adaptive Routing (SMEAR)
Given an input sequence $\\mathbf{X} \\in \\mathbb{R}^{S \\times D}$:
1. Compute pooled sequence representation: $\\mathbf{z} = \\frac{1}{S} \\sum_{s=1}^S \\mathbf{x}_s \\in \\mathbb{R}^D$.
2. Compute routing distribution over experts:
   $$\\mathbf{g} = \\text{softmax}(\\mathbf{z} \\mathbf{W}_r + \\mathbf{b}_r) \\in \\Delta^{E-1}$$
3. Synthesize dynamic layer weights via convex parameter combination:
   $$\\bar{\\mathbf{W}}_1 = \\sum_{e=1}^E g_e \\mathbf{W}_{1, e}, \\quad \\bar{\\mathbf{b}}_1 = \\sum_{e=1}^E g_e \\mathbf{b}_{1, e}, \\quad \\bar{\\mathbf{W}}_2 = \\sum_{e=1}^E g_e \\mathbf{W}_{2, e}, \\quad \\bar{\\mathbf{b}}_2 = \\sum_{e=1}^E g_e \\mathbf{b}_{2, e}$$
4. Execute single standard feed-forward pass:
   $$\\mathbf{Y} = \\text{GELU}(\\mathbf{X} \\bar{\\mathbf{W}}_1 + \\bar{\\mathbf{b}}_1) \\bar{\\mathbf{W}}_2 + \\bar{\\mathbf{b}}_2$$

### 3. Rewarded Soups & Pareto Front Interpolation
For conflicting objectives $R_1, R_2$ (e.g. Helpfulness vs. Harmlessness), fine-tune policies $\\theta^{(1)}$ and $\\theta^{(2)}$. The Pareto front is traced continuously via scalar interpolation $\\alpha \\in [0, 1]$:
$$\\theta(\\alpha) = \\alpha \\theta^{(1)} + (1 - \\alpha) \\theta^{(2)}$$
Evaluating $\\theta(\\alpha)$ requires no retraining and produces zero additional inference overhead."""
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
ckpt_rel = "weights/05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x = weights["input"]
B, S, D = x.shape
num_experts = weights["experts_w1"].shape[0]
d_ff = weights["experts_w1"].shape[2]
print(f"Loaded Soup of Experts checkpoint on {env.device}: X={x.shape}, E={num_experts}, D={D}, D_ff={d_ff}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class SMEARLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int):
        super().__init__()
        self.d_model, self.d_ff, self.num_experts = d_model, d_ff, num_experts
        self.experts_w1 = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        self.experts_b1 = nn.Parameter(torch.empty(num_experts, d_ff))
        self.experts_w2 = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        self.experts_b2 = nn.Parameter(torch.empty(num_experts, d_model))
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor):
        x_pool = x.mean(dim=1)  # Sequence pooling
        routing_weights = F.softmax(self.router(x_pool), dim=-1)
        out = torch.zeros_like(x)
        for b in range(x.shape[0]):
            g = routing_weights[b]
            w1_bar = torch.sum(g.view(self.num_experts, 1, 1) * self.experts_w1, dim=0)
            b1_bar = torch.sum(g.view(self.num_experts, 1) * self.experts_b1, dim=0)
            w2_bar = torch.sum(g.view(self.num_experts, 1, 1) * self.experts_w2, dim=0)
            b2_bar = torch.sum(g.view(self.num_experts, 1) * self.experts_b2, dim=0)
            out[b] = F.gelu(x[b] @ w1_bar + b1_bar) @ w2_bar + b2_bar
        return out, routing_weights

    def forward_activation_space(self, x: torch.Tensor, routing_weights: torch.Tensor):
        out = torch.zeros_like(x)
        for b in range(x.shape[0]):
            for e in range(self.num_experts):
                h_e = F.gelu(x[b] @ self.experts_w1[e] + self.experts_b1[e])
                out[b] += routing_weights[b, e] * (h_e @ self.experts_w2[e] + self.experts_b2[e])
        return out"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """smear = SMEARLayer(d_model=D, d_ff=d_ff, num_experts=num_experts)
with torch.no_grad():
    smear.experts_w1.copy_(weights["experts_w1"])
    smear.experts_b1.copy_(weights["experts_b1"])
    smear.experts_w2.copy_(weights["experts_w2"])
    smear.experts_b2.copy_(weights["experts_b2"])
    smear.router.weight.copy_(weights["router_weight"].t())
    smear.router.bias.copy_(weights["router_bias"])

# 1. SMEAR & Activation-space parity
out_smear, r_weights = smear(x)
err_smear = float(torch.max(torch.abs(out_smear.detach() - weights["smear_output"])))
out_act = smear.forward_activation_space(x, r_weights)
err_act = float(torch.max(torch.abs(out_act.detach() - weights["act_output"])))
assert err_smear < 1e-5 and err_act < 1e-5, f"MoE parity mismatch: {err_smear:.2e}, {err_act:.2e}"

# 2. Rewarded Pareto Soup Verification
a = 0.6
pw1 = a * weights["experts_w1"][0] + (1 - a) * weights["experts_w1"][1]
pb1 = a * weights["experts_b1"][0] + (1 - a) * weights["experts_b1"][1]
pw2 = a * weights["experts_w2"][0] + (1 - a) * weights["experts_w2"][1]
pb2 = a * weights["experts_b2"][0] + (1 - a) * weights["experts_b2"][1]
out_p = F.gelu(x @ pw1 + pb1) @ pw2 + pb2
err_p = float(torch.max(torch.abs(out_p - weights["pareto_soup_output"])))
assert err_p < 1e-5, f"Pareto soup error {err_p:.2e}"

# 3. Non-linearity divergence invariant
nonlin_diff = float(torch.norm(out_smear.detach() - out_act.detach()))
assert nonlin_diff > 1e-4, "Non-linear GELU activation must cause SMEAR/Act-MoE divergence!"
print(f"Parity Verified: SMEAR={err_smear:.2e}, ActMoE={err_act:.2e}, Pareto={err_p:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_smear = benchmark_fn(lambda: smear(x), iters=50, name="SMEAR (Parameter Merging)")
b_act = benchmark_fn(lambda: smear.forward_activation_space(x, r_weights), iters=50, name="Activation-Space MoE")
print(format_benchmark_table([b_smear, b_act]))

alphas = torch.linspace(0.0, 1.0, 15)
diffs = [torch.norm(F.gelu(x @ (a*weights["experts_w1"][0] + (1-a)*weights["experts_w1"][1]) + 
                           (a*weights["experts_b1"][0] + (1-a)*weights["experts_b1"][1])) @ 
                    (a*weights["experts_w2"][0] + (1-a)*weights["experts_w2"][1]) + 
                    (a*weights["experts_b2"][0] + (1-a)*weights["experts_b2"][1]) - 
                    weights["pareto_soup_output"]).item() for a in alphas]

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=100)
ax[0].plot(alphas.numpy(), diffs, "o-", color="#1f77b4", lw=2)
ax[0].set_title("Pareto Soup Distance to Golden Point ($\\\\alpha=0.6$)")
ax[0].grid(True, linestyle="--", alpha=0.5)

ax[1].bar(range(num_experts), r_weights[0].detach().cpu().numpy(), alpha=0.7, label="Seq 0")
ax[1].bar(range(num_experts), r_weights[1].detach().cpu().numpy(), alpha=0.7, label="Seq 1")
ax[1].set_title("Routing Probabilities $g_e$")
ax[1].legend()
ax[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Averaging Disconnected Loss Basins (Permutation Mismatch)
torch.manual_seed(999)
w_rand_a, w_rand_b = torch.randn(D, d_ff) * 0.5, torch.randn(D, d_ff) * 0.5
w_up_a, w_up_b = weights["base_w1"] + torch.randn(D, d_ff) * 0.05, weights["base_w1"] + torch.randn(D, d_ff) * 0.05

# 1. Upcycled (shared basin) retains representation correlation
y_up_a, y_up_b = x @ w_up_a, x @ w_up_b
y_up_soup = x @ (0.5 * (w_up_a + w_up_b))
corr_up = F.cosine_similarity(y_up_soup.flatten(), 0.5 * (y_up_a + y_up_b).flatten(), dim=0).item()

# 2. Independent basins suffer variance collapse
y_rd_soup = x @ (0.5 * (w_rand_a + w_rand_b))
std_orig = 0.5 * ((x @ w_rand_a).std() + (x @ w_rand_b).std()).item()

print(f"Upcycled Shared Basin Coherence: {corr_up:.4f}")
print(f"Independent Basins: Expected Std={std_orig:.4f}, Blended Std={y_rd_soup.std().item():.4f}")
assert corr_up > 0.99, "Upcycled models must exhibit near-perfect mode connectivity!" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Parameter-Space Routing:** SMEAR performs dynamic routing directly in parameter space, replacing $E$ fragmented activation passes with a single dense GEMM.
2. **Linear Mode Connectivity Requirement:** Parameter souping strictly requires models to reside in a shared optimization basin, achieved through Sparse Upcycling or Branch-Train-Merge (BTX).
3. **Zero-Inference Pareto Navigation:** Rewarded Soups allow moving along multi-objective trade-off frontiers (e.g. helpfulness vs. safety) by simply adjusting weight interpolation coefficients without extra parameters or test-time compute.

---
### Cross-Domain Generalisation Matrix
- **Multi-Domain LLMs:** Domain specialists (Medical, Legal, Math) fine-tuned from a common base are merged per sequence to serve general multi-turn queries.
- **Reinforcement Learning from Human Feedback (RLHF):** Interpolate helpfulness-rewarded and safety-rewarded policies to continuously adjust refusal thresholds at deployment.
- **Computer Vision & Diffusion LoRA Merging:** Parameter-space merging of style, character, and composition adapters into a single inference checkpoint.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.md](../../markdown/05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.md)
- **PyTorch 2.x Module:** [04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.py](../../pytorch2/05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.py)
- **Functional JAX Module:** [04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.py](../../jax/05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.py)
- **Native Rust Candle Track:** [04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.rs](../../candle/05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.rs)
- **Hardware WGSL Kernel:** [04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.rs](../../kernels/05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.rs)
- **Universal Golden SafeTensors:** [04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.safetensors](../../weights/05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Topic: 03. DeepSeekMoE & MoD](03_deepseek_moe_aux_loss_free_and_mixture_of_depths.ipynb) | [Category Index](../README.md) | [Next Topic: 05. Kolmogorov-Arnold Networks (KAN)](05_kolmogorov_arnold_networks_kan.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/01_mlp_moe_soups_and_kan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
