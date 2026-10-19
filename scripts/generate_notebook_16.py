"""Script to procedurally generate the strict 8-cell notebook for Topic 16.

Ensures total lines <= 220, valid JSON, cell IDs, and all semantic requirements.
"""

import json
from pathlib import Path

cells = []

# Cell 1: Header, Citations, Purpose & Topic Bridges
c1 = [
    "# Training Optimisations: Progressive Stacking, Data Pruning, MRL, and Plasticity\n",
    "\n",
    "### Historical Lineage & Theoretical Foundations\n",
    "- **Matryoshka Representation Learning (MRL)** (*Kusupati et al., 2022*): Nested multi-scale representation learning encoding coarse-to-fine semantics across prefix dimensions $[d_1 \\subset d_2 \\subset \\dots \\subset d_K]$.\n",
    "- **EL2N & Early Pruning** (*Paul et al., 2021*): Error $L_2$ norm $\\|p(x) - y\\|_2$ at early training accurately identifies hard, learnable samples, enabling $50\\%$ data pruning without accuracy loss.\n",
    "- **RHO-Loss** (*Mindermann et al., 2022*): Reducible Holdout Loss $\\mathcal{L}_\\text{model}(x) - \\mathcal{L}_\\text{ref}(x)$ discards irreducible label noise that misleads naive loss-based pruning.\n",
    "- **Plasticity Loss & ReDo** (*D逼et et al., 2023; Sokar et al., 2023*): Deep networks lose capacity (plasticity) under continual training; Recycling Dormant Neurons (ReDo) reinitializes inactive units.\n",
    "\n",
    "> [!WARNING]\n",
    "> **Decision Boundary (When NOT to Use Naive Loss Pruning):** Never prune datasets using raw sample loss alone when labels contain noise; noisy/corrupted samples exhibit the highest training loss and will disproportionately dominate the retained set. Use RHO-Loss with a clean reference model instead.\n",
    "\n",
    "**Topic Connections:** `🔗 Upstream`: [Grokking & Edge of Stability](01_grokking_phase_transitions_edge_of_stability_and_implicit_bias.ipynb) | `🔗 Downstream`: [Self-Optimising Hypergradients](03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting.ipynb)\n"
]
cells.append({"cell_type": "markdown", "id": "cell_0", "metadata": {}, "source": c1})

# Cell 2: Mathematical Foundations & Governing Equations
c2 = [
    "## 2. Mathematical Foundations & Governing Equations\n",
    "\n",
    "1. **Matryoshka Prefix Projection:** Multi-scale loss over prefix dimensions $\\mathcal{D} = \\{d_1, \\dots, d_K\\}$:\n",
    "$$\\hat{y}^{(m)} = z_{1:d_m} W_{1:d_m, :} + b, \\quad \\mathcal{L}_\\text{MRL} = \\sum_{m=1}^K c_m \\mathcal{L}(\\hat{y}^{(m)}, y)$$\n",
    "2. **EL2N Score (Error $L_2$ Norm):** At early training epoch $t_0 \\ll T$:\n",
    "$$\\text{EL2N}(x, y) = \\|p_{\\theta_{t_0}}(x) - y\\|_2 = \\sqrt{\\sum_{k=1}^C (p_k(x) - y_k)^2}$$\n",
    "3. **RHO-Loss (Reducible Holdout Loss):** Filtering irreducible noise via reference model $\\theta_\\text{ref}$:\n",
    "$$\\text{RHO}(x) = \\mathcal{L}(f_\\theta(x), y) - \\mathcal{L}(f_{\\theta_\\text{ref}}(x), y)$$\n",
    "4. **ReDo Plasticity & Dormant Score:** Mean activation score across batch $\\mathcal{B}$ and threshold $\\tau$:\n",
    "$$s_j = \\frac{1}{|\\mathcal{B}|} \\sum_{i \\in \\mathcal{B}} |a_i^{(j)}|, \\quad \\text{dormant}_j = \\mathbb{I}(s_j < \\tau)$$\n",
    "5. **Asymptotic Complexity:** MRL slice $\\mathcal{O}(B d_m C)$; EL2N $\\mathcal{O}(B C)$; ReDo tracking $\\mathcal{O}(B D)$.\n"
]
cells.append({"cell_type": "markdown", "id": "cell_1", "metadata": {}, "source": c2})

# Cell 3: Deterministic Setup & Micro-Data Generator
c3 = [
    "import sys\n",
    "from pathlib import Path\n",
    "import numpy as np, torch, torch.nn as nn\n",
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
    "ckpt_path = repo_root / \"weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/02_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity.safetensors\"\n",
    "weights, test_in, test_out, meta = core.parity.load_reference_checkpoint(ckpt_path, device=device)\n",
    "print(f\"Device: {device} | Checkpoint: {ckpt_path.name} | Loaded {len(weights)} tensors\")\n"
]
cells.append({"cell_type": "code", "execution_count": None, "id": "cell_2", "metadata": {}, "outputs": [], "source": c3})

# Cell 4: First-Principles Implementation & Native Parity
c4 = [
    "class LinearAnchor(nn.Module):\n",
    "    def __init__(self, in_features: int = 8, out_features: int = 4):\n",
    "        super().__init__()\n",
    "        self.fc = nn.Linear(in_features, out_features)\n",
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
    "        return self.fc(x)\n",
    "\n",
    "def mrl_slice_forward(z: torch.Tensor, w: torch.Tensor, b: torch.Tensor, dim_slice: int) -> torch.Tensor:\n",
    "    return z[:, :dim_slice] @ w[:dim_slice, :] + b\n",
    "\n",
    "def compute_el2n(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:\n",
    "    return torch.norm(probs - targets, p=2, dim=-1)\n",
    "\n",
    "def compute_rho_loss(loss_model: torch.Tensor, loss_ref: torch.Tensor) -> torch.Tensor:\n",
    "    return loss_model - loss_ref\n",
    "\n",
    "def detect_dormant_neurons(activations: torch.Tensor, tau: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:\n",
    "    mean_act = torch.mean(torch.abs(activations), dim=0)\n",
    "    dormant_mask = (mean_act < tau).to(torch.float32)\n",
    "    return mean_act, dormant_mask\n",
    "print(\"Training optimisations routines defined successfully.\")\n"
]
cells.append({"cell_type": "code", "execution_count": None, "id": "cell_3", "metadata": {}, "outputs": [], "source": c4})

# Cell 5: Quantitative Mathematical Assertions & Parity
c5 = [
    "anchor = LinearAnchor(8, 4).to(device)\n",
    "anchor.fc.weight.data.copy_(weights['anchor_w'].T)\n",
    "anchor.fc.bias.data.copy_(weights['anchor_b'])\n",
    "\n",
    "out_d2 = mrl_slice_forward(weights['z_emb'], weights['mrl_w'], weights['mrl_b'], dim_slice=2)\n",
    "out_d4 = mrl_slice_forward(weights['z_emb'], weights['mrl_w'], weights['mrl_b'], dim_slice=4)\n",
    "out_d8 = mrl_slice_forward(weights['z_emb'], weights['mrl_w'], weights['mrl_b'], dim_slice=8)\n",
    "err_mrl = max(float(torch.max(torch.abs(out_d2 - weights['golden_logits_d2'])).item()),\n",
    "              float(torch.max(torch.abs(out_d4 - weights['golden_logits_d4'])).item()),\n",
    "              float(torch.max(torch.abs(out_d8 - weights['golden_logits_d8'])).item()))\n",
    "\n",
    "el2n = compute_el2n(weights['p_probs'], weights['y_targets'])\n",
    "err_el2n = float(torch.max(torch.abs(el2n - weights['golden_el2n'])).item())\n",
    "\n",
    "rho = compute_rho_loss(weights['loss_model'], weights['loss_ref'])\n",
    "err_rho = float(torch.max(torch.abs(rho - weights['golden_rho_loss'])).item())\n",
    "\n",
    "mean_act, mask = detect_dormant_neurons(weights['activations'], tau=0.15)\n",
    "err_redo = max(float(torch.max(torch.abs(mean_act - weights['mean_act'])).item()),\n",
    "               float(torch.max(torch.abs(mask - weights['dormant_mask'])).item()))\n",
    "\n",
    "anchor_out = anchor(weights['anchor_x'])\n",
    "err_anchor = float(torch.max(torch.abs(anchor_out - weights['golden_anchor_out'])).item())\n",
    "\n",
    "for name, err in [('MRL (2/4/8)', err_mrl), ('EL2N', err_el2n), ('RHO-Loss', err_rho), ('ReDo', err_redo), ('Anchor', err_anchor)]:\n",
    "    assert err < 1e-5, f'{name} invariant failed: {err:.2e}'\n",
    "print(f'All 5 invariants verified! MRL: {err_mrl:.2e}, EL2N: {err_el2n:.2e}, RHO: {err_rho:.2e}, ReDo: {err_redo:.2e}, Anchor: {err_anchor:.2e}')\n"
]
cells.append({"cell_type": "code", "execution_count": None, "id": "cell_4", "metadata": {}, "outputs": [], "source": c5})

# Cell 6: Performance Profile & Diagnostic Visualizations
c6 = [
    "track_fns = {\n",
    "    'Linear Anchor': lambda: anchor(weights['anchor_x']),\n",
    "    'MRL Slice (d=4)': lambda: mrl_slice_forward(weights['z_emb'], weights['mrl_w'], weights['mrl_b'], 4),\n",
    "    'EL2N Score': lambda: compute_el2n(weights['p_probs'], weights['y_targets']),\n",
    "    'RHO-Loss Score': lambda: compute_rho_loss(weights['loss_model'], weights['loss_ref']),\n",
    "    'ReDo Dormant Detection': lambda: detect_dormant_neurons(weights['activations'], 0.15),\n",
    "}\n",
    "bench_results = core.benchmark.compare_tracks(track_fns, warmup=3, iters=15, batch_size=4)\n",
    "print(core.benchmark.format_benchmark_table(bench_results))\n"
]
cells.append({"cell_type": "code", "execution_count": None, "id": "cell_5", "metadata": {}, "outputs": [], "source": c6})

# Cell 7: Pathological Misuse vs. Corrected Implementation
c7 = [
    "# Pathological Anti-Pattern: Naive loss pruning retains corrupted label noise over clean learnable samples\n",
    "# Sample 0: Clean hard sample. Sample 1: Corrupted label noise (irreducible).\n",
    "loss_curr = np.array([1.8, 3.2])  # Both have high training loss\n",
    "loss_clean_ref = np.array([0.2, 3.1]) # Reference model easily predicts sample 0, struggles on corrupted sample 1\n",
    "\n",
    "naive_selected = int(np.argmax(loss_curr)) # Naive selects sample 1 (corrupted noise!)\n",
    "rho_scores = loss_curr - loss_clean_ref    # RHO: sample 0 has 1.6 reducible loss, sample 1 has 0.1\n",
    "rho_selected = int(np.argmax(rho_scores)) # RHO selects sample 0 (genuine learning signal!)\n",
    "\n",
    "print(f'[FAILURE DEMONSTRATED] Naive loss pruning chose sample {naive_selected} (irreducible label noise)!')\n",
    "print(f'[CORRECT] RHO-Loss chose sample {rho_selected} with reducible gain {rho_scores[rho_selected]:.2f}!')\n",
    "assert naive_selected == 1 and rho_selected == 0, 'RHO-loss vs naive loss contrast assertion failed.'\n",
    "print('[REMEDY] Use RHO-loss with holdout reference model to discount irreducible aleatoric uncertainty.')\n"
]
cells.append({"cell_type": "code", "execution_count": None, "id": "cell_6", "metadata": {}, "outputs": [], "source": c7})

# Cell 8: Takeaways, Cross-Domain Isomorphism & Footer
c8 = [
    "## 8. Algorithmic Takeaways & Cross-Domain Isomorphisms\n",
    "\n",
    "- **Takeaways**: MRL embeds hierarchical granularity in a single forward pass; EL2N and RHO-loss enable compute-efficient dataset pruning; ReDo restores lost plasticity in continual regimes.\n",
    "\n",
    "| Scientific Domain | Matryoshka Representation (MRL) | Data Pruning / RHO-Loss | Dormant Neuron Recycling (ReDo) |\n",
    "| :--- | :--- | :--- | :--- |\n",
    "| **Vector Search & Retrieval** | 8x smaller index size with sub-vector re-ranking | Deduplicate dense web crawls without semantic loss | Prevent index collapse on streaming document additions |\n",
    "| **Genomic Embeddings** | Low-dim coarse taxon match, high-dim variant resolution | Filter non-coding sequencing artifacts | Revive dead pathways in continuous variant learning |\n",
    "| **Multimodal Vision-Language** | Fast thumbnail search followed by high-res grounding | Exclude miscaptioned noisy pairs from pretraining | Re-allocate unspecialized visual tokens during fine-tuning |\n",
    "\n",
    "---\n",
    "**Navigation:** [← Grokking & Edge of Stability](01_grokking_phase_transitions_edge_of_stability_and_implicit_bias.ipynb) | [↑ Master Index](../../README.md) | [Self-Optimising Hypergradients →](03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting.ipynb)\n"
]
cells.append({"cell_type": "markdown", "id": "cell_7", "metadata": {}, "source": c8})

nb_data = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_dir = Path("notebooks/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "02_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity.ipynb"

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(nb_data, f, indent=1)

print(f"Generated notebook at {out_file}")
