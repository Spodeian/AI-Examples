"""Script to build Topic 06 Category 01 notebook and update Topic 05 notebook navigation link."""

import json
from pathlib import Path


def update_topic_05_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "05_parametric_vs_nonparametric_and_neural_processes.ipynb"
    )
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cell_8 = nb["cells"][7]
    new_source = []
    for line in cell_8["source"]:
        if "Next: 06. AI by Purpose: Prediction vs Prescription vs Control vs Combinatorial vs Symbolic (Upcoming in Category 01)" in line:
            new_line = line.replace(
                "Next: 06. AI by Purpose: Prediction vs Prescription vs Control vs Combinatorial vs Symbolic (Upcoming in Category 01)",
                "[Next: 06. AI by Purpose: Prediction, Prescription, Control, Combinatorial, Symbolic](06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.ipynb)",
            )
            new_source.append(new_line)
        else:
            new_source.append(line)
    cell_8["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated Topic 05 notebook navigation link.")


def create_topic_06_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.ipynb"
    )

    cell_1_md = r"""# AI by Purpose: Prediction, Prescription, Control, Combinatorial, Symbolic

[![Track](https://img.shields.io/badge/Track-PyTorch_2.x_%7C_JAX_%7C_Candle_%7C_Kernels-blue.svg)](../../README.md)
[![Parity](https://img.shields.io/badge/Parity-Transitive_%3C_1e--5-success.svg)](../../tests/test_cross_track_parity.py)

### Historical Lineage & Seminal Citations
- Fisher (1936, Annals of Eugenics): Foundational formulation of statistical discrimination and predictive conditional estimation.
- Elmachtoub & Grigas (2022, Smart "Predict, Then Optimize", Management Science): Convex surrogate loss (SPO+) directly bridging prediction and downstream linear programming.
- Pearl (2009, Causality: Models, Reasoning, and Inference, Cambridge University Press): Structural causal models and interventional uplift identification.
- Martius & Lampert (2016, Extrapolation and Learning Equations, arXiv): Equation Learner (EQL) architectures for analytical law discovery.

### Problem Statement & Purpose
Investigates the architectural divergence driven by functional **purpose**: minimizing statistical error (Predictive MSE), directly minimizing operational decision regret through combinatorial solvers (Prescriptive SPO+), isolating actionable treatment interventions from baseline confounding (Causal Control), and distilling analytical governing laws via functional basis primitives (Symbolic EQL).

### Decision Boundary: When NOT to Use
> [!WARNING]
> Do NOT use standard Mean Squared Error (MSE) when training neural network cost predictors feeding into downstream Linear Programs (LPs) or combinatorial solvers; symmetric MSE ignores polyhedral decision boundaries, yielding catastrophic decision regret. Use the convex SPO+ surrogate loss. Do NOT use standard predictive regressors for causal policy interventions; observational confounding leads to invalid treatment effect estimates.

### Topic Connections
🔗 Upstream: [05. Parametric vs. Nonparametric & Neural Processes](05_parametric_vs_nonparametric_and_neural_processes.ipynb) | 🔗 Downstream: [07. Input & Output Spaces](07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.ipynb)"""

    cell_2_md = r"""## Mathematical Foundations & First-Principles Derivations

### 1. Decision Regret & Smart "Predict, Then Optimize" (SPO+)
Let $\hat{\mathbf{c}} = f_\theta(\mathbf{x})$ denote predicted cost vectors and $\mathcal{S}$ a bounded polyhedron. The downstream decision is $\mathbf{w}^*(\hat{\mathbf{c}}) \in \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}$.
True **Decision Regret**:
$$\mathrm{Regret}(\hat{\mathbf{c}}, \mathbf{c}^*) = \mathbf{c}^{*\top} \mathbf{w}^*(\hat{\mathbf{c}}) - \mathbf{c}^{*\top} \mathbf{w}^*(\mathbf{c}^*) \ge 0$$
Because $\nabla_{\hat{\mathbf{c}}} \mathbf{w}^*(\hat{\mathbf{c}}) = \mathbf{0}$ almost everywhere, direct backpropagation fails. Elmachtoub & Grigas (2022) prove the **SPO+ convex upper bound**:
$$\mathcal{L}_{\mathrm{SPO}+}(\hat{\mathbf{c}}, \mathbf{c}^*) = \max_{\mathbf{w} \in \mathcal{S}} (\mathbf{c}^* - 2 \hat{\mathbf{c}})^\top \mathbf{w} + 2 \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}^*) - \mathbf{c}^{*\top} \mathbf{w}^*(\mathbf{c}^*) \ge \mathrm{Regret}(\hat{\mathbf{c}}, \mathbf{c}^*)$$
For simplex decisions $\mathbf{w} \in \Delta_K$, $\max_{\mathbf{w} \in \Delta_K} \mathbf{v}^\top \mathbf{w} = \max_k v_k$, yielding closed-form subgradients:
$$\nabla_{\hat{\mathbf{c}}} \mathcal{L}_{\mathrm{SPO}+}(\hat{\mathbf{c}}, \mathbf{c}^*) = 2 \left( \mathbf{w}^*(\mathbf{c}^*) - \mathbf{w}^*(2\hat{\mathbf{c}} - \mathbf{c}^*) \right)$$

### 2. Causal Uplift & Counterfactual Disentanglement
Disentangles baseline observational risk $\mu_0(\mathbf{x})$ from causal treatment uplift $\tau(\mathbf{x})$:
$$Y(T) = \mu_0(\mathbf{x}) + T \cdot \tau(\mathbf{x})$$

### 3. Symbolic Equation Learner (EQL)
Expands inputs into non-linear functional primitives $\mathbf{\Phi}(\mathbf{x}) = [x_0, \sin(x_1), \cos(x_2), x_0 x_3]^\top$ to extract closed-form analytical equations $y = \mathbf{\Phi}(\mathbf{x})^\top \mathbf{W} + b$."""

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
weights_path = repo_root / "weights" / "01_paradigms_and_dichotomies" / "06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.safetensors"
weights, ref_in, ref_out, meta = core.parity.load_reference_checkpoint(weights_path, device=device)
print(f"Device: {device} | Checkpoint: {weights_path.name} | Meta: {meta}")"""

    cell_4_code = """from jaxtyping import Float
from torch import Tensor


class CostPredictor(nn.Module):
    def __init__(self, w1: Float[Tensor, "dim h"], b1: Float[Tensor, "*h"], w2: Float[Tensor, "h out"], b2: Float[Tensor, "*out"]) -> None:
        super().__init__()
        self.w1, self.b1, self.w2, self.b2 = nn.Parameter(w1), nn.Parameter(b1), nn.Parameter(w2), nn.Parameter(b2)

    def forward(self, x: Float[Tensor, "b dim"]) -> Float[Tensor, "b out"]:
        return F.softplus(F.relu(x @ self.w1 + self.b1) @ self.w2 + self.b2)


class SPOPlusEstimator:
    @staticmethod
    def solve_linear_program(cost_vectors: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        cost_dim = cost_vectors.shape[-1]
        opt_idx = torch.argmin(cost_vectors, dim=-1)
        return F.one_hot(opt_idx, num_classes=cost_dim).float()

    @staticmethod
    def decision_regret(c_pred: Float[Tensor, "b d"], c_star: Float[Tensor, "b d"]) -> Float[Tensor, "*b"]:
        w_pred = SPOPlusEstimator.solve_linear_program(c_pred)
        w_star = SPOPlusEstimator.solve_linear_program(c_star)
        return torch.sum(c_star * w_pred, dim=-1) - torch.sum(c_star * w_star, dim=-1)

    @staticmethod
    def spo_plus_loss(c_pred: Float[Tensor, "b d"], c_star: Float[Tensor, "b d"]) -> Float[Tensor, "*b"]:
        w_star = SPOPlusEstimator.solve_linear_program(c_star)
        cost_oracle = torch.sum(c_star * w_star, dim=-1)
        v_vec = c_star - 2.0 * c_pred
        max_term = torch.max(v_vec, dim=-1).values
        lin_term = 2.0 * torch.sum(c_pred * w_star, dim=-1) - cost_oracle
        return max_term + lin_term


class CausalUpliftEstimator(nn.Module):
    def __init__(self, w_tau: Float[Tensor, "d 1"], b_tau: Float[Tensor, "*out"], w_mu0: Float[Tensor, "d 1"], b_mu0: Float[Tensor, "*out"]) -> None:
        super().__init__()
        self.w_tau, self.b_tau, self.w_mu0, self.b_mu0 = nn.Parameter(w_tau), nn.Parameter(b_tau), nn.Parameter(w_mu0), nn.Parameter(b_mu0)

    def predict_treatment_effect(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b 1"]:
        return x @ self.w_tau + self.b_tau

    def forward(self, x: Float[Tensor, "b d"], treatment_t: Float[Tensor, "b 1"]) -> Float[Tensor, "b 1"]:
        mu0 = x @ self.w_mu0 + self.b_mu0
        tau = self.predict_treatment_effect(x)
        return mu0 + treatment_t * tau


class SymbolicEquationLearner(nn.Module):
    def __init__(self, w_eql: Float[Tensor, "p 1"], b_eql: Float[Tensor, "*out"]) -> None:
        super().__init__()
        self.w_eql, self.b_eql = nn.Parameter(w_eql), nn.Parameter(b_eql)

    @staticmethod
    def expand_primitives(x: Float[Tensor, "b d"]) -> Float[Tensor, "b p"]:
        return torch.cat([x[:, 0:1], torch.sin(x[:, 1:2]), torch.cos(x[:, 2:3]), x[:, 0:1] * x[:, 3:4]], dim=-1)

    def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b 1"]:
        return self.expand_primitives(x) @ self.w_eql + self.b_eql


cost_model = CostPredictor(weights["w_cost1"], weights["b_cost1"], weights["w_cost2"], weights["b_cost2"]).to(device)
causal_model = CausalUpliftEstimator(weights["w_tau"], weights["b_tau"], weights["w_mu0"], weights["b_mu0"]).to(device)
eql_model = SymbolicEquationLearner(weights["w_eql"], weights["b_eql"]).to(device)
print("CostPredictor, CausalUpliftEstimator, and SymbolicEquationLearner initialized.")"""

    cell_5_code = """# 1. 4-Way Transitive Parity (< 1e-5) against golden checkpoint
x_test = ref_in
c_star = weights["c_star"]
treatment_t = weights["treatment_t"]

# Predictive / Prescriptive Cost Backbone
c_pred = cost_model(x_test)
assert float(torch.max(torch.abs(c_pred - weights["c_pred"])).item()) < 1e-5

# SPO+ Loss & Decision Regret Invariants
regret = SPOPlusEstimator.decision_regret(c_pred, c_star)
assert float(torch.max(torch.abs(regret - weights["decision_regret"])).item()) < 1e-5
assert float(torch.min(regret).item()) >= -1e-6, "Decision regret must be non-negative!"

spo_loss = SPOPlusEstimator.spo_plus_loss(c_pred, c_star)
assert float(torch.max(torch.abs(spo_loss - weights["spo_plus_loss"])).item()) < 1e-5
assert torch.all(spo_loss >= regret - 1e-5), "SPO+ must strictly upper bound decision regret!"

# Causal Uplift & Counterfactual Parity
tau_pred = causal_model.predict_treatment_effect(x_test)
assert float(torch.max(torch.abs(tau_pred - weights["tau_pred"])).item()) < 1e-5
y_factual = causal_model(x_test, treatment_t)
assert float(torch.max(torch.abs(y_factual - weights["y_factual"])).item()) < 1e-5

# Symbolic EQL Functional Parity
y_sym = eql_model(x_test)
assert float(torch.max(torch.abs(y_sym - weights["y_symbolic"])).item()) < 1e-5

print("All mathematical invariants (SPO+ Upper Bound, Non-negative Regret, Transitive Parity) verified.")"""

    cell_6_code = """# Multi-Track Micro-Benchmark Table
def run_cost_pred():
    return cost_model(x_test)

def run_spo_loss():
    return SPOPlusEstimator.spo_plus_loss(c_pred, c_star)

def run_causal():
    return causal_model(x_test, treatment_t)

def run_eql():
    return eql_model(x_test)

track_fns = {
    "Cost Predictor Backbone": run_cost_pred,
    "SPO+ Surrogate Loss": run_spo_loss,
    "Causal Uplift Model": run_causal,
    "Symbolic EQL Layer": run_eql,
}
bench_results = core.benchmark.compare_tracks(track_fns, warmup=3, iters=15, batch_size=8)
print(core.benchmark.format_benchmark_table(bench_results))"""

    cell_7_code = """# Pathological Misuse: Prediction MSE vs Decision Regret
# Case: Predictor trained with MSE flips optimal downstream LP vertex near decision boundary
c_true = torch.tensor([[1.0, 1.05, 5.0, 5.0]], device=device)  # Optimal vertex: index 0
c_hat_mse = torch.tensor([[1.08, 1.02, 5.0, 5.0]], device=device)  # Low MSE (0.003), but flips argmin to index 1!
c_hat_spo = torch.tensor([[0.90, 1.15, 4.8, 5.2]], device=device)  # Higher MSE (0.026), but correctly preserves argmin at 0!

regret_mse = SPOPlusEstimator.decision_regret(c_hat_mse, c_true).item()
regret_spo = SPOPlusEstimator.decision_regret(c_hat_spo, c_true).item()

print(f"[FAILURE DEMONSTRATED] MSE prediction error flipped LP vertex: Regret(MSE)={regret_mse:.4f} vs Regret(SPO)={regret_spo:.4f}!")
assert regret_mse > regret_spo, "Expected MSE predictor to yield strictly higher regret than decision-aligned predictor!"

# Correct Pattern: torch.compile and export verification on CostPredictor
ok_exp, _ = core.export.verify_torch_export(cost_model, x_test)
ok_comp, _ = core.export.verify_torch_compile_fullgraph(cost_model, x_test)
assert ok_exp and ok_comp, "Compilation or export verification failed"
print("[CORRECT] CostPredictor successfully verified with torch.compile(fullgraph=True) and torch.export.")"""

    cell_8_md = r"""### Mathematical & Architectural Synthesis

<svg viewBox="0 0 650 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg">
  <rect x="5" y="5" width="640" height="110" rx="8" fill="#0f172a" stroke="#334155" stroke-width="1.5"/>
  <rect x="25" y="25" width="135" height="35" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
  <text x="92" y="47" fill="#38bdf8" font-size="11" font-weight="bold" text-anchor="middle">Predictive (MSE)</text>
  <rect x="175" y="25" width="135" height="35" rx="6" fill="#1e293b" stroke="#10b981" stroke-width="1.5"/>
  <text x="242" y="47" fill="#10b981" font-size="11" font-weight="bold" text-anchor="middle">Prescriptive (SPO+)</text>
  <rect x="325" y="25" width="135" height="35" rx="6" fill="#1e293b" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="392" y="47" fill="#f59e0b" font-size="11" font-weight="bold" text-anchor="middle">Causal Uplift</text>
  <rect x="475" y="25" width="150" height="35" rx="6" fill="#1e293b" stroke="#a855f7" stroke-width="1.5"/>
  <text x="550" y="47" fill="#a855f7" font-size="11" font-weight="bold" text-anchor="middle">Symbolic (EQL)</text>
  <rect x="25" y="70" width="600" height="32" rx="4" fill="#0f172a" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4 4"/>
  <text x="325" y="90" fill="#cbd5e1" font-size="10" text-anchor="middle">MSE: Statistical residual | SPO+: Convex regret upper bound | Causal: do(T) uplift | EQL: Analytical basis</text>
</svg>

### Key Algorithmic Takeaways
1. **Decision Regret Alignment:** Statistical MSE does not penalize vertex-flipping rank inversions in downstream linear programs. Direct training with the convex SPO+ surrogate loss minimizes operational decision regret.
2. **Causal Confounder Separation:** Observational prediction confluences baseline risk with policy impact; explicit causal uplift heads $\tau(\mathbf{x})$ are required for actionable interventions.
3. **Symbolic Interpretability:** Replacing opaque neural representations with Equation Learner (EQL) functional primitives allows recovering closed-form analytical equations verifiable by automated formal solvers.

---
**Curriculum Navigation:**
&larr; [Previous: 05. Parametric vs. Nonparametric & Neural Processes](05_parametric_vs_nonparametric_and_neural_processes.ipynb) | [Master Index: README.md](../../README.md) | Next: 07. Input & Output Spaces: Tokens, Bytes, Patches, Coords, Pointers, and Manifolds (Upcoming in Category 01) &rarr;"""

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
    print(f"Created Topic 06 notebook at: {nb_path}")


if __name__ == "__main__":
    update_topic_05_notebook()
    create_topic_06_notebook()
