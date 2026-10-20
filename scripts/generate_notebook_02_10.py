"""Generate 8-cell educational notebook for Topic 10:
10_survival_analysis_and_contextual_bandits.ipynb
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf

def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 10. Survival Analysis & Contextual Bandits

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Edward L. Kaplan & Paul Meier (1958) *Nonparametric estimation from incomplete observations*; William R. Thompson (1933) *On the likelihood that one unknown probability exceeds another*; David R. Cox (1972) *Regression Models and Life-Tables*; Peter Auer et al. (2002) *Finite-time Analysis of the Multiarmed Bandit Problem*; Lihong Li et al. (2010) *A Contextual-Bandit Approach to Personalized News Recommendation*; Jared L. Katzman et al. (2018) *DeepSurv*.
- **Purpose:** Unifies time-to-event estimation under right-censoring and online exploration-exploitation trade-offs via Kaplan-Meier, Cox Proportional Hazards (DeepSurv), and LinUCB bandits with Sherman-Morrison rank-1 acceleration.

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not drop right-censored observations or treat censoring times as true event times in standard regression models; doing so severely underestimates survival times and skews hazard ratios. Do not compute nested $O(N^2)$ pairwise risk sets when evaluating Cox partial likelihood on large cohorts ($N > 10^4$); sort event times descending and accumulate prefix LogCumSumExp in $O(N \\log N)$."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Kaplan-Meier Product-Limit Survival Estimator
For distinct ordered event times $t_1 < t_2 < \\dots < t_K$ with $d_i$ events and $n_i$ individuals at risk:
$$\\hat{S}(t) = \\prod_{t_i \\le t} \\left( 1 - \\frac{d_i}{n_i} \\right), \\quad \\widehat{\\text{Var}}\\{\\hat{S}(t)\\} = \\hat{S}(t)^2 \\sum_{t_i \\le t} \\frac{d_i}{n_i(n_i - d_i)}$$

### 2. Cox Proportional Hazards & DeepSurv Partial Likelihood
With hazard $h(t | \\mathbf{x}) = h_0(t) \\exp(g_\\theta(\\mathbf{x}))$, the baseline hazard cancels out of the partial log-likelihood:
$$\\mathcal{L}_{\\text{Cox}}(\\theta) = -\\sum_{i: E_i = 1} \\left[ g_\\theta(\\mathbf{x}_i) - \\ln \\sum_{j \\in \\mathcal{R}(T_i)} \\exp(g_\\theta(\\mathbf{x}_j)) \\right]$$

### 3. LinUCB Contextual Bandit with Sherman-Morrison Acceleration
Given ridge system $\\mathbf{A}_a = \\mathbf{X}_a^\\top \\mathbf{X}_a + \\lambda \\mathbf{I}$ and $\\mathbf{b}_a = \\mathbf{X}_a^\\top \\mathbf{r}_a$:
$$a_t = \\arg\\max_{a} \\left[ \\mathbf{x}_t^\\top \\mathbf{A}_a^{-1}\\mathbf{b}_a + \\alpha \\sqrt{\\mathbf{x}_t^\\top \\mathbf{A}_a^{-1} \\mathbf{x}_t} \\right]$$
Rank-1 rank updates execute in $\\mathcal{O}(d^2)$ via Sherman-Morrison:
$$\\mathbf{A}_{a, \\text{new}}^{-1} = \\mathbf{A}_a^{-1} - \\frac{\\mathbf{A}_a^{-1} \\mathbf{x}_t \\mathbf{x}_t^\\top \\mathbf{A}_a^{-1}}{1 + \\mathbf{x}_t^\\top \\mathbf{A}_a^{-1} \\mathbf{x}_t}$$

**Complexity:** KM: $\\mathcal{O}(N \\log N)$; Cox prefix logcumsumexp: $\\mathcal{O}(N \\log N)$; LinUCB update: $\\mathcal{O}(d^2)$."""
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import core

env = core.setup_env(seed=42)
N, D, K_arms = 32, 4, 3
X_surv = torch.randn(N, D)
w_true = torch.tensor([0.4, -0.6, 0.8, -0.3])
risk_true = X_surv @ w_true + 0.2
base_times = torch.rand(N) * 20.0 + 1.0
T_surv = base_times / (torch.exp(risk_true) ** 0.5)
E_surv = (torch.rand(N) > 0.25).float()  # 75% uncensored events

A_arms = torch.stack([torch.eye(D) * 1.0 for _ in range(K_arms)])
b_arms = torch.randn(K_arms, D) * 0.5
x_context = torch.tensor([0.5, -0.2, 0.8, 0.1])
print(f"Setup complete on {env.device}: Cohort N={N}, Arms={K_arms}, Context dim={D}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class SurvivalBanditToolkit:
    @staticmethod
    def kaplan_meier(T: torch.Tensor, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unique_times = torch.unique(T[E == 1], sorted=True)
        surv_probs, cur_s = [], 1.0
        for t_i in unique_times:
            n_i = (T >= t_i).sum().float()
            d_i = ((T == t_i) & (E == 1)).sum().float()
            cur_s *= (1.0 - d_i / n_i).item()
            surv_probs.append(cur_s)
        return unique_times, torch.tensor(surv_probs)

    @staticmethod
    def cox_loss(risk: torch.Tensor, T: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(T, descending=True)
        sorted_risk, sorted_E = risk[order], E[order]
        log_risk_set = torch.logcumsumexp(sorted_risk, dim=0)
        event_mask = sorted_E == 1
        return -torch.sum(sorted_risk[event_mask] - log_risk_set[event_mask])

    @staticmethod
    def linucb_score(x: torch.Tensor, A_arms: torch.Tensor, b_arms: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, int]:
        scores = torch.zeros(A_arms.shape[0])
        for a in range(A_arms.shape[0]):
            A_inv = torch.linalg.inv(A_arms[a])
            theta = A_inv @ b_arms[a]
            mean_r = torch.dot(x, theta)
            var_r = torch.sqrt(torch.clamp(x @ A_inv @ x, min=0.0))
            scores[a] = mean_r + alpha * var_r
        return scores, torch.argmax(scores).item()

    @staticmethod
    def sherman_morrison(A_inv: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        A_inv_x = A_inv @ x
        denom = 1.0 + torch.dot(x, A_inv_x)
        return A_inv - torch.outer(A_inv_x, A_inv_x) / denom"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# Quantitative Invariants & Precision Verification
km_t, km_s = SurvivalBanditToolkit.kaplan_meier(T_surv, E_surv)

# Invariant 1: Kaplan-Meier is monotonically non-increasing and bounded in [0, 1]
assert (km_s >= 0.0).all() and (km_s <= 1.0).all(), "KM survival probabilities out of bounds"
diffs = km_s[1:] - km_s[:-1]
assert (diffs <= 1e-6).all(), "KM curve violated monotonicity"

# Invariant 2: Cox negative partial log-likelihood is finite and positive
loss_val = SurvivalBanditToolkit.cox_loss(risk_true, T_surv, E_surv)
assert torch.isfinite(loss_val) and loss_val > 0.0, "Cox loss must be strictly positive and finite"

# Invariant 3: Sherman-Morrison rank-1 update matches exact matrix inverse
A_test = A_arms[0].clone()
A_inv_test = torch.linalg.inv(A_test)
A_inv_sm = SurvivalBanditToolkit.sherman_morrison(A_inv_test, x_context)
A_inv_exact = torch.linalg.inv(A_test + torch.outer(x_context, x_context))
sm_err = torch.max(torch.abs(A_inv_sm - A_inv_exact)).item()
assert sm_err < 1e-5, f"Sherman-Morrison inverse mismatch: {sm_err}"

scores, best_arm = SurvivalBanditToolkit.linucb_score(x_context, A_arms, b_arms, alpha=1.0)
print(f"Verification passed: Final S(t)={km_s[-1].item():.4f}, Cox Loss={loss_val.item():.4f}, SM err={sm_err:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_km = benchmark_fn(lambda: SurvivalBanditToolkit.kaplan_meier(T_surv, E_surv), iters=30, name="Kaplan-Meier (N=32)")
b_cox = benchmark_fn(lambda: SurvivalBanditToolkit.cox_loss(risk_true, T_surv, E_surv), iters=30, name="Cox Loss (Prefix Scan, N=32)")
b_sm = benchmark_fn(lambda: SurvivalBanditToolkit.sherman_morrison(A_inv_test, x_context), iters=30, name="Sherman-Morrison O(d^2) Update")
print(format_benchmark_table([b_km, b_cox, b_sm]))"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Dropping Right-Censored Data vs Proper Kaplan-Meier
# ANTI-PATTERN: Discarding censored patients severely underestimates survival rates!
uncensored_times = T_surv[E_surv == 1]
# Naive empirical survival: proportion of patients surviving past median time
t_median = torch.median(T_surv).item()
naive_survival_estimate = (uncensored_times > t_median).float().mean().item()

# Proper Kaplan-Meier estimate at median time
idx_median = (km_t <= t_median).nonzero()
km_median_estimate = km_s[idx_median[-1]].item() if len(idx_median) > 0 else 1.0

bias = abs(naive_survival_estimate - km_median_estimate)
print(f"[NAIVE DISCARD] Estimated Survival at t={t_median:.2f}: {naive_survival_estimate:.4f} (Biased downward!)")
print(f"[KAPLAN-MEIER]  Estimated Survival at t={t_median:.2f}: {km_median_estimate:.4f} (Unbiased product-limit)")
print(f"Absolute Estimation Distortion from discarding censored cases: {bias:.4f}")
assert bias > 0.05, "Censoring discard must demonstrate severe bias" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **The Censoring Imperative:** Right-censored data cannot be dropped or treated as failure at censoring time; product-limit KM and Cox partial likelihood preserve unbiased estimators.
2. **Prefix LogCumSumExp Acceleration:** Sorting event times transforms $O(N^2)$ pairwise risk set summation into a single $O(N \\log N)$ prefix scan, enabling deep neural survival architectures (DeepSurv).
3. **Sherman-Morrison Rank-1 Efficiency:** Updating Gram matrix inverses in $O(d^2)$ operations enables low-latency online bandit recommendation at scale.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [10_survival_analysis_and_contextual_bandits.md](../../markdown/02_classical_and_probabilistic_ml/10_survival_analysis_and_contextual_bandits.md)
- **PyTorch 2.x Module:** [10_survival_analysis_and_contextual_bandits.py](../../pytorch2/02_classical_and_probabilistic_ml/10_survival_analysis_and_contextual_bandits.py)
- **Functional JAX Module:** [10_survival_analysis_and_contextual_bandits.py](../../jax/02_classical_and_probabilistic_ml/10_survival_analysis_and_contextual_bandits.py)
- **Native Rust Candle Track:** [10_survival_analysis_and_contextual_bandits.rs](../../candle/02_classical_and_probabilistic_ml/10_survival_analysis_and_contextual_bandits.rs)
- **Hardware WGSL Kernel:** [10_survival_analysis_and_contextual_bandits.rs](../../kernels/02_classical_and_probabilistic_ml/10_survival_analysis_and_contextual_bandits.rs)
- **Golden Reference Checkpoint:** [10_survival_analysis_and_contextual_bandits.safetensors](../../weights/02_classical_and_probabilistic_ml/10_survival_analysis_and_contextual_bandits.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 09. Hidden Markov Models & Kalman Filters](09_hidden_markov_models_kalman_ukf_and_particle_filters.ipynb) | [Category Index](../README.md) | [Next: 11. Probabilistic Graphical Models & Factor Graphs](11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "10_survival_analysis_and_contextual_bandits.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)

if __name__ == "__main__":
    build_notebook()
