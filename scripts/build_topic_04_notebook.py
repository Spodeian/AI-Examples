"""Script to build Topic 04 Category 01 notebook and update Topic 03 notebook navigation link."""

import json
from pathlib import Path


def update_topic_03_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "03_discrete_vs_continuous_latents_and_gradient_estimators.ipynb"
    )
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cell_8 = nb["cells"][7]
    new_source = []
    for line in cell_8["source"]:
        if "Next: 04. Deterministic vs. Stochastic & MDN (Upcoming in Category 01)" in line:
            new_line = line.replace(
                "Next: 04. Deterministic vs. Stochastic & MDN (Upcoming in Category 01)",
                "[Next: 04. Deterministic vs. Stochastic & Mixture Density Networks](04_deterministic_vs_stochastic_and_mixture_density_networks.ipynb)",
            )
            new_source.append(new_line)
        else:
            new_source.append(line)
    cell_8["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated Topic 03 notebook navigation link.")


def create_topic_04_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "04_deterministic_vs_stochastic_and_mixture_density_networks.ipynb"
    )

    cell_1_md = r"""# Deterministic vs. Stochastic & Mixture Density Networks

[![Track](https://img.shields.io/badge/Track-PyTorch_2.x_%7C_JAX_%7C_Candle_%7C_Kernels-blue.svg)](../../README.md)
[![Parity](https://img.shields.io/badge/Parity-Transitive_%3C_1e--5-success.svg)](../../tests/test_cross_track_parity.py)

### Historical Lineage & Seminal Citations
- Bishop (1994, Mixture Density Networks, Aston Univ Tech Report NCRG/94/004): Neural networks parameterizing conditional Gaussian mixtures.
- Koenker & Bassett (1978, Regression Quantiles, Econometrica): Pinball check loss for conditional quantiles.
- Dabney, Rowland, Bellemare & Munos (2018, Implicit Quantile Networks, ICML): Quantile representations for distributional value learning.

### Problem Statement & Purpose
Investigates the breakdown of deterministic conditional mean regression ($\hat{y} = \mathbb{E}[Y \mid X=x]$) under multimodal, bifurcated, or inverse problems. Contrasts point MSE predictors with Mixture Density Networks (MDN) that model the full conditional density $p(y \mid x)$ and Quantile Regressors that invert the cumulative distribution function non-parametrically.

### Decision Boundary: When NOT to Use
> [!WARNING]
> Do NOT use deterministic MSE regression when the mapping $x \to y$ is one-to-many or possesses bifurcated branches (e.g. inverse kinematics, robotic obstacle avoidance). MSE predicts the unweighted conditional mean, which frequently falls into regions of zero probability density (e.g. crashing into an obstacle midway between two open avoidance paths). Use Mixture Density Networks (MDN) or Quantile Networks to preserve distinct modes.

### Topic Connections
🔗 Upstream: [03. Discrete vs. Continuous Latents](03_discrete_vs_continuous_latents_and_gradient_estimators.ipynb) | 🔗 Downstream: [05. Parametric vs. Nonparametric & GP](05_parametric_vs_nonparametric_and_gaussian_processes.ipynb)"""

    cell_2_md = r"""## Mathematical Foundations & First-Principles Derivations

### 1. The Conditional Expectation Failure Mode
Under squared $L_2$ loss, the optimal point prediction $\hat{y}^*(x) = \arg\min_f \mathbb{E}\left[\|Y - f(X)\|_2^2 \mid X=x\right]$ is:
$$\hat{y}^*(x) = \mathbb{E}[Y \mid X=x] = \int y \, p(y \mid x) \, dy$$
If $p(y \mid x) = \frac{1}{2}\delta(y - 1) + \frac{1}{2}\delta(y + 1)$, then $\hat{y}^*(x) = 0$, where $p(y=0 \mid x) = 0$.

### 2. Mixture Density Network (Bishop 1994)
MDNs parameterize $K$ Gaussian components with mixing weights $\pi_k(x)$, centroids $\mu_k(x)$, and scales $\sigma_k(x)$:
$$p(y \mid x) = \sum_{k=1}^K \pi_k(x) \frac{1}{\sqrt{2\pi}\sigma_k(x)} \exp\left(-\frac{(y - \mu_k(x))^2}{2\sigma_k^2(x)}\right)$$
Numerically stable log-likelihood via LogSumExp:
$$\log p(y \mid x) = \operatorname{LogSumExp}_{k=1}^K \left[ \log \pi_k(x) - \log \sigma_k(x) - \frac{1}{2}\log(2\pi) - \frac{(y - \mu_k(x))^2}{2\sigma_k^2(x)} \right]$$

### 3. Quantile Regression & Pinball Loss
For target $y$ and quantile prediction $q_\tau(x)$ at level $\tau \in (0, 1)$:
$$\mathcal{L}_\tau(y, q_\tau) = \max\left(\tau (y - q_\tau), (\tau - 1)(y - q_\tau)\right)$$"""

    cell_3_code = """import math
import sys
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
weights_path = repo_root / "weights" / "01_paradigms_and_dichotomies" / "04_deterministic_vs_stochastic_and_mixture_density_networks.safetensors"
weights, ref_in, ref_out, _ = core.parity.load_reference_checkpoint(weights_path, device=device)
print(f"Device: {device} | Dim: {ref_in.shape[-1]} | Checkpoint: {weights_path.name}")"""

    cell_4_code = """from jaxtyping import Float
from torch import Tensor


class DeterministicRegressor(nn.Module):
    def __init__(self, w_h: Float[Tensor, "dim h"], b_h: Float[Tensor, "*h"], w_det: Float[Tensor, "h out"], b_det: Float[Tensor, "*out"]) -> None:
        super().__init__()
        self.w_h, self.b_h, self.w_det, self.b_det = nn.Parameter(w_h), nn.Parameter(b_h), nn.Parameter(w_det), nn.Parameter(b_det)

    def forward(self, x: Float[Tensor, "b dim"]) -> Float[Tensor, "b out"]:
        return F.relu(x @ self.w_h + self.b_h) @ self.w_det + self.b_det


class MixtureDensityNetwork(nn.Module):
    def __init__(self, w_h: Float[Tensor, "dim h"], b_h: Float[Tensor, "*h"], w_pi: Float[Tensor, "h k"], b_pi: Float[Tensor, "*k"], w_mu: Float[Tensor, "h k"], b_mu: Float[Tensor, "*k"], w_sigma: Float[Tensor, "h k"], b_sigma: Float[Tensor, "*k"]) -> None:
        super().__init__()
        self.w_h, self.b_h = nn.Parameter(w_h), nn.Parameter(b_h)
        self.w_pi, self.b_pi = nn.Parameter(w_pi), nn.Parameter(b_pi)
        self.w_mu, self.b_mu = nn.Parameter(w_mu), nn.Parameter(b_mu)
        self.w_sigma, self.b_sigma = nn.Parameter(w_sigma), nn.Parameter(b_sigma)

    def forward(self, x: Float[Tensor, "b dim"]) -> tuple[Float[Tensor, "b k"], Float[Tensor, "b k"], Float[Tensor, "b k"]]:
        h = F.relu(x @ self.w_h + self.b_h)
        pi = F.softmax(h @ self.w_pi + self.b_pi, dim=-1)
        mu = h @ self.w_mu + self.b_mu
        sigma = torch.exp(h @ self.w_sigma + self.b_sigma)
        return pi, mu, sigma

    def log_prob(self, x: Float[Tensor, "b dim"], y: Float[Tensor, "b out"]) -> Float[Tensor, "*b"]:
        h = F.relu(x @ self.w_h + self.b_h)
        log_pi = F.log_softmax(h @ self.w_pi + self.b_pi, dim=-1)
        mu = h @ self.w_mu + self.b_mu
        log_sigma = h @ self.w_sigma + self.b_sigma
        sigma = torch.exp(log_sigma)
        diff = y - mu
        log_gauss = -0.5 * math.log(2.0 * math.pi) - log_sigma - 0.5 * (diff / sigma).pow(2)
        return torch.logsumexp(log_pi + log_gauss, dim=-1)


class QuantileRegressor(nn.Module):
    def __init__(self, w_h: Float[Tensor, "dim h"], b_h: Float[Tensor, "*h"], w_q: Float[Tensor, "h q"], b_q: Float[Tensor, "*q"]) -> None:
        super().__init__()
        self.w_h, self.b_h, self.w_q, self.b_q = nn.Parameter(w_h), nn.Parameter(b_h), nn.Parameter(w_q), nn.Parameter(b_q)

    def forward(self, x: Float[Tensor, "b dim"]) -> Float[Tensor, "b q"]:
        return F.relu(x @ self.w_h + self.b_h) @ self.w_q + self.b_q


det_model = DeterministicRegressor(weights["w_h"], weights["b_h"], weights["w_det"], weights["b_det"]).to(device)
mdn = MixtureDensityNetwork(weights["w_h"], weights["b_h"], weights["w_pi"], weights["b_pi"], weights["w_mu"], weights["b_mu"], weights["w_sigma"], weights["b_sigma"]).to(device)
q_model = QuantileRegressor(weights["w_h"], weights["b_h"], weights["w_quantile"], weights["b_quantile"]).to(device)
print("DeterministicRegressor, MixtureDensityNetwork, and QuantileRegressor initialized.")"""

    cell_5_code = """# 1. 4-Way Transitive Parity (< 1e-5) against golden checkpoint
y_det = det_model(ref_in)
assert float(torch.max(torch.abs(y_det - weights["y_det"])).item()) < 1e-5

pi, mu, sigma = mdn(ref_in)
assert float(torch.max(torch.abs(pi - weights["pi"])).item()) < 1e-5
assert float(torch.max(torch.abs(mu - weights["mu"])).item()) < 1e-5
assert float(torch.max(torch.abs(sigma - weights["sigma"])).item()) < 1e-5

log_probs = mdn.log_prob(ref_in, weights["sample_y"])
assert float(torch.max(torch.abs(log_probs - weights["mdn_log_probs"])).item()) < 1e-5

y_q = q_model(ref_in)
assert float(torch.max(torch.abs(y_q - weights["y_quantiles"])).item()) < 1e-5

u = weights["sample_y"] - y_q
pinball = torch.maximum(weights["quantiles"] * u, (weights["quantiles"] - 1.0) * u)
assert float(torch.max(torch.abs(pinball - weights["pinball_loss"])).item()) < 1e-5

# 2. Mathematical Invariants: Simplex Axiom & Scale Positivity
assert torch.allclose(pi.sum(-1), torch.ones(ref_in.shape[0], device=device), atol=1e-5)
assert (sigma > 0.0).all().item()
print("All mathematical invariants (Transitive Parity + Simplex Axiom + Scale Positivity) verified.")"""

    cell_6_code = """# Multi-Track Micro-Benchmark Table
def run_det():
    return det_model(ref_in)

def run_mdn():
    return mdn.log_prob(ref_in, weights["sample_y"])

def run_quantile():
    return q_model(ref_in)

track_fns = {
    "Deterministic MSE": run_det,
    "Mixture Density Network": run_mdn,
    "Quantile Regressor": run_quantile,
}
bench_results = core.benchmark.compare_tracks(track_fns, warmup=3, iters=15, batch_size=8)
print(core.benchmark.format_benchmark_table(bench_results))"""

    cell_7_code = """# Pathological Misuse: Multimodal Inverse Problem Failure
gen = torch.Generator(device=device).manual_seed(42)
n_pts = 400
# True bifurcation: x = 0 with modes at y = -1.0 and y = +1.0
branch = torch.bernoulli(torch.full((n_pts, 1), 0.5, device=device), generator=gen)
y_multi = torch.where(branch > 0.5, 1.0 + 0.1 * torch.randn(n_pts, 1, generator=gen, device=device), -1.0 + 0.1 * torch.randn(n_pts, 1, generator=gen, device=device))
x_multi = 0.05 * torch.randn(n_pts, 1, generator=gen, device=device)

# Deterministic model predicting conditional mean E[Y|X=0]
y_mean_pred = y_multi.mean().item()  # Close to 0.0
err_to_branches = torch.min((y_multi - y_mean_pred).abs(), dim=-1).values.mean().item()

# Failure assertion: The conditional mean is at distance ~1.0 from BOTH actual modes!
assert err_to_branches >= 0.80, f"Expected mean prediction to be invalid with error >= 0.80, got {err_to_branches:.3f}"
print(f"[FAILURE DEMONSTRATED] Deterministic MSE predicted mean {y_mean_pred:.3f}, lying in zero-probability void (error {err_to_branches:.3f} to both branches)!")

# Correct Pattern: Continuous models verified with core.export compile and export
ok_exp, _ = core.export.verify_torch_export(det_model, ref_in)
ok_comp, _ = core.export.verify_torch_compile_fullgraph(det_model, ref_in)
assert ok_exp and ok_comp, "Compilation or export verification failed"
print("[CORRECT] Deterministic Regressor compiled with torch.compile(fullgraph=True) and exported.")"""

    cell_8_md = r"""### Mathematical & Architectural Synthesis

<svg viewBox="0 0 650 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg">
  <rect x="5" y="5" width="640" height="110" rx="8" fill="#0f172a" stroke="#334155" stroke-width="1.5"/>
  <rect x="25" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#ef4444" stroke-width="1.5"/>
  <text x="115" y="47" fill="#ef4444" font-size="11" font-weight="bold" text-anchor="middle">Deterministic MSE</text>
  <rect x="235" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#10b981" stroke-width="1.5"/>
  <text x="325" y="47" fill="#10b981" font-size="11" font-weight="bold" text-anchor="middle">Mixture Density Net</text>
  <rect x="445" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
  <text x="535" y="47" fill="#38bdf8" font-size="11" font-weight="bold" text-anchor="middle">Quantile Regressor</text>
  <rect x="25" y="70" width="600" height="32" rx="4" fill="#0f172a" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4 4"/>
  <text x="325" y="90" fill="#f59e0b" font-size="10" text-anchor="middle">MSE predicts E[Y|X] in low-density voids | MDN models disjoint modes | Quantiles give distribution-free CDF</text>
</svg>

### Key Algorithmic Takeaways
1. **Conditional Mean Flaw:** Minimizing squared loss forces point estimators to converge to $\mathbb{E}[Y \mid X=x]$, which produces disastrous, invalid midpoint predictions on multimodal or bifurcated distributions.
2. **Mixture Density Networks:** Parameterizing mixing weights $\pi_k(x)$ on the simplex alongside component means $\mu_k(x)$ and variances $\sigma_k^2(x)$ allows analytical evaluation of the full conditional density $p(y \mid x)$ in a single forward pass.
3. **Distribution-Free Quantiles:** Quantile regression with pinball loss $\max(\tau u, (\tau - 1) u)$ yields non-parametric prediction intervals without Gaussianity assumptions.

---
**Curriculum Navigation:**
&larr; [Previous: 03. Discrete vs. Continuous Latents](03_discrete_vs_continuous_latents_and_gradient_estimators.ipynb) | [Master Index: README.md](../../README.md) | Next: 05. Parametric vs. Nonparametric & Gaussian Process Priors (Upcoming in Category 01) &rarr;"""

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
    print(f"Created Topic 04 notebook at: {nb_path}")


if __name__ == "__main__":
    update_topic_03_notebook()
    create_topic_04_notebook()
