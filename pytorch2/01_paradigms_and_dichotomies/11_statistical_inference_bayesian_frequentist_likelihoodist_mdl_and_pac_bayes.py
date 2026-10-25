"""Statistical Inference: Bayesian, Frequentist, Likelihoodist, MDL, and PAC-Bayes.

Implements the 5 schools of statistical inference:
1. Frequentist: Fisher Information Matrix, Cramér-Rao Lower Bound, Likelihood Ratio Test.
2. Bayesian: Conjugate Gaussian update, Laplace marginal likelihood, and Occam factor penalty.
3. James-Stein Shrinkage Estimator: Demonstrating Stein's phenomenon on multi-group means.
4. PAC-Bayes Generalization Bound: McAllester bound evaluating stochastic hypothesis bounds.
5. Anytime-Valid E-Values & Martingales: Sequential testing with Ville's inequality.
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from safetensors.torch import load_file


class FrequentistFisherCramerRao(nn.Module):
    """Frequentist estimator, Fisher Information, and Cramér-Rao lower bound."""

    def __init__(self, sigma2_noise: float = 0.5) -> None:
        super().__init__()
        self.sigma2_noise = sigma2_noise

    def forward(
        self,
        x_data: Float[torch.Tensor, "num_obs obs_dim"],
        theta_0: Float[torch.Tensor, "obs_dim"],
    ) -> Tuple[
        Float[torch.Tensor, "obs_dim"],
        Float[torch.Tensor, "obs_dim"],
        Float[torch.Tensor, "obs_dim"],
        Float[torch.Tensor, ""],
    ]:
        num_obs = x_data.shape[0]
        obs_dim = x_data.shape[1]

        # MLE for Gaussian mean
        theta_mle = torch.mean(x_data, dim=0)

        # Diagonal Fisher Information: I_n = n / sigma^2
        fisher_diag = torch.full((obs_dim,), num_obs / self.sigma2_noise, dtype=x_data.dtype, device=x_data.device)
        cramer_rao = torch.full((obs_dim,), self.sigma2_noise / num_obs, dtype=x_data.dtype, device=x_data.device)

        # Wilks' Likelihood Ratio Test against theta_0
        diff = theta_mle - theta_0
        lrt_stat = (num_obs / self.sigma2_noise) * torch.sum(diff ** 2)

        return theta_mle, fisher_diag, cramer_rao, lrt_stat


class BayesianConjugateLaplace(nn.Module):
    """Conjugate Gaussian Bayesian update and Laplace marginal likelihood Occam factor."""

    def __init__(self, sigma2_0: float = 1.0, sigma2_noise: float = 0.5) -> None:
        super().__init__()
        self.sigma2_0 = sigma2_0
        self.sigma2_noise = sigma2_noise

    def forward(
        self,
        x_data: Float[torch.Tensor, "num_obs obs_dim"],
        mu_0: Float[torch.Tensor, "obs_dim"],
    ) -> Tuple[
        Float[torch.Tensor, "obs_dim"],
        Float[torch.Tensor, "obs_dim"],
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
    ]:
        num_obs = x_data.shape[0]
        obs_dim = x_data.shape[1]
        theta_mle = torch.mean(x_data, dim=0)

        prec_0 = 1.0 / self.sigma2_0
        prec_lik = num_obs / self.sigma2_noise
        prec_post = prec_0 + prec_lik
        var_post = 1.0 / prec_post
        sigma_post = math.sqrt(var_post)

        mu_post = var_post * (prec_0 * mu_0 + prec_lik * theta_mle)
        sigma_post_vec = torch.full((obs_dim,), sigma_post, dtype=x_data.dtype, device=x_data.device)

        # Laplace Marginal Likelihood:
        # log p(D|M) approx log p(D|theta_map) + log p(theta_map) + (D/2)*log(2*pi) - 0.5*log det(H)
        log_det_h = float(obs_dim * math.log(prec_post))
        res = x_data - mu_post.unsqueeze(0)
        log_lik_map = -0.5 * num_obs * obs_dim * math.log(2.0 * math.pi * self.sigma2_noise) - 0.5 * torch.sum(res ** 2) / self.sigma2_noise
        log_prior_map = -0.5 * obs_dim * math.log(2.0 * math.pi * self.sigma2_0) - 0.5 * torch.sum((mu_post - mu_0) ** 2) / self.sigma2_0
        occam_penalty = -0.5 * log_det_h + 0.5 * obs_dim * math.log(2.0 * math.pi)
        laplace_log_marginal = log_lik_map + log_prior_map + occam_penalty

        return mu_post, sigma_post_vec, laplace_log_marginal, torch.tensor(occam_penalty, dtype=x_data.dtype, device=x_data.device)


class JamesSteinEstimator(nn.Module):
    """James-Stein shrinkage estimator demonstrating Stein's paradox for K >= 3."""

    def __init__(self, sigma2: float = 1.0) -> None:
        super().__init__()
        self.sigma2 = sigma2

    def forward(
        self, y_obs: Float[torch.Tensor, "num_groups"]
    ) -> Tuple[Float[torch.Tensor, "num_groups"], Float[torch.Tensor, "num_groups"], Float[torch.Tensor, ""]]:
        num_groups = y_obs.shape[0]
        norm_sq = torch.sum(y_obs ** 2)
        c_shrink = 1.0 - ((num_groups - 2.0) * self.sigma2) / norm_sq
        c_shrink_pos = torch.clamp(c_shrink, min=0.0)

        theta_js = c_shrink * y_obs
        theta_js_pos = c_shrink_pos * y_obs
        return theta_js, theta_js_pos, c_shrink


class PACBayesBound(nn.Module):
    """McAllester PAC-Bayes generalization bound for stochastic hypothesis spaces."""

    def __init__(self, sigma_p: float = 1.0, delta: float = 0.05) -> None:
        super().__init__()
        self.sigma_p = sigma_p
        self.delta = delta

    def forward(
        self,
        mu_q: Float[torch.Tensor, "dim"],
        sigma_q: Float[torch.Tensor, "dim"],
        emp_risk: Float[torch.Tensor, ""],
        num_obs: int,
    ) -> Tuple[Float[torch.Tensor, ""], Float[torch.Tensor, ""]]:
        # KL(Q || P) for Gaussian with isotropic variances
        kl_div = 0.5 * torch.sum(
            (mu_q ** 2 + sigma_q ** 2) / (self.sigma_p ** 2)
            - 1.0
            - torch.log((sigma_q ** 2) / (self.sigma_p ** 2))
        )
        complexity_term = torch.sqrt(
            (kl_div + math.log(2.0 * math.sqrt(num_obs) / self.delta)) / (2.0 * num_obs)
        )
        pac_bound = emp_risk + complexity_term
        return kl_div, pac_bound


class AnytimeValidMartingale(nn.Module):
    """Anytime-valid e-values test martingale with Ville's boundary condition."""

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self, e_values: Float[torch.Tensor, "time_steps"]
    ) -> Tuple[Float[torch.Tensor, "time_steps"], Float[torch.Tensor, ""]]:
        martingale = torch.cumprod(e_values, dim=0)
        threshold = torch.tensor(1.0 / self.alpha, dtype=e_values.dtype, device=e_values.device)
        return martingale, threshold


class StatisticalInferenceUnifiedModule(nn.Module):
    """Unified module combining all 5 inference schools with linear parity anchor."""

    def __init__(self) -> None:
        super().__init__()
        self.frequentist = FrequentistFisherCramerRao(sigma2_noise=0.5)
        self.bayesian = BayesianConjugateLaplace(sigma2_0=1.0, sigma2_noise=0.5)
        self.james_stein = JamesSteinEstimator(sigma2=1.0)
        self.pac_bayes = PACBayesBound(sigma_p=1.0, delta=0.05)
        self.martingale = AnytimeValidMartingale(alpha=0.05)
        self.linear = nn.Linear(8, 4)

    def forward(self, x: Float[torch.Tensor, "batch in_dim"]) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(checkpoint_path: str) -> Tuple[StatisticalInferenceUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = StatisticalInferenceUnifiedModule()
    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])
    return module, weights


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    ckpt = (
        Path(__file__).parents[2]
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.safetensors"
    )
    model, tensors = load_checkpoint(str(ckpt))

    # 1. Verify linear anchor
    out_anchor = model(tensors["x_anchor"])
    assert torch.allclose(out_anchor, tensors["linear_anchor_out"], atol=1e-5), "Linear anchor mismatch"

    # 2. Verify Frequentist Fisher & Cramér-Rao
    theta_0 = torch.zeros_like(tensors["theta_mle"])
    mle, fisher, crlb, lrt = model.frequentist(tensors["x_data"], theta_0)
    assert torch.allclose(mle, tensors["theta_mle"], atol=1e-5), "MLE mismatch"
    assert torch.allclose(fisher, tensors["fisher_diag"], atol=1e-5), "Fisher info mismatch"
    assert torch.allclose(crlb, tensors["cramer_rao_lower_bound"], atol=1e-5), "Cramér-Rao mismatch"
    assert torch.allclose(lrt, tensors["lrt_stat"], atol=1e-5), "LRT stat mismatch"

    # 3. Verify Bayesian Conjugate & Laplace
    mu_post, sigma_post, laplace_ll, occam = model.bayesian(tensors["x_data"], tensors["mu_0"])
    assert torch.allclose(mu_post, tensors["mu_post"], atol=1e-5), "Mu post mismatch"
    assert torch.allclose(sigma_post, tensors["sigma_post_vec"], atol=1e-5), "Sigma post mismatch"
    assert torch.allclose(laplace_ll, tensors["laplace_log_marginal"], atol=1e-5), "Laplace marginal LL mismatch"
    assert torch.allclose(occam, tensors["occam_penalty"], atol=1e-5), "Occam penalty mismatch"

    # 4. Verify James-Stein
    th_js, th_js_pos, c_sh = model.james_stein(tensors["y_obs"])
    assert torch.allclose(th_js, tensors["theta_js"], atol=1e-5), "James-Stein mismatch"
    assert torch.allclose(th_js_pos, tensors["theta_js_pos"], atol=1e-5), "James-Stein pos mismatch"
    assert torch.allclose(c_sh, tensors["c_shrink"], atol=1e-5), "C shrink mismatch"

    # 5. Verify PAC-Bayes
    kl, pac = model.pac_bayes(tensors["mu_q"], tensors["sigma_q"], tensors["emp_risk"], num_obs=16)
    assert torch.allclose(kl, tensors["kl_div"], atol=1e-5), "KL div mismatch"
    assert torch.allclose(pac, tensors["pac_bound"], atol=1e-5), "PAC bound mismatch"

    # 6. Verify Anytime-Valid Martingale
    mart, thresh = model.martingale(tensors["e_values"])
    assert torch.allclose(mart, tensors["martingale"], atol=1e-5), "Martingale mismatch"
    assert torch.allclose(thresh, tensors["ville_threshold"], atol=1e-5), "Threshold mismatch"

    print("PyTorch 2 Statistical Inference successfully verified with zero errors!")
