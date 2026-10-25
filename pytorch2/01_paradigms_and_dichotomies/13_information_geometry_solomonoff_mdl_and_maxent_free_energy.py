"""Information Geometry, Solomonoff MDL, and MaxEnt Free Energy (Topic 13).

Implements the fundamental mathematical pillars of information geometry and description length:
1. Amari's Dually Flat Manifold: Primal natural coordinates theta, dual expectation coordinates eta = grad psi,
   Legendre-Fenchel conjugacy psi(theta) + phi(eta) - theta^T eta = 0, Fisher-Rao Riemannian metric G(theta) = grad^2 psi,
   and Bregman divergence equivalence D_psi(theta_2, theta_1) = KL(p_1 || p_2).
2. Dual alpha-Geodesics: e-geodesic (alpha = 1, flat in theta, reverse KL / mode-seeking) vs.
   m-geodesic (alpha = -1, flat in eta, forward KL / mean-seeking).
3. Algorithmic Information & Shtarkov NML / Prequential MDL: Minimax regret coding and sequential predictive log-loss.
4. Variational Free Energy: Perception and action balancing epistemic accuracy and complexity F(q) = KL(q || p) + E_q[-log p(x|z)].
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from safetensors.torch import load_file


class AmariDuallyFlatGeometry(nn.Module):
    """Amari dually flat manifold for exponential families on the simplex."""

    def __init__(self) -> None:
        super().__init__()

    def log_partition(self, theta: Float[torch.Tensor, "dim"]) -> Float[torch.Tensor, "1"]:
        return torch.logsumexp(theta, dim=0, keepdim=True)

    def expectation_param(self, theta: Float[torch.Tensor, "dim"]) -> Float[torch.Tensor, "dim"]:
        return F.softmax(theta, dim=0)

    def dual_potential(self, eta: Float[torch.Tensor, "dim"]) -> Float[torch.Tensor, "1"]:
        # Negative Shannon entropy phi(eta) = sum(eta * log(eta))
        return torch.sum(eta * torch.log(eta)).unsqueeze(0)

    def legendre_fenchel_gap(
        self, theta: Float[torch.Tensor, "dim"], eta: Float[torch.Tensor, "dim"]
    ) -> Float[torch.Tensor, "1"]:
        psi = self.log_partition(theta)
        phi = self.dual_potential(eta)
        inner = torch.sum(theta * eta)
        return torch.abs(psi + phi - inner)

    def fisher_rao_metric(self, eta: Float[torch.Tensor, "dim"]) -> Float[torch.Tensor, "dim dim"]:
        return torch.diag(eta) - torch.outer(eta, eta)

    def e_geodesic(
        self,
        theta_1: Float[torch.Tensor, "dim"],
        theta_2: Float[torch.Tensor, "dim"],
        t: float = 0.5,
    ) -> Float[torch.Tensor, "dim"]:
        # alpha = 1: affine in natural coordinates
        return (1.0 - t) * theta_1 + t * theta_2

    def m_geodesic_eta(
        self,
        eta_1: Float[torch.Tensor, "dim"],
        eta_2: Float[torch.Tensor, "dim"],
        t: float = 0.5,
    ) -> Float[torch.Tensor, "dim"]:
        # alpha = -1: affine in expectation coordinates
        return (1.0 - t) * eta_1 + t * eta_2

    def kl_divergence(
        self, eta_p: Float[torch.Tensor, "dim"], eta_q: Float[torch.Tensor, "dim"]
    ) -> Float[torch.Tensor, "1"]:
        return torch.sum(eta_p * torch.log(eta_p / eta_q)).unsqueeze(0)

    def bregman_divergence(
        self,
        theta_1: Float[torch.Tensor, "dim"],
        theta_2: Float[torch.Tensor, "dim"],
        eta_1: Float[torch.Tensor, "dim"],
    ) -> Float[torch.Tensor, "1"]:
        psi_1 = torch.logsumexp(theta_1, dim=0)
        psi_2 = torch.logsumexp(theta_2, dim=0)
        return (psi_2 - psi_1 - torch.sum(eta_1 * (theta_2 - theta_1))).unsqueeze(0)


class ShtarkovNMLAndPrequentialMDL(nn.Module):
    """Normalized Maximum Likelihood (Shtarkov Minimax Regret) and Prequential MDL."""

    def __init__(self) -> None:
        super().__init__()

    def shtarkov_log_complexity(self, n: int) -> Float[torch.Tensor, "1"]:
        comp_sum = 0.0
        for k in range(n + 1):
            comb = math.comb(n, k)
            if k == 0 or k == n:
                max_lik = 1.0
            else:
                p_hat = k / n
                max_lik = (p_hat ** k) * ((1.0 - p_hat) ** (n - k))
            comp_sum += comb * max_lik
        return torch.tensor([math.log(comp_sum)], dtype=torch.float32)

    def prequential_codelength(
        self, stream_bits: Float[torch.Tensor, "n"]
    ) -> Float[torch.Tensor, "1"]:
        n_seq = stream_bits.shape[0]
        preq_loss = 0.0
        s_prev = 0.0
        for t in range(1, n_seq + 1):
            bit = float(stream_bits[t - 1])
            p_pred = (s_prev + 0.5) / (t - 1 + 1.0)
            prob_t = p_pred if bit == 1.0 else (1.0 - p_pred)
            preq_loss -= math.log(prob_t)
            s_prev += bit
        return torch.tensor([preq_loss], dtype=stream_bits.dtype, device=stream_bits.device)


class VariationalFreeEnergy(nn.Module):
    """Friston Variational Free Energy F(q) = KL(q || p_prior) + E_q[-log p(x|z)]."""

    def __init__(self, sigma_obs: float = 0.2) -> None:
        super().__init__()
        self.sigma_obs = sigma_obs

    def forward(
        self,
        q_mu: Float[torch.Tensor, "latent_dim"],
        q_sigma: Float[torch.Tensor, "latent_dim"],
        p_prior_mu: Float[torch.Tensor, "latent_dim"],
        p_prior_sigma: Float[torch.Tensor, "latent_dim"],
        obs_x: Float[torch.Tensor, "obs_dim"],
        a_mat: Float[torch.Tensor, "obs_dim latent_dim"],
    ) -> Tuple[Float[torch.Tensor, "1"], Float[torch.Tensor, "1"], Float[torch.Tensor, "1"]]:
        kl_qp = torch.sum(
            torch.log(p_prior_sigma / q_sigma)
            + (q_sigma ** 2 + (q_mu - p_prior_mu) ** 2) / (2.0 * p_prior_sigma ** 2)
            - 0.5
        ).unsqueeze(0)

        obs_dim = obs_x.shape[0]
        res_mean = obs_x - torch.matmul(a_mat, q_mu)
        sq_err_mean = torch.sum(res_mean ** 2)
        tr_cov = torch.sum(torch.diagonal(torch.matmul(a_mat.t(), a_mat)) * (q_sigma ** 2))
        recon_loss = (
            0.5 * obs_dim * math.log(2.0 * math.pi * (self.sigma_obs ** 2))
            + (sq_err_mean + tr_cov) / (2.0 * (self.sigma_obs ** 2))
        ).unsqueeze(0)

        free_energy = kl_qp + recon_loss
        return kl_qp, recon_loss, free_energy


class InformationGeometryUnifiedModule(nn.Module):
    """Unified module combining Amari geometry, MDL, and Free Energy with linear anchor."""

    def __init__(self) -> None:
        super().__init__()
        self.amari = AmariDuallyFlatGeometry()
        self.mdl = ShtarkovNMLAndPrequentialMDL()
        self.free_energy = VariationalFreeEnergy(sigma_obs=0.2)
        self.linear = nn.Linear(8, 4)

    def forward(self, x: Float[torch.Tensor, "batch in_dim"]) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[InformationGeometryUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = InformationGeometryUnifiedModule()
    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])
    return module, weights


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    ckpt = (
        Path(__file__).parents[2]
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "13_information_geometry_solomonoff_mdl_and_maxent_free_energy.safetensors"
    )
    model, tensors = load_checkpoint(str(ckpt))

    # 1. Linear Anchor Parity
    out_anchor = model(tensors["x_anchor"])
    assert torch.allclose(out_anchor, tensors["linear_anchor_out"], atol=1e-5), "Linear anchor mismatch"

    # 2. Amari Dually Flat Geometry
    psi = model.amari.log_partition(tensors["theta"])
    assert torch.allclose(psi, tensors["psi"], atol=1e-5), "psi mismatch"
    eta = model.amari.expectation_param(tensors["theta"])
    assert torch.allclose(eta, tensors["eta"], atol=1e-5), "eta mismatch"
    phi = model.amari.dual_potential(eta)
    assert torch.allclose(phi, tensors["phi"], atol=1e-5), "phi mismatch"
    gap = model.amari.legendre_fenchel_gap(tensors["theta"], eta)
    assert torch.allclose(gap, tensors["legendre_gap"], atol=1e-5), "Legendre gap mismatch"
    fisher = model.amari.fisher_rao_metric(eta)
    assert torch.allclose(fisher, tensors["fisher_metric"], atol=1e-5), "Fisher metric mismatch"

    # 3. Dual alpha-geodesics & Bregman divergence
    e_mid = model.amari.e_geodesic(tensors["theta_1"], tensors["theta_2"], t=0.5)
    assert torch.allclose(e_mid, tensors["e_geodesic_mid"], atol=1e-5), "e-geodesic mismatch"
    m_mid_eta = model.amari.m_geodesic_eta(tensors["eta_1"], tensors["eta_2"], t=0.5)
    assert torch.allclose(m_mid_eta, tensors["m_geodesic_mid_eta"], atol=1e-5), "m-geodesic mismatch"
    kl_fwd = model.amari.kl_divergence(tensors["eta_1"], tensors["eta_2"])
    assert torch.allclose(kl_fwd, tensors["kl_forward"], atol=1e-5), "KL forward mismatch"
    kl_rev = model.amari.kl_divergence(tensors["eta_2"], tensors["eta_1"])
    assert torch.allclose(kl_rev, tensors["kl_reverse"], atol=1e-5), "KL reverse mismatch"
    bregman = model.amari.bregman_divergence(tensors["theta_1"], tensors["theta_2"], tensors["eta_1"])
    assert torch.allclose(bregman, tensors["bregman_div"], atol=1e-5), "Bregman div mismatch"

    # 4. Shtarkov NML & Prequential MDL
    nml = model.mdl.shtarkov_log_complexity(8)
    assert torch.allclose(nml, tensors["nml_log_comp"], atol=1e-5), "Shtarkov NML mismatch"
    preq = model.mdl.prequential_codelength(tensors["stream_bits"])
    assert torch.allclose(preq, tensors["prequential_codelength"], atol=1e-5), "Prequential MDL mismatch"

    # 5. Variational Free Energy
    kl_qp, recon, fe = model.free_energy(
        tensors["q_mu"],
        tensors["q_sigma"],
        tensors["p_prior_mu"],
        tensors["p_prior_sigma"],
        tensors["obs_x"],
        tensors["a_mat"],
    )
    assert torch.allclose(kl_qp, tensors["kl_qp"], atol=1e-5), "Free energy KL mismatch"
    assert torch.allclose(recon, tensors["recon_loss"], atol=1e-5), "Free energy recon mismatch"
    assert torch.allclose(fe, tensors["free_energy"], atol=1e-5), "Free energy mismatch"

    print("PyTorch 2 Information Geometry, MDL & Free Energy verified successfully!")
