"""Inductive vs. Transductive Learning and Online Regret vs. Batch IID.

Implements the fundamental paradigm dichotomies:
1. Transductive Graph Learning: Zhou et al. (2003) Label Spreading via normalized graph Laplacian F* = (I - alpha * S)^{-1} Y.
2. Inductive Out-of-Sample Prediction: Linear ridge regression mapping new unseen features to classes.
3. Online Adversarial Regret Minimization: McMahan et al. (2013) FTRL-Proximal with per-coordinate rates and exact L1 sparsity.
4. Concept Drift Detection: Page-Hinkley / CUSUM test for detecting non-stationary distribution shifts.
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from safetensors.torch import load_file


class TransductiveLabelSpreading(nn.Module):
    """Transductive graph label spreading exploiting manifold structure of unlabeled data."""

    def __init__(self, alpha: float = 0.8) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        laplacian_s: Float[torch.Tensor, "num_nodes num_nodes"],
        y_init: Float[torch.Tensor, "num_nodes num_classes"],
    ) -> Tuple[
        Float[torch.Tensor, "num_nodes num_classes"],
        Float[torch.Tensor, "num_nodes num_classes"],
        Float[torch.Tensor, "num_nodes"],
    ]:
        num_nodes = laplacian_s.shape[0]
        eye_n = torch.eye(num_nodes, dtype=laplacian_s.dtype, device=laplacian_s.device)
        inv_mat = torch.linalg.inv(eye_n - self.alpha * laplacian_s)
        f_star = torch.matmul(inv_mat, y_init)
        f_probs = torch.softmax(f_star, dim=-1)
        preds = torch.argmax(f_probs, dim=-1).to(torch.float32)
        return f_star, f_probs, preds


class InductiveLinearPredictor(nn.Module):
    """Inductive predictor mapping out-of-sample points directly without graph retraining."""

    def __init__(self, w_ind: Float[torch.Tensor, "feature_dim"]) -> None:
        super().__init__()
        self.register_buffer("w_ind", w_ind)

    def forward(
        self, x_test: Float[torch.Tensor, "batch_size feature_dim"]
    ) -> Tuple[Float[torch.Tensor, "batch_size"], Float[torch.Tensor, "batch_size"]]:
        scores = torch.matmul(x_test, self.w_ind)
        preds = (scores >= 0.5).to(torch.float32)
        return scores, preds


class FTRLProximalOptimizer(nn.Module):
    """Follow-The-Regularized-Leader (FTRL-Proximal) online optimizer with exact L1 sparsity."""

    def __init__(
        self,
        dim: int,
        alpha: float = 0.1,
        beta: float = 1.0,
        lambda1: float = 0.5,
        lambda2: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(
        self, grad_stream: Float[torch.Tensor, "time_steps feature_dim"]
    ) -> Tuple[
        Float[torch.Tensor, "feature_dim"],
        Float[torch.Tensor, "feature_dim"],
        Float[torch.Tensor, "feature_dim"],
    ]:
        z = torch.zeros(self.dim, dtype=grad_stream.dtype, device=grad_stream.device)
        n = torch.zeros(self.dim, dtype=grad_stream.dtype, device=grad_stream.device)

        for t in range(grad_stream.shape[0]):
            g = grad_stream[t]
            w_t = torch.zeros(self.dim, dtype=grad_stream.dtype, device=grad_stream.device)
            for i in range(self.dim):
                if abs(float(z[i])) <= self.lambda1:
                    w_t[i] = 0.0
                else:
                    sign_z = 1.0 if z[i] > 0 else -1.0
                    denom = (self.beta + math.sqrt(float(n[i]))) / self.alpha + self.lambda2
                    w_t[i] = - (z[i] - sign_z * self.lambda1) / denom

            sigma_t = (torch.sqrt(n + g ** 2) - torch.sqrt(n)) / self.alpha
            z = z + g - sigma_t * w_t
            n = n + g ** 2

        final_w = torch.zeros(self.dim, dtype=grad_stream.dtype, device=grad_stream.device)
        for i in range(self.dim):
            if abs(float(z[i])) <= self.lambda1:
                final_w[i] = 0.0
            else:
                sign_z = 1.0 if z[i] > 0 else -1.0
                denom = (self.beta + math.sqrt(float(n[i]))) / self.alpha + self.lambda2
                final_w[i] = - (z[i] - sign_z * self.lambda1) / denom

        return final_w, z, n


class PageHinkleyCUSUMDetector(nn.Module):
    """Page-Hinkley cumulative sum (CUSUM) test for streaming concept drift detection."""

    def __init__(self, delta: float = 0.05, threshold: float = 0.5, mean_est: float = 0.10) -> None:
        super().__init__()
        self.delta = delta
        self.threshold = threshold
        self.mean_est = mean_est

    def forward(
        self, stream_errors: Float[torch.Tensor, "time_steps"]
    ) -> Tuple[
        Float[torch.Tensor, "time_steps"],
        Float[torch.Tensor, "time_steps"],
        Float[torch.Tensor, "time_steps"],
    ]:
        time_steps = stream_errors.shape[0]
        cusum_s = torch.zeros(time_steps, dtype=stream_errors.dtype, device=stream_errors.device)
        ph_stat = torch.zeros(time_steps, dtype=stream_errors.dtype, device=stream_errors.device)
        drift_flags = torch.zeros(time_steps, dtype=stream_errors.dtype, device=stream_errors.device)

        cum_s = 0.0
        min_s = 0.0
        for t in range(time_steps):
            cum_s += float(stream_errors[t] - self.mean_est - self.delta)
            if cum_s < min_s:
                min_s = cum_s
            stat = cum_s - min_s
            cusum_s[t] = cum_s
            ph_stat[t] = stat
            drift_flags[t] = 1.0 if stat > self.threshold else 0.0

        return cusum_s, ph_stat, drift_flags


class InductiveTransductiveOnlineUnifiedModule(nn.Module):
    """Unified module combining transductive, inductive, FTRL, and CUSUM with linear anchor."""

    def __init__(self, w_ind: torch.Tensor) -> None:
        super().__init__()
        self.transductive = TransductiveLabelSpreading(alpha=0.8)
        self.inductive = InductiveLinearPredictor(w_ind)
        self.ftrl = FTRLProximalOptimizer(dim=4, alpha=0.1, beta=1.0, lambda1=0.5, lambda2=0.1)
        self.cusum = PageHinkleyCUSUMDetector(delta=0.05, threshold=0.5, mean_est=0.10)
        self.linear = nn.Linear(8, 4)

    def forward(self, x: Float[torch.Tensor, "batch in_dim"]) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[InductiveTransductiveOnlineUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = InductiveTransductiveOnlineUnifiedModule(weights["w_ind"])
    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])
    return module, weights


if __name__ == "__main__":
    ckpt = (
        Path(__file__).parents[2]
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "12_inductive_vs_transductive_and_online_regret_vs_batch_iid.safetensors"
    )
    model, tensors = load_checkpoint(str(ckpt))

    # 1. Linear Anchor Parity
    out_anchor = model(tensors["x_anchor"])
    assert torch.allclose(out_anchor, tensors["linear_anchor_out"], atol=1e-5), "Linear anchor mismatch"

    # 2. Transductive Label Spreading
    f_star, f_probs, t_preds = model.transductive(tensors["laplacian_s"], tensors["y_init"])
    assert torch.allclose(f_star, tensors["f_star"], atol=1e-5), "F* mismatch"
    assert torch.allclose(f_probs, tensors["f_probs"], atol=1e-5), "F probs mismatch"
    assert torch.allclose(t_preds, tensors["transductive_preds"], atol=1e-5), "Transductive preds mismatch"

    # 3. Inductive Predictor
    x_unlabeled = tensors["node_coords"][4:]
    ind_scores, ind_preds = model.inductive(x_unlabeled)
    assert torch.allclose(ind_scores, tensors["inductive_scores"], atol=1e-5), "Inductive scores mismatch"
    assert torch.allclose(ind_preds, tensors["inductive_preds"], atol=1e-5), "Inductive preds mismatch"

    # 4. FTRL-Proximal Online Optimizer
    w_ftrl, z_acc, n_acc = model.ftrl(tensors["grad_stream"])
    assert torch.allclose(w_ftrl, tensors["final_w_ftrl"], atol=1e-5), "FTRL w mismatch"
    assert torch.allclose(z_acc, tensors["z_accum"], atol=1e-5), "FTRL z mismatch"
    assert torch.allclose(n_acc, tensors["n_accum"], atol=1e-5), "FTRL n mismatch"

    # 5. Page-Hinkley CUSUM Drift Detector
    s_cusum, ph_s, d_flags = model.cusum(tensors["stream_errors"])
    assert torch.allclose(s_cusum, tensors["cusum_s"], atol=1e-5), "CUSUM s mismatch"
    assert torch.allclose(ph_s, tensors["ph_stat"], atol=1e-5), "PH stat mismatch"
    assert torch.allclose(d_flags, tensors["drift_flags"], atol=1e-5), "Drift flags mismatch"

    print("PyTorch 2 Inductive vs. Transductive & Online Regret successfully verified with zero errors!")
