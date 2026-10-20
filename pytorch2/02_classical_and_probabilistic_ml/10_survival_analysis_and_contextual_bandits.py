"""Survival Analysis and Contextual Bandits in PyTorch 2.x.

Features:
- Nonparametric Kaplan-Meier survival estimator
- Cox Proportional Hazards (DeepSurv) negative partial log-likelihood using logcumsumexp over risk sets
- Linear Upper Confidence Bound (LinUCB) contextual bandit policy with Sherman-Morrison rank-1 update
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import safetensors.torch


def kaplan_meier_survival(
    T: torch.Tensor,
    E: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Kaplan-Meier survival curve S(t) = prod_{t_i <= t} (1 - d_i / n_i)."""
    event_mask = E == 1
    unique_times = torch.unique(T[event_mask], sorted=True)
    surv_probs = []
    current_s = torch.tensor(1.0, device=T.device, dtype=T.dtype)

    for t_i in unique_times:
        n_i = (T >= t_i).sum().to(T.dtype)
        d_i = ((T == t_i) & (E == 1)).sum().to(T.dtype)
        current_s = current_s * (1.0 - d_i / n_i)
        surv_probs.append(current_s)

    return unique_times, torch.stack(surv_probs)


def cox_ph_loss(
    risk_scores: torch.Tensor,
    T: torch.Tensor,
    E: torch.Tensor,
) -> torch.Tensor:
    """Computes negative Cox partial log-likelihood via prefix logcumsumexp on descending event times."""
    order = torch.argsort(T, descending=True)
    sorted_risk = risk_scores[order]
    sorted_E = E[order]

    # In descending order, risk set R(T_i) for item i is all items 0..i
    log_risk_set = torch.logcumsumexp(sorted_risk, dim=0)

    event_mask = sorted_E == 1
    loss = -torch.sum(sorted_risk[event_mask] - log_risk_set[event_mask])
    return loss


def linucb_score_arms(
    x_context: torch.Tensor,
    A_arms: torch.Tensor,
    b_arms: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes UCB exploration scores across disjoint linear bandit arms."""
    num_arms = A_arms.shape[0]
    scores = torch.zeros(num_arms, device=x_context.device, dtype=x_context.dtype)

    for a in range(num_arms):
        A_inv = torch.linalg.inv(A_arms[a])
        theta_hat = A_inv @ b_arms[a]
        mean_reward = torch.dot(x_context, theta_hat)
        conf_width = alpha * torch.sqrt(x_context @ A_inv @ x_context)
        scores[a] = mean_reward + conf_width

    selected = torch.argmax(scores)
    return scores, selected


def sherman_morrison_rank1_update(
    A_inv: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Updates matrix inverse A_new^{-1} in O(d^2) time after rank-1 addition A + x x^T."""
    # (A + x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
    A_inv_x = A_inv @ x
    denom = 1.0 + torch.dot(x, A_inv_x)
    return A_inv - torch.outer(A_inv_x, A_inv_x) / denom


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/10_survival_analysis_and_contextual_bandits.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    X_surv = chk["X_surv"]
    T_surv = chk["T_surv"]
    E_surv = chk["E_surv"]
    w_surv = chk["w_surv"]
    b_surv = chk["b_surv"]
    golden_risk = chk["risk_scores"]
    golden_cox_loss = chk["cox_loss"][0].item()
    golden_km_times = chk["km_times"]
    golden_km_survival = chk["km_survival"]

    # 1. Cox Partial Log-Likelihood Parity
    risk = X_surv @ w_surv + b_surv
    diff_risk = torch.max(torch.abs(risk - golden_risk)).item()
    assert diff_risk < 1e-5, f"Risk score mismatch: {diff_risk:.2e}"

    loss = cox_ph_loss(risk, T_surv, E_surv).item()
    diff_loss = abs(loss - golden_cox_loss)
    assert diff_loss < 1e-4, f"Cox partial log likelihood mismatch: {diff_loss:.2e}"

    # 2. Kaplan-Meier Parity
    km_t, km_s = kaplan_meier_survival(T_surv, E_surv)
    assert torch.equal(km_t, golden_km_times), "KM event times mismatch"
    diff_km = torch.max(torch.abs(km_s - golden_km_survival)).item()
    assert diff_km < 1e-5, f"Kaplan-Meier survival mismatch: {diff_km:.2e}"

    # 3. LinUCB Parity
    A_arms = chk["A_arms"]
    b_arms = chk["b_arms"]
    x_ctx = chk["x_context"]
    golden_scores = chk["linucb_scores"]
    golden_arm = chk["selected_arm"][0].item()

    scores, selected = linucb_score_arms(x_ctx, A_arms, b_arms, alpha=1.0)
    diff_scores = torch.max(torch.abs(scores - golden_scores)).item()
    assert diff_scores < 1e-5, f"LinUCB scores mismatch: {diff_scores:.2e}"
    assert selected.item() == golden_arm, f"Selected arm mismatch: {selected} vs {golden_arm}"

    # 4. Sherman-Morrison Inversion Invariant Check
    A_inv_0 = torch.linalg.inv(A_arms[0])
    A_inv_updated = sherman_morrison_rank1_update(A_inv_0, x_ctx)
    A_exact_inv = torch.linalg.inv(A_arms[0] + torch.outer(x_ctx, x_ctx))
    diff_sm = torch.max(torch.abs(A_inv_updated - A_exact_inv)).item()
    assert diff_sm < 1e-5, f"Sherman-Morrison update error: {diff_sm:.2e}"

    print("PyTorch Topic 10 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
