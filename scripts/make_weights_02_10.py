"""Generates reference SafeTensors checkpoint for Category 02 Topic 10:
10_survival_analysis_and_contextual_bandits.safetensors
"""

from __future__ import annotations
from pathlib import Path
import torch
import safetensors.torch


def compute_km_survival(T: torch.Tensor, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Kaplan-Meier survival curve at unique event times."""
    event_mask = E == 1
    unique_times = torch.unique(T[event_mask], sorted=True)
    surv_probs = []
    current_s = 1.0

    for t_i in unique_times:
        n_i = (T >= t_i).sum().float()
        d_i = ((T == t_i) & (E == 1)).sum().float()
        current_s = current_s * (1.0 - d_i / n_i)
        surv_probs.append(current_s)

    return unique_times, torch.tensor(surv_probs, dtype=torch.float32)


def compute_cox_partial_log_likelihood(
    risk_scores: torch.Tensor,
    T: torch.Tensor,
    E: torch.Tensor,
) -> torch.Tensor:
    """Computes negative Cox partial log-likelihood using logsumexp over risk sets."""
    order = torch.argsort(T, descending=True)
    sorted_risk = risk_scores[order]
    sorted_E = E[order]

    # Suffix logsumexp for risk sets
    # Because times are descending, risk set for item i is 0..i
    # Wait: R(T_i) = {j: T_j >= T_i}.
    # If sorted descending (largest time first):
    # For item i with time T_i, all items j <= i have T_j >= T_i!
    # So R(T_i) is the prefix 0..i in descending order!
    # Let's verify:
    # Let times be [10, 8, 5, 2].
    # For time 5 (index 2): items with T >= 5 are indices 0, 1, 2! Correct!
    # So logsumexp of prefix 0..i.
    
    # Prefix logsumexp:
    log_risk_set = torch.logcumsumexp(sorted_risk, dim=0)

    # Loss is sum over uncensored events: -(risk_i - log_sum_R(i))
    event_mask = sorted_E == 1
    loss = -torch.sum(sorted_risk[event_mask] - log_risk_set[event_mask])
    return loss


def compute_linucb_scores(
    x_context: torch.Tensor,
    A_arms: torch.Tensor,
    b_arms: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes LinUCB exploration scores across arms."""
    num_arms = A_arms.shape[0]
    scores = torch.zeros(num_arms)

    for a in range(num_arms):
        A_inv = torch.linalg.inv(A_arms[a])
        theta_hat = A_inv @ b_arms[a]
        mean_reward = torch.dot(x_context, theta_hat)
        conf_width = alpha * torch.sqrt(x_context @ A_inv @ x_context)
        scores[a] = mean_reward + conf_width

    selected = torch.argmax(scores)
    return scores, selected


def generate_checkpoint():
    torch.manual_seed(42)
    N, D = 32, 4
    K_arms = 3

    # Survival data
    X_surv = torch.randn(N, D)
    w_surv = torch.tensor([0.4, -0.6, 0.8, -0.3], dtype=torch.float32)
    b_surv = torch.tensor([0.2], dtype=torch.float32)

    risk_scores = X_surv @ w_surv + b_surv
    hazard_ratios = torch.exp(risk_scores)

    # Synthetic survival times and censoring
    base_times = torch.rand(N) * 20.0 + 1.0
    # Higher risk -> shorter event time
    T_surv = base_times / (hazard_ratios ** 0.5)
    # Binary event indicators (approx 75% uncensored, 25% censored)
    E_surv = (torch.rand(N) > 0.25).float()

    cox_loss = compute_cox_partial_log_likelihood(risk_scores, T_surv, E_surv)
    km_times, km_survival = compute_km_survival(T_surv, E_surv)

    # Contextual Bandits (LinUCB)
    A_arms = torch.stack([torch.eye(D) * 1.0 for _ in range(K_arms)])
    # Add some historical context to arms
    b_arms = torch.zeros(K_arms, D)
    for a in range(K_arms):
        hist_x = torch.randn(5, D)
        hist_r = torch.rand(5)
        A_arms[a] += hist_x.T @ hist_x
        b_arms[a] += hist_x.T @ hist_r

    x_context = torch.tensor([0.5, -0.2, 0.8, 0.1], dtype=torch.float32)
    linucb_scores, selected_arm = compute_linucb_scores(x_context, A_arms, b_arms, alpha=1.0)

    # Golden linear mapping for test_cross_track_parity.py contract
    test_input = torch.randn(8, D)
    weight = torch.randn(D, D)
    bias = torch.randn(D)
    test_output = test_input @ weight + bias

    checkpoint_data = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": test_input.clone().contiguous(),
        "test_output": test_output.clone().contiguous(),
        "X_surv": X_surv.clone().contiguous(),
        "T_surv": T_surv.clone().contiguous(),
        "E_surv": E_surv.clone().contiguous(),
        "w_surv": w_surv.clone().contiguous(),
        "b_surv": b_surv.clone().contiguous(),
        "risk_scores": risk_scores.clone().contiguous(),
        "hazard_ratios": hazard_ratios.clone().contiguous(),
        "cox_loss": cox_loss.unsqueeze(0).clone().contiguous(),
        "km_times": km_times.clone().contiguous(),
        "km_survival": km_survival.clone().contiguous(),
        "A_arms": A_arms.clone().contiguous(),
        "b_arms": b_arms.clone().contiguous(),
        "x_context": x_context.clone().contiguous(),
        "linucb_scores": linucb_scores.clone().contiguous(),
        "selected_arm": selected_arm.unsqueeze(0).clone().contiguous(),
    }

    out_dir = Path("weights/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "10_survival_analysis_and_contextual_bandits.safetensors"
    safetensors.torch.save_file(checkpoint_data, out_path)
    print("Saved Topic 10 reference checkpoint to", out_path)


if __name__ == "__main__":
    generate_checkpoint()
