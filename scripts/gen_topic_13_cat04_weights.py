"""Generate golden reference safetensors checkpoint for Category 04 Topic 13:
06_sharpness_aware_swad_and_weight_averaging_dynamics
in 04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation."""

from pathlib import Path
import torch
from safetensors.torch import save_file


def step_sam(
    param: torch.Tensor,
    grad_at_w: torch.Tensor,
    grad_at_pert: torch.Tensor,
    rho: float = 0.05,
    lr: float = 0.01,
    weight_decay: float = 0.001,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sharpness-Aware Minimization (SAM) update step.
    
    1. Perturbation: epsilon = rho * grad_at_w / (||grad_at_w||_2 + eps)
    2. Update: param_next = param - lr * (grad_at_pert + weight_decay * param)
    """
    grad_norm = torch.norm(grad_at_w, p=2) + eps
    epsilon = rho * (grad_at_w / grad_norm)
    update = grad_at_pert + weight_decay * param
    param_next = param - lr * update
    return param_next, epsilon


def step_lookahead(
    slow_param: torch.Tensor,
    fast_param: torch.Tensor,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lookahead synchronization step:
    slow_next = slow + alpha * (fast - slow)
    fast_next = slow_next
    """
    slow_next = slow_param + alpha * (fast_param - slow_param)
    fast_next = slow_next.clone()
    return slow_next, fast_next


def step_swa_ema(
    swa_avg: torch.Tensor,
    ema_avg: torch.Tensor,
    current_param: torch.Tensor,
    n_models: int,
    decay: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SWA running average and EMA step."""
    new_swa = (swa_avg * n_models + current_param) / (n_models + 1)
    new_ema = decay * ema_avg + (1.0 - decay) * current_param
    return new_swa, new_ema


def uniform_model_soup(models: torch.Tensor) -> torch.Tensor:
    """Average a stack of M models: models shape [M, d]."""
    return torch.mean(models, dim=0)


def main():
    torch.manual_seed(42)

    # 1. SAM Invariant
    param_sam = torch.tensor([1.2, -0.5, 0.8, -1.4], dtype=torch.float32)
    grad_at_w = torch.tensor([0.3, -0.1, 0.4, -0.2], dtype=torch.float32)
    grad_at_pert = torch.tensor([0.35, -0.08, 0.42, -0.18], dtype=torch.float32)
    golden_param_sam, golden_epsilon_sam = step_sam(
        param_sam, grad_at_w, grad_at_pert, rho=0.05, lr=0.01, weight_decay=0.001
    )

    # 2. Lookahead Invariant
    slow_p = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float32)
    fast_p = torch.tensor([1.4, 1.8, -0.6, 0.9], dtype=torch.float32)
    golden_slow_p, golden_fast_p = step_lookahead(slow_p, fast_p, alpha=0.5)

    # 3. SWA & EMA Invariant
    swa_avg = torch.tensor([0.8, 1.2, -0.5, 0.4], dtype=torch.float32)
    ema_avg = torch.tensor([0.85, 1.15, -0.48, 0.42], dtype=torch.float32)
    curr_param = torch.tensor([0.9, 1.1, -0.4, 0.45], dtype=torch.float32)
    golden_swa, golden_ema = step_swa_ema(swa_avg, ema_avg, curr_param, n_models=4, decay=0.99)

    # 4. Model Soup Invariant
    soup_models = torch.tensor([
        [1.0, 2.0, -1.0, 0.5],
        [1.2, 1.9, -0.9, 0.6],
        [0.9, 2.1, -1.1, 0.4],
    ], dtype=torch.float32)
    golden_soup = uniform_model_soup(soup_models)

    # 5. Linear Anchor for cross-track parity
    anchor_x = torch.randn(2, 8, dtype=torch.float32)
    anchor_w = torch.randn(8, 4, dtype=torch.float32)
    anchor_b = torch.randn(4, dtype=torch.float32)
    golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

    tensors = {
        "param_sam": param_sam,
        "grad_at_w": grad_at_w,
        "grad_at_pert": grad_at_pert,
        "golden_param_sam": golden_param_sam,
        "golden_epsilon_sam": golden_epsilon_sam,
        "slow_p": slow_p,
        "fast_p": fast_p,
        "golden_slow_p": golden_slow_p,
        "golden_fast_p": golden_fast_p,
        "swa_avg": swa_avg,
        "ema_avg": ema_avg,
        "curr_param": curr_param,
        "golden_swa": golden_swa,
        "golden_ema": golden_ema,
        "soup_models": soup_models,
        "golden_soup": golden_soup,
        "anchor_x": anchor_x,
        "anchor_w": anchor_w,
        "anchor_b": anchor_b,
        "golden_anchor_out": golden_anchor_out,
        # Standard Parity keys
        "weight": anchor_w.clone(),
        "bias": anchor_b.clone(),
        "test_input": anchor_x.clone(),
        "test_output": golden_anchor_out.clone(),
    }

    out_dir = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "06_sharpness_aware_swad_and_weight_averaging_dynamics.safetensors"
    save_file(tensors, str(out_file))
    print(f"Saved {len(tensors)} tensors to {out_file}")


if __name__ == "__main__":
    main()
