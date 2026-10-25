"""PyTorch 2.x implementation for Category 05 Topic 04:
Soup of Experts, SMEAR, Rewarded Soups, BTX, and Sparse Upcycling
in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class SoupOfExperts(nn.Module):
    """Soup of Experts (SMEAR: Soft Merging of Experts with Adaptive Routing).
    
    References:
        - Muqeeth et al. (2023): Soft Merging of Experts with Adaptive Routing (SMEAR)
        - Chronopoulou et al. (2023): AdapterSoup: Weight-Averaging of Adapters
        - Ostapenko et al. (2024): PHATGOOSE: Gradient-Optimized Routing of Subnetworks
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts

        # Parameter tensors for all experts: (E, D, D_ff), (E, D_ff), etc.
        self.experts_w1 = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        self.experts_b1 = nn.Parameter(torch.empty(num_experts, d_ff))
        self.experts_w2 = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        self.experts_b2 = nn.Parameter(torch.empty(num_experts, d_model))

        # Sequence-level router
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, S, D)
            
        Returns:
            Tuple of:
                - output: Merged parameter execution output of shape (B, S, D)
                - routing_weights: Sequence routing probabilities of shape (B, E)
        """
        b_sz, _, _ = x.shape
        x_pool = x.mean(dim=1)  # (B, D)
        routing_weights = F.softmax(self.router(x_pool), dim=-1)  # (B, E)

        out = torch.zeros_like(x)
        for b in range(b_sz):
            g = routing_weights[b]  # (E,)
            # Merge parameters dynamically in weight space
            merged_w1 = torch.sum(g.view(self.num_experts, 1, 1) * self.experts_w1, dim=0)
            merged_b1 = torch.sum(g.view(self.num_experts, 1) * self.experts_b1, dim=0)
            merged_w2 = torch.sum(g.view(self.num_experts, 1, 1) * self.experts_w2, dim=0)
            merged_b2 = torch.sum(g.view(self.num_experts, 1) * self.experts_b2, dim=0)

            # Execute single dense forward pass with merged parameters
            h = F.gelu(x[b] @ merged_w1 + merged_b1)
            out[b] = h @ merged_w2 + merged_b2

        return out, routing_weights


class ActivationMoE(nn.Module):
    """Conventional Activation-Space MoE for direct empirical contrast with SMEAR."""

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts

        self.experts_w1 = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        self.experts_b1 = nn.Parameter(torch.empty(num_experts, d_ff))
        self.experts_w2 = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        self.experts_b2 = nn.Parameter(torch.empty(num_experts, d_model))

        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_sz, _, _ = x.shape
        x_pool = x.mean(dim=1)
        routing_weights = F.softmax(self.router(x_pool), dim=-1)

        out = torch.zeros_like(x)
        for b in range(b_sz):
            g = routing_weights[b]
            for e in range(self.num_experts):
                h = F.gelu(x[b] @ self.experts_w1[e] + self.experts_b1[e])
                out_e = h @ self.experts_w2[e] + self.experts_b2[e]
                out[b] += g[e] * out_e

        return out


def evaluate_pareto_soup(
    x: torch.Tensor,
    experts_w1: torch.Tensor,
    experts_b1: torch.Tensor,
    experts_w2: torch.Tensor,
    experts_b2: torch.Tensor,
    alpha: float = 0.6,
) -> torch.Tensor:
    """Evaluates static weight interpolation (Rewarded Pareto Soup) between expert 0 and 1."""
    w1 = alpha * experts_w1[0] + (1.0 - alpha) * experts_w1[1]
    b1 = alpha * experts_b1[0] + (1.0 - alpha) * experts_b1[1]
    w2 = alpha * experts_w2[0] + (1.0 - alpha) * experts_w2[1]
    b2 = alpha * experts_b2[0] + (1.0 - alpha) * experts_b2[1]

    h = F.gelu(x @ w1 + b1)
    return h @ w2 + b2


def verify_against_checkpoint(checkpoint_path: Path | str | None = None) -> None:
    """Verifies PyTorch forward passes against golden safetensors checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.safetensors")

    weights = load_file(str(checkpoint_path))
    x = weights["input"]
    d_model = 16
    d_ff = 32
    num_experts = 4

    # 1. Soup of Experts (SMEAR) Verification
    smear = SoupOfExperts(d_model=d_model, d_ff=d_ff, num_experts=num_experts)
    with torch.no_grad():
        smear.experts_w1.copy_(weights["experts_w1"])
        smear.experts_b1.copy_(weights["experts_b1"])
        smear.experts_w2.copy_(weights["experts_w2"])
        smear.experts_b2.copy_(weights["experts_b2"])
        smear.router.weight.copy_(weights["router_weight"].t())
        smear.router.bias.copy_(weights["router_bias"])

        out_smear, _ = smear(x)
        err_smear = (out_smear - weights["smear_output"]).abs().max().item()
        assert err_smear < 1e-5, f"SMEAR output error {err_smear} >= 1e-5"

    # 2. Activation-Space MoE Verification
    act_moe = ActivationMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts)
    with torch.no_grad():
        act_moe.experts_w1.copy_(weights["experts_w1"])
        act_moe.experts_b1.copy_(weights["experts_b1"])
        act_moe.experts_w2.copy_(weights["experts_w2"])
        act_moe.experts_b2.copy_(weights["experts_b2"])
        act_moe.router.weight.copy_(weights["router_weight"].t())
        act_moe.router.bias.copy_(weights["router_bias"])

        out_act = act_moe(x)
        err_act = (out_act - weights["act_output"]).abs().max().item()
        assert err_act < 1e-5, f"Activation MoE output error {err_act} >= 1e-5"

    # 3. Rewarded Pareto Soup Verification
    out_pareto = evaluate_pareto_soup(
        x,
        weights["experts_w1"],
        weights["experts_b1"],
        weights["experts_w2"],
        weights["experts_b2"],
        alpha=0.6,
    )
    err_pareto = (out_pareto - weights["pareto_soup_output"]).abs().max().item()
    assert err_pareto < 1e-5, f"Pareto Soup error {err_pareto} >= 1e-5"

    print("All Category 05 Topic 04 PyTorch models verified against checkpoint with < 1e-5 max error!")


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    #!/usr/bin/env python3
    """Generate golden weights and inputs/outputs for Category 05 Topic 04:
    Soup of Experts, SMEAR, Rewarded Soups, BTX, and Sparse Upcycling.
    """

    from pathlib import Path
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file


    def main():
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 4
        d_model = 16
        d_ff = 32
        num_experts = 4

        # Input tensor: (B, S, D)
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

        # 1. Base pre-trained model (for Sparse Upcycling)
        base_w1 = torch.randn(d_model, d_ff, dtype=torch.float32) * 0.2
        base_b1 = torch.randn(d_ff, dtype=torch.float32) * 0.1
        base_w2 = torch.randn(d_ff, d_model, dtype=torch.float32) * 0.2
        base_b2 = torch.randn(d_model, dtype=torch.float32) * 0.1

        # 2. Upcycled & branched experts (shared pre-trained basin + task fine-tuning delta)
        experts_w1 = torch.zeros(num_experts, d_model, d_ff, dtype=torch.float32)
        experts_b1 = torch.zeros(num_experts, d_ff, dtype=torch.float32)
        experts_w2 = torch.zeros(num_experts, d_ff, d_model, dtype=torch.float32)
        experts_b2 = torch.zeros(num_experts, d_model, dtype=torch.float32)

        for e in range(num_experts):
            # Small fine-tuning perturbation from common base
            delta_w1 = torch.randn(d_model, d_ff) * 0.05
            delta_b1 = torch.randn(d_ff) * 0.02
            delta_w2 = torch.randn(d_ff, d_model) * 0.05
            delta_b2 = torch.randn(d_model) * 0.02

            experts_w1[e] = base_w1 + delta_w1
            experts_b1[e] = base_b1 + delta_b1
            experts_w2[e] = base_w2 + delta_w2
            experts_b2[e] = base_b2 + delta_b2

        # 3. Sequence-level router weights
        router_w = torch.randn(d_model, num_experts, dtype=torch.float32) * 0.2
        router_b = torch.randn(num_experts, dtype=torch.float32) * 0.1

        # ==============================================================
        # A. Soup of Experts (SMEAR) Dynamic Parameter Merging
        # ==============================================================
        # For each sequence b: pool sequence representations and compute routing weights
        x_pool = x.mean(dim=1)  # (B, D)
        routing_logits = x_pool @ router_w + router_b  # (B, E)
        routing_weights = F.softmax(routing_logits, dim=-1)  # (B, E)

        smear_output = torch.zeros_like(x)
        for b in range(batch_size):
            g = routing_weights[b]  # (E,)
            # Dynamically merge weights in parameter space
            merged_w1 = torch.sum(g.view(num_experts, 1, 1) * experts_w1, dim=0)  # (D, D_ff)
            merged_b1 = torch.sum(g.view(num_experts, 1) * experts_b1, dim=0)        # (D_ff,)
            merged_w2 = torch.sum(g.view(num_experts, 1, 1) * experts_w2, dim=0)  # (D_ff, D)
            merged_b2 = torch.sum(g.view(num_experts, 1) * experts_b2, dim=0)        # (D,)

            # Single dense forward pass for sequence b
            h = F.gelu(x[b] @ merged_w1 + merged_b1)
            smear_output[b] = h @ merged_w2 + merged_b2

        # ==============================================================
        # B. Contrast: Activation-Space MoE Forward
        # ==============================================================
        act_output = torch.zeros_like(x)
        for b in range(batch_size):
            g = routing_weights[b]
            for e in range(num_experts):
                h_e = F.gelu(x[b] @ experts_w1[e] + experts_b1[e])
                out_e = h_e @ experts_w2[e] + experts_b2[e]
                act_output[b] += g[e] * out_e

        # ==============================================================
        # C. Rewarded Pareto Soup (Static Weight Interpolation)
        # ==============================================================
        # Interpolating between Expert 0 and Expert 1 with alpha = 0.6
        alpha = 0.6
        pareto_w1 = alpha * experts_w1[0] + (1.0 - alpha) * experts_w1[1]
        pareto_b1 = alpha * experts_b1[0] + (1.0 - alpha) * experts_b1[1]
        pareto_w2 = alpha * experts_w2[0] + (1.0 - alpha) * experts_w2[1]
        pareto_b2 = alpha * experts_b2[0] + (1.0 - alpha) * experts_b2[1]

        h_p = F.gelu(x @ pareto_w1 + pareto_b1)
        pareto_soup_output = h_p @ pareto_w2 + pareto_b2

        # Standard linear parity tensors for test_cross_track_parity.py
        D_core = 4
        test_input = torch.randn(8, D_core)
        weight = torch.randn(D_core, D_core)
        bias = torch.randn(D_core)
        test_output = test_input @ weight + bias

        # Output directory
        out_dir = Path("weights/05_architectures/01_mlp_moe_soups_and_kan")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.safetensors"

        tensors = {
            "weight": weight.clone().contiguous(),
            "bias": bias.clone().contiguous(),
            "test_input": test_input.clone().contiguous(),
            "test_output": test_output.clone().contiguous(),
            "input": x,
            "base_w1": base_w1,
            "base_b1": base_b1,
            "base_w2": base_w2,
            "base_b2": base_b2,
            "experts_w1": experts_w1,
            "experts_b1": experts_b1,
            "experts_w2": experts_w2,
            "experts_b2": experts_b2,
            "router_weight": router_w,
            "router_bias": router_b,
            "smear_output": smear_output,
            "act_output": act_output,
            "pareto_soup_output": pareto_soup_output,
        }

        save_file(tensors, str(out_file))
        print(f"Generated {out_file} successfully.")
        print(f"smear_output shape: {smear_output.shape}, norm: {smear_output.norm():.6f}")
        print(f"act_output shape: {act_output.shape}, norm: {act_output.norm():.6f}")
        print(f"pareto_soup_output shape: {pareto_soup_output.shape}, norm: {pareto_soup_output.norm():.6f}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_against_checkpoint()
