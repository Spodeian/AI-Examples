"""PyTorch 2.x implementation for Category 05 Topic 03:
DeepSeekMoE, Auxiliary-Loss-Free Load Balancing, and Mixture of Depths (MoD)
in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class ExpertMLP(nn.Module):
    """Feedforward expert MLP with GELU activation."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class DeepSeekMoE(nn.Module):
    """DeepSeekMoE with Fine-Grained + Shared Experts and Auxiliary-Loss-Free Load Balancing.
    
    References:
        - Dai et al. (2024): DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts
        - DeepSeek-V3 Technical Report (2024): Auxiliary-Loss-Free Load Balancing Strategy
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_shared: int = 1,
        num_routed: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_shared = num_shared
        self.num_routed = num_routed
        self.top_k = top_k

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([ExpertMLP(d_model, d_ff) for _ in range(num_shared)])

        # Routed experts and router
        self.router = nn.Linear(d_model, num_routed)
        self.routed_experts = nn.ModuleList([ExpertMLP(d_model, d_ff) for _ in range(num_routed)])

        # Dynamic bias for aux-loss-free balancing (non-parameter buffer updated outside autograd)
        self.register_buffer("b_dyn", torch.zeros(num_routed))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, S, D) or (T, D)
            
        Returns:
            Tuple of:
                - output: Same shape as x
                - expert_counts: 1D tensor of shape (num_routed,) with token frequencies
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        num_tokens = x_flat.shape[0]

        # 1. Unconditionally evaluate all shared experts
        out_flat = torch.zeros_like(x_flat)
        for shared in self.shared_experts:
            out_flat += shared(x_flat)

        # 2. Router logits and aux-loss-free top-k selection
        router_logits = self.router(x_flat)  # (T, N_r)
        biased_logits = router_logits + self.b_dyn  # Used ONLY for index selection

        _, topk_indices = torch.topk(biased_logits, k=self.top_k, dim=-1)  # (T, k)

        # Invariant: Gating weights use UNBIASED logits to prevent gradient distortion
        unbiased_topk_logits = torch.gather(router_logits, dim=-1, index=topk_indices)
        topk_weights = F.softmax(unbiased_topk_logits, dim=-1)  # (T, k)

        # 3. Accumulate routed experts
        expert_counts = torch.zeros(self.num_routed, device=x.device)
        for t in range(num_tokens):
            for k_idx in range(self.top_k):
                e = topk_indices[t, k_idx].item()
                w = topk_weights[t, k_idx]
                expert_counts[e] += 1.0
                out_flat[t] += w * self.routed_experts[e](x_flat[t : t + 1]).squeeze(0)

        output = out_flat.view(orig_shape)
        return output, expert_counts

    def update_dynamic_bias(self, expert_counts: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
        """Updates the dynamic bias vector outside autograd based on load discrepancy."""
        with torch.no_grad():
            avg_count = expert_counts.mean()
            self.b_dyn.add_(gamma * torch.sign(avg_count - expert_counts))
        return self.b_dyn


class MixtureOfDepths(nn.Module):
    """Mixture of Depths (MoD) Layer (Raposo et al., DeepMind 2024).
    
    Dynamically routes a subset of tokens through the block while allowing the remaining
    tokens to bypass via an identity residual skip connection, enforcing fixed capacity C.
    """

    def __init__(self, d_model: int, block: nn.Module, capacity: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.block = block
        self.capacity = capacity
        self.router = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, S, D)
            
        Returns:
            Output tensor of shape (B, S, D) with capacity tokens computed and rest bypassed.
        """
        b_sz, s_len, _ = x.shape
        router_logits = self.router(x).squeeze(-1)  # (B, S)
        router_weights = torch.sigmoid(router_logits)  # (B, S)

        # Select top-C tokens per sequence to execute block
        _, topk_indices = torch.topk(router_weights, k=self.capacity, dim=-1)  # (B, C)

        out = x.clone()
        for b in range(b_sz):
            for c_idx in range(self.capacity):
                s = topk_indices[b, c_idx].item()
                token_in = x[b, s : s + 1]
                gate = router_weights[b, s]
                block_out = self.block(token_in)
                out[b, s] = token_in.squeeze(0) + gate * block_out.squeeze(0)

        return out


def verify_against_checkpoint(checkpoint_path: Path | str | None = None) -> None:
    """Verifies PyTorch forward passes against golden safetensors checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.safetensors")

    weights = load_file(str(checkpoint_path))
    x = weights["input"]
    d_model = 16
    d_ff = 16
    num_shared = 1
    num_routed = 4
    top_k = 2

    # 1. DeepSeekMoE Verification
    ds_moe = DeepSeekMoE(d_model=d_model, d_ff=d_ff, num_shared=num_shared, num_routed=num_routed, top_k=top_k)
    with torch.no_grad():
        ds_moe.b_dyn.copy_(weights["b_dyn"])
        ds_moe.router.weight.copy_(weights["router_weight"].t())
        ds_moe.router.bias.copy_(weights["router_bias"])
        for s in range(num_shared):
            ds_moe.shared_experts[s].fc1.weight.copy_(weights["shared_w1"][s].t())
            ds_moe.shared_experts[s].fc1.bias.copy_(weights["shared_b1"][s])
            ds_moe.shared_experts[s].fc2.weight.copy_(weights["shared_w2"][s].t())
            ds_moe.shared_experts[s].fc2.bias.copy_(weights["shared_b2"][s])
        for r in range(num_routed):
            ds_moe.routed_experts[r].fc1.weight.copy_(weights["routed_w1"][r].t())
            ds_moe.routed_experts[r].fc1.bias.copy_(weights["routed_b1"][r])
            ds_moe.routed_experts[r].fc2.weight.copy_(weights["routed_w2"][r].t())
            ds_moe.routed_experts[r].fc2.bias.copy_(weights["routed_b2"][r])

        out_ds, counts = ds_moe(x)
        err_ds = (out_ds - weights["deepseek_moe_output"]).abs().max().item()
        assert err_ds < 1e-5, f"DeepSeekMoE output error {err_ds} >= 1e-5"

        # Update dynamic bias
        b_updated = ds_moe.update_dynamic_bias(counts, gamma=0.1)
        err_b = (b_updated - weights["b_dyn_updated"]).abs().max().item()
        assert err_b < 1e-5, f"Dynamic bias update error {err_b} >= 1e-5"

    # 2. Mixture of Depths Verification
    shared_block = ds_moe.shared_experts[0]
    mod = MixtureOfDepths(d_model=d_model, block=shared_block, capacity=2)
    with torch.no_grad():
        mod.router.weight.copy_(weights["mod_weight"].t())
        mod.router.bias.copy_(weights["mod_bias"])

        out_mod = mod(x)
        err_mod = (out_mod - weights["mod_output"]).abs().max().item()
        assert err_mod < 1e-5, f"Mixture of Depths error {err_mod} >= 1e-5"

    print("All Category 05 Topic 03 PyTorch models verified against checkpoint with < 1e-5 max error!")


if __name__ == "__main__":
    verify_against_checkpoint()
