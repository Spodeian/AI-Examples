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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    #!/usr/bin/env python3
    """Generate golden weights and inputs/outputs for Category 05 Topic 03:
    DeepSeekMoE, Auxiliary-Loss-Free Load Balancing, and Mixture of Depths (MoD).
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
        d_ff = 16
        num_shared = 1
        num_routed = 4
        top_k = 2
        tokens = batch_size * seq_len  # 8

        # Input tensor
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
        x_flat = x.view(tokens, d_model)

        # 1. Router weights for routed experts: (D, N_r)
        router_w = torch.randn(d_model, num_routed, dtype=torch.float32) * 0.2
        router_b = torch.randn(num_routed, dtype=torch.float32) * 0.1

        # Dynamic routing bias b_dyn for aux-loss-free balancing: (N_r,)
        b_dyn = torch.tensor([0.1, -0.05, 0.05, -0.1], dtype=torch.float32)

        # 2. Shared expert weights: (N_s, D, D_ff), (N_s, D_ff), (N_s, D_ff, D), (N_s, D)
        shared_w1 = torch.randn(num_shared, d_model, d_ff, dtype=torch.float32) * 0.2
        shared_b1 = torch.randn(num_shared, d_ff, dtype=torch.float32) * 0.1
        shared_w2 = torch.randn(num_shared, d_ff, d_model, dtype=torch.float32) * 0.2
        shared_b2 = torch.randn(num_shared, d_model, dtype=torch.float32) * 0.1

        # 3. Routed expert weights: (N_r, D, D_ff), (N_r, D_ff), (N_r, D_ff, D), (N_r, D)
        routed_w1 = torch.randn(num_routed, d_model, d_ff, dtype=torch.float32) * 0.2
        routed_b1 = torch.randn(num_routed, d_ff, dtype=torch.float32) * 0.1
        routed_w2 = torch.randn(num_routed, d_ff, d_model, dtype=torch.float32) * 0.2
        routed_b2 = torch.randn(num_routed, d_model, dtype=torch.float32) * 0.1

        # 4. Mixture of Depths (MoD) router: (D, 1)
        mod_w = torch.randn(d_model, 1, dtype=torch.float32) * 0.2
        mod_b = torch.randn(1, dtype=torch.float32) * 0.1

        # Helpers for expert evaluations
        def eval_shared(s_idx: int, inp: torch.Tensor) -> torch.Tensor:
            h = F.gelu(inp @ shared_w1[s_idx] + shared_b1[s_idx])
            return h @ shared_w2[s_idx] + shared_b2[s_idx]

        def eval_routed(r_idx: int, inp: torch.Tensor) -> torch.Tensor:
            h = F.gelu(inp @ routed_w1[r_idx] + routed_b1[r_idx])
            return h @ routed_w2[r_idx] + routed_b2[r_idx]

        # ==============================================================
        # A. DeepSeekMoE with Aux-Loss-Free Load Balancing
        # ==============================================================
        # Router logits: (T, N_r)
        router_logits = x_flat @ router_w + router_b

        # Aux-loss-free top-k selection uses biased scores (logits + b_dyn)
        biased_logits = router_logits + b_dyn
        _, topk_indices = torch.topk(biased_logits, k=top_k, dim=-1)  # (T, k)

        # CRITICAL INVARIANT: Gating weights use UNBIASED router_logits!
        unbiased_topk_logits = torch.gather(router_logits, dim=-1, index=topk_indices)
        topk_weights = F.softmax(unbiased_topk_logits, dim=-1)  # (T, k)

        deepseek_out_flat = torch.zeros_like(x_flat)
        expert_counts = torch.zeros(num_routed, dtype=torch.float32)

        for t in range(tokens):
            # 1. Unconditionally evaluate all shared experts
            for s in range(num_shared):
                deepseek_out_flat[t] += eval_shared(s, x_flat[t : t + 1]).squeeze(0)

            # 2. Add weighted routed experts
            for k_idx in range(top_k):
                e = topk_indices[t, k_idx].item()
                w = topk_weights[t, k_idx]
                expert_counts[e] += 1.0
                deepseek_out_flat[t] += w * eval_routed(e, x_flat[t : t + 1]).squeeze(0)

        deepseek_moe_output = deepseek_out_flat.view(batch_size, seq_len, d_model)

        # Dynamic bias update simulation:
        # b_dyn_new = b_dyn + gamma * sign(avg_count - count)
        gamma = 0.1
        avg_count = expert_counts.mean()
        b_dyn_updated = b_dyn + gamma * torch.sign(avg_count - expert_counts)

        # ==============================================================
        # B. Mixture of Depths (MoD)
        # ==============================================================
        # For each sequence, select top C = 2 tokens to execute the shared FFN block
        mod_capacity = 2
        mod_router_logits = (x @ mod_w + mod_b).squeeze(-1)  # (B, S)
        mod_router_weights = torch.sigmoid(mod_router_logits)  # (B, S)

        _, mod_topk_indices = torch.topk(mod_router_weights, k=mod_capacity, dim=-1)  # (B, C)

        mod_output = x.clone()
        for b in range(batch_size):
            for c_idx in range(mod_capacity):
                s = mod_topk_indices[b, c_idx].item()
                token_in = x[b, s : s + 1]
                gate = mod_router_weights[b, s]
                # Execute shared expert block scaled by router gate
                block_out = eval_shared(0, token_in)
                mod_output[b, s] = token_in.squeeze(0) + gate * block_out.squeeze(0)

        # Standard linear parity tensors for test_cross_track_parity.py
        D_core = 4
        test_input = torch.randn(8, D_core)
        weight = torch.randn(D_core, D_core)
        bias = torch.randn(D_core)
        test_output = test_input @ weight + bias

        # Output directory
        out_dir = Path("weights/05_architectures/01_mlp_moe_soups_and_kan")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "03_deepseek_moe_aux_loss_free_and_mixture_of_depths.safetensors"

        tensors = {
            "weight": weight.clone().contiguous(),
            "bias": bias.clone().contiguous(),
            "test_input": test_input.clone().contiguous(),
            "test_output": test_output.clone().contiguous(),
            "input": x,
            "router_weight": router_w,
            "router_bias": router_b,
            "b_dyn": b_dyn,
            "b_dyn_updated": b_dyn_updated,
            "shared_w1": shared_w1,
            "shared_b1": shared_b1,
            "shared_w2": shared_w2,
            "shared_b2": shared_b2,
            "routed_w1": routed_w1,
            "routed_b1": routed_b1,
            "routed_w2": routed_w2,
            "routed_b2": routed_b2,
            "mod_weight": mod_w,
            "mod_bias": mod_b,
            "deepseek_moe_output": deepseek_moe_output,
            "mod_output": mod_output,
        }

        save_file(tensors, str(out_file))
        print(f"Generated {out_file} successfully.")
        print(f"deepseek_moe_output shape: {deepseek_moe_output.shape}, norm: {deepseek_moe_output.norm():.6f}")
        print(f"mod_output shape: {mod_output.shape}, norm: {mod_output.norm():.6f}")
        print(f"b_dyn_updated: {b_dyn_updated.tolist()}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_against_checkpoint()
