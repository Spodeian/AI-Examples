"""PyTorch 2.x implementation for Category 05 Topic 02:
Mixture of Experts (Top-K, Expert Choice, and Soft MoE) in 05_architectures/01_mlp_moe_soups_and_kan.
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


class TopKMoE(nn.Module):
    """Sparse Top-K Mixture of Experts with Auxiliary Load-Balancing and Router Z-loss.
    
    References:
        - Shazeer et al. (2017): Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer
        - Fedus et al. (2021): Switch Transformers
        - Zoph et al. (2022): ST-MoE
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([ExpertMLP(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, S, D) or (T, D)
            
        Returns:
            Tuple of:
                - output: Same shape as x
                - aux_loss: Scalar auxiliary load balancing loss E * sum(f_i * P_i)
                - z_loss: Scalar router z-loss (1/T) * sum(logsumexp(H)^2)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        num_tokens = x_flat.shape[0]

        # Router logits and probabilities
        router_logits = self.router(x_flat)  # (T, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (T, E)

        # Top-k selection
        topk_vals, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)  # (T, k)
        topk_weights = F.softmax(topk_vals, dim=-1)  # (T, k)

        # Accumulate expert outputs
        out_flat = torch.zeros_like(x_flat)
        for t in range(num_tokens):
            for k_idx in range(self.top_k):
                e = topk_indices[t, k_idx].item()
                w = topk_weights[t, k_idx]
                out_flat[t] += w * self.experts[e](x_flat[t : t + 1]).squeeze(0)

        # Auxiliary load balancing loss: E * sum_i(f_i * P_i)
        f_count = torch.zeros(self.num_experts, dtype=torch.float32, device=x.device)
        for t in range(num_tokens):
            for k_idx in range(self.top_k):
                e = topk_indices[t, k_idx].item()
                f_count[e] += 1.0
        f_i = f_count / num_tokens
        P_i = router_probs.mean(dim=0)
        aux_loss = (self.num_experts * torch.sum(f_i * P_i)).view(1)

        # Router Z-loss
        lse = torch.logsumexp(router_logits, dim=-1)
        z_loss = torch.mean(lse**2).view(1)

        output = out_flat.view(orig_shape)
        return output, aux_loss, z_loss


class ExpertChoiceMoE(nn.Module):
    """Expert Choice Mixture of Experts (Zhou et al., 2022).
    
    Instead of tokens selecting top-k experts, each expert chooses its top-C tokens.
    Guarantees perfectly balanced expert computation with zero dropped tokens or padding.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, capacity: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity = capacity

        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([ExpertMLP(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)

        router_logits = self.router(x_flat)  # (T, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (T, E)

        # Transposed routing matrix: (E, T)
        G = router_probs.t()
        out_flat = torch.zeros_like(x_flat)

        for e in range(self.num_experts):
            expert_scores = G[e]  # (T,)
            _, selected_tokens = torch.topk(expert_scores, k=self.capacity, dim=-1)
            for t_idx in selected_tokens:
                t = t_idx.item()
                score = G[e, t]
                out_flat[t] += score * self.experts[e](x_flat[t : t + 1]).squeeze(0)

        return out_flat.view(orig_shape)


class SoftMoE(nn.Module):
    """Soft Mixture of Experts (Puigcerver et al., 2023).
    
    Fully differentiable MoE without hard discrete token dispatch.
    Tokens are softly dispatched into expert slots via softmax over tokens,
    processed by individual experts, and softly combined back into output tokens.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        slots_per_expert: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.total_slots = num_experts * slots_per_expert

        self.proj_disp = nn.Linear(d_model, self.total_slots)
        self.proj_comb = nn.Linear(d_model, self.total_slots)
        self.experts = nn.ModuleList([ExpertMLP(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)

        # Dispatch weights: softmax across tokens (dim=0)
        disp_logits = self.proj_disp(x_flat)  # (T, total_slots)
        Phi = F.softmax(disp_logits, dim=0)  # (T, total_slots)

        # Combine weights: softmax across slots (dim=1)
        comb_logits = self.proj_comb(x_flat)  # (T, total_slots)
        Psi = F.softmax(comb_logits, dim=1)  # (T, total_slots)

        # Slot inputs: (total_slots, d_model)
        slot_inputs = Phi.t() @ x_flat

        # Slot evaluation
        slot_outputs = torch.zeros_like(slot_inputs)
        for j in range(self.total_slots):
            e = j // self.slots_per_expert
            slot_outputs[j] = self.experts[e](slot_inputs[j : j + 1]).squeeze(0)

        # Soft combination back to tokens
        out_flat = Psi @ slot_outputs
        return out_flat.view(orig_shape)


def verify_against_checkpoint(checkpoint_path: Path | str | None = None) -> None:
    """Verifies PyTorch forward passes against golden safetensors checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.safetensors")
    
    weights = load_file(str(checkpoint_path))
    x = weights["input"]
    num_experts = 4
    d_model = 16
    d_ff = 32
    top_k = 2

    # Instantiate TopK MoE
    topk_moe = TopKMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k)
    with torch.no_grad():
        topk_moe.router.weight.copy_(weights["router_weight"].t())
        topk_moe.router.bias.copy_(weights["router_bias"])
        for e in range(num_experts):
            topk_moe.experts[e].fc1.weight.copy_(weights["experts_w1"][e].t())
            topk_moe.experts[e].fc1.bias.copy_(weights["experts_b1"][e])
            topk_moe.experts[e].fc2.weight.copy_(weights["experts_w2"][e].t())
            topk_moe.experts[e].fc2.bias.copy_(weights["experts_b2"][e])

        out_topk, aux_loss, z_loss = topk_moe(x)
        err_topk = (out_topk - weights["topk_output"]).abs().max().item()
        err_aux = (aux_loss - weights["aux_loss"]).abs().max().item()
        err_z = (z_loss - weights["z_loss"]).abs().max().item()
        assert err_topk < 1e-5, f"TopK MoE parity error {err_topk} >= 1e-5"
        assert err_aux < 1e-5, f"Aux loss parity error {err_aux} >= 1e-5"
        assert err_z < 1e-5, f"Z-loss parity error {err_z} >= 1e-5"

    # Instantiate Expert Choice MoE
    ec_moe = ExpertChoiceMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, capacity=2)
    with torch.no_grad():
        ec_moe.router.weight.copy_(weights["router_weight"].t())
        ec_moe.router.bias.copy_(weights["router_bias"])
        for e in range(num_experts):
            ec_moe.experts[e].fc1.weight.copy_(weights["experts_w1"][e].t())
            ec_moe.experts[e].fc1.bias.copy_(weights["experts_b1"][e])
            ec_moe.experts[e].fc2.weight.copy_(weights["experts_w2"][e].t())
            ec_moe.experts[e].fc2.bias.copy_(weights["experts_b2"][e])

        out_ec = ec_moe(x)
        err_ec = (out_ec - weights["expert_choice_output"]).abs().max().item()
        assert err_ec < 1e-5, f"Expert Choice MoE parity error {err_ec} >= 1e-5"

    # Instantiate Soft MoE
    soft_moe = SoftMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, slots_per_expert=1)
    with torch.no_grad():
        soft_moe.proj_disp.weight.copy_(weights["soft_disp_weight"].t())
        soft_moe.proj_disp.bias.copy_(weights["soft_disp_bias"])
        soft_moe.proj_comb.weight.copy_(weights["soft_comb_weight"].t())
        soft_moe.proj_comb.bias.copy_(weights["soft_comb_bias"])
        for e in range(num_experts):
            soft_moe.experts[e].fc1.weight.copy_(weights["experts_w1"][e].t())
            soft_moe.experts[e].fc1.bias.copy_(weights["experts_b1"][e])
            soft_moe.experts[e].fc2.weight.copy_(weights["experts_w2"][e].t())
            soft_moe.experts[e].fc2.bias.copy_(weights["experts_b2"][e])

        out_soft = soft_moe(x)
        err_soft = (out_soft - weights["soft_moe_output"]).abs().max().item()
        assert err_soft < 1e-5, f"Soft MoE parity error {err_soft} >= 1e-5"

    print("All Category 05 Topic 02 PyTorch models verified against checkpoint with < 1e-5 max error!")


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    #!/usr/bin/env python3
    """Generate golden weights and inputs/outputs for Category 05 Topic 02:
    Mixture of Experts (Top-K, Expert Choice, and Soft MoE).
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
        top_k = 2
        slots_per_expert = 1
        total_slots = num_experts * slots_per_expert
        tokens = batch_size * seq_len  # 8

        # Input tensor
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

        # 1. Router weights
        router_w = torch.randn(d_model, num_experts, dtype=torch.float32) * 0.2
        router_b = torch.randn(num_experts, dtype=torch.float32) * 0.1

        # 2. Expert weights: E experts, each with W1 (D -> D_ff), b1 (D_ff), W2 (D_ff -> D), b2 (D)
        experts_w1 = torch.randn(num_experts, d_model, d_ff, dtype=torch.float32) * 0.2
        experts_b1 = torch.randn(num_experts, d_ff, dtype=torch.float32) * 0.1
        experts_w2 = torch.randn(num_experts, d_ff, d_model, dtype=torch.float32) * 0.2
        experts_b2 = torch.randn(num_experts, d_model, dtype=torch.float32) * 0.1

        # 3. Soft MoE projection weights
        soft_disp_w = torch.randn(d_model, total_slots, dtype=torch.float32) * 0.2
        soft_disp_b = torch.randn(total_slots, dtype=torch.float32) * 0.1
        soft_comb_w = torch.randn(d_model, total_slots, dtype=torch.float32) * 0.2
        soft_comb_b = torch.randn(total_slots, dtype=torch.float32) * 0.1

        # Flatten tokens for MoE routing
        x_flat = x.view(tokens, d_model)

        # Helper for expert evaluation
        def eval_expert(e_idx: int, inp: torch.Tensor) -> torch.Tensor:
            # inp: (..., d_model)
            h = F.gelu(inp @ experts_w1[e_idx] + experts_b1[e_idx])
            out = h @ experts_w2[e_idx] + experts_b2[e_idx]
            return out

        # ==========================================
        # A. Top-K Sparse MoE (Token Choice)
        # ==========================================
        router_logits = x_flat @ router_w + router_b  # (T, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (T, E)

        topk_vals, topk_indices = torch.topk(router_logits, k=top_k, dim=-1)  # (T, k)
        topk_weights = F.softmax(topk_vals, dim=-1)  # (T, k) normalized over top-k

        topk_out_flat = torch.zeros_like(x_flat)
        for t in range(tokens):
            for k_idx in range(top_k):
                e = topk_indices[t, k_idx].item()
                w = topk_weights[t, k_idx]
                topk_out_flat[t] += w * eval_expert(e, x_flat[t : t + 1]).squeeze(0)

        topk_output = topk_out_flat.view(batch_size, seq_len, d_model)

        # Load balancing auxiliary loss: L_aux = E * sum_i (f_i * P_i)
        # f_i = fraction of tokens routed to expert i
        # P_i = average routing probability to expert i
        f_count = torch.zeros(num_experts, dtype=torch.float32)
        for t in range(tokens):
            for k_idx in range(top_k):
                e = topk_indices[t, k_idx].item()
                f_count[e] += 1.0
        f_i = f_count / tokens
        P_i = router_probs.mean(dim=0)
        aux_loss = (num_experts * torch.sum(f_i * P_i)).view(1)

        # Router Z-loss: (1/T) sum_t (log sum_i exp(H_ti))^2
        lse = torch.logsumexp(router_logits, dim=-1)
        z_loss = torch.mean(lse**2).view(1)

        # ==========================================
        # B. Expert Choice Routing
        # ==========================================
        # Each expert picks top C tokens
        # capacity C = 2
        C = 2
        # In Zhou et al., routing gate score G = softmax(H, dim=-1)
        # Transposed: G_t is (E, T)
        G = router_probs.t()  # (E, T)
        ec_out_flat = torch.zeros_like(x_flat)
        for e in range(num_experts):
            expert_scores = G[e]  # (T,)
            _, selected_tokens = torch.topk(expert_scores, k=C, dim=-1)
            # evaluate expert on these tokens
            for t_idx in selected_tokens:
                t = t_idx.item()
                score = G[e, t]
                ec_out_flat[t] += score * eval_expert(e, x_flat[t : t + 1]).squeeze(0)

        expert_choice_output = ec_out_flat.view(batch_size, seq_len, d_model)

        # ==========================================
        # C. Soft MoE
        # ==========================================
        # Dispatch weights: softmax across tokens (dim=0)
        disp_logits = x_flat @ soft_disp_w + soft_disp_b  # (T, total_slots)
        Phi = F.softmax(disp_logits, dim=0)  # (T, total_slots)

        # Combine weights: softmax across slots (dim=1)
        comb_logits = x_flat @ soft_comb_w + soft_comb_b  # (T, total_slots)
        Psi = F.softmax(comb_logits, dim=1)  # (T, total_slots)

        # Slot inputs: (total_slots, d_model) = Phi^T @ x_flat
        slot_inputs = Phi.t() @ x_flat  # (total_slots, d_model)

        # Each slot is evaluated by its expert
        slot_outputs = torch.zeros_like(slot_inputs)
        for j in range(total_slots):
            e = j // slots_per_expert
            slot_outputs[j] = eval_expert(e, slot_inputs[j : j + 1]).squeeze(0)

        # Combine to tokens: Y = Psi @ slot_outputs (T, d_model)
        soft_out_flat = Psi @ slot_outputs
        soft_moe_output = soft_out_flat.view(batch_size, seq_len, d_model)

        # Standard linear parity tensors for test_cross_track_parity.py
        D_core = 4
        test_input = torch.randn(8, D_core)
        weight = torch.randn(D_core, D_core)
        bias = torch.randn(D_core)
        test_output = test_input @ weight + bias

        # Output directory
        out_dir = Path("weights/05_architectures/01_mlp_moe_soups_and_kan")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "02_mixture_of_experts_top_k_expert_choice_and_soft_moe.safetensors"

        tensors = {
            "weight": weight.clone().contiguous(),
            "bias": bias.clone().contiguous(),
            "test_input": test_input.clone().contiguous(),
            "test_output": test_output.clone().contiguous(),
            "input": x,
            "router_weight": router_w,
            "router_bias": router_b,
            "experts_w1": experts_w1,
            "experts_b1": experts_b1,
            "experts_w2": experts_w2,
            "experts_b2": experts_b2,
            "soft_disp_weight": soft_disp_w,
            "soft_disp_bias": soft_disp_b,
            "soft_comb_weight": soft_comb_w,
            "soft_comb_bias": soft_comb_b,
            "topk_output": topk_output,
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "expert_choice_output": expert_choice_output,
            "soft_moe_output": soft_moe_output,
        }

        save_file(tensors, str(out_file))
        print(f"Generated {out_file} successfully.")
        print(f"topk_output shape: {topk_output.shape}, norm: {topk_output.norm():.6f}")
        print(f"aux_loss: {aux_loss.item():.6f}, z_loss: {z_loss.item():.6f}")
        print(f"expert_choice_output shape: {expert_choice_output.shape}, norm: {expert_choice_output.norm():.6f}")
        print(f"soft_moe_output shape: {soft_moe_output.shape}, norm: {soft_moe_output.norm():.6f}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_against_checkpoint()
