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


if __name__ == "__main__":
    main()
