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


if __name__ == "__main__":
    main()
