"""Generate golden weights and reference I/O tensors for Topic 06:
06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.

Pillars:
1. Predictive vs Prescriptive AI: Cost Prediction vs Downstream Decision Regret.
2. Smart "Predict, Then Optimize" (SPO+ Loss): Convex surrogate directly optimizing decisions.
3. Causal Uplift Estimator: Distinguishing observational prediction from interventional effects.
4. Symbolic Equation Operator (EQL): Discovering functional equations with non-linear primitives.
5. Canonical Linear Anchor for cross-track parity harness.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

import core


def generate_topic_06_cat01_weights(output_path: Path) -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    in_dim = 4
    hidden_dim = 4
    cost_dim = 4  # 4 combinatorial decision options (e.g. routing paths)

    # 1. Input batches: feature representations x and oracle cost vectors c_star
    x = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)
    c_star = torch.randn(batch_size, cost_dim, generator=gen, dtype=torch.float32).abs() + 0.1  # strictly positive costs

    # 2. Canonical linear reference anchor
    test_input = x
    weight = torch.randn(in_dim, 1, generator=gen, dtype=torch.float32)
    bias = torch.randn(1, generator=gen, dtype=torch.float32)
    test_output = torch.matmul(test_input, weight) + bias

    # 3. Cost Predictor Backbone (Predictive / Prescriptive Head)
    w_cost1 = torch.randn(in_dim, hidden_dim, generator=gen, dtype=torch.float32)
    b_cost1 = torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    w_cost2 = torch.randn(hidden_dim, cost_dim, generator=gen, dtype=torch.float32)
    b_cost2 = torch.randn(cost_dim, generator=gen, dtype=torch.float32)

    h_cost = F.relu(torch.matmul(x, w_cost1) + b_cost1)
    c_pred = F.softplus(torch.matmul(h_cost, w_cost2) + b_cost2)  # [B, cost_dim] predicted costs

    # 4. Downstream Combinatorial Optimization & SPO+ Loss (Elmachtoub & Grigas 2022)
    # Domain: Simplex decisions w in Delta_4 (selecting minimum cost path)
    # Oracle decision: w*(c*) = one_hot(argmin_i c*_i)
    # Predicted decision: w*(c_pred) = one_hot(argmin_i c_pred_i)
    opt_idx_oracle = torch.argmin(c_star, dim=-1)  # [B]
    w_star_oracle = F.one_hot(opt_idx_oracle, num_classes=cost_dim).float()  # [B, cost_dim]

    opt_idx_pred = torch.argmin(c_pred, dim=-1)
    w_star_pred = F.one_hot(opt_idx_pred, num_classes=cost_dim).float()  # [B, cost_dim]

    # Exact Decision Regret: c*^T w*(c_pred) - c*^T w*(c*) >= 0
    cost_actual = torch.sum(c_star * w_star_pred, dim=-1)  # [B]
    cost_oracle = torch.sum(c_star * w_star_oracle, dim=-1)  # [B]
    decision_regret = cost_actual - cost_oracle  # [B]

    # SPO+ Surrogate Loss:
    # L_SPO+(c_hat, c*) = max_{w in Delta} (c* - 2*c_hat)^T w + 2*c_hat^T w*(c*) - c*^T w*(c*)
    # For unit simplex Delta, max_w v^T w is simply max_i v_i!
    v_vec = c_star - 2.0 * c_pred  # [B, cost_dim]
    max_term = torch.max(v_vec, dim=-1).values  # [B]
    lin_term = 2.0 * torch.sum(c_pred * w_star_oracle, dim=-1) - cost_oracle  # [B]
    spo_plus_loss = max_term + lin_term  # [B]

    # 5. Causal Uplift Estimator
    # Maps x to treatment effect tau(x) and baseline outcome mu_0(x)
    w_tau = torch.randn(in_dim, 1, generator=gen, dtype=torch.float32)
    b_tau = torch.randn(1, generator=gen, dtype=torch.float32)
    tau_pred = torch.matmul(x, w_tau) + b_tau  # [B, 1]

    treatment_t = torch.bernoulli(torch.full((batch_size, 1), 0.5), generator=gen)  # [B, 1] in {0, 1}
    # Factual outcome: y = mu_0(x) + T * tau(x)
    w_mu0 = torch.randn(in_dim, 1, generator=gen, dtype=torch.float32)
    b_mu0 = torch.randn(1, generator=gen, dtype=torch.float32)
    mu0_pred = torch.matmul(x, w_mu0) + b_mu0  # [B, 1]
    y_factual = mu0_pred + treatment_t * tau_pred  # [B, 1]

    # 6. Symbolic Equation Operator (EQL Primitive Layer)
    # Primitives: [x[:, 0:1], sin(x[:, 1:2]), cos(x[:, 2:3]), x[:, 0:1] * x[:, 3:4]]
    eql_features = torch.cat([
        x[:, 0:1],
        torch.sin(x[:, 1:2]),
        torch.cos(x[:, 2:3]),
        x[:, 0:1] * x[:, 3:4],
    ], dim=-1)  # [B, 4]
    w_eql = torch.randn(4, 1, generator=gen, dtype=torch.float32)
    b_eql = torch.randn(1, generator=gen, dtype=torch.float32)
    y_symbolic = torch.matmul(eql_features, w_eql) + b_eql  # [B, 1]

    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "c_star": c_star.detach(),
        "w_cost1": w_cost1.detach(),
        "b_cost1": b_cost1.detach(),
        "w_cost2": w_cost2.detach(),
        "b_cost2": b_cost2.detach(),
        "c_pred": c_pred.detach(),
        "w_star_oracle": w_star_oracle.detach(),
        "w_star_pred": w_star_pred.detach(),
        "decision_regret": decision_regret.detach(),
        "spo_plus_loss": spo_plus_loss.detach(),
        "w_tau": w_tau.detach(),
        "b_tau": b_tau.detach(),
        "tau_pred": tau_pred.detach(),
        "w_mu0": w_mu0.detach(),
        "b_mu0": b_mu0.detach(),
        "mu0_pred": mu0_pred.detach(),
        "treatment_t": treatment_t.detach(),
        "y_factual": y_factual.detach(),
        "w_eql": w_eql.detach(),
        "b_eql": b_eql.detach(),
        "eql_features": eql_features.detach(),
        "y_symbolic": y_symbolic.detach(),
    }

    core.parity.save_reference_checkpoint(
        path=output_path,
        weights_dict=weights_dict,
        input_tensor=test_input,
        output_tensor=test_output,
        metadata={
            "topic": "06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic",
            "category": "01_paradigms_and_dichotomies",
            "batch_size": str(batch_size),
            "in_dim": str(in_dim),
            "hidden_dim": str(hidden_dim),
            "cost_dim": str(cost_dim),
        },
    )

    print(f"Generated Topic 06 Category 01 weights at: {output_path}")
    print(f"c_pred shape: {c_pred.shape}")
    print(f"spo_plus_loss shape: {spo_plus_loss.shape}")
    print(f"tau_pred shape: {tau_pred.shape}")
    print(f"y_symbolic shape: {y_symbolic.shape}")


if __name__ == "__main__":
    out_file = (
        Path(__file__).resolve().parent.parent
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.safetensors"
    )
    generate_topic_06_cat01_weights(out_file)
