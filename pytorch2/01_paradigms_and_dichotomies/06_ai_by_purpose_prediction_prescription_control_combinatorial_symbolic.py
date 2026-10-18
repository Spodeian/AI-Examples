"""Production PyTorch 2.x implementation for Category 01 Topic 06:
06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class CostPredictor(nn.Module):
    """Predictive / Prescriptive Cost Predictor Backbone.

    Inductive Bias:
        Maps environmental features x to downstream cost vectors c_pred.
        Trained either via intermediate MSE or directly via decision-focused SPO+ loss.
    """

    def __init__(
        self,
        w_cost1: Float[Tensor, "in_dim hidden_dim"],
        b_cost1: Float[Tensor, "*hidden_dim"],
        w_cost2: Float[Tensor, "hidden_dim cost_dim"],
        b_cost2: Float[Tensor, "*cost_dim"],
    ) -> None:
        super().__init__()
        self.w_cost1 = nn.Parameter(w_cost1.clone())
        self.b_cost1 = nn.Parameter(b_cost1.clone())
        self.w_cost2 = nn.Parameter(w_cost2.clone())
        self.b_cost2 = nn.Parameter(b_cost2.clone())

    def forward(
        self, x: Float[Tensor, "batch in_dim"]
    ) -> Float[Tensor, "batch cost_dim"]:
        h = F.relu(torch.matmul(x, self.w_cost1) + self.b_cost1)
        return F.softplus(torch.matmul(h, self.w_cost2) + self.b_cost2)


class SPOPlusEstimator:
    """Smart 'Predict, Then Optimize' (SPO+) Loss & Decision Regret (Elmachtoub & Grigas 2022).

    Inductive Bias:
        Convex upper bound on downstream decision regret that directly backpropagates
        gradients through linear/combinatorial program vertices without intermediate MSE misalignment.
    """

    @staticmethod
    def solve_linear_program(
        cost_vectors: Float[Tensor, "batch cost_dim"],
    ) -> Float[Tensor, "batch cost_dim"]:
        """Exact LP vertex solver over unit simplex Delta: argmin_w c^T w."""
        cost_dim = cost_vectors.shape[-1]
        opt_idx = torch.argmin(cost_vectors, dim=-1)
        return F.one_hot(opt_idx, num_classes=cost_dim).float()

    @staticmethod
    def decision_regret(
        c_pred: Float[Tensor, "batch cost_dim"],
        c_star: Float[Tensor, "batch cost_dim"],
    ) -> Float[Tensor, "*batch"]:
        """Exact Decision Regret: c*^T w*(c_pred) - c*^T w*(c*) >= 0."""
        w_pred = SPOPlusEstimator.solve_linear_program(c_pred)
        w_star = SPOPlusEstimator.solve_linear_program(c_star)

        cost_actual = torch.sum(c_star * w_pred, dim=-1)
        cost_oracle = torch.sum(c_star * w_star, dim=-1)
        return cost_actual - cost_oracle

    @staticmethod
    def spo_plus_loss(
        c_pred: Float[Tensor, "batch cost_dim"],
        c_star: Float[Tensor, "batch cost_dim"],
    ) -> Float[Tensor, "*batch"]:
        """SPO+ Surrogate Loss: max_w (c* - 2 c_hat)^T w + 2 c_hat^T w*(c*) - c*^T w*(c*)."""
        w_star = SPOPlusEstimator.solve_linear_program(c_star)
        cost_oracle = torch.sum(c_star * w_star, dim=-1)

        v_vec = c_star - 2.0 * c_pred
        max_term = torch.max(v_vec, dim=-1).values
        lin_term = 2.0 * torch.sum(c_pred * w_star, dim=-1) - cost_oracle
        return max_term + lin_term


class CausalUpliftEstimator(nn.Module):
    """Causal Treatment Effect & Counterfactual Outcome Estimator (Pearl 2009).

    Inductive Bias:
        Separates baseline confounding mu_0(x) from interventional treatment effect tau(x).
    """

    def __init__(
        self,
        w_tau: Float[Tensor, "in_dim 1"],
        b_tau: Float[Tensor, "*out_dim"],
        w_mu0: Float[Tensor, "in_dim 1"],
        b_mu0: Float[Tensor, "*out_dim"],
    ) -> None:
        super().__init__()
        self.w_tau = nn.Parameter(w_tau.clone())
        self.b_tau = nn.Parameter(b_tau.clone())
        self.w_mu0 = nn.Parameter(w_mu0.clone())
        self.b_mu0 = nn.Parameter(b_mu0.clone())

    def predict_treatment_effect(
        self, x: Float[Tensor, "batch in_dim"]
    ) -> Float[Tensor, "batch 1"]:
        return torch.matmul(x, self.w_tau) + self.b_tau

    def forward(
        self,
        x: Float[Tensor, "batch in_dim"],
        treatment_t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch 1"]:
        mu0 = torch.matmul(x, self.w_mu0) + self.b_mu0
        tau = self.predict_treatment_effect(x)
        return mu0 + treatment_t * tau


class SymbolicEquationLearner(nn.Module):
    """Equation Learner (EQL) with Non-linear Functional Primitives (Martius & Lampert 2016).

    Inductive Bias:
        Constructs analytical closed-form equations using unary and product primitives.
    """

    def __init__(
        self,
        w_eql: Float[Tensor, "num_primitives 1"],
        b_eql: Float[Tensor, "*out_dim"],
    ) -> None:
        super().__init__()
        self.w_eql = nn.Parameter(w_eql.clone())
        self.b_eql = nn.Parameter(b_eql.clone())

    @staticmethod
    def expand_primitives(
        x: Float[Tensor, "batch in_dim"],
    ) -> Float[Tensor, "batch num_primitives"]:
        return torch.cat([
            x[:, 0:1],
            torch.sin(x[:, 1:2]),
            torch.cos(x[:, 2:3]),
            x[:, 0:1] * x[:, 3:4],
        ], dim=-1)

    def forward(
        self, x: Float[Tensor, "batch in_dim"]
    ) -> Float[Tensor, "batch 1"]:
        features = self.expand_primitives(x)
        return torch.matmul(features, self.w_eql) + self.b_eql
