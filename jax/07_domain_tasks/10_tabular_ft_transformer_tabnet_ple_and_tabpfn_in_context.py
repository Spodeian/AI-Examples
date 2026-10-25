"""Tabular Ft Transformer Tabnet Ple and Tabpfn In Context — JAX reference implementation.

Topic: 07_domain_tasks/10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context
Golden weights: weights/07_domain_tasks/10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "07_domain_tasks/10_tabular_ft_transformer_tabnet_ple_and_tabpfn_in_context"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ft_transformer_tokenize(
    x_num: jnp.ndarray,
    w_num: jnp.ndarray,
    b_num: jnp.ndarray,
    cat_tokens: jnp.ndarray,
    cls_token: jnp.ndarray,
) -> jnp.ndarray:
    """FT-Transformer Feature Tokenizer in JAX."""
    b, _ = x_num.shape
    d = cls_token.shape[-1]
    num_tokens = jnp.expand_dims(x_num, axis=-1) * jnp.expand_dims(w_num, axis=0) + jnp.expand_dims(b_num, axis=0)
    cls_expanded = jnp.broadcast_to(cls_token, (b, 1, d))
    return jnp.concatenate([cls_expanded, num_tokens, cat_tokens], axis=1)


def sparsemax(z: jnp.ndarray) -> jnp.ndarray:
    """Exact Sparsemax activation projection onto probability simplex in JAX."""
    z_sorted = jnp.sort(z, axis=-1)[:, ::-1]
    d = z.shape[-1]
    range_d = jnp.arange(1, d + 1, dtype=z.dtype).reshape(1, -1)
    cumsum_z = jnp.cumsum(z_sorted, axis=-1)
    condition = 1.0 + range_d * z_sorted > cumsum_z
    k_z = jnp.sum(condition, axis=-1, keepdims=True)
    tau_z = (jnp.take_along_axis(cumsum_z, (k_z - 1).astype(int), axis=-1) - 1.0) / k_z
    return jnp.maximum(z - tau_z, 0.0)


def tabnet_attentive_mask(
    p_prior: jnp.ndarray,
    h_features: jnp.ndarray,
    gamma: float = 1.3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """TabNet Attentive Transformer feature selection mask and prior update in JAX."""
    logits = p_prior * h_features
    mask = sparsemax(logits)
    p_next = p_prior * (gamma - mask)
    return mask, p_next


def ple_task_routing(
    expert_outputs: jnp.ndarray,
    gate_logits: jnp.ndarray,
    expert_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Progressive Layered Extraction (PLE) gated task aggregation in JAX."""
    candidates = expert_outputs[:, expert_indices, :]
    weights = jax.nn.softmax(gate_logits, axis=-1)[:, :, None]
    return jnp.sum(candidates * weights, axis=1)


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    raw = load_file(str(path))
    tensors = {k: jnp.array(v) for k, v in raw.items()}

    # Anchor check
    out = linear_anchor(tensors["test_input"], tensors["weight"], tensors["bias"])
    err = float(jnp.max(jnp.abs(out - tensors["test_output"])))
    assert err < 1e-5, f"test_output parity error: {err}"

    # 1. FT-Transformer
    tok = ft_transformer_tokenize(
        tensors["x_num"],
        tensors["w_num"],
        tensors["b_num"],
        tensors["cat_tokens"],
        tensors["cls_token"],
    )
    err_tok = float(jnp.max(jnp.abs(tok - tensors["tokenized_ref"])))
    assert err_tok < 1e-5, f"tokenized parity error: {err_tok}"

    # 2. Sparsemax
    sm = sparsemax(tensors["z_sparsemax"])
    err_sm = float(jnp.max(jnp.abs(sm - tensors["sparsemax_ref"])))
    assert err_sm < 1e-5, f"sparsemax parity error: {err_sm}"

    # 3. TabNet
    mask, p_next = tabnet_attentive_mask(tensors["p_prior"], tensors["h_feat"], gamma=1.3)
    err_mask = float(jnp.max(jnp.abs(mask - tensors["tabnet_mask_ref"])))
    err_p = float(jnp.max(jnp.abs(p_next - tensors["tabnet_p_next_ref"])))
    assert err_mask < 1e-5, f"tabnet mask parity error: {err_mask}"
    assert err_p < 1e-5, f"tabnet p_next parity error: {err_p}"

    # 4. PLE
    ple = ple_task_routing(tensors["expert_outputs"], tensors["gate_logits"], tensors["expert_indices"])
    err_ple = float(jnp.max(jnp.abs(ple - tensors["ple_routed_ref"])))
    assert err_ple < 1e-5, f"ple parity error: {err_ple}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
