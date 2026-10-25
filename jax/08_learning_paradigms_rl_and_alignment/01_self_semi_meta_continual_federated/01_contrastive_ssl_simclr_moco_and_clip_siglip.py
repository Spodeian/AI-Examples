"""Contrastive Ssl Simclr Moco and CLIP Siglip — JAX reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/01_contrastive_ssl_simclr_moco_and_clip_siglip
Golden weights: weights/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/01_contrastive_ssl_simclr_moco_and_clip_siglip.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/01_contrastive_ssl_simclr_moco_and_clip_siglip"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def normalize(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


def simclr_nt_xent_loss(z1: jnp.ndarray, z2: jnp.ndarray, temperature: float = 0.07) -> jnp.ndarray:
    """SimCLR NT-Xent Loss in JAX."""
    z1_norm = normalize(z1)
    z2_norm = normalize(z2)
    sim_matrix = (z1_norm @ z2_norm.T) / temperature
    log_probs = jax.nn.log_softmax(sim_matrix, axis=-1)
    return -jnp.mean(jnp.diag(log_probs))


def moco_contrastive_loss(
    q: jnp.ndarray,
    k_pos: jnp.ndarray,
    queue: jnp.ndarray,
    temperature: float = 0.07,
) -> jnp.ndarray:
    """MoCo Memory Queue Contrastive Loss in JAX."""
    q_norm = normalize(q)
    k_norm = normalize(k_pos)
    queue_norm = normalize(queue)

    l_pos = jnp.sum(q_norm * k_norm, axis=-1, keepdims=True) / temperature
    l_neg = (q_norm @ queue_norm.T) / temperature
    logits = jnp.concatenate([l_pos, l_neg], axis=1)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[:, 0])


def clip_symmetric_loss(
    image_embeds: jnp.ndarray,
    text_embeds: jnp.ndarray,
    log_temp: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    """CLIP Symmetric Cross-Entropy Loss in JAX."""
    i_norm = normalize(image_embeds)
    t_norm = normalize(text_embeds)
    temp = jnp.exp(jnp.clip(log_temp, a_max=4.6052))

    logits = (i_norm @ t_norm.T) * temp + bias
    loss_i2t = -jnp.mean(jnp.diag(jax.nn.log_softmax(logits, axis=-1)))
    loss_t2i = -jnp.mean(jnp.diag(jax.nn.log_softmax(logits.T, axis=-1)))
    return 0.5 * (loss_i2t + loss_t2i)


def siglip_pairwise_loss(
    image_embeds: jnp.ndarray,
    text_embeds: jnp.ndarray,
    log_temp: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    """SigLIP Pairwise Sigmoid Loss in JAX."""
    b = image_embeds.shape[0]
    i_norm = normalize(image_embeds)
    t_norm = normalize(text_embeds)
    temp = jnp.exp(jnp.clip(log_temp, a_max=4.6052))

    logits = (i_norm @ t_norm.T) * temp + bias
    labels = 2.0 * jnp.eye(b) - 1.0
    loss = jax.nn.softplus(-labels * logits)
    return jnp.mean(loss)


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    raw = load_file(str(path))
    tensors = {k: jnp.array(v) for k, v in raw.items()}

    # Anchor check
    out = linear_anchor(tensors["test_input"], tensors["weight"], tensors["bias"])
    err = float(jnp.max(jnp.abs(out - tensors["test_output"])))
    assert err < 1e-5, f"test_output parity error: {err}"

    # 1. SimCLR
    simclr = simclr_nt_xent_loss(tensors["z1"], tensors["z2"], temperature=0.1)
    err_simclr = float(jnp.abs(simclr - tensors["simclr_ref"]))
    assert err_simclr < 1e-5, f"simclr parity error: {err_simclr}"

    # 2. MoCo
    moco = moco_contrastive_loss(
        tensors["moco_q"], tensors["moco_k_pos"], tensors["moco_queue"], temperature=0.1
    )
    err_moco = float(jnp.abs(moco - tensors["moco_ref"]))
    assert err_moco < 1e-5, f"moco parity error: {err_moco}"

    # 3. CLIP
    clip = clip_symmetric_loss(
        tensors["img_embeds"], tensors["txt_embeds"], tensors["log_temp"], tensors["clip_bias"]
    )
    err_clip = float(jnp.abs(clip - tensors["clip_ref"]))
    assert err_clip < 1e-5, f"clip parity error: {err_clip}"

    # 4. SigLIP
    siglip = siglip_pairwise_loss(
        tensors["img_embeds"], tensors["txt_embeds"], tensors["log_temp"], tensors["clip_bias"]
    )
    err_siglip = float(jnp.abs(siglip - tensors["siglip_ref"]))
    assert err_siglip < 1e-5, f"siglip parity error: {err_siglip}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
