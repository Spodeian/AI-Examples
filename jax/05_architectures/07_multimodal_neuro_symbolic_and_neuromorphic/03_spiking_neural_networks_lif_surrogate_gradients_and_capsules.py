"""Spiking Neural Networks, LIF, Surrogate Gradients, and Capsules — JAX reference implementation.

Topic: 05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules
Golden weights: weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.safetensors

``python jax/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.matmul(x, weight) + bias


def lif_simulation(
    currents: jax.Array,
    beta: float = 0.8,
    v_th: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Leaky Integrate-and-Fire (LIF) discrete simulation in JAX."""
    t_steps, b, d = currents.shape
    v = jnp.zeros((b, d), dtype=currents.dtype)
    s = jnp.zeros((b, d), dtype=currents.dtype)

    all_v = []
    all_s = []

    for t in range(t_steps):
        v = beta * v * (1.0 - s) + currents[t]
        s = (v >= v_th).astype(currents.dtype)
        all_v.append(v)
        all_s.append(s)

    return jnp.stack(all_s, axis=0), jnp.stack(all_v, axis=0)


def capsule_squash(s: jax.Array) -> jax.Array:
    """Capsule vector squashing non-linearity."""
    norm_sq = jnp.sum(s ** 2, axis=-1, keepdims=True)
    norm = jnp.sqrt(norm_sq + 1e-8)
    scale = norm_sq / (1.0 + norm_sq)
    return scale * (s / norm)


def capsule_dynamic_routing(u_hat: jax.Array, num_iters: int = 3) -> jax.Array:
    """Dynamic routing agreement over capsule prediction vectors in JAX."""
    b, n_in, n_out, d_out = u_hat.shape
    b_logits = jnp.zeros((b, n_in, n_out), dtype=u_hat.dtype)

    v = jnp.zeros((b, n_out, d_out), dtype=u_hat.dtype)
    for i in range(num_iters):
        c = jax.nn.softmax(b_logits, axis=-1)
        s = jnp.sum(jnp.expand_dims(c, -1) * u_hat, axis=1)
        v = capsule_squash(s)
        if i < num_iters - 1:
            agreement = jnp.sum(u_hat * jnp.expand_dims(v, 1), axis=-1)
            b_logits = b_logits + agreement

    return v


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation reproduces the golden checkpoint."""
    golden = {k: jnp.asarray(v) for k, v in load_file(str(path)).items()}

    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear_anchor parity failed: max abs error {err:.2e}"

    spikes, voltages = lif_simulation(golden["currents"], beta=0.8, v_th=1.0)
    assert float(jnp.max(jnp.abs(spikes - golden["lif_spikes"]))) < 1e-5
    assert float(jnp.max(jnp.abs(voltages - golden["lif_voltages"]))) < 1e-5

    v_caps = capsule_dynamic_routing(golden["u_hat"], num_iters=3)
    assert float(jnp.max(jnp.abs(v_caps - golden["v_capsules"]))) < 1e-5

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
