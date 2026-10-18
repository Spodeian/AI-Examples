"""Functional JAX implementation for Category 01 Topic 08:
08_computation_spaces_latent_reasoning_and_jepa
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safetensors.torch import load_file

try:
    import jax
    import jax.numpy as jnp
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
    _ = jax
except (ImportError, AttributeError):
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False

try:
    import scipy.special as scipy_sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def gelu(x: Any) -> Any:
    """Exact Gaussian Error Linear Unit matching PyTorch."""
    if HAS_JAX:
        return jax.nn.gelu(x, approximate=False)
    if HAS_SCIPY:
        x_arr = jnp.asarray(x, dtype=jnp.float32)
        return x_arr * 0.5 * (1.0 + scipy_sp.erf(x_arr / (2.0 ** 0.5)))
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    return 0.5 * x_arr * (1.0 + jnp.tanh((2.0 / jnp.pi) ** 0.5 * (x_arr + 0.044715 * jnp.power(x_arr, 3))))


def rms_norm(z: Any, eps: float = 1e-5) -> Any:
    """Root Mean Square Normalization along the innermost dimension."""
    variance = jnp.mean(jnp.square(z), axis=-1, keepdims=True)
    return z / jnp.sqrt(variance + eps)



def continuous_latent_reasoning(
    x: Any,
    w_proj: Any,
    b_proj: Any,
    w_z: Any,
    w_anc: Any,
    b_z: Any,
    steps: int = 3,
) -> tuple[Any, Any]:
    """Runs K-step input-anchored recurrent latent deliberation."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    wp = jnp.asarray(w_proj, dtype=jnp.float32)
    bp = jnp.asarray(b_proj, dtype=jnp.float32)
    wz = jnp.asarray(w_z, dtype=jnp.float32)
    wa = jnp.asarray(w_anc, dtype=jnp.float32)
    bz = jnp.asarray(b_z, dtype=jnp.float32)

    e_x = gelu(jnp.dot(x_arr, wp) + bp)
    z = e_x
    trajectories = [z]

    for _ in range(steps):
        delta = gelu(jnp.dot(z, wz) + jnp.dot(e_x, wa) + bz)
        z = rms_norm(z + delta)
        trajectories.append(z)

    traj_arr = jnp.stack(trajectories, axis=1)
    return traj_arr, z


def ponder_halting(
    trajectories: Any,
    w_h: Any,
    b_h: Any,
) -> Any:
    """Evaluates adaptive computation halting probabilities for steps k = 1..K."""
    traj = jnp.asarray(trajectories, dtype=jnp.float32)
    wh = jnp.asarray(w_h, dtype=jnp.float32)
    bh = jnp.asarray(b_h, dtype=jnp.float32)

    batch = traj.shape[0]
    steps = traj.shape[1] - 1

    hazards = []
    for k in range(steps):
        z_k = traj[:, k + 1]
        lam = 1.0 / (1.0 + jnp.exp(-(jnp.dot(z_k, wh) + bh)))
        hazards.append(lam)

    hazards_arr = jnp.concatenate(hazards, axis=-1)
    halting_probs = []
    unhalted = jnp.ones((batch, 1), dtype=jnp.float32)

    for k in range(steps):
        lam = hazards_arr[:, k : k + 1]
        p_k = lam * unhalted
        halting_probs.append(p_k)
        unhalted = unhalted * (1.0 - lam)

    return jnp.concatenate(halting_probs, axis=-1)


def jepa_predict(
    s_x: Any,
    w_p1: Any,
    b_p1: Any,
    w_p2: Any,
    b_p2: Any,
) -> Any:
    """Predicts target representation in latent space: GELU(s_x @ w_p1 + b_p1) @ w_p2 + b_p2."""
    sx = jnp.asarray(s_x, dtype=jnp.float32)
    wp1 = jnp.asarray(w_p1, dtype=jnp.float32)
    bp1 = jnp.asarray(b_p1, dtype=jnp.float32)
    wp2 = jnp.asarray(w_p2, dtype=jnp.float32)
    bp2 = jnp.asarray(b_p2, dtype=jnp.float32)

    h = gelu(jnp.dot(sx, wp1) + bp1)
    return jnp.dot(h, wp2) + bp2


def jepa_loss(
    pred: Any,
    target: Any,
) -> Any:
    """Evaluates squared L2 representation prediction loss."""
    p = jnp.asarray(pred, dtype=jnp.float32)
    t = jnp.asarray(target, dtype=jnp.float32)
    return jnp.sum(jnp.square(p - t), axis=-1)


def linear_forward(
    x: Any,
    weight: Any,
    bias: Any,
) -> Any:
    """Canonical linear forward pass matching test_output: x @ weight + bias."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_arr = jnp.asarray(weight, dtype=jnp.float32)
    b_arr = jnp.asarray(bias, dtype=jnp.float32)
    return jnp.dot(x_arr, w_arr) + b_arr


if __name__ == "__main__":
    ckpt_path = Path("weights/01_paradigms_and_dichotomies/08_computation_spaces_latent_reasoning_and_jepa.safetensors")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    tensors = load_file(str(ckpt_path))
    x = tensors["test_input"].numpy()
    w = tensors["layers.weight"].numpy()
    b = tensors["layers.bias"].numpy()
    y_gold = tensors["test_output"].numpy()

    y_pred = linear_forward(x, w, b)
    max_diff = float(jnp.max(jnp.abs(y_pred - y_gold)))
    assert max_diff < 1e-5, f"JAX linear anchor mismatch: {max_diff}"

    trajectories, final_thought = continuous_latent_reasoning(
        x,
        tensors["layers.w_proj"].numpy(),
        tensors["layers.b_proj"].numpy(),
        tensors["layers.w_z"].numpy(),
        tensors["layers.w_anc"].numpy(),
        tensors["layers.b_z"].numpy(),
        steps=3,
    )
    gold_traj = tensors["layers.latent_trajectories"].numpy()
    traj_diff = float(jnp.max(jnp.abs(trajectories - gold_traj)))
    assert traj_diff < 1e-4, f"JAX trajectory mismatch: {traj_diff}"

    halting_probs = ponder_halting(
        trajectories,
        tensors["layers.w_h"].numpy(),
        tensors["layers.b_h"].numpy(),
    )
    gold_halt = tensors["layers.halting_probs"].numpy()
    halt_diff = float(jnp.max(jnp.abs(halting_probs - gold_halt)))
    assert halt_diff < 1e-4, f"JAX halting probs mismatch: {halt_diff}"

    jepa_pred = jepa_predict(
        final_thought,
        tensors["layers.w_p1"].numpy(),
        tensors["layers.b_p1"].numpy(),
        tensors["layers.w_p2"].numpy(),
        tensors["layers.b_p2"].numpy(),
    )
    gold_jepa = tensors["layers.jepa_pred"].numpy()
    jepa_diff = float(jnp.max(jnp.abs(jepa_pred - gold_jepa)))
    assert jepa_diff < 1e-4, f"JAX JEPA pred mismatch: {jepa_diff}"

    print("Functional JAX Module 08 passed all smoke tests and golden parity assertions!")
