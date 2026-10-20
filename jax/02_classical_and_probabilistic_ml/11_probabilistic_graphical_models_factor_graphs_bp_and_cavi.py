"""Probabilistic Graphical Models: Factor Graphs, Belief Propagation, and CAVI in JAX.

Features:
- Exact Sum-Product Belief Propagation on tree factor graphs
- Max-Product MAP inference
- Damped Loopy Belief Propagation on cyclic graphs
- Coordinate Ascent Mean-Field Variational Inference (CAVI)
"""

from __future__ import annotations
from pathlib import Path
import jax
import jax.numpy as jnp
import safetensors.torch
import numpy as np


@jax.jit
def sum_product_tree_bp(
    unary: jnp.ndarray,       # [4, 2]
    edges: jnp.ndarray,       # [3, 2, 2]
) -> jnp.ndarray:
    """Computes exact marginals for a 4-node tree rooted at node 1."""
    # Leaves to root 1
    m_0_1 = unary[0] @ edges[0]
    m_2_1 = edges[1] @ unary[2]
    m_3_1 = edges[2] @ unary[3]

    b1_unnorm = unary[1] * m_0_1 * m_2_1 * m_3_1
    b1 = b1_unnorm / jnp.sum(b1_unnorm)

    # Root 1 to leaves
    prod_1_to_0 = unary[1] * m_2_1 * m_3_1
    m_1_0 = edges[0] @ prod_1_to_0
    b0_unnorm = unary[0] * m_1_0
    b0 = b0_unnorm / jnp.sum(b0_unnorm)

    prod_1_to_2 = unary[1] * m_0_1 * m_3_1
    m_1_2 = prod_1_to_2 @ edges[1]
    b2_unnorm = unary[2] * m_1_2
    b2 = b2_unnorm / jnp.sum(b2_unnorm)

    prod_1_to_3 = unary[1] * m_0_1 * m_2_1
    m_1_3 = prod_1_to_3 @ edges[2]
    b3_unnorm = unary[3] * m_1_3
    b3 = b3_unnorm / jnp.sum(b3_unnorm)

    return jnp.stack([b0, b1, b2, b3])


def max_product_tree_map(
    unary: jnp.ndarray,       # [4, 2]
    edges: jnp.ndarray,       # [3, 2, 2]
) -> jnp.ndarray:
    """Computes exact MAP state configuration on tree."""
    best_config = None
    max_p = -1.0
    for x0 in (0, 1):
        for x1 in (0, 1):
            for x2 in (0, 1):
                for x3 in (0, 1):
                    p = float(
                        unary[0, x0] * unary[1, x1] * unary[2, x2] * unary[3, x3]
                        * edges[0, x0, x1] * edges[1, x1, x2] * edges[2, x1, x3]
                    )
                    if p > max_p:
                        max_p = p
                        best_config = [x0, x1, x2, x3]
    return jnp.array(best_config, dtype=jnp.int32)


def damped_loopy_bp_triangle(
    unary: jnp.ndarray,       # [3, 2]
    edges: jnp.ndarray,       # [3, 2, 2]
    alpha: float = 0.5,
    iters: int = 20,
) -> jnp.ndarray:
    """Damped Loopy BP on triangle graph with alpha smoothing."""
    m = np.ones((6, 2), dtype=np.float32) * 0.5
    u_np = np.array(unary)
    e_np = np.array(edges)

    for _ in range(iters):
        # 0 -> 1
        raw_0_1 = (u_np[0] * m[4]) @ e_np[0]
        raw_0_1 /= raw_0_1.sum()
        m[0] = (1.0 - alpha) * m[0] + alpha * raw_0_1

        # 1 -> 0
        raw_1_0 = e_np[0] @ (u_np[1] * m[3])
        raw_1_0 /= raw_1_0.sum()
        m[1] = (1.0 - alpha) * m[1] + alpha * raw_1_0

        # 1 -> 2
        raw_1_2 = (u_np[1] * m[0]) @ e_np[1]
        raw_1_2 /= raw_1_2.sum()
        m[2] = (1.0 - alpha) * m[2] + alpha * raw_1_2

        # 2 -> 1
        raw_2_1 = e_np[1] @ (u_np[2] * m[5])
        raw_2_1 /= raw_2_1.sum()
        m[3] = (1.0 - alpha) * m[3] + alpha * raw_2_1

        # 2 -> 0
        raw_2_0 = (u_np[2] * m[2]) @ e_np[2]
        raw_2_0 /= raw_2_0.sum()
        m[4] = (1.0 - alpha) * m[4] + alpha * raw_2_0

        # 0 -> 2
        raw_0_2 = e_np[2] @ (u_np[0] * m[1])
        raw_0_2 /= raw_0_2.sum()
        m[5] = (1.0 - alpha) * m[5] + alpha * raw_0_2

    b0 = u_np[0] * m[1] * m[4]
    b0 /= b0.sum()

    b1 = u_np[1] * m[0] * m[3]
    b1 /= b1.sum()

    b2 = u_np[2] * m[2] * m[5]
    b2 /= b2.sum()

    return jnp.array(np.stack([b0, b1, b2]))


@jax.jit
def cavi_gmm_step(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma_sq: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes CAVI responsibilities and ELBO."""
    diff = jnp.expand_dims(x, 1) - jnp.expand_dims(mu, 0)
    log_p = -0.5 * (diff ** 2) / sigma_sq
    r = jax.nn.softmax(log_p, axis=1)
    elbo = jnp.sum(r * (log_p - jnp.log(jnp.clip(r, a_min=1e-8))))
    return r, elbo


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    unary_tree = jnp.array(chk["unary_tree"].numpy())
    edge_tree = jnp.array(chk["edge_tree"].numpy())
    golden_tree_marg = chk["exact_marginals"].numpy()
    golden_map = chk["map_config"].numpy()

    # 1. Sum-Product Tree BP Parity
    marg = sum_product_tree_bp(unary_tree, edge_tree)
    diff_tree = float(np.max(np.abs(np.array(marg) - golden_tree_marg)))
    assert diff_tree < 1e-5, f"JAX tree marginals mismatch: {diff_tree:.2e}"

    # 2. Max-Product MAP Parity
    map_res = max_product_tree_map(unary_tree, edge_tree)
    assert np.array_equal(np.array(map_res), golden_map), "JAX MAP configuration mismatch"

    # 3. Damped Loopy BP Parity
    unary_tri = jnp.array(chk["unary_tri"].numpy())
    edge_tri = jnp.array(chk["edge_tri"].numpy())
    golden_loopy = chk["loopy_marginals"].numpy()
    loopy_res = damped_loopy_bp_triangle(unary_tri, edge_tri, alpha=0.5, iters=20)
    diff_loopy = float(np.max(np.abs(np.array(loopy_res) - golden_loopy)))
    assert diff_loopy < 1e-5, f"JAX Loopy BP marginals mismatch: {diff_loopy:.2e}"

    # 4. CAVI Parity
    data_cavi = jnp.array(chk["data_cavi"].numpy())
    mu_cavi = jnp.array(chk["mu_cavi"].numpy())
    golden_resp = chk["responsibilities"].numpy()
    golden_elbo = float(chk["elbo_val"][0].numpy())
    resp, elbo = cavi_gmm_step(data_cavi, mu_cavi)
    diff_resp = float(np.max(np.abs(np.array(resp) - golden_resp)))
    diff_elbo = abs(float(elbo) - golden_elbo)
    assert diff_resp < 1e-5, f"JAX CAVI responsibilities mismatch: {diff_resp:.2e}"
    assert diff_elbo < 1e-4, f"JAX CAVI ELBO mismatch: {diff_elbo:.2e}"

    print("JAX Topic 11 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
