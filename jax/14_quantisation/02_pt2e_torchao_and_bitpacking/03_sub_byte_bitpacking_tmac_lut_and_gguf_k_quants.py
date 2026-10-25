"""Sub Byte Bitpacking T-MAC LUT and GGUF K Quants — JAX implementation.

Topic: 14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants
Golden weights: weights/14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.safetensors
"""

from __future__ import annotations

from pathlib import Path
import jax.numpy as jnp
from safetensors.torch import load_file

TOPIC = "14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(x, weight) + bias


def bitpack_2x4bit(v_int4: jnp.ndarray) -> jnp.ndarray:
    v = v_int4.astype(jnp.int32)
    packed = (v[0::2] & 0x0F) | ((v[1::2] & 0x0F) << 4)
    return packed.astype(jnp.float32)


def bitunpack_2x4bit(packed: jnp.ndarray) -> jnp.ndarray:
    p = packed.astype(jnp.int32)
    v0 = (p & 0x0F).astype(jnp.float32)
    v1 = ((p >> 4) & 0x0F).astype(jnp.float32)
    stacked = jnp.stack([v0, v1], axis=-1)
    return stacked.flatten()


def bitpack_5x_ternary(trits: jnp.ndarray) -> jnp.ndarray:
    t = trits.astype(jnp.int32).reshape(-1, 5)
    code = t[:, 0] + 3 * t[:, 1] + 9 * t[:, 2] + 27 * t[:, 3] + 81 * t[:, 4]
    return code.astype(jnp.float32)


def bitunpack_5x_ternary(packed_codes: jnp.ndarray) -> jnp.ndarray:
    c = packed_codes.astype(jnp.int32)
    t0 = c % 3
    c1 = c // 3
    t1 = c1 % 3
    c2 = c1 // 3
    t2 = c2 % 3
    c3 = c2 // 3
    t3 = c3 % 3
    t4 = c3 // 3
    stacked = jnp.stack([t0, t1, t2, t3, t4], axis=-1)
    return stacked.flatten().astype(jnp.float32)


def tmac_precompute_lut(activations_group4: jnp.ndarray) -> jnp.ndarray:
    a = activations_group4.flatten()
    lut_entries = []
    for i in range(16):
        val = 0.0
        for k in range(4):
            if (i >> k) & 1:
                val += float(a[k])
        lut_entries.append(val)
    return jnp.array(lut_entries, dtype=jnp.float32)


def tmac_lut_lookup(lut: jnp.ndarray, weight_indices: jnp.ndarray) -> jnp.ndarray:
    idx = weight_indices.astype(jnp.int32)
    return lut[idx]


def gguf_q4_k_quantize_block(
    w_block256: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    w = w_block256.flatten()
    d = float(jnp.max(jnp.abs(w))) / 127.0
    d = max(d, 1e-8)
    sub_scales = []
    dequants = []

    for k in range(8):
        chunk = w[k * 32 : (k + 1) * 32]
        cmax = float(jnp.max(jnp.abs(chunk)))
        dk = cmax / (7.0 * d) if cmax > 0.0 else 1.0
        eff_scale = d * dk
        q = jnp.clip(jnp.round(chunk / eff_scale), -8.0, 7.0)
        deq = q * eff_scale
        sub_scales.append(dk)
        dequants.append(deq)

    dequant_array = jnp.concatenate(dequants)
    super_scale_array = jnp.array([d], dtype=jnp.float32)
    sub_scales_array = jnp.array(sub_scales, dtype=jnp.float32)
    return dequant_array, super_scale_array, sub_scales_array


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. 2x4-bit packing
    p2x4 = bitpack_2x4bit(g["raw_nibbles"])
    assert float(jnp.max(jnp.abs(p2x4 - g["packed_2x4bit"]))) < 1e-5
    unp2x4 = bitunpack_2x4bit(g["packed_2x4bit"])
    assert float(jnp.max(jnp.abs(unp2x4 - g["unpacked_nibbles"]))) < 1e-5

    # 3. 5x ternary packing
    p_trit = bitpack_5x_ternary(g["raw_trits"])
    assert float(jnp.max(jnp.abs(p_trit - g["packed_ternary"]))) < 1e-5
    unp_trit = bitunpack_5x_ternary(g["packed_ternary"])
    assert float(jnp.max(jnp.abs(unp_trit - g["unpacked_trits"]))) < 1e-5

    # 4. T-MAC LUT
    lut = tmac_precompute_lut(g["tmac_act"])
    assert float(jnp.max(jnp.abs(lut - g["tmac_lut"]))) < 1e-5
    tmac_out = tmac_lut_lookup(lut, g["tmac_weight_indices"])
    assert float(jnp.max(jnp.abs(tmac_out - g["tmac_dot_ref"]))) < 1e-5

    # 5. GGUF Q4_K
    deq, sup_s, sub_s = gguf_q4_k_quantize_block(g["gguf_raw_w"])
    assert float(jnp.max(jnp.abs(deq - g["gguf_dequant"]))) < 1e-5
    assert float(jnp.max(jnp.abs(sup_s - g["gguf_super_scale"]))) < 1e-5
    assert float(jnp.max(jnp.abs(sub_s - g["gguf_sub_scales"]))) < 1e-5

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
