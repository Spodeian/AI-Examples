"""Safe Serialization Weights Only Safetensors and Mmap — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/01_safe_serialization_weights_only_safetensors_and_mmap.
Run ``python jax/17_deployment_low_level_and_capstones/01_safe_serialization_weights_only_safetensors_and_mmap.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/01_safe_serialization_weights_only_safetensors_and_mmap.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def safetensors_pack_tensors(tensors: dict[str, jnp.ndarray]) -> tuple[bytes, int]:
    header = {}
    offset = 0
    payload_buffers = []

    for name, tensor in tensors.items():
        arr = np.asarray(tensor, dtype=np.float32)
        raw_bytes = arr.tobytes()
        num_bytes = len(raw_bytes)
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + num_bytes],
        }
        offset += num_bytes
        payload_buffers.append(raw_bytes)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)
    header_len_bytes = struct.pack("<Q", header_len)

    wire_bytes = header_len_bytes + header_json + b"".join(payload_buffers)
    return wire_bytes, header_len


def safetensors_unpack_tensors(wire_bytes: bytes) -> dict[str, jnp.ndarray]:
    header_len = struct.unpack("<Q", wire_bytes[:8])[0]
    header_json_bytes = wire_bytes[8 : 8 + header_len]
    header = json.loads(header_json_bytes.decode("utf-8"))

    data_start = 8 + header_len
    unpacked = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start_offset = data_start + meta["data_offsets"][0]
        end_offset = data_start + meta["data_offsets"][1]
        chunk = wire_bytes[start_offset:end_offset]
        shape = tuple(meta["shape"])
        arr = np.frombuffer(chunk, dtype=np.float32).reshape(shape)
        unpacked[name] = jnp.array(arr)

    return unpacked


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Projection check with golden weights
    y_ref = g["x_query"] @ g["w_proj"] + g["b_proj"]
    err = float(jnp.max(jnp.abs(y_ref - g["y_ref"])))
    assert err < 1e-5, f"y_ref projection error {err:.2e}"

    # Pack and unpack roundtrip in JAX
    test_dict = {"w_proj": g["w_proj"], "b_proj": g["b_proj"]}
    wire_bytes, header_len = safetensors_pack_tensors(test_dict)
    unpacked = safetensors_unpack_tensors(wire_bytes)
    y_unpacked = g["x_query"] @ unpacked["w_proj"] + unpacked["b_proj"]
    err = float(jnp.max(jnp.abs(y_unpacked - g["y_ref"])))
    assert err < 1e-5, f"unpacked roundtrip error {err:.2e}"

    print("17_deployment_low_level_and_capstones/01_safe_serialization_weights_only_safetensors_and_mmap: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
