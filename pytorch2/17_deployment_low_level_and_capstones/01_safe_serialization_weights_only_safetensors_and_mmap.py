"""Safe Serialization Weights Only Safetensors and Mmap — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/01_safe_serialization_weights_only_safetensors_and_mmap
Golden weights: weights/17_deployment_low_level_and_capstones/01_safe_serialization_weights_only_safetensors_and_mmap.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/01_safe_serialization_weights_only_safetensors_and_mmap.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/01_safe_serialization_weights_only_safetensors_and_mmap.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/01_safe_serialization_weights_only_safetensors_and_mmap"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def safetensors_pack_tensors(tensors: dict[str, torch.Tensor]) -> tuple[bytes, int]:
    """Serializes tensors into the safetensors binary wire format.

    Format:
        - 8 bytes: unsigned 64-bit little-endian integer N (header length in bytes)
        - N bytes: UTF-8 encoded JSON header metadata
        - Remaining bytes: Raw contiguous tensor binary payloads

    Returns:
        wire_bytes: Complete binary representation.
        header_len: Length of JSON header in bytes.
    """
    header = {}
    offset = 0
    payload_buffers = []

    for name, tensor in tensors.items():
        tensor_contig = tensor.detach().contiguous().cpu()
        raw_bytes = tensor_contig.numpy().tobytes()
        num_bytes = len(raw_bytes)
        dtype_str = "F32" if tensor.dtype == torch.float32 else str(tensor.dtype)
        header[name] = {
            "dtype": dtype_str,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + num_bytes],
        }
        offset += num_bytes
        payload_buffers.append(raw_bytes)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)
    header_len_bytes = struct.pack("<Q", header_len)

    wire_bytes = header_len_bytes + header_json + b"".join(payload_buffers)
    return wire_bytes, header_len


def safetensors_unpack_tensors(wire_bytes: bytes) -> dict[str, torch.Tensor]:
    """Deserializes tensors from the safetensors binary wire format."""
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
        tensor = torch.frombuffer(bytearray(chunk), dtype=torch.float32).reshape(shape)
        unpacked[name] = tensor.clone()

    return unpacked


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    x_query = torch.randn(4, 8)
    w_proj = torch.randn(8, 8) * 0.25
    b_proj = torch.randn(8) * 0.1
    y_ref = x_query @ w_proj + b_proj

    test_dict = {"w_proj": w_proj, "b_proj": b_proj}
    wire_bytes, header_len = safetensors_pack_tensors(test_dict)
    unpacked = safetensors_unpack_tensors(wire_bytes)

    # Invariant: projection with unpacked tensors matches y_ref exactly
    y_unpacked = x_query @ unpacked["w_proj"] + unpacked["b_proj"]
    assert float(torch.max(torch.abs(y_unpacked - y_ref))) < 1e-6

    header_size_tensor = torch.tensor([header_len], dtype=torch.int64)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_query": x_query,
        "w_proj": w_proj,
        "b_proj": b_proj,
        "y_ref": y_ref,
        "header_size": header_size_tensor,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
