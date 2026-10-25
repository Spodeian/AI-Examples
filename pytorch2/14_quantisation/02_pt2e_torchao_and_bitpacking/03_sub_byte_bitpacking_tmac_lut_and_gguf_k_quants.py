"""Sub Byte Bitpacking T-MAC LUT and GGUF K Quants — PyTorch 2.x reference implementation.

Topic: 14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants
Golden weights: weights/14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.safetensors

``python pytorch2/14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def bitpack_2x4bit(v_int4: torch.Tensor) -> torch.Tensor:
    """Packs pairs of 4-bit unsigned integers [0, 15] into a single byte."""
    v = v_int4.long()
    packed = (v[0::2] & 0x0F) | ((v[1::2] & 0x0F) << 4)
    return packed.float()


def bitunpack_2x4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpacks a byte array into interleaved 4-bit unsigned integers."""
    p = packed.long()
    out = torch.empty(len(p) * 2, dtype=torch.float32, device=packed.device)
    out[0::2] = (p & 0x0F).float()
    out[1::2] = ((p >> 4) & 0x0F).float()
    return out


def bitpack_5x_ternary(trits: torch.Tensor) -> torch.Tensor:
    """Packs groups of 5 ternary trits {0, 1, 2} into a single byte (3^5 = 243 < 256)."""
    t = trits.long().reshape(-1, 5)
    code = t[:, 0] + 3 * t[:, 1] + 9 * t[:, 2] + 27 * t[:, 3] + 81 * t[:, 4]
    return code.float()


def bitunpack_5x_ternary(packed_codes: torch.Tensor) -> torch.Tensor:
    """Unpacks a base-3 packed byte array into groups of 5 ternary trits."""
    c = packed_codes.long()
    t0 = c % 3
    c1 = c // 3
    t1 = c1 % 3
    c2 = c1 // 3
    t2 = c2 % 3
    c3 = c2 // 3
    t3 = c3 % 3
    t4 = c3 // 3
    return torch.stack([t0, t1, t2, t3, t4], dim=-1).flatten().float()


def tmac_precompute_lut(activations_group4: torch.Tensor) -> torch.Tensor:
    """Precomputes 16-entry addition table for a group of 4 activations."""
    lut = torch.zeros(16, dtype=torch.float32, device=activations_group4.device)
    a = activations_group4.flatten()
    for i in range(16):
        val = 0.0
        for k in range(4):
            if (i >> k) & 1:
                val += float(a[k])
        lut[i] = val
    return lut


def tmac_lut_lookup(lut: torch.Tensor, weight_indices: torch.Tensor) -> torch.Tensor:
    """Computes multiplier-free matrix multiplication via direct table lookup."""
    idx = weight_indices.long()
    return lut[idx]


def gguf_q4_k_quantize_block(
    w_block256: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GGUF Q4_K_M hierarchical super-block quantization (256 elements in 8 sub-blocks)."""
    w = w_block256.flatten()
    d = float(torch.max(torch.abs(w))) / 127.0
    d = max(d, 1e-8)
    sub_scales = []
    dequants = []

    for k in range(8):
        chunk = w[k * 32 : (k + 1) * 32]
        cmax = float(torch.max(torch.abs(chunk)))
        dk = cmax / (7.0 * d) if cmax > 0.0 else 1.0
        eff_scale = d * dk
        q = torch.clamp(torch.round(chunk / eff_scale), -8.0, 7.0)
        deq = q * eff_scale
        sub_scales.append(dk)
        dequants.append(deq)

    dequant_tensor = torch.cat(dequants)
    super_scale_tensor = torch.tensor([d], dtype=torch.float32)
    sub_scales_tensor = torch.tensor(sub_scales, dtype=torch.float32)
    return dequant_tensor, super_scale_tensor, sub_scales_tensor


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. 2x4-bit Bitpacking
    raw_nibbles = (torch.randint(0, 16, (64,)).float())
    packed_2x4bit = bitpack_2x4bit(raw_nibbles)
    unpacked_nibbles = bitunpack_2x4bit(packed_2x4bit)

    # 2. 5x Ternary Bitpacking
    raw_trits = (torch.randint(0, 3, (50,)).float())
    packed_ternary = bitpack_5x_ternary(raw_trits)
    unpacked_trits = bitunpack_5x_ternary(packed_ternary)

    # 3. T-MAC 4-bit Table Lookup
    tmac_act = torch.tensor([1.2, -0.8, 2.5, -1.4], dtype=torch.float32)
    tmac_lut = tmac_precompute_lut(tmac_act)
    tmac_weight_indices = torch.tensor([0, 5, 10, 15], dtype=torch.float32)
    tmac_dot_ref = tmac_lut_lookup(tmac_lut, tmac_weight_indices)

    # 4. GGUF Q4_K Super-Block Quantization
    gguf_raw_w = torch.randn(256) * 0.35
    gguf_dequant, gguf_super_scale, gguf_sub_scales = gguf_q4_k_quantize_block(gguf_raw_w)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "raw_nibbles": raw_nibbles,
        "packed_2x4bit": packed_2x4bit,
        "unpacked_nibbles": unpacked_nibbles,
        "raw_trits": raw_trits,
        "packed_ternary": packed_ternary,
        "unpacked_trits": unpacked_trits,
        "tmac_act": tmac_act,
        "tmac_lut": tmac_lut,
        "tmac_weight_indices": tmac_weight_indices,
        "tmac_dot_ref": tmac_dot_ref,
        "gguf_raw_w": gguf_raw_w,
        "gguf_dequant": gguf_dequant,
        "gguf_super_scale": gguf_super_scale,
        "gguf_sub_scales": gguf_sub_scales,
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
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
