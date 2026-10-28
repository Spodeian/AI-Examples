//! Multi-Precision Weights Quantizer & Zero-Copy Tensor Transformation Engine.
//! Supports FP8 E4M3FN, INT8 symmetric, BitNet 1.58b ternary, and NF4 NormalFloat.

use crate::spec::PrecisionFormat;
use serde::{Deserialize, Serialize};

/// 16 non-uniform quantile levels for NF4 NormalFloat4 quantization.
pub const NF4_QUANTILES: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.2844414,
    -0.18477343,
    -0.09105004,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];

/// Quantized tensor representation with associated scale factors and packed raw bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub precision: PrecisionFormat,
    pub scale: f32,
    pub zero_point: i32,
    pub raw_bytes: Vec<u8>,
}

impl QuantizedTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Quantizes FP32 weights into FP8 E4M3FN (8-bit with delayed max-abs scale).
pub fn quantize_fp32_to_fp8_e4m3fn(data: &[f32], name: &str, shape: &[usize]) -> QuantizedTensor {
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs / 448.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let mut raw_bytes = Vec::with_capacity(data.len());
    for &x in data {
        let scaled = (x * inv_scale).round().clamp(-448.0, 448.0);
        // Map [-448, 448] to signed byte representation [-128, 127] scaled by 448/127
        let byte_val = ((scaled / 448.0) * 127.0).round().clamp(-128.0, 127.0) as i8;
        raw_bytes.push(byte_val as u8);
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::FP8E4M3FN,
        scale,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes FP8 E4M3FN back to FP32.
pub fn dequantize_fp8_e4m3fn_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(tensor.raw_bytes.len());
    for &b in &tensor.raw_bytes {
        let byte_val = b as i8 as f32;
        let scaled = (byte_val / 127.0) * 448.0;
        out.push(scaled * tensor.scale);
    }
    out
}

/// Quantizes FP32 weights to symmetric INT8 with max-abs scaling.
pub fn quantize_fp32_to_int8_symmetric(
    data: &[f32],
    name: &str,
    shape: &[usize],
) -> QuantizedTensor {
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let mut raw_bytes = Vec::with_capacity(data.len());
    for &x in data {
        let quantized = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
        raw_bytes.push(quantized as u8);
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::INT8Sym,
        scale,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes INT8 symmetric weights back to FP32.
pub fn dequantize_int8_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    tensor
        .raw_bytes
        .iter()
        .map(|&b| (b as i8 as f32) * tensor.scale)
        .collect()
}

/// Quantizes FP32 weights into BitNet 1.58b ternary format {-1, 0, +1}.
pub fn quantize_fp32_to_bitnet_1_58(data: &[f32], name: &str, shape: &[usize]) -> QuantizedTensor {
    let sum_abs: f32 = data.iter().map(|x| x.abs()).sum();
    let gamma = if !data.is_empty() {
        sum_abs / data.len() as f32
    } else {
        1.0
    };
    let inv_gamma = 1.0 / (gamma + 1e-8);

    let mut raw_bytes = Vec::with_capacity(data.len());
    for &x in data {
        let scaled = (x * inv_gamma).round().clamp(-1.0, 1.0) as i8;
        raw_bytes.push(scaled as u8);
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::INT2,
        scale: gamma,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes BitNet 1.58b ternary weights back to FP32.
pub fn dequantize_bitnet_1_58_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    tensor
        .raw_bytes
        .iter()
        .map(|&b| (b as i8 as f32) * tensor.scale)
        .collect()
}

/// Quantizes FP32 weights into NF4 (NormalFloat4) format with packed nibbles.
pub fn quantize_fp32_to_nf4(data: &[f32], name: &str, shape: &[usize]) -> QuantizedTensor {
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let mut raw_bytes = Vec::with_capacity(data.len().div_ceil(2));
    let mut i = 0;
    while i < data.len() {
        let val1 = (data[i] * inv_scale).clamp(-1.0, 1.0);
        let idx1 = find_closest_nf4_index(val1);

        let idx2 = if i + 1 < data.len() {
            let val2 = (data[i + 1] * inv_scale).clamp(-1.0, 1.0);
            find_closest_nf4_index(val2)
        } else {
            0
        };

        // Pack two 4-bit nibbles into one byte
        let packed_byte = (idx1 & 0x0F) | ((idx2 & 0x0F) << 4);
        raw_bytes.push(packed_byte);
        i += 2;
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::NF4,
        scale,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes NF4 packed bytes back to FP32.
pub fn dequantize_nf4_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    let total_elements = tensor.num_elements();
    let mut out = Vec::with_capacity(total_elements);

    for &packed in &tensor.raw_bytes {
        let idx1 = (packed & 0x0F) as usize;
        let idx2 = ((packed >> 4) & 0x0F) as usize;

        if out.len() < total_elements {
            out.push(NF4_QUANTILES[idx1] * tensor.scale);
        }
        if out.len() < total_elements {
            out.push(NF4_QUANTILES[idx2] * tensor.scale);
        }
    }
    out
}

fn find_closest_nf4_index(val: f32) -> u8 {
    let mut best_idx = 0;
    let mut min_diff = (val - NF4_QUANTILES[0]).abs();

    for (i, &q) in NF4_QUANTILES.iter().enumerate().skip(1) {
        let diff = (val - q).abs();
        if diff < min_diff {
            min_diff = diff;
            best_idx = i;
        }
    }
    best_idx as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_quantization_roundtrip() {
        let input = vec![0.5, -1.2, 3.4, -0.05, 0.0, 2.1];
        let shape = vec![6];
        let q = quantize_fp32_to_int8_symmetric(&input, "test_weight", &shape);
        assert_eq!(q.raw_bytes.len(), 6);

        let deq = dequantize_int8_to_fp32(&q);
        assert_eq!(deq.len(), input.len());
        for (orig, rec) in input.iter().zip(deq.iter()) {
            assert!(
                (orig - rec).abs() < 0.05,
                "Mismatch: orig={}, rec={}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_bitnet_1_58_quantization() {
        let input = vec![1.5, -0.9, 0.1, -1.2, 0.0, 2.0];
        let shape = vec![6];
        let q = quantize_fp32_to_bitnet_1_58(&input, "ternary_linear", &shape);

        let deq = dequantize_bitnet_1_58_to_fp32(&q);
        assert_eq!(deq.len(), 6);
        for &val in &q.raw_bytes {
            let s = val as i8;
            assert!(s == -1 || s == 0 || s == 1);
        }
    }

    #[test]
    fn test_nf4_quantization_roundtrip() {
        let input = vec![0.8, -0.5, 0.0, 0.25, -0.95, 0.6];
        let shape = vec![6];
        let q = quantize_fp32_to_nf4(&input, "nf4_linear", &shape);
        assert_eq!(q.raw_bytes.len(), 3); // 6 elements packed into 3 bytes

        let deq = dequantize_nf4_to_fp32(&q);
        assert_eq!(deq.len(), 6);
        for (orig, rec) in input.iter().zip(deq.iter()) {
            assert!(
                (orig - rec).abs() < 0.15,
                "NF4 Mismatch: orig={}, rec={}",
                orig,
                rec
            );
        }
    }
}
