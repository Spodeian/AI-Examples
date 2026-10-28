//! Complete 27-Format Numeric & Precision Lattice according to GLOSSARY_AND_NOTATION.md.

use serde::{Deserialize, Serialize};

/// Bit-level numerical representation format across deep learning, edge, and scientific computing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionFormat {
    // 1. IEEE 754 & Industry Standard Floating Point
    #[serde(rename = "fp64", alias = "f_p64")]
    FP64,
    #[default]
    #[serde(rename = "fp32", alias = "f_p32")]
    FP32,
    #[serde(rename = "tf32", alias = "t_f32")]
    TF32,
    #[serde(rename = "fp16", alias = "f_p16")]
    FP16,
    #[serde(rename = "bf16", alias = "b_f16")]
    BF16,

    // 2. 8-Bit & 4-Bit Minifloats
    #[serde(rename = "fp8_e4m3", alias = "fp8_e4m3fn", alias = "f_p8_e4_m3_f_n")]
    FP8E4M3FN,
    #[serde(rename = "fp8_e4m3fnuz", alias = "f_p8_e4_m3_f_n_u_z")]
    FP8E4M3FNUZ,
    #[serde(rename = "fp8_e5m2", alias = "f_p8_e5_m2")]
    FP8E5M2,
    #[serde(rename = "fp8_e5m2fnuz", alias = "f_p8_e5_m2_f_n_u_z")]
    FP8E5M2FNUZ,
    #[serde(rename = "fp4_e2m1", alias = "f_p4_e2_m1")]
    FP4E2M1,

    // 3. OCP Microscaling Formats (Shared E8M0 scale per 32 elements)
    #[serde(rename = "mxfp8", alias = "m_x_f_p8")]
    MXFP8,
    #[serde(rename = "mxfp6_e3m2", alias = "m_x_f_p6_e3_m2")]
    MXFP6E3M2,
    #[serde(rename = "mxfp6_e2m3", alias = "m_x_f_p6_e2_m3")]
    MXFP6E2M3,
    #[serde(rename = "mxfp4_e2m1", alias = "m_x_f_p4_e2_m1")]
    MXFP4E2M1,

    // 4. Fixed-Point, Integer & Sub-Bit Formats
    #[serde(rename = "int32", alias = "i_n_t32")]
    INT32,
    #[serde(rename = "int16", alias = "i_n_t16")]
    INT16,
    #[serde(rename = "int8_sym", alias = "int8", alias = "i_n_t8_sym")]
    INT8Sym,
    #[serde(rename = "int8_asym", alias = "i_n_t8_asym")]
    INT8Asym,
    #[serde(rename = "uint8", alias = "u_i_n_t8")]
    UINT8,
    #[serde(rename = "int4_sym", alias = "int4", alias = "i_n_t4_sym")]
    INT4Sym,
    #[serde(rename = "int4_asym", alias = "i_n_t4_asym")]
    INT4Asym,
    #[serde(rename = "nf4", alias = "n_f4")]
    NF4,
    #[serde(rename = "sf4", alias = "s_f4")]
    SF4,
    #[serde(rename = "int2", alias = "i_n_t2")]
    INT2,
    #[serde(rename = "bitnet158b", alias = "bitnet_158b", alias = "bit_net158b")]
    BitNet158b, // Native Ternary {-1, 0, +1}
    #[serde(rename = "binary_xnor")]
    BinaryXnor, // 1-bit {-1, +1}

    // 5. Alternative Arithmetic Systems
    #[serde(rename = "posit8_0")]
    Posit8_0,
    #[serde(rename = "posit8_1")]
    Posit8_1,
    #[serde(rename = "posit16_1")]
    Posit16_1,
    #[serde(rename = "lns8", alias = "l_n_s8")]
    LNS8, // 8-bit Logarithmic Number System
    #[serde(rename = "e8_lattice_qtip", alias = "e8_lattice_q_t_i_p")]
    E8LatticeQTIP,

    // 6. Cryptographic & Formal Verification Substrates
    #[serde(rename = "zkml_fp", alias = "z_k_m_l_finite_field_fp")]
    ZKMLFiniteFieldFp,
    #[serde(rename = "ckks", alias = "c_k_k_s_ring_l_w_e")]
    CKKSRingLWE,
    #[serde(rename = "ibp_crown", alias = "i_b_p_c_r_o_w_n_interval_bounds")]
    IBPCROWNIntervalBounds,
}

impl PrecisionFormat {
    /// Nominal total bits per element (or effective bits for microscaling/ternary).
    pub fn bits(self) -> f32 {
        match self {
            Self::FP64 => 64.0,
            Self::FP32 | Self::TF32 | Self::INT32 => 32.0,
            Self::FP16 | Self::BF16 | Self::INT16 | Self::Posit16_1 => 16.0,
            Self::FP8E4M3FN
            | Self::FP8E4M3FNUZ
            | Self::FP8E5M2
            | Self::FP8E5M2FNUZ
            | Self::MXFP8
            | Self::INT8Sym
            | Self::INT8Asym
            | Self::UINT8
            | Self::Posit8_0
            | Self::Posit8_1
            | Self::LNS8 => 8.0,
            Self::MXFP6E3M2 | Self::MXFP6E2M3 => 6.25, // 6 bits + shared scale amortized
            Self::FP4E2M1
            | Self::MXFP4E2M1
            | Self::INT4Sym
            | Self::INT4Asym
            | Self::NF4
            | Self::SF4 => 4.25,
            Self::E8LatticeQTIP => 2.5,
            Self::INT2 => 2.0,
            Self::BitNet158b => 1.58, // log2(3)
            Self::BinaryXnor => 1.0,
            Self::ZKMLFiniteFieldFp => 256.0,
            Self::CKKSRingLWE => 128.0,
            Self::IBPCROWNIntervalBounds => 64.0, // lower + upper FP32
        }
    }

    pub fn is_floating_point(self) -> bool {
        matches!(
            self,
            Self::FP64
                | Self::FP32
                | Self::TF32
                | Self::FP16
                | Self::BF16
                | Self::FP8E4M3FN
                | Self::FP8E4M3FNUZ
                | Self::FP8E5M2
                | Self::FP8E5M2FNUZ
                | Self::FP4E2M1
                | Self::MXFP8
                | Self::MXFP6E3M2
                | Self::MXFP6E2M3
                | Self::MXFP4E2M1
        )
    }

    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Self::INT32
                | Self::INT16
                | Self::INT8Sym
                | Self::INT8Asym
                | Self::UINT8
                | Self::INT4Sym
                | Self::INT4Asym
                | Self::INT2
        )
    }

    pub fn is_ternary_or_binary(self) -> bool {
        matches!(self, Self::BitNet158b | Self::BinaryXnor)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::FP64 => "FP64 (Double Precision)",
            Self::FP32 => "FP32 (Single Precision)",
            Self::TF32 => "TF32 (TensorFloat-32)",
            Self::FP16 => "FP16 (Half Precision)",
            Self::BF16 => "BF16 (BFloat16)",
            Self::FP8E4M3FN => "FP8 E4M3FN (Forward Weights/Activations)",
            Self::FP8E4M3FNUZ => "FP8 E4M3FNUZ (AMD/PyTorch)",
            Self::FP8E5M2 => "FP8 E5M2 (Gradients & Backward Pass)",
            Self::FP8E5M2FNUZ => "FP8 E5M2FNUZ (Unsigned Zero)",
            Self::FP4E2M1 => "FP4 E2M1 (Extreme Minifloat)",
            Self::MXFP8 => "MXFP8 (Microscaling Block-32)",
            Self::MXFP6E3M2 => "MXFP6 E3M2 (Microscaling 6-bit)",
            Self::MXFP6E2M3 => "MXFP6 E2M3 (Microscaling High-Mantissa)",
            Self::MXFP4E2M1 => "MXFP4 E2M1 (OCP 4-bit Microscaled)",
            Self::INT32 => "INT32 (Signed Integer)",
            Self::INT16 => "INT16 (Audio / Fixed DSP)",
            Self::INT8Sym => "INT8 Symmetric (Standard CPU/GPU PTQ)",
            Self::INT8Asym => "INT8 Asymmetric (ReLU Zero-Offset)",
            Self::UINT8 => "UINT8 (Unsigned Image / Index)",
            Self::INT4Sym => "INT4 Symmetric (GPTQ / AWQ)",
            Self::INT4Asym => "INT4 Asymmetric (Zero-Point Offset)",
            Self::NF4 => "NF4 (NormalFloat-4 QLoRA)",
            Self::SF4 => "SF4 (Signed Float 4-bit)",
            Self::INT2 => "INT2 (2-bit Compressed)",
            Self::BitNet158b => "BitNet b1.58 (Native Ternary {-1, 0, +1})",
            Self::BinaryXnor => "1-Bit Binary (XNOR Popcount)",
            Self::Posit8_0 => "Posit(8, 0) Dynamic Golden",
            Self::Posit8_1 => "Posit(8, 1) Broad Dynamic Range",
            Self::Posit16_1 => "Posit(16, 1) High Precision Scientific",
            Self::LNS8 => "LNS8 (8-bit Logarithmic Number System)",
            Self::E8LatticeQTIP => "E8 Lattice QTIP (Vector Quantization)",
            Self::ZKMLFiniteFieldFp => "ZKML Finite Field F_p (Freivalds Verification)",
            Self::CKKSRingLWE => "CKKS Homomorphic Encryption (Ring-LWE)",
            Self::IBPCROWNIntervalBounds => "CROWN / IBP (Formal Neural Verification Bounds)",
        }
    }
}

/// Precision mapping configuration across different functional roles in the architecture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericLatticeConfig {
    pub default_weight_precision: PrecisionFormat,
    pub default_activation_precision: PrecisionFormat,
    pub reduction_precision: PrecisionFormat, // e.g. FP32 for LayerNorm / Softmax reductions
    pub master_weight_precision: PrecisionFormat, // e.g. FP32 or BF16 for optimizer moments
    #[serde(default)]
    pub per_node_overrides: std::collections::HashMap<String, PrecisionFormat>,
}

impl Default for NumericLatticeConfig {
    fn default() -> Self {
        Self {
            default_weight_precision: PrecisionFormat::BF16,
            default_activation_precision: PrecisionFormat::BF16,
            reduction_precision: PrecisionFormat::FP32,
            master_weight_precision: PrecisionFormat::FP32,
            per_node_overrides: std::collections::HashMap::new(),
        }
    }
}
