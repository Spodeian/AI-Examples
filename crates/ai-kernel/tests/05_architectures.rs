//! Category 05 Architectures Integration Test Suite for ai-kernel.

use ai_kernel::c05_architectures::t01_residual_mlp_mlp_mixer_and_gmlp as t01;
use candle_core::{safetensors::load, Device, Result};

#[path = "common/mod.rs"]
mod common;

#[test]
fn test_01_residual_mlp_mlp_mixer_and_gmlp_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t01::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. Spatial Gating Unit Kernel Verification
    let w_sgu = tensors.get("w_sgu").unwrap().to_vec2::<f32>()?;
    let w_sgu_flat: Vec<f32> = w_sgu.into_iter().flatten().collect();
    let b_sgu = tensors.get("b_sgu").unwrap().to_vec2::<f32>()?;
    let b_sgu_flat: Vec<f32> = b_sgu.into_iter().flatten().collect();

    let b_sz = 2;
    let s_sz = 4;
    let d_sz = 4;
    let total_elements = b_sz * s_sz * d_sz;

    let z1 = vec![0.5f32; total_elements];
    let z2 = vec![1.0f32; total_elements];

    let sgu_cpu = t01::spatial_gating_cpu(&z1, &z2, &w_sgu_flat, &b_sgu_flat, b_sz, s_sz, d_sz);
    assert_eq!(sgu_cpu.len(), total_elements);

    let sgu_auto = pollster::block_on(
        t01::spatial_gating_auto(None, &z1, &z2, &w_sgu_flat, &b_sgu_flat, b_sz, s_sz, d_sz)
    );

    for (i, (&c, &a)) in sgu_cpu.iter().zip(sgu_auto.iter()).enumerate() {
        assert!((c - a).abs() < 1e-5, "SGU CPU vs Auto mismatch at {i}: {c} vs {a}");
    }

    Ok(())
}
