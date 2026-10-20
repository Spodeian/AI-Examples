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

use ai_kernel::c05_architectures::t02_mixture_of_experts_top_k_expert_choice_and_soft_moe as t02;

#[test]
fn test_02_mixture_of_experts_top_k_expert_choice_and_soft_moe_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.safetensors",
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

    let anchor_calc = t02::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. Soft MoE Dispatch Kernel Verification
    let num_tokens = 8;
    let d_model = 16;
    let total_slots = 4;

    let x = vec![0.25f32; num_tokens * d_model];
    let phi = vec![0.125f32; num_tokens * total_slots];

    let dispatch_cpu = t02::soft_moe_dispatch_cpu(&x, &phi, num_tokens, d_model, total_slots);
    assert_eq!(dispatch_cpu.len(), total_slots * d_model);

    let dispatch_auto = t02::soft_moe_dispatch_auto(None, &x, &phi, num_tokens, d_model, total_slots);
    for (i, (&c, &a)) in dispatch_cpu.iter().zip(dispatch_auto.iter()).enumerate() {
        assert!((c - a).abs() < 1e-5, "Dispatch CPU vs Auto mismatch at {i}: {c} vs {a}");
    }

    // 3. Soft MoE Combine Kernel Verification
    let slot_outputs = vec![0.5f32; total_slots * d_model];
    let psi = vec![0.25f32; num_tokens * total_slots];

    let combine_cpu = t02::soft_moe_combine_cpu(&psi, &slot_outputs, num_tokens, d_model, total_slots);
    assert_eq!(combine_cpu.len(), num_tokens * d_model);

    Ok(())
}

use ai_kernel::c05_architectures::t03_deepseek_moe_aux_loss_free_and_mixture_of_depths as t03;

#[test]
fn test_03_deepseek_moe_aux_loss_free_and_mixture_of_depths_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.safetensors",
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

    let anchor_calc = t03::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. MoD Selective Gating Kernel Verification
    let num_tokens = 8;
    let d_model = 16;
    let total_elements = num_tokens * d_model;

    let x = vec![0.5f32; total_elements];
    let block_out = vec![1.2f32; total_elements];
    let gate = vec![0.8f32; num_tokens];
    let mask = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // 50% capacity

    let gating_cpu = t03::mod_selective_gating_cpu(&x, &block_out, &gate, &mask, num_tokens, d_model);
    assert_eq!(gating_cpu.len(), total_elements);

    let gating_auto = t03::mod_selective_gating_auto(None, &x, &block_out, &gate, &mask, num_tokens, d_model);
    for (i, (&c, &a)) in gating_cpu.iter().zip(gating_auto.iter()).enumerate() {
        assert!((c - a).abs() < 1e-5, "MoD Gating CPU vs Auto mismatch at {i}: {c} vs {a}");
    }

    // 3. Dynamic Bias Update Verification
    let b_dyn = vec![0.1f32, -0.05, 0.05, -0.1];
    let counts = vec![3.0f32, 5.0, 4.0, 4.0];
    let b_new = t03::dynamic_bias_update_cpu(&b_dyn, &counts, 0.1);
    assert_eq!(b_new.len(), 4);

    Ok(())
}

use ai_kernel::c05_architectures::t04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling as t04;

#[test]
fn test_04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.safetensors",
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

    let anchor_calc = t04::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. Dynamic Parameter Merging (SMEAR Kernel Verification)
    let experts_w1 = tensors.get("experts_w1").unwrap().to_vec3::<f32>()?;
    let experts_w1_flat: Vec<f32> = experts_w1.into_iter().flatten().flatten().collect();
    let num_experts = 4;
    let in_dim = 16;
    let out_dim = 32;

    let routing_weights = vec![0.25f32, 0.25, 0.25, 0.25];

    let merged_cpu = t04::soup_weight_merge_cpu(&experts_w1_flat, &routing_weights, num_experts, in_dim, out_dim);
    assert_eq!(merged_cpu.len(), in_dim * out_dim);

    let merged_auto = t04::soup_weight_merge_auto(None, &experts_w1_flat, &routing_weights, num_experts, in_dim, out_dim);
    for (i, (&c, &a)) in merged_cpu.iter().zip(merged_auto.iter()).enumerate() {
        assert!((c - a).abs() < 1e-5, "Merged weights CPU vs Auto mismatch at {i}: {c} vs {a}");
    }

    Ok(())
}


