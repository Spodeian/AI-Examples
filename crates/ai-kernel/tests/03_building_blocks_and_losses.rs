//! Category 03 Building Blocks & Losses Integration Test Suite for ai-kernel.

use ai_kernel::c03_building_blocks_and_losses::t01_activation_functions_zoo as t01;
use candle_core::{safetensors::load, Device, Result};

#[path = "common/mod.rs"]
mod common;

#[test]
fn test_01_activation_functions_zoo_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/01_activation_functions_zoo.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let x_anchor = tensors.get("x_anchor").unwrap().to_vec2::<f32>()?;
    let x_anchor_flat: Vec<f32> = x_anchor.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let anchor_gold = tensors.get("linear_anchor_out").unwrap().to_vec2::<f32>()?;
    let anchor_gold_flat: Vec<f32> = anchor_gold.into_iter().flatten().collect();

    let anchor_calc = t01::linear_anchor_forward_cpu(&x_anchor_flat, &weight_flat, &bias, 4, 8, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(anchor_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. SwiGLU & GeGLU on CPU
    let x_gate = tensors.get("x_gate").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let w_gate = tensors.get("w_gate").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let w_up = tensors.get("w_up").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let w_down = tensors.get("w_down").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let swiglu_gold = tensors.get("swiglu_out").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let geglu_gold = tensors.get("geglu_out").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();

    let swiglu_calc = t01::swiglu_cpu(&x_gate, &w_gate, &w_up, &w_down, 4, 4, 6, 4);
    for (i, (&calc, &gold)) in swiglu_calc.iter().zip(swiglu_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "SwiGLU error at {i}: {calc} vs {gold}");
    }

    let geglu_calc = t01::geglu_cpu(&x_gate, &w_gate, &w_up, &w_down, 4, 4, 6, 4);
    for (i, (&calc, &gold)) in geglu_calc.iter().zip(geglu_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-4, "GeGLU error at {i}: {calc} vs {gold}");
    }

    // 3. Snake on CPU and Auto
    let x_snake = tensors.get("x_snake").unwrap().to_vec1::<f32>()?;
    let snake_a = tensors.get("snake_a").unwrap().to_vec1::<f32>()?[0];
    let snake_gold = tensors.get("snake_out").unwrap().to_vec1::<f32>()?;

    let snake_calc = t01::snake_cpu(&x_snake, snake_a);
    for (i, (&calc, &gold)) in snake_calc.iter().zip(snake_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Snake CPU error at {i}: {calc} vs {gold}");
    }

    let snake_auto_res = pollster::block_on(t01::snake_auto(None, &x_snake, snake_a));
    for (i, (&calc, &gold)) in snake_auto_res.iter().zip(snake_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Snake Auto error at {i}: {calc} vs {gold}");
    }

    // 4. SELU on CPU
    let x_selu = tensors.get("x_selu").unwrap().to_vec1::<f32>()?;
    let selu_scale = tensors.get("selu_scale").unwrap().to_vec1::<f32>()?[0];
    let selu_alpha = tensors.get("selu_alpha").unwrap().to_vec1::<f32>()?[0];
    let selu_gold = tensors.get("selu_out").unwrap().to_vec1::<f32>()?;

    let selu_calc = t01::selu_cpu(&x_selu, selu_scale, selu_alpha);
    for (i, (&calc, &gold)) in selu_calc.iter().zip(selu_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "SELU error at {i}: {calc} vs {gold}");
    }

    // 5. Rectified & Smooth on CPU
    let x_dead = tensors.get("x_dead").unwrap().to_vec1::<f32>()?;
    let relu_gold = tensors.get("relu_out").unwrap().to_vec1::<f32>()?;
    let leaky_gold = tensors.get("leaky_relu_out").unwrap().to_vec1::<f32>()?;
    let silu_gold = tensors.get("silu_out").unwrap().to_vec1::<f32>()?;

    let relu_calc = t01::relu_cpu(&x_dead);
    for (i, (&calc, &gold)) in relu_calc.iter().zip(relu_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "ReLU error at {i}: {calc} vs {gold}");
    }

    let leaky_calc = t01::leaky_relu_cpu(&x_dead, 0.1);
    for (i, (&calc, &gold)) in leaky_calc.iter().zip(leaky_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "LeakyReLU error at {i}: {calc} vs {gold}");
    }

    let silu_calc = t01::silu_cpu(&x_dead);
    for (i, (&calc, &gold)) in silu_calc.iter().zip(silu_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "SiLU error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

use ai_kernel::c03_building_blocks_and_losses::t02_normalization_layers_complete as t02;

#[test]
fn test_02_normalization_layers_complete_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/02_normalization_layers_complete.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let test_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let gamma = tensors.get("gamma").unwrap().to_vec1::<f32>()?;
    let beta = tensors.get("beta").unwrap().to_vec1::<f32>()?;
    let dyt_alpha = tensors.get("dyt_alpha").unwrap().to_vec1::<f32>()?;
    let dyt_gamma = tensors.get("dyt_gamma").unwrap().to_vec1::<f32>()?;
    let dyt_beta = tensors.get("dyt_beta").unwrap().to_vec1::<f32>()?;
    let cond = tensors.get("cond_vector").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let w_adaln = tensors.get("w_adaln").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let b_adaln = tensors.get("b_adaln").unwrap().to_vec1::<f32>()?;
    let w_mat = tensors.get("w_matrix").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let eps = tensors.get("eps").unwrap().to_vec1::<f32>()?[0];

    // 1. Linear Anchor on CPU
    let anchor_calc = t02::linear_anchor_forward_cpu(&test_in, &weight, &bias, 4, 8, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(test_out.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. LayerNorm on CPU
    let ln_gold = tensors.get("y_layernorm").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let ln_calc = t02::layernorm_cpu(&test_in, &gamma, &beta, 4, 8, eps);
    for (i, (&calc, &gold)) in ln_calc.iter().zip(ln_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "LayerNorm error at {i}: {calc} vs {gold}");
    }

    // 3. RMSNorm on CPU and Auto
    let rms_gold = tensors.get("y_rmsnorm").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let rms_calc = t02::rmsnorm_cpu(&test_in, &gamma, 4, 8, eps);
    for (i, (&calc, &gold)) in rms_calc.iter().zip(rms_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "RMSNorm CPU error at {i}: {calc} vs {gold}");
    }
    let rms_auto = pollster::block_on(t02::rmsnorm_auto(None, &test_in, &gamma, 4, 8, eps));
    for (i, (&calc, &gold)) in rms_auto.iter().zip(rms_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "RMSNorm Auto error at {i}: {calc} vs {gold}");
    }

    // 4. DyT on CPU and Auto
    let dyt_gold = tensors.get("y_dyt").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let dyt_calc = t02::dyt_cpu(&test_in, &dyt_alpha, &dyt_gamma, &dyt_beta, 4, 8);
    for (i, (&calc, &gold)) in dyt_calc.iter().zip(dyt_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "DyT CPU error at {i}: {calc} vs {gold}");
    }
    let dyt_auto = pollster::block_on(t02::dyt_auto(None, &test_in, &dyt_alpha, &dyt_gamma, &dyt_beta, 4, 8));
    for (i, (&calc, &gold)) in dyt_auto.iter().zip(dyt_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "DyT Auto error at {i}: {calc} vs {gold}");
    }

    // 5. adaLN-Zero on CPU
    let adaln_gold = tensors.get("y_adaln_zero").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let adaln_calc = t02::adaln_zero_cpu(&test_in, &cond, &w_adaln, &b_adaln, 4, 8, 4, eps);
    for (i, (&calc, &gold)) in adaln_calc.iter().zip(adaln_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "adaLN error at {i}: {calc} vs {gold}");
    }

    // 6. QK-Norm on CPU
    let qk_gold = tensors.get("y_qknorm").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let qk_calc = t02::qknorm_cpu(&test_in, 4, 8, eps);
    for (i, (&calc, &gold)) in qk_calc.iter().zip(qk_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "QK-Norm error at {i}: {calc} vs {gold}");
    }

    // 7. Scaled Weight Standardization on CPU
    let ws_gold = tensors.get("w_standardized").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let ws_calc = t02::scaled_weight_standardization_cpu(&w_mat, 8, 8, eps);
    for (i, (&calc, &gold)) in ws_calc.iter().zip(ws_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Weight Standardization error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

use ai_kernel::c03_building_blocks_and_losses::t03_weight_initialization_and_mup_scaling as t03;

#[test]
fn test_03_weight_initialization_and_mup_scaling_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/03_weight_initialization_and_mup_scaling.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let test_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let res_branch = tensors.get("residual_branch").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let ls_gamma = tensors.get("layerscale_gamma").unwrap().to_vec1::<f32>()?;
    let rz_alpha = tensors.get("rezero_alpha").unwrap().to_vec1::<f32>()?[0];
    let res_scale = tensors.get("residual_scale").unwrap().to_vec1::<f32>()?[0];
    let x_hid = tensors.get("x_hidden").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let w_mup_ro = tensors.get("w_mup_readout").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let width_scale = tensors.get("width_scale").unwrap().to_vec1::<f32>()?[0];

    // 1. Linear Anchor on CPU
    let anchor_calc = t03::linear_anchor_forward_cpu(&test_in, &weight, &bias, 4, 8, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(test_out.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. LayerScale on CPU and Auto
    let ls_gold = tensors.get("y_layerscale").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let ls_calc = t03::layerscale_cpu(&test_in, &res_branch, &ls_gamma, 4, 8);
    for (i, (&calc, &gold)) in ls_calc.iter().zip(ls_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "LayerScale CPU error at {i}: {calc} vs {gold}");
    }
    let ls_auto = pollster::block_on(t03::layerscale_auto(None, &test_in, &res_branch, &ls_gamma, 4, 8));
    for (i, (&calc, &gold)) in ls_auto.iter().zip(ls_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "LayerScale Auto error at {i}: {calc} vs {gold}");
    }

    // 3. ReZero on CPU
    let rz_gold = tensors.get("y_rezero").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let rz_calc = t03::rezero_cpu(&test_in, &res_branch, rz_alpha, 4, 8);
    for (i, (&calc, &gold)) in rz_calc.iter().zip(rz_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "ReZero error at {i}: {calc} vs {gold}");
    }

    // 4. Residual 1/sqrt(2L) Scale on CPU
    let scaled_gold = tensors.get("y_residual_scaled").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let scaled_calc = t03::residual_scale_cpu(&test_in, &res_branch, res_scale, 4, 8);
    for (i, (&calc, &gold)) in scaled_calc.iter().zip(scaled_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Residual scale error at {i}: {calc} vs {gold}");
    }

    // 5. muP Readout on CPU
    let ro_gold = tensors.get("y_mup_readout").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let ro_calc = t03::mup_readout_cpu(&x_hid, &w_mup_ro, width_scale, 4, 8, 4);
    for (i, (&calc, &gold)) in ro_calc.iter().zip(ro_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "muP Readout error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

use ai_kernel::c03_building_blocks_and_losses::t04_positional_and_structural_encodings as t04;

#[test]
fn test_04_positional_and_structural_encodings_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/04_positional_and_structural_encodings.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let test_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let pe = tensors.get("sinusoidal_pe").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let cos = tensors.get("cos_rope").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let sin = tensors.get("sin_rope").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let slopes = tensors.get("alibi_slopes").unwrap().to_vec1::<f32>()?;
    let adj = tensors.get("graph_adj").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();

    // 1. Linear Anchor on CPU
    let anchor_calc = t04::linear_anchor_forward_cpu(&test_in, &weight, &bias, 4, 8, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(test_out.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. Sinusoidal PE on CPU
    let sin_gold = tensors.get("y_sinusoidal").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let sin_calc = t04::sinusoidal_pe_cpu(&test_in, &pe, 4, 8);
    for (i, (&calc, &gold)) in sin_calc.iter().zip(sin_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Sinusoidal PE error at {i}: {calc} vs {gold}");
    }

    // 3. RoPE on CPU and Auto
    let rp_gold = tensors.get("y_rope").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let rp_calc = t04::rope_cpu(&test_in, &cos, &sin, 4, 8);
    for (i, (&calc, &gold)) in rp_calc.iter().zip(rp_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "RoPE CPU error at {i}: {calc} vs {gold}");
    }
    let rp_auto = pollster::block_on(t04::rope_auto(None, &test_in, &cos, &sin, 4, 8));
    for (i, (&calc, &gold)) in rp_auto.iter().zip(rp_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "RoPE Auto error at {i}: {calc} vs {gold}");
    }

    // 4. ALiBi on CPU
    let alibi_gold = tensors.get("alibi_bias").unwrap().to_vec3::<f32>()?.into_iter().flatten().flatten().collect::<Vec<f32>>();
    let alibi_calc = t04::alibi_cpu(4, &slopes, 2);
    for (i, (&calc, &gold)) in alibi_calc.iter().zip(alibi_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "ALiBi error at {i}: {calc} vs {gold}");
    }

    // 5. RWPE on CPU
    let rw_gold = tensors.get("y_rwpe").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let rw_calc = t04::rwpe_cpu(&adj, 4, 4);
    for (i, (&calc, &gold)) in rw_calc.iter().zip(rw_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "RWPE error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

use ai_kernel::c03_building_blocks_and_losses::t05_pooling_and_downsampling_operators as t05;

#[test]
fn test_05_pooling_and_downsampling_operators_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/05_pooling_and_downsampling_operators.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let test_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;

    let x_gem = tensors.get("x_gem").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let p_gem = tensors.get("p_gem").unwrap().to_vec1::<f32>()?[0];
    let y_gem_gold = tensors.get("y_gem").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let x_blur = tensors.get("x_blur").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let blur_kernel = tensors.get("blur_kernel").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let y_blur_gold = tensors.get("y_blur").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let x_pixel = tensors.get("x_pixel").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let y_pixel_gold = tensors.get("y_pixel").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let x_pma = tensors.get("x_pma").unwrap().to_vec3::<f32>()?.into_iter().flatten().flatten().collect::<Vec<f32>>();
    let s_seed = tensors.get("s_seed").unwrap().to_vec3::<f32>()?.into_iter().flatten().flatten().collect::<Vec<f32>>();
    let y_pma_gold = tensors.get("y_pma").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();

    // 1. Linear Anchor on CPU
    let anchor_calc = t05::linear_anchor_forward_cpu(&test_in, &weight, &bias, 4, 8, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(test_out.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. GeM on CPU and Auto
    let gem_calc = t05::gem_pool_cpu(&x_gem, 2, 2, 4, 4, p_gem, 1e-6);
    for (i, (&calc, &gold)) in gem_calc.iter().zip(y_gem_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "GeM CPU error at {i}: {calc} vs {gold}");
    }
    let gem_auto = pollster::block_on(t05::gem_pool_auto(None, &x_gem, 2, 2, 4, 4, p_gem, 1e-6));
    for (i, (&calc, &gold)) in gem_auto.iter().zip(y_gem_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "GeM Auto error at {i}: {calc} vs {gold}");
    }

    // 3. BlurPool on CPU
    let blur_calc = t05::blur_pool_cpu(&x_blur, &blur_kernel, 2, 2, 6, 6, 2);
    for (i, (&calc, &gold)) in blur_calc.iter().zip(y_blur_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "BlurPool error at {i}: {calc} vs {gold}");
    }

    // 4. PixelUnshuffle on CPU
    let pixel_calc = t05::pixel_unshuffle_cpu(&x_pixel, 2, 2, 4, 4, 2);
    for (i, (&calc, &gold)) in pixel_calc.iter().zip(y_pixel_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "PixelUnshuffle error at {i}: {calc} vs {gold}");
    }

    // 5. PMA Attention Pooling on CPU
    let pma_calc = t05::pma_cpu(&x_pma, &s_seed, 2, 4, 8);
    for (i, (&calc, &gold)) in pma_calc.iter().zip(y_pma_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "PMA error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

use ai_kernel::c03_building_blocks_and_losses::t06_regularization_and_training_stabilizers as t06;

#[test]
fn test_06_regularization_and_training_stabilizers_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/06_regularization_and_training_stabilizers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let test_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;

    let x_drop = tensors.get("x_drop").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let drop_mask = tensors.get("drop_mask").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let drop_rate = tensors.get("drop_rate").unwrap().to_vec1::<f32>()?[0];
    let y_drop_gold = tensors.get("y_drop").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let x_id = tensors.get("x_identity").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let x_br = tensors.get("x_branch").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let dp_mask = tensors.get("droppath_mask").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let dp_rate = tensors.get("droppath_rate").unwrap().to_vec1::<f32>()?[0];
    let y_droppath_gold = tensors.get("y_droppath").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let x_log = tensors.get("x_logits").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let softcap = tensors.get("softcap_val").unwrap().to_vec1::<f32>()?[0];
    let y_softcap_gold = tensors.get("y_softcap").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let logits_ls = tensors.get("logits_ls").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let targets_ls = tensors.get("targets_ls").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let alpha_ls = tensors.get("smooth_alpha").unwrap().to_vec1::<f32>()?[0];
    let loss_ls_gold = tensors.get("loss_ls").unwrap().to_vec1::<f32>()?[0];

    let router_logits = tensors.get("router_logits").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let z_coef = tensors.get("z_coef").unwrap().to_vec1::<f32>()?[0];
    let loss_z_gold = tensors.get("loss_z").unwrap().to_vec1::<f32>()?[0];

    // 1. Linear Anchor on CPU
    let anchor_calc = t06::linear_anchor_forward_cpu(&test_in, &weight, &bias, 4, 8, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(test_out.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. Inverted Dropout on CPU
    let drop_calc = t06::dropout_cpu(&x_drop, &drop_mask, drop_rate);
    for (i, (&calc, &gold)) in drop_calc.iter().zip(y_drop_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Dropout error at {i}: {calc} vs {gold}");
    }

    // 3. DropPath on CPU
    let dp_calc = t06::droppath_cpu(&x_id, &x_br, &dp_mask, 4, 8, dp_rate);
    for (i, (&calc, &gold)) in dp_calc.iter().zip(y_droppath_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "DropPath error at {i}: {calc} vs {gold}");
    }

    // 4. SoftCap on CPU and Auto
    let sc_calc = t06::softcapping_cpu(&x_log, softcap);
    for (i, (&calc, &gold)) in sc_calc.iter().zip(y_softcap_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "SoftCap CPU error at {i}: {calc} vs {gold}");
    }
    let sc_auto = pollster::block_on(t06::softcapping_auto(None, &x_log, softcap));
    for (i, (&calc, &gold)) in sc_auto.iter().zip(y_softcap_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "SoftCap Auto error at {i}: {calc} vs {gold}");
    }

    // 5. Label Smoothing on CPU
    let ls_calc = t06::label_smoothing_loss_cpu(&logits_ls, &targets_ls, 4, 4, alpha_ls);
    assert!((ls_calc - loss_ls_gold).abs() < 1e-5, "Label Smoothing error: {ls_calc} vs {loss_ls_gold}");

    // 6. Router Z-Loss on CPU
    let z_calc = t06::router_z_loss_cpu(&router_logits, 4, 4, z_coef);
    assert!((z_calc - loss_z_gold).abs() < 1e-5, "Router Z-Loss error: {z_calc} vs {loss_z_gold}");

    Ok(())
}

use ai_kernel::c03_building_blocks_and_losses::t07_losses_regression_classification_and_barron_robust as t07;

#[test]
fn test_07_losses_regression_classification_and_barron_robust_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/07_losses_regression_classification_and_barron_robust.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let test_out = tensors.get("test_output").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let weight = tensors.get("weight").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;

    let r_barron = tensors.get("r_barron").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let alpha_b = tensors.get("alpha_barron").unwrap().to_vec1::<f32>()?[0];
    let c_b = tensors.get("c_barron").unwrap().to_vec1::<f32>()?[0];
    let loss_barron_gold = tensors.get("loss_barron").unwrap().to_vec1::<f32>()?[0];

    let r_huber = tensors.get("r_huber").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let delta_h = tensors.get("delta_huber").unwrap().to_vec1::<f32>()?[0];
    let loss_huber_gold = tensors.get("loss_huber").unwrap().to_vec1::<f32>()?[0];

    let logits_clf = tensors.get("logits_clf").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let targets_clf = tensors.get("targets_clf").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gamma_f = tensors.get("gamma_focal").unwrap().to_vec1::<f32>()?[0];
    let loss_focal_gold = tensors.get("loss_focal").unwrap().to_vec1::<f32>()?[0];

    let eps_p = tensors.get("eps_poly").unwrap().to_vec1::<f32>()?[0];
    let loss_poly_gold = tensors.get("loss_poly").unwrap().to_vec1::<f32>()?[0];

    let logits_asl = tensors.get("logits_asl").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let targets_asl = tensors.get("targets_asl").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gamma_pos = tensors.get("gamma_pos").unwrap().to_vec1::<f32>()?[0];
    let gamma_neg = tensors.get("gamma_neg").unwrap().to_vec1::<f32>()?[0];
    let margin_asl = tensors.get("margin_asl").unwrap().to_vec1::<f32>()?[0];
    let loss_asl_gold = tensors.get("loss_asl").unwrap().to_vec1::<f32>()?[0];

    // 1. Linear Anchor on CPU
    let anchor_calc = t07::linear_anchor_forward_cpu(&test_in, &weight, &bias, 4, 8, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(test_out.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. Barron Robust Loss on CPU and Auto
    let loss_b_calc = t07::barron_robust_loss_cpu(&r_barron, alpha_b, c_b);
    assert!((loss_b_calc - loss_barron_gold).abs() < 1e-5, "Barron CPU error: {loss_b_calc} vs {loss_barron_gold}");

    let loss_b_auto = pollster::block_on(t07::barron_robust_auto(None, &r_barron, alpha_b, c_b));
    assert!((loss_b_auto - loss_barron_gold).abs() < 1e-5, "Barron Auto error: {loss_b_auto} vs {loss_barron_gold}");

    // 3. Huber Loss on CPU
    let loss_h_calc = t07::huber_loss_cpu(&r_huber, delta_h);
    assert!((loss_h_calc - loss_huber_gold).abs() < 1e-5, "Huber CPU error: {loss_h_calc} vs {loss_huber_gold}");

    // 4. Focal Loss on CPU
    let loss_f_calc = t07::focal_loss_cpu(&logits_clf, &targets_clf, 4, 4, gamma_f);
    assert!((loss_f_calc - loss_focal_gold).abs() < 1e-5, "Focal CPU error: {loss_f_calc} vs {loss_focal_gold}");

    // 5. PolyLoss on CPU
    let loss_p_calc = t07::polyloss_cpu(&logits_clf, &targets_clf, 4, 4, eps_p);
    assert!((loss_p_calc - loss_poly_gold).abs() < 1e-5, "PolyLoss CPU error: {loss_p_calc} vs {loss_poly_gold}");

    // 6. Asymmetric Loss on CPU
    let loss_asl_calc = t07::asymmetric_loss_cpu(&logits_asl, &targets_asl, gamma_pos, gamma_neg, margin_asl, 1e-8);
    assert!((loss_asl_calc - loss_asl_gold).abs() < 1e-5, "ASL CPU error: {loss_asl_calc} vs {loss_asl_gold}");

    Ok(())
}

use ai_kernel::c03_building_blocks_and_losses::t08_losses_metric_contrastive_and_structured_vision as t08;

#[test]
fn test_08_losses_metric_contrastive_and_structured_vision_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/08_losses_metric_contrastive_and_structured_vision.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let test_out = tensors.get("test_output").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let weight = tensors.get("weight").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;

    let q_repr = tensors.get("q_repr").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let k_repr = tensors.get("k_repr").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let tau = tensors.get("tau_val").unwrap().to_vec1::<f32>()?[0];
    let loss_infonce_gold = tensors.get("loss_infonce").unwrap().to_vec1::<f32>()?[0];

    let siglip_scale = tensors.get("siglip_scale").unwrap().to_vec1::<f32>()?[0];
    let siglip_bias = tensors.get("siglip_bias").unwrap().to_vec1::<f32>()?[0];
    let loss_siglip_gold = tensors.get("loss_siglip").unwrap().to_vec1::<f32>()?[0];

    let arc_emb = tensors.get("arc_emb").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let arc_w = tensors.get("arc_weights").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let arc_targets: Vec<f32> = tensors
        .get("arc_targets")
        .unwrap()
        .flatten_all()?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|x| x as f32)
        .collect();
    let arc_s = tensors.get("arc_s").unwrap().to_vec1::<f32>()?[0];
    let arc_m = tensors.get("arc_m").unwrap().to_vec1::<f32>()?[0];
    let loss_arcface_gold = tensors.get("loss_arcface").unwrap().to_vec1::<f32>()?[0];

    let dice_probs = tensors.get("dice_probs").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let dice_targets = tensors.get("dice_targets").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let loss_dice_gold = tensors.get("loss_dice").unwrap().to_vec1::<f32>()?[0];

    let boxes1 = tensors.get("boxes1").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let boxes2 = tensors.get("boxes2").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let loss_giou_gold = tensors.get("loss_giou").unwrap().to_vec1::<f32>()?[0];

    // 1. Linear Anchor on CPU
    let anchor_calc = t08::linear_anchor_forward_cpu(&test_in, &weight, &bias, 4, 8, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(test_out.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Linear anchor error at {i}: {calc} vs {gold}");
    }

    // 2. InfoNCE Loss on CPU
    let loss_infonce_calc = t08::infonce_loss_cpu(&q_repr, &k_repr, 4, 8, tau);
    assert!((loss_infonce_calc - loss_infonce_gold).abs() < 1e-5, "InfoNCE error: {loss_infonce_calc} vs {loss_infonce_gold}");

    // 3. SigLIP Loss on CPU
    let loss_siglip_calc = t08::siglip_loss_cpu(&q_repr, &k_repr, 4, 8, siglip_scale, siglip_bias);
    assert!((loss_siglip_calc - loss_siglip_gold).abs() < 1e-5, "SigLIP error: {loss_siglip_calc} vs {loss_siglip_gold}");

    // 4. ArcFace Loss on CPU
    let loss_arc_calc = t08::arcface_loss_cpu(&arc_emb, &arc_w, &arc_targets, 4, 5, 8, arc_s, arc_m);
    assert!((loss_arc_calc - loss_arcface_gold).abs() < 1e-5, "ArcFace error: {loss_arc_calc} vs {loss_arcface_gold}");

    // 5. Soft-Dice on CPU and Auto
    let loss_dice_calc = t08::soft_dice_loss_cpu(&dice_probs, &dice_targets, 1e-6);
    assert!((loss_dice_calc - loss_dice_gold).abs() < 1e-5, "Soft-Dice CPU error: {loss_dice_calc} vs {loss_dice_gold}");

    let loss_dice_auto = pollster::block_on(t08::soft_dice_auto(None, &dice_probs, &dice_targets, 1e-6));
    assert!((loss_dice_auto - loss_dice_gold).abs() < 1e-5, "Soft-Dice Auto error: {loss_dice_auto} vs {loss_dice_gold}");

    // 6. GIoU Loss on CPU
    let loss_giou_calc = t08::giou_loss_cpu(&boxes1, &boxes2, 4);
    assert!((loss_giou_calc - loss_giou_gold).abs() < 1e-5, "GIoU error: {loss_giou_calc} vs {loss_giou_gold}");

    Ok(())
}
