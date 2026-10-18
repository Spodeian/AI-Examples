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
