//! Category 04 Training Methods & Optimisers Integration Test Suite for ai-kernel.

use ai_kernel::c04_training_methods_and_optimisers::t01_first_order_and_adaptive_optimisers as t01;
use ai_kernel::KernelDevice;
use candle_core::{safetensors::load, Device, Result};

#[path = "common/mod.rs"]
mod common;

#[test]
fn test_01_first_order_and_adaptive_optimisers_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let param = tensors.get("param").unwrap().to_vec1::<f32>()?;
    let grad = tensors.get("grad").unwrap().to_vec1::<f32>()?;
    let m_init = tensors.get("m_init").unwrap().to_vec1::<f32>()?;
    let v_init = tensors.get("v_init").unwrap().to_vec1::<f32>()?;

    let gold_p_adamw = tensors.get("p_adamw").unwrap().to_vec1::<f32>()?;
    let gold_m_adamw = tensors.get("m_adamw").unwrap().to_vec1::<f32>()?;
    let gold_v_adamw = tensors.get("v_adamw").unwrap().to_vec1::<f32>()?;

    // 1. CPU Fused AdamW Step
    let n = param.len();
    let mut cpu_p = vec![0.0f32; n];
    let mut cpu_m = vec![0.0f32; n];
    let mut cpu_v = vec![0.0f32; n];

    t01::adamw_cpu(
        &param,
        &grad,
        &m_init,
        &v_init,
        &mut cpu_p,
        &mut cpu_m,
        &mut cpu_v,
        2,
        0.001,
        0.01,
        0.9,
        0.999,
        1e-8,
    );

    for i in 0..n {
        assert!((cpu_p[i] - gold_p_adamw[i]).abs() < 1e-5, "CPU p_adamw error at {i}: {} vs {}", cpu_p[i], gold_p_adamw[i]);
        assert!((cpu_m[i] - gold_m_adamw[i]).abs() < 1e-5, "CPU m_adamw error at {i}: {} vs {}", cpu_m[i], gold_m_adamw[i]);
        assert!((cpu_v[i] - gold_v_adamw[i]).abs() < 1e-5, "CPU v_adamw error at {i}: {} vs {}", cpu_v[i], gold_v_adamw[i]);
    }

    // 2. Auto-dispatcher (WGPU with CPU fallback)
    let kdev = KernelDevice::Cpu;
    let (auto_p, auto_m, auto_v) = t01::adamw_auto(
        &kdev,
        &param,
        &grad,
        &m_init,
        &v_init,
        2,
        0.001,
        0.01,
        0.9,
        0.999,
        1e-8,
    );

    for i in 0..n {
        assert!((auto_p[i] - gold_p_adamw[i]).abs() < 1e-5, "Auto p_adamw error at {i}: {} vs {}", auto_p[i], gold_p_adamw[i]);
        assert!((auto_m[i] - gold_m_adamw[i]).abs() < 1e-5, "Auto m_adamw error at {i}: {} vs {}", auto_m[i], gold_m_adamw[i]);
        assert!((auto_v[i] - gold_v_adamw[i]).abs() < 1e-5, "Auto v_adamw error at {i}: {} vs {}", auto_v[i], gold_v_adamw[i]);
    }

    Ok(())
}

#[test]
fn test_02_modern_sign_and_factored_optimisers_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t02_modern_sign_and_factored_optimisers as t02;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let param = tensors.get("param_1d").unwrap().to_vec1::<f32>()?;
    let grad = tensors.get("grad_1d").unwrap().to_vec1::<f32>()?;
    let m_init = tensors.get("m_1d").unwrap().to_vec1::<f32>()?;

    let gold_p_lion = tensors.get("p_lion").unwrap().to_vec1::<f32>()?;
    let gold_m_lion = tensors.get("m_lion").unwrap().to_vec1::<f32>()?;

    // 1. CPU Fused Lion Step
    let n = param.len();
    let mut cpu_p = vec![0.0f32; n];
    let mut cpu_m = vec![0.0f32; n];

    t02::lion_cpu(
        &param,
        &grad,
        &m_init,
        &mut cpu_p,
        &mut cpu_m,
        0.0001,
        0.1,
        0.9,
        0.99,
    );

    for i in 0..n {
        assert!((cpu_p[i] - gold_p_lion[i]).abs() < 1e-5, "CPU p_lion error at {i}: {} vs {}", cpu_p[i], gold_p_lion[i]);
        assert!((cpu_m[i] - gold_m_lion[i]).abs() < 1e-5, "CPU m_lion error at {i}: {} vs {}", cpu_m[i], gold_m_lion[i]);
    }

    // 2. Auto-dispatcher (WGPU with CPU fallback)
    let kdev = KernelDevice::Cpu;
    let (auto_p, auto_m) = t02::lion_auto(
        &kdev,
        &param,
        &grad,
        &m_init,
        0.0001,
        0.1,
        0.9,
        0.99,
    );

    for i in 0..n {
        assert!((auto_p[i] - gold_p_lion[i]).abs() < 1e-5, "Auto p_lion error at {i}: {} vs {}", auto_p[i], gold_p_lion[i]);
        assert!((auto_m[i] - gold_m_lion[i]).abs() < 1e-5, "Auto m_lion error at {i}: {} vs {}", auto_m[i], gold_m_lion[i]);
    }

    Ok(())
}

