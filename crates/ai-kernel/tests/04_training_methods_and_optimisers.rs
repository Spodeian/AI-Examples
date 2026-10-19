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

#[test]
fn test_03_large_batch_and_layerwise_optimisers_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t03_large_batch_and_layerwise_optimisers as t03;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let param = tensors.get("param").unwrap().to_vec1::<f32>()?;
    let grad = tensors.get("grad").unwrap().to_vec1::<f32>()?;
    let v_init_lars = tensors.get("v_init_lars").unwrap().to_vec1::<f32>()?;
    let m_init_lamb = tensors.get("m_init_lamb").unwrap().to_vec1::<f32>()?;
    let v_init_lamb = tensors.get("v_init_lamb").unwrap().to_vec1::<f32>()?;

    let gold_p_lars = tensors.get("p_lars").unwrap().to_vec1::<f32>()?;
    let gold_v_lars = tensors.get("v_lars").unwrap().to_vec1::<f32>()?;
    let gold_ratio_lars = tensors.get("ratio_lars").unwrap().to_vec1::<f32>()?[0];

    let gold_p_lamb = tensors.get("p_lamb").unwrap().to_vec1::<f32>()?;
    let gold_m_lamb = tensors.get("m_lamb").unwrap().to_vec1::<f32>()?;
    let gold_v_lamb = tensors.get("v_lamb").unwrap().to_vec1::<f32>()?;
    let gold_ratio_lamb = tensors.get("ratio_lamb").unwrap().to_vec1::<f32>()?[0];

    let n = param.len();

    // 1. LARS CPU & Auto
    let mut cpu_p_lars = vec![0.0f32; n];
    let mut cpu_v_lars = vec![0.0f32; n];
    let cpu_ratio_lars = t03::lars_cpu(
        &param,
        &grad,
        &v_init_lars,
        &mut cpu_p_lars,
        &mut cpu_v_lars,
        0.1,
        0.9,
        0.0001,
        0.001,
        1e-8,
    );
    assert!((cpu_ratio_lars - gold_ratio_lars).abs() < 1e-5);
    for i in 0..n {
        assert!((cpu_p_lars[i] - gold_p_lars[i]).abs() < 1e-5);
        assert!((cpu_v_lars[i] - gold_v_lars[i]).abs() < 1e-5);
    }

    let kdev = KernelDevice::Cpu;
    let (auto_p_lars, auto_v_lars, auto_ratio_lars) = t03::lars_auto(
        &kdev,
        &param,
        &grad,
        &v_init_lars,
        0.1,
        0.9,
        0.0001,
        0.001,
        1e-8,
    );
    assert!((auto_ratio_lars - gold_ratio_lars).abs() < 1e-5);
    for i in 0..n {
        assert!((auto_p_lars[i] - gold_p_lars[i]).abs() < 1e-5);
        assert!((auto_v_lars[i] - gold_v_lars[i]).abs() < 1e-5);
    }

    // 2. LAMB CPU & Auto
    let mut cpu_p_lamb = vec![0.0f32; n];
    let mut cpu_m_lamb = vec![0.0f32; n];
    let mut cpu_v_lamb = vec![0.0f32; n];
    let cpu_ratio_lamb = t03::lamb_cpu(
        &param,
        &grad,
        &m_init_lamb,
        &v_init_lamb,
        &mut cpu_p_lamb,
        &mut cpu_m_lamb,
        &mut cpu_v_lamb,
        2,
        0.001,
        0.9,
        0.999,
        1e-6,
        0.01,
    );
    assert!((cpu_ratio_lamb - gold_ratio_lamb).abs() < 2e-4);
    for i in 0..n {
        assert!((cpu_p_lamb[i] - gold_p_lamb[i]).abs() < 1e-5);
        assert!((cpu_m_lamb[i] - gold_m_lamb[i]).abs() < 1e-5);
        assert!((cpu_v_lamb[i] - gold_v_lamb[i]).abs() < 1e-5);
    }

    let (auto_p_lamb, auto_m_lamb, auto_v_lamb, auto_ratio_lamb) = t03::lamb_auto(
        &kdev,
        &param,
        &grad,
        &m_init_lamb,
        &v_init_lamb,
        2,
        0.001,
        0.9,
        0.999,
        1e-6,
        0.01,
    );
    assert!((auto_ratio_lamb - gold_ratio_lamb).abs() < 2e-4);
    for i in 0..n {
        assert!((auto_p_lamb[i] - gold_p_lamb[i]).abs() < 1e-5);
        assert!((auto_m_lamb[i] - gold_m_lamb[i]).abs() < 1e-5);
        assert!((auto_v_lamb[i] - gold_v_lamb[i]).abs() < 1e-5);
    }

    // 3. Gradient Noise Scale CPU
    let per_sample_grads = tensors.get("per_sample_grads").unwrap().to_vec2::<f32>()?;
    let flat_grads: Vec<f32> = per_sample_grads.iter().flatten().copied().collect();
    let (g_sq, tr_sig, b_sim) = t03::gradient_noise_scale_cpu(&flat_grads, 8, 4);

    let gold_g_sq = tensors.get("g_norm_sq").unwrap().to_vec1::<f32>()?[0];
    let gold_tr_sig = tensors.get("trace_sigma").unwrap().to_vec1::<f32>()?[0];
    let gold_b_sim = tensors.get("b_simple").unwrap().to_vec1::<f32>()?[0];

    assert!((g_sq - gold_g_sq).abs() < 1e-5);
    assert!((tr_sig - gold_tr_sig).abs() < 1e-5);
    assert!((b_sim - gold_b_sim).abs() < 1e-5);

    Ok(())
}

#[test]
fn test_04_orthogonal_and_preconditioned_matrix_optimisers_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t04_orthogonal_and_preconditioned_matrix_optimisers as t04;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let param_2d = tensors.get("param_2d").unwrap().to_vec2::<f32>()?;
    let grad_2d = tensors.get("grad_2d").unwrap().to_vec2::<f32>()?;
    let flat_p: Vec<f32> = param_2d.iter().flatten().copied().collect();
    let flat_g: Vec<f32> = grad_2d.iter().flatten().copied().collect();

    let m = 4;
    let n = 4;
    let len = m * n;

    // 1. Muon CPU & Auto
    let m_init_muon = tensors.get("m_init_muon").unwrap().to_vec2::<f32>()?;
    let flat_m_muon: Vec<f32> = m_init_muon.iter().flatten().copied().collect();

    let gold_p_muon = tensors.get("p_muon").unwrap().to_vec2::<f32>()?;
    let gold_m_muon = tensors.get("m_muon").unwrap().to_vec2::<f32>()?;
    let flat_gold_p_muon: Vec<f32> = gold_p_muon.iter().flatten().copied().collect();
    let flat_gold_m_muon: Vec<f32> = gold_m_muon.iter().flatten().copied().collect();

    let mut cpu_p_muon = vec![0.0f32; len];
    let mut cpu_m_muon = vec![0.0f32; len];
    t04::muon_cpu(
        &flat_p,
        &flat_g,
        &flat_m_muon,
        &mut cpu_p_muon,
        &mut cpu_m_muon,
        m,
        n,
        0.02,
        0.95,
        true,
        0.01,
        5,
    );
    for i in 0..len {
        assert!((cpu_p_muon[i] - flat_gold_p_muon[i]).abs() < 1e-5);
        assert!((cpu_m_muon[i] - flat_gold_m_muon[i]).abs() < 1e-5);
    }

    let kdev = KernelDevice::Cpu;
    let (auto_p_muon, auto_m_muon) = t04::muon_auto(
        &kdev,
        &flat_p,
        &flat_g,
        &flat_m_muon,
        m,
        n,
        0.02,
        0.95,
        true,
        0.01,
        5,
    );
    for i in 0..len {
        assert!((auto_p_muon[i] - flat_gold_p_muon[i]).abs() < 1e-5);
        assert!((auto_m_muon[i] - flat_gold_m_muon[i]).abs() < 1e-5);
    }

    // 2. Shampoo CPU
    let l_init = tensors.get("l_init").unwrap().to_vec2::<f32>()?;
    let r_init = tensors.get("r_init").unwrap().to_vec2::<f32>()?;
    let flat_l_init: Vec<f32> = l_init.iter().flatten().copied().collect();
    let flat_r_init: Vec<f32> = r_init.iter().flatten().copied().collect();

    let gold_p_shampoo = tensors.get("p_shampoo").unwrap().to_vec2::<f32>()?;
    let gold_l_shampoo = tensors.get("l_shampoo").unwrap().to_vec2::<f32>()?;
    let gold_r_shampoo = tensors.get("r_shampoo").unwrap().to_vec2::<f32>()?;
    let flat_gold_p_shampoo: Vec<f32> = gold_p_shampoo.iter().flatten().copied().collect();
    let flat_gold_l_shampoo: Vec<f32> = gold_l_shampoo.iter().flatten().copied().collect();
    let flat_gold_r_shampoo: Vec<f32> = gold_r_shampoo.iter().flatten().copied().collect();

    let mut cpu_p_shampoo = vec![0.0f32; len];
    let mut cpu_l_shampoo = vec![0.0f32; m * m];
    let mut cpu_r_shampoo = vec![0.0f32; n * n];
    t04::shampoo_cpu(
        &flat_p,
        &flat_g,
        &flat_l_init,
        &flat_r_init,
        &mut cpu_p_shampoo,
        &mut cpu_l_shampoo,
        &mut cpu_r_shampoo,
        m,
        n,
        0.005,
        0.95,
        1e-6,
        0.01,
    );
    for i in 0..len {
        assert!((cpu_p_shampoo[i] - flat_gold_p_shampoo[i]).abs() < 1e-5);
    }
    for i in 0..(m * m) {
        assert!((cpu_l_shampoo[i] - flat_gold_l_shampoo[i]).abs() < 1e-5);
    }
    for i in 0..(n * n) {
        assert!((cpu_r_shampoo[i] - flat_gold_r_shampoo[i]).abs() < 1e-5);
    }

    // 3. GaLore CPU
    let r = 2;
    let m_init_galore = tensors.get("m_init_galore").unwrap().to_vec2::<f32>()?;
    let v_init_galore = tensors.get("v_init_galore").unwrap().to_vec2::<f32>()?;
    let proj = tensors.get("proj_matrix").unwrap().to_vec2::<f32>()?;
    let flat_m_galore: Vec<f32> = m_init_galore.iter().flatten().copied().collect();
    let flat_v_galore: Vec<f32> = v_init_galore.iter().flatten().copied().collect();
    let flat_proj: Vec<f32> = proj.iter().flatten().copied().collect();

    let gold_p_galore = tensors.get("p_galore").unwrap().to_vec2::<f32>()?;
    let gold_m_galore = tensors.get("m_galore").unwrap().to_vec2::<f32>()?;
    let gold_v_galore = tensors.get("v_galore").unwrap().to_vec2::<f32>()?;
    let flat_gold_p_galore: Vec<f32> = gold_p_galore.iter().flatten().copied().collect();
    let flat_gold_m_galore: Vec<f32> = gold_m_galore.iter().flatten().copied().collect();
    let flat_gold_v_galore: Vec<f32> = gold_v_galore.iter().flatten().copied().collect();

    let mut cpu_p_galore = vec![0.0f32; len];
    let mut cpu_m_galore = vec![0.0f32; r * n];
    let mut cpu_v_galore = vec![0.0f32; r * n];
    t04::galore_cpu(
        &flat_p,
        &flat_g,
        &flat_m_galore,
        &flat_v_galore,
        &flat_proj,
        &mut cpu_p_galore,
        &mut cpu_m_galore,
        &mut cpu_v_galore,
        m,
        n,
        r,
        2,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
    );
    for i in 0..len {
        assert!((cpu_p_galore[i] - flat_gold_p_galore[i]).abs() < 1e-5);
    }
    for i in 0..(r * n) {
        assert!((cpu_m_galore[i] - flat_gold_m_galore[i]).abs() < 1e-5);
        assert!((cpu_v_galore[i] - flat_gold_v_galore[i]).abs() < 1e-5);
    }

    Ok(())
}



