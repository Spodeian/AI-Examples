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

#[test]
fn test_05_second_order_and_curvature_optimisers_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t05_second_order_and_curvature_optimisers as t05;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let theta = tensors.get("theta").unwrap().to_vec1::<f32>()?;
    let grad = tensors.get("grad").unwrap().to_vec1::<f32>()?;
    let momentum = tensors.get("momentum").unwrap().to_vec1::<f32>()?;
    let hess_diag = tensors.get("hess_diag").unwrap().to_vec1::<f32>()?;
    let hess_diag_est = tensors.get("hess_diag_est").unwrap().to_vec1::<f32>()?;
    let ada_v = tensors.get("ada_v").unwrap().to_vec1::<f32>()?;
    let ada_d = tensors.get("ada_d").unwrap().to_vec1::<f32>()?;
    let lbfgs_s = tensors.get("lbfgs_s").unwrap().to_vec2::<f32>()?;
    let lbfgs_y = tensors.get("lbfgs_y").unwrap().to_vec2::<f32>()?;
    let flat_s: Vec<f32> = lbfgs_s.iter().flatten().copied().collect();
    let flat_y: Vec<f32> = lbfgs_y.iter().flatten().copied().collect();

    let lm_jacobian = tensors.get("lm_jacobian").unwrap().to_vec2::<f32>()?;
    let flat_j: Vec<f32> = lm_jacobian.iter().flatten().copied().collect();
    let lm_residual = tensors.get("lm_residual").unwrap().to_vec1::<f32>()?;

    let gold_lbfgs = tensors.get("golden_lbfgs_theta").unwrap().to_vec1::<f32>()?;
    let gold_sophia_th = tensors.get("golden_sophia_theta").unwrap().to_vec1::<f32>()?;
    let gold_sophia_m = tensors.get("golden_sophia_m").unwrap().to_vec1::<f32>()?;
    let gold_sophia_h = tensors.get("golden_sophia_h").unwrap().to_vec1::<f32>()?;

    let gold_ada_th = tensors.get("golden_adahessian_theta").unwrap().to_vec1::<f32>()?;
    let gold_ada_m = tensors.get("golden_adahessian_m").unwrap().to_vec1::<f32>()?;
    let gold_ada_v = tensors.get("golden_adahessian_v").unwrap().to_vec1::<f32>()?;

    let gold_lm_th = tensors.get("golden_lm_theta").unwrap().to_vec1::<f32>()?;

    let d = theta.len();
    let m_hist = lbfgs_s.len();
    let n_res = lm_jacobian.len();

    // 1. L-BFGS CPU & Auto
    let cpu_lbfgs = t05::step_lbfgs_cpu(&theta, &flat_s, &flat_y, &grad, 0.1, m_hist, d);
    let auto_lbfgs = t05::step_lbfgs_auto(&kdev, &theta, &flat_s, &flat_y, &grad, 0.1, m_hist, d);
    for j in 0..d {
        assert!((cpu_lbfgs[j] - gold_lbfgs[j]).abs() < 1e-5);
        assert!((auto_lbfgs[j] - gold_lbfgs[j]).abs() < 1e-5);
    }

    // 2. Sophia-G CPU & Auto
    let (cpu_s_th, cpu_s_m, cpu_s_h) = t05::step_sophia_cpu(
        &theta, &grad, &momentum, &hess_diag, &hess_diag_est, 0.01, 0.96, 0.99, 0.01, 0.04, 1e-6,
    );
    let (auto_s_th, auto_s_m, auto_s_h) = t05::step_sophia_auto(
        &kdev, &theta, &grad, &momentum, &hess_diag, &hess_diag_est, 0.01, 0.96, 0.99, 0.01, 0.04, 1e-6,
    );
    for j in 0..d {
        assert!((cpu_s_th[j] - gold_sophia_th[j]).abs() < 1e-5);
        assert!((auto_s_th[j] - gold_sophia_th[j]).abs() < 1e-5);
        assert!((cpu_s_m[j] - gold_sophia_m[j]).abs() < 1e-5);
        assert!((auto_s_m[j] - gold_sophia_m[j]).abs() < 1e-5);
        assert!((cpu_s_h[j] - gold_sophia_h[j]).abs() < 1e-5);
        assert!((auto_s_h[j] - gold_sophia_h[j]).abs() < 1e-5);
    }

    // 3. AdaHessian CPU & Auto
    let (cpu_a_th, cpu_a_m, cpu_a_v) = t05::step_adahessian_cpu(
        &theta, &grad, &momentum, &ada_v, &ada_d, 0.01, 0.9, 0.999, 0.01, 1, 1e-4,
    );
    let (auto_a_th, auto_a_m, auto_a_v) = t05::step_adahessian_auto(
        &kdev, &theta, &grad, &momentum, &ada_v, &ada_d, 0.01, 0.9, 0.999, 0.01, 1, 1e-4,
    );
    for j in 0..d {
        assert!((cpu_a_th[j] - gold_ada_th[j]).abs() < 1e-5);
        assert!((auto_a_th[j] - gold_ada_th[j]).abs() < 1e-5);
        assert!((cpu_a_m[j] - gold_ada_m[j]).abs() < 1e-5);
        assert!((auto_a_m[j] - gold_ada_m[j]).abs() < 1e-5);
        assert!((cpu_a_v[j] - gold_ada_v[j]).abs() < 1e-5);
        assert!((auto_a_v[j] - gold_ada_v[j]).abs() < 1e-5);
    }

    // 4. Levenberg-Marquardt CPU & Auto
    let cpu_lm = t05::step_lm_cpu(&theta, &flat_j, &lm_residual, 0.1, n_res, d);
    let auto_lm = t05::step_lm_auto(&kdev, &theta, &flat_j, &lm_residual, 0.1, n_res, d);
    for j in 0..d {
        assert!((cpu_lm[j] - gold_lm_th[j]).abs() < 1e-5);
        assert!((auto_lm[j] - gold_lm_th[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_06_schedule_free_and_parameter_free_optimisers_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t06_schedule_free_and_parameter_free_optimisers as t06;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let sf_x = tensors.get("sf_x").unwrap().to_vec1::<f32>()?;
    let sf_z = tensors.get("sf_z").unwrap().to_vec1::<f32>()?;
    let sf_y = tensors.get("sf_y").unwrap().to_vec1::<f32>()?;
    let sf_grad = tensors.get("sf_grad").unwrap().to_vec1::<f32>()?;
    let sf_m = tensors.get("sf_m").unwrap().to_vec1::<f32>()?;
    let sf_v = tensors.get("sf_v").unwrap().to_vec1::<f32>()?;

    let polyak_param = tensors.get("polyak_param").unwrap().to_vec1::<f32>()?;
    let polyak_grad = tensors.get("polyak_grad").unwrap().to_vec1::<f32>()?;
    let f_val = tensors.get("f_val").unwrap().to_vec1::<f32>()?[0];
    let f_opt = tensors.get("f_opt").unwrap().to_vec1::<f32>()?[0];

    let bb_param = tensors.get("bb_param").unwrap().to_vec1::<f32>()?;
    let bb_grad = tensors.get("bb_grad").unwrap().to_vec1::<f32>()?;
    let bb_s = tensors.get("bb_s").unwrap().to_vec1::<f32>()?;
    let bb_y = tensors.get("bb_y").unwrap().to_vec1::<f32>()?;

    let d_param = tensors.get("d_param").unwrap().to_vec1::<f32>()?;
    let d_grad = tensors.get("d_grad").unwrap().to_vec1::<f32>()?;
    let d_est = tensors.get("d_est").unwrap().to_vec1::<f32>()?[0];

    let gold_sf_x = tensors.get("golden_sf_x").unwrap().to_vec1::<f32>()?;
    let gold_sf_z = tensors.get("golden_sf_z").unwrap().to_vec1::<f32>()?;
    let gold_sf_y = tensors.get("golden_sf_y").unwrap().to_vec1::<f32>()?;
    let gold_polyak = tensors.get("golden_polyak_param").unwrap().to_vec1::<f32>()?;
    let gold_bb = tensors.get("golden_bb_param").unwrap().to_vec1::<f32>()?;
    let gold_dadapt = tensors.get("golden_d_adapt_param").unwrap().to_vec1::<f32>()?;

    let d = sf_x.len();

    // 1. Schedule-Free AdamW CPU & Auto
    let (cpu_x, cpu_z, cpu_y, _, _) = t06::step_schedule_free_cpu(
        &sf_x, &sf_z, &sf_y, &sf_grad, &sf_m, &sf_v, 0.01, 0.9, 0.999, 0.01, 1, 1e-8,
    );
    let (auto_x, auto_z, auto_y, _, _) = t06::step_schedule_free_auto(
        &kdev, &sf_x, &sf_z, &sf_y, &sf_grad, &sf_m, &sf_v, 0.01, 0.9, 0.999, 0.01, 1, 1e-8,
    );
    for j in 0..d {
        assert!((cpu_x[j] - gold_sf_x[j]).abs() < 1e-5);
        assert!((auto_x[j] - gold_sf_x[j]).abs() < 1e-5);
        assert!((cpu_z[j] - gold_sf_z[j]).abs() < 1e-5);
        assert!((auto_z[j] - gold_sf_z[j]).abs() < 1e-5);
        assert!((cpu_y[j] - gold_sf_y[j]).abs() < 1e-5);
        assert!((auto_y[j] - gold_sf_y[j]).abs() < 1e-5);
    }

    // 2. Polyak CPU & Auto
    let cpu_polyak = t06::step_polyak_cpu(&polyak_param, &polyak_grad, f_val, f_opt, 1e-8);
    let auto_polyak = t06::step_polyak_auto(&kdev, &polyak_param, &polyak_grad, f_val, f_opt, 1e-8);
    for j in 0..d {
        assert!((cpu_polyak[j] - gold_polyak[j]).abs() < 1e-5);
        assert!((auto_polyak[j] - gold_polyak[j]).abs() < 1e-5);
    }

    // 3. Barzilai-Borwein CPU & Auto
    let cpu_bb = t06::step_barzilai_borwein_cpu(&bb_param, &bb_grad, &bb_s, &bb_y, 1e-8);
    let auto_bb = t06::step_barzilai_borwein_auto(&kdev, &bb_param, &bb_grad, &bb_s, &bb_y, 1e-8);
    for j in 0..d {
        assert!((cpu_bb[j] - gold_bb[j]).abs() < 1e-5);
        assert!((auto_bb[j] - gold_bb[j]).abs() < 1e-5);
    }

    // 4. D-Adaptation CPU & Auto
    let (cpu_dadapt, _) = t06::step_d_adaptation_cpu(&d_param, &d_grad, d_est, 0.01, 1e-8);
    let (auto_dadapt, _) = t06::step_d_adaptation_auto(&kdev, &d_param, &d_grad, d_est, 0.01, 1e-8);
    for j in 0..d {
        assert!((cpu_dadapt[j] - gold_dadapt[j]).abs() < 1e-5);
        assert!((auto_dadapt[j] - gold_dadapt[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_07_learning_rate_schedulers_and_lr_finder_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t07_learning_rate_schedulers_and_lr_finder as t07;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/07_learning_rate_schedulers_and_lr_finder.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let param_cosine = tensors.get("param_cosine").unwrap().to_vec1::<f32>()?;
    let grad_cosine = tensors.get("grad_cosine").unwrap().to_vec1::<f32>()?;
    let gold_lr_cos = tensors.get("golden_lr_cosine").unwrap().to_vec1::<f32>()?[0];
    let gold_p_cos = tensors.get("golden_param_cosine").unwrap().to_vec1::<f32>()?;

    let param_wsd = tensors.get("param_wsd").unwrap().to_vec1::<f32>()?;
    let grad_wsd = tensors.get("grad_wsd").unwrap().to_vec1::<f32>()?;
    let gold_lr_wsd = tensors.get("golden_lr_wsd").unwrap().to_vec1::<f32>()?[0];
    let gold_p_wsd = tensors.get("golden_param_wsd").unwrap().to_vec1::<f32>()?;

    let param_onecycle = tensors.get("param_onecycle").unwrap().to_vec1::<f32>()?;
    let grad_onecycle = tensors.get("grad_onecycle").unwrap().to_vec1::<f32>()?;
    let gold_lr_oc = tensors.get("golden_lr_onecycle").unwrap().to_vec1::<f32>()?[0];
    let gold_p_oc = tensors.get("golden_param_onecycle").unwrap().to_vec1::<f32>()?;

    let param_autoclip = tensors.get("param_autoclip").unwrap().to_vec1::<f32>()?;
    let grad_autoclip = tensors.get("grad_autoclip").unwrap().to_vec1::<f32>()?;
    let norm_history = tensors.get("norm_history").unwrap().to_vec1::<f32>()?;
    let gold_clipped_g = tensors.get("golden_clipped_grad").unwrap().to_vec1::<f32>()?;
    let gold_p_ac = tensors.get("golden_param_autoclip").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let d = param_cosine.len();

    // 1. Cosine Annealing with Warmup CPU & Auto
    let cpu_lr_cos = t07::schedule_cosine_warmup_cpu(30, 100, 10, 1e-5, 0.01);
    let auto_lr_cos = t07::schedule_cosine_warmup_auto(&kdev, 30, 100, 10, 1e-5, 0.01);
    assert!((cpu_lr_cos - gold_lr_cos).abs() < 1e-5);
    assert!((auto_lr_cos - gold_lr_cos).abs() < 1e-5);
    let cpu_p_cos = t07::step_sgd_cpu(&param_cosine, &grad_cosine, cpu_lr_cos);
    let auto_p_cos = t07::step_sgd_auto(&kdev, &param_cosine, &grad_cosine, auto_lr_cos);
    for j in 0..d {
        assert!((cpu_p_cos[j] - gold_p_cos[j]).abs() < 1e-5);
        assert!((auto_p_cos[j] - gold_p_cos[j]).abs() < 1e-5);
    }

    // 2. WSD CPU & Auto
    let cpu_lr_wsd = t07::schedule_wsd_cpu(85, 100, 10, 20, 1e-4, 0.01);
    let auto_lr_wsd = t07::schedule_wsd_auto(&kdev, 85, 100, 10, 20, 1e-4, 0.01);
    assert!((cpu_lr_wsd - gold_lr_wsd).abs() < 1e-5);
    assert!((auto_lr_wsd - gold_lr_wsd).abs() < 1e-5);
    let cpu_p_wsd = t07::step_sgd_cpu(&param_wsd, &grad_wsd, cpu_lr_wsd);
    let auto_p_wsd = t07::step_sgd_auto(&kdev, &param_wsd, &grad_wsd, auto_lr_wsd);
    for j in 0..d {
        assert!((cpu_p_wsd[j] - gold_p_wsd[j]).abs() < 1e-5);
        assert!((auto_p_wsd[j] - gold_p_wsd[j]).abs() < 1e-5);
    }

    // 3. OneCycleLR CPU & Auto
    let cpu_lr_oc = t07::schedule_onecycle_cpu(40, 100, 30, 0.01, 25.0, 1000.0);
    let auto_lr_oc = t07::schedule_onecycle_auto(&kdev, 40, 100, 30, 0.01, 25.0, 1000.0);
    assert!((cpu_lr_oc - gold_lr_oc).abs() < 1e-5);
    assert!((auto_lr_oc - gold_lr_oc).abs() < 1e-5);
    let cpu_p_oc = t07::step_sgd_cpu(&param_onecycle, &grad_onecycle, cpu_lr_oc);
    let auto_p_oc = t07::step_sgd_auto(&kdev, &param_onecycle, &grad_onecycle, auto_lr_oc);
    for j in 0..d {
        assert!((cpu_p_oc[j] - gold_p_oc[j]).abs() < 1e-5);
        assert!((auto_p_oc[j] - gold_p_oc[j]).abs() < 1e-5);
    }

    // 4. AutoClip CPU & Auto
    let (cpu_p_ac, cpu_cg, _) = t07::step_autoclip_cpu(&param_autoclip, &grad_autoclip, &norm_history, 90.0, 0.01, 1e-8);
    let (auto_p_ac, auto_cg, _) = t07::step_autoclip_auto(&kdev, &param_autoclip, &grad_autoclip, &norm_history, 90.0, 0.01, 1e-8);
    for j in 0..d {
        assert!((cpu_cg[j] - gold_clipped_g[j]).abs() < 1e-5);
        assert!((auto_cg[j] - gold_clipped_g[j]).abs() < 1e-5);
        assert!((cpu_p_ac[j] - gold_p_ac[j]).abs() < 1e-5);
        assert!((auto_p_ac[j] - gold_p_ac[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t07::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}



