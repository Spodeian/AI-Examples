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

#[test]
fn test_08_constrained_proximal_frank_wolfe_admm_and_mirror_descent_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t08_constrained_proximal_frank_wolfe_admm_and_mirror_descent as t08;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let param_pgd = tensors.get("param_pgd").unwrap().to_vec1::<f32>()?;
    let grad_pgd = tensors.get("grad_pgd").unwrap().to_vec1::<f32>()?;
    let gold_p_pgd = tensors.get("golden_param_pgd").unwrap().to_vec1::<f32>()?;

    let param_fw = tensors.get("param_fw").unwrap().to_vec1::<f32>()?;
    let grad_fw = tensors.get("grad_fw").unwrap().to_vec1::<f32>()?;
    let gold_p_fw = tensors.get("golden_param_fw").unwrap().to_vec1::<f32>()?;

    let param_ista = tensors.get("param_ista").unwrap().to_vec1::<f32>()?;
    let grad_ista = tensors.get("grad_ista").unwrap().to_vec1::<f32>()?;
    let gold_p_ista = tensors.get("golden_param_ista").unwrap().to_vec1::<f32>()?;

    let param_md = tensors.get("param_md").unwrap().to_vec1::<f32>()?;
    let grad_md = tensors.get("grad_md").unwrap().to_vec1::<f32>()?;
    let gold_p_md = tensors.get("golden_param_md").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let d = param_pgd.len();

    // 1. PGD CPU & Auto
    let cpu_pgd = t08::step_pgd_cpu(&param_pgd, &grad_pgd, 0.1, 1.0, 1e-8);
    let auto_pgd = t08::step_pgd_auto(&kdev, &param_pgd, &grad_pgd, 0.1, 1.0, 1e-8);
    for j in 0..d {
        assert!((cpu_pgd[j] - gold_p_pgd[j]).abs() < 1e-5);
        assert!((auto_pgd[j] - gold_p_pgd[j]).abs() < 1e-5);
    }

    // 2. Frank-Wolfe CPU & Auto
    let cpu_fw = t08::step_frank_wolfe_cpu(&param_fw, &grad_fw, 1.0, 0.2);
    let auto_fw = t08::step_frank_wolfe_auto(&kdev, &param_fw, &grad_fw, 1.0, 0.2);
    for j in 0..d {
        assert!((cpu_fw[j] - gold_p_fw[j]).abs() < 1e-5);
        assert!((auto_fw[j] - gold_p_fw[j]).abs() < 1e-5);
    }

    // 3. ISTA CPU & Auto
    let cpu_ista = t08::step_ista_cpu(&param_ista, &grad_ista, 0.05, 0.1);
    let auto_ista = t08::step_ista_auto(&kdev, &param_ista, &grad_ista, 0.05, 0.1);
    for j in 0..d {
        assert!((cpu_ista[j] - gold_p_ista[j]).abs() < 1e-5);
        assert!((auto_ista[j] - gold_p_ista[j]).abs() < 1e-5);
    }

    // 4. Mirror Descent CPU & Auto
    let cpu_md = t08::step_mirror_descent_cpu(&param_md, &grad_md, 0.1, 1e-12);
    let auto_md = t08::step_mirror_descent_auto(&kdev, &param_md, &grad_md, 0.1, 1e-12);
    for j in 0..d {
        assert!((cpu_md[j] - gold_p_md[j]).abs() < 1e-5);
        assert!((auto_md[j] - gold_p_md[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t08::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_09_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t09_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery as t09;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let param = tensors.get("param").unwrap().to_vec1::<f32>()?;
    let g1 = tensors.get("g1").unwrap().to_vec1::<f32>()?;
    let g2 = tensors.get("g2").unwrap().to_vec1::<f32>()?;
    let losses = tensors.get("L").unwrap().to_vec1::<f32>()?;
    let z_star = tensors.get("z_star").unwrap().to_vec1::<f32>()?;
    let weights_tch = tensors.get("weights_tch").unwrap().to_vec1::<f32>()?;
    let weights_lin = tensors.get("weights_lin").unwrap().to_vec1::<f32>()?;

    let gold_alpha = tensors.get("golden_alpha_mgda").unwrap().to_vec1::<f32>()?[0];
    let gold_g_mgda = tensors.get("golden_g_mgda").unwrap().to_vec1::<f32>()?;
    let gold_p_mgda = tensors.get("golden_param_mgda").unwrap().to_vec1::<f32>()?;
    let gold_g_pc = tensors.get("golden_g_pc").unwrap().to_vec1::<f32>()?;
    let gold_p_pc = tensors.get("golden_param_pc").unwrap().to_vec1::<f32>()?;
    let gold_g_tch = tensors.get("golden_g_tch").unwrap().to_vec1::<f32>()?;
    let gold_p_tch = tensors.get("golden_param_tch").unwrap().to_vec1::<f32>()?;
    let gold_g_lin = tensors.get("golden_g_lin").unwrap().to_vec1::<f32>()?;
    let gold_p_lin = tensors.get("golden_param_lin").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let d = param.len();

    // 1. MGDA CPU & Auto
    let (cpu_p_mgda, cpu_g_mgda, cpu_alpha) = t09::step_mgda_2task_cpu(&param, &g1, &g2, 0.01, 1e-12);
    let (auto_p_mgda, auto_g_mgda, auto_alpha) = t09::step_mgda_2task_auto(&kdev, &param, &g1, &g2, 0.01, 1e-12);
    assert!((cpu_alpha - gold_alpha).abs() < 1e-5);
    assert!((auto_alpha - gold_alpha).abs() < 1e-5);
    for j in 0..d {
        assert!((cpu_g_mgda[j] - gold_g_mgda[j]).abs() < 1e-5);
        assert!((auto_g_mgda[j] - gold_g_mgda[j]).abs() < 1e-5);
        assert!((cpu_p_mgda[j] - gold_p_mgda[j]).abs() < 1e-5);
        assert!((auto_p_mgda[j] - gold_p_mgda[j]).abs() < 1e-5);
    }

    // 2. PCGrad CPU & Auto
    let (cpu_p_pc, cpu_g_pc) = t09::step_pcgrad_2task_cpu(&param, &g1, &g2, 0.01, 1e-12);
    let (auto_p_pc, auto_g_pc) = t09::step_pcgrad_2task_auto(&kdev, &param, &g1, &g2, 0.01, 1e-12);
    for j in 0..d {
        assert!((cpu_g_pc[j] - gold_g_pc[j]).abs() < 1e-5);
        assert!((auto_g_pc[j] - gold_g_pc[j]).abs() < 1e-5);
        assert!((cpu_p_pc[j] - gold_p_pc[j]).abs() < 1e-5);
        assert!((auto_p_pc[j] - gold_p_pc[j]).abs() < 1e-5);
    }

    // 3. Tchebycheff CPU & Auto
    let (cpu_p_tch, cpu_g_tch) = t09::step_tchebycheff_cpu(&param, &g1, &g2, &losses, &z_star, &weights_tch, 0.05, 0.01);
    let (auto_p_tch, auto_g_tch) = t09::step_tchebycheff_auto(&kdev, &param, &g1, &g2, &losses, &z_star, &weights_tch, 0.05, 0.01);
    for j in 0..d {
        assert!((cpu_g_tch[j] - gold_g_tch[j]).abs() < 1e-5);
        assert!((auto_g_tch[j] - gold_g_tch[j]).abs() < 1e-5);
        assert!((cpu_p_tch[j] - gold_p_tch[j]).abs() < 1e-5);
        assert!((auto_p_tch[j] - gold_p_tch[j]).abs() < 1e-5);
    }

    // 4. Linear Scalarization CPU & Auto
    let (cpu_p_lin, cpu_g_lin) = t09::step_linear_scalarization_cpu(&param, &g1, &g2, &weights_lin, 0.01);
    let (auto_p_lin, auto_g_lin) = t09::step_linear_scalarization_auto(&kdev, &param, &g1, &g2, &weights_lin, 0.01);
    for j in 0..d {
        assert!((cpu_g_lin[j] - gold_g_lin[j]).abs() < 1e-5);
        assert!((auto_g_lin[j] - gold_g_lin[j]).abs() < 1e-5);
        assert!((cpu_p_lin[j] - gold_p_lin[j]).abs() < 1e-5);
        assert!((auto_p_lin[j] - gold_p_lin[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t09::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_10_coordinate_alternating_projections_and_majorization_minimization_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t10_coordinate_alternating_projections_and_majorization_minimization as t10;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/03_coordinate_alternating_projections_and_majorization_minimization.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let x0 = tensors.get("x0").unwrap().to_vec1::<f32>()?;
    let normal = tensors.get("normal").unwrap().to_vec1::<f32>()?;
    let offset = tensors.get("offset").unwrap().to_vec1::<f32>()?[0];

    let param_bcd = tensors.get("param_bcd").unwrap().to_vec1::<f32>()?;
    let b_bcd = tensors.get("b_bcd").unwrap().to_vec1::<f32>()?;
    let a_diag = tensors.get("A_diag").unwrap().to_vec1::<f32>()?;

    let param_irls = tensors.get("param_irls").unwrap().to_vec1::<f32>()?;
    let target_irls = tensors.get("target_irls").unwrap().to_vec1::<f32>()?;

    let param_cccp = tensors.get("param_cccp").unwrap().to_vec1::<f32>()?;

    let gold_dykstra_x = tensors.get("golden_dykstra_x").unwrap().to_vec1::<f32>()?;
    let gold_dykstra_y = tensors.get("golden_dykstra_y").unwrap().to_vec1::<f32>()?;
    let gold_dykstra_p = tensors.get("golden_dykstra_p").unwrap().to_vec1::<f32>()?;
    let gold_dykstra_q = tensors.get("golden_dykstra_q").unwrap().to_vec1::<f32>()?;

    let gold_param_bcd = tensors.get("golden_param_bcd").unwrap().to_vec1::<f32>()?;
    let gold_param_irls = tensors.get("golden_param_irls").unwrap().to_vec1::<f32>()?;
    let gold_param_cccp = tensors.get("golden_param_cccp").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let d = x0.len();

    // 1. Dykstra CPU & Auto
    let p0 = vec![0.0f32; d];
    let q0 = vec![0.0f32; d];
    let (cpu_x, cpu_y, cpu_p, cpu_q) = t10::step_dykstra_cpu(&x0, &p0, &q0, &normal, offset, 0.0, 2.0);
    let (auto_x, auto_y, auto_p, auto_q) = t10::step_dykstra_auto(&kdev, &x0, &p0, &q0, &normal, offset, 0.0, 2.0);
    for j in 0..d {
        assert!((cpu_x[j] - gold_dykstra_x[j]).abs() < 1e-5);
        assert!((auto_x[j] - gold_dykstra_x[j]).abs() < 1e-5);
        assert!((cpu_y[j] - gold_dykstra_y[j]).abs() < 1e-5);
        assert!((auto_y[j] - gold_dykstra_y[j]).abs() < 1e-5);
        assert!((cpu_p[j] - gold_dykstra_p[j]).abs() < 1e-5);
        assert!((auto_p[j] - gold_dykstra_p[j]).abs() < 1e-5);
        assert!((cpu_q[j] - gold_dykstra_q[j]).abs() < 1e-5);
        assert!((auto_q[j] - gold_dykstra_q[j]).abs() < 1e-5);
    }

    // 2. BCD CPU & Auto
    let cpu_bcd = t10::step_bcd_cpu(&param_bcd, &b_bcd, &a_diag, 0.2);
    let auto_bcd = t10::step_bcd_auto(&kdev, &param_bcd, &b_bcd, &a_diag, 0.2);
    for j in 0..d {
        assert!((cpu_bcd[j] - gold_param_bcd[j]).abs() < 1e-5);
        assert!((auto_bcd[j] - gold_param_bcd[j]).abs() < 1e-5);
    }

    // 3. IRLS CPU & Auto
    let cpu_irls = t10::step_irls_cpu(&param_irls, &target_irls, 0.5, 1e-4);
    let auto_irls = t10::step_irls_auto(&kdev, &param_irls, &target_irls, 0.5, 1e-4);
    for j in 0..d {
        assert!((cpu_irls[j] - gold_param_irls[j]).abs() < 1e-5);
        assert!((auto_irls[j] - gold_param_irls[j]).abs() < 1e-5);
    }

    // 4. CCCP CPU & Auto
    let cpu_cccp = t10::step_cccp_cpu(&param_cccp);
    let auto_cccp = t10::step_cccp_auto(&kdev, &param_cccp);
    for j in 0..d {
        assert!((cpu_cccp[j] - gold_param_cccp[j]).abs() < 1e-5);
        assert!((auto_cccp[j] - gold_param_cccp[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t10::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_11_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t11_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn as t11;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let log_alpha = tensors.get("log_alpha").unwrap().to_vec2::<f32>()?;
    let flat_log_alpha: Vec<f32> = log_alpha.iter().flatten().copied().collect();

    let candidates = tensors.get("candidates").unwrap().to_vec2::<f32>()?;
    let flat_candidates: Vec<f32> = candidates.iter().flatten().copied().collect();
    let scores = tensors.get("scores").unwrap().to_vec1::<f32>()?;

    let delta_e = tensors.get("delta_e").unwrap().to_vec1::<f32>()?;
    let sa_temp = tensors.get("sa_temp").unwrap().to_vec1::<f32>()?[0];

    let pt_e1 = tensors.get("pt_e1").unwrap().to_vec1::<f32>()?[0];
    let pt_e2 = tensors.get("pt_e2").unwrap().to_vec1::<f32>()?[0];
    let pt_t1 = tensors.get("pt_t1").unwrap().to_vec1::<f32>()?[0];
    let pt_t2 = tensors.get("pt_t2").unwrap().to_vec1::<f32>()?[0];

    let gold_sinkhorn = tensors.get("golden_sinkhorn_out").unwrap().to_vec2::<f32>()?;
    let flat_gold_sinkhorn: Vec<f32> = gold_sinkhorn.iter().flatten().copied().collect();

    let gold_cem_mean = tensors.get("golden_cem_mean").unwrap().to_vec1::<f32>()?;
    let gold_cem_std = tensors.get("golden_cem_std").unwrap().to_vec1::<f32>()?;
    let gold_sa_probs = tensors.get("golden_sa_probs").unwrap().to_vec1::<f32>()?;
    let gold_pt_prob = tensors.get("golden_pt_swap_prob").unwrap().to_vec1::<f32>()?[0];

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    // 1. Sinkhorn CPU & Auto
    let cpu_sinkhorn = t11::run_sinkhorn_cpu(&flat_log_alpha, 4, 4, 10, 0.5);
    let auto_sinkhorn = t11::run_sinkhorn_auto(&kdev, &flat_log_alpha, 4, 4, 10, 0.5);
    for j in 0..flat_gold_sinkhorn.len() {
        assert!((cpu_sinkhorn[j] - flat_gold_sinkhorn[j]).abs() < 1e-5);
        assert!((auto_sinkhorn[j] - flat_gold_sinkhorn[j]).abs() < 1e-5);
    }

    // 2. CEM CPU & Auto
    let (cpu_mean, cpu_std) = t11::step_cem_cpu(&flat_candidates, &scores, 20, 4, 0.25);
    let (auto_mean, auto_std) = t11::step_cem_auto(&kdev, &flat_candidates, &scores, 20, 4, 0.25);
    for j in 0..4 {
        assert!((cpu_mean[j] - gold_cem_mean[j]).abs() < 1e-5);
        assert!((auto_mean[j] - gold_cem_mean[j]).abs() < 1e-5);
        assert!((cpu_std[j] - gold_cem_std[j]).abs() < 1e-5);
        assert!((auto_std[j] - gold_cem_std[j]).abs() < 1e-5);
    }

    // 3. Simulated Annealing CPU & Auto
    let cpu_sa = t11::sa_acceptance_prob_cpu(&delta_e, sa_temp);
    let auto_sa = t11::sa_acceptance_prob_auto(&kdev, &delta_e, sa_temp);
    for j in 0..delta_e.len() {
        assert!((cpu_sa[j] - gold_sa_probs[j]).abs() < 1e-5);
        assert!((auto_sa[j] - gold_sa_probs[j]).abs() < 1e-5);
    }

    // 4. Parallel Tempering CPU & Auto
    let cpu_pt = t11::parallel_tempering_swap_prob_cpu(pt_e1, pt_e2, pt_t1, pt_t2);
    let auto_pt = t11::parallel_tempering_swap_prob_auto(&kdev, pt_e1, pt_e2, pt_t1, pt_t2);
    assert!((cpu_pt - gold_pt_prob).abs() < 1e-5);
    assert!((auto_pt - gold_pt_prob).abs() < 1e-5);

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t11::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_12_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t12_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed as t12;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let sha_losses = tensors.get("sha_losses").unwrap().to_vec1::<f32>()?;
    let golden_sha_idx = tensors.get("golden_sha_idx").unwrap().to_vec1::<f32>()?;
    let golden_sha_losses = tensors.get("golden_sha_losses").unwrap().to_vec1::<f32>()?;

    let golden_hb_schedule = tensors.get("golden_hb_schedule").unwrap().to_vec2::<f32>()?;
    let flat_golden_hb_schedule: Vec<f32> = golden_hb_schedule.iter().flatten().copied().collect();

    let x_query = tensors.get("x_query").unwrap().to_vec1::<f32>()?;
    let good_samples = tensors.get("good_samples").unwrap().to_vec1::<f32>()?;
    let bad_samples = tensors.get("bad_samples").unwrap().to_vec1::<f32>()?;
    let golden_tpe_log_ratio = tensors.get("golden_tpe_log_ratio").unwrap().to_vec1::<f32>()?;

    let rung_losses = tensors.get("rung_losses").unwrap().to_vec1::<f32>()?;
    let candidate_losses = tensors.get("candidate_losses").unwrap().to_vec1::<f32>()?;
    let golden_asha_promoted = tensors.get("golden_asha_promoted").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    // 1. SHA CPU & Auto
    let (cpu_idx, cpu_losses) = t12::compute_sha_top_k_cpu(&sha_losses, 3);
    let (auto_idx, auto_losses) = t12::compute_sha_top_k_auto(&kdev, &sha_losses, 3);
    for j in 0..golden_sha_idx.len() {
        assert!((cpu_idx[j] - golden_sha_idx[j]).abs() < 1e-5);
        assert!((auto_idx[j] - golden_sha_idx[j]).abs() < 1e-5);
        assert!((cpu_losses[j] - golden_sha_losses[j]).abs() < 1e-5);
        assert!((auto_losses[j] - golden_sha_losses[j]).abs() < 1e-5);
    }

    // 2. Hyperband CPU & Auto
    let cpu_hb = t12::compute_hyperband_schedule_cpu(81, 3);
    let auto_hb = t12::compute_hyperband_schedule_auto(&kdev, 81, 3);
    for j in 0..flat_golden_hb_schedule.len() {
        assert!((cpu_hb[j] - flat_golden_hb_schedule[j]).abs() < 1e-5);
        assert!((auto_hb[j] - flat_golden_hb_schedule[j]).abs() < 1e-5);
    }

    // 3. TPE CPU & Auto
    let cpu_tpe = t12::compute_tpe_log_ratio_cpu(&x_query, &good_samples, &bad_samples, 0.5);
    let auto_tpe = t12::compute_tpe_log_ratio_auto(&kdev, &x_query, &good_samples, &bad_samples, 0.5);
    for j in 0..golden_tpe_log_ratio.len() {
        assert!((cpu_tpe[j] - golden_tpe_log_ratio[j]).abs() < 1e-5);
        assert!((auto_tpe[j] - golden_tpe_log_ratio[j]).abs() < 1e-5);
    }

    // 4. ASHA CPU & Auto
    let cpu_p0 = t12::check_asha_promotion_cpu(&rung_losses, candidate_losses[0], 3);
    let cpu_p1 = t12::check_asha_promotion_cpu(&rung_losses, candidate_losses[1], 3);
    let auto_p0 = t12::check_asha_promotion_auto(&kdev, &rung_losses, candidate_losses[0], 3);
    let auto_p1 = t12::check_asha_promotion_auto(&kdev, &rung_losses, candidate_losses[1], 3);
    assert!((cpu_p0 - golden_asha_promoted[0]).abs() < 1e-5);
    assert!((cpu_p1 - golden_asha_promoted[1]).abs() < 1e-5);
    assert!((auto_p0 - golden_asha_promoted[0]).abs() < 1e-5);
    assert!((auto_p1 - golden_asha_promoted[1]).abs() < 1e-5);

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t12::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_13_sharpness_aware_swad_and_weight_averaging_dynamics_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t13_sharpness_aware_swad_and_weight_averaging_dynamics as t13;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/06_sharpness_aware_swad_and_weight_averaging_dynamics.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let param_sam = tensors.get("param_sam").unwrap().to_vec1::<f32>()?;
    let grad_at_w = tensors.get("grad_at_w").unwrap().to_vec1::<f32>()?;
    let grad_at_pert = tensors.get("grad_at_pert").unwrap().to_vec1::<f32>()?;
    let golden_param_sam = tensors.get("golden_param_sam").unwrap().to_vec1::<f32>()?;
    let golden_epsilon_sam = tensors.get("golden_epsilon_sam").unwrap().to_vec1::<f32>()?;

    let slow_p = tensors.get("slow_p").unwrap().to_vec1::<f32>()?;
    let fast_p = tensors.get("fast_p").unwrap().to_vec1::<f32>()?;
    let golden_slow_p = tensors.get("golden_slow_p").unwrap().to_vec1::<f32>()?;
    let golden_fast_p = tensors.get("golden_fast_p").unwrap().to_vec1::<f32>()?;

    let swa_avg = tensors.get("swa_avg").unwrap().to_vec1::<f32>()?;
    let ema_avg = tensors.get("ema_avg").unwrap().to_vec1::<f32>()?;
    let curr_param = tensors.get("curr_param").unwrap().to_vec1::<f32>()?;
    let golden_swa = tensors.get("golden_swa").unwrap().to_vec1::<f32>()?;
    let golden_ema = tensors.get("golden_ema").unwrap().to_vec1::<f32>()?;

    let soup_models = tensors.get("soup_models").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_soup = tensors.get("golden_soup").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    // 1. SAM CPU & Auto
    let (cpu_p, cpu_eps) = t13::step_sam_cpu(&param_sam, &grad_at_w, &grad_at_pert, 0.05, 0.01, 0.001, 1e-12);
    let (auto_p, auto_eps) = t13::step_sam_auto(&kdev, &param_sam, &grad_at_w, &grad_at_pert, 0.05, 0.01, 0.001, 1e-12);
    for j in 0..golden_param_sam.len() {
        assert!((cpu_p[j] - golden_param_sam[j]).abs() < 1e-5);
        assert!((auto_p[j] - golden_param_sam[j]).abs() < 1e-5);
        assert!((cpu_eps[j] - golden_epsilon_sam[j]).abs() < 1e-5);
        assert!((auto_eps[j] - golden_epsilon_sam[j]).abs() < 1e-5);
    }

    // 2. Lookahead CPU & Auto
    let (cpu_slow, cpu_fast) = t13::step_lookahead_cpu(&slow_p, &fast_p, 0.5);
    let (auto_slow, auto_fast) = t13::step_lookahead_auto(&kdev, &slow_p, &fast_p, 0.5);
    for j in 0..golden_slow_p.len() {
        assert!((cpu_slow[j] - golden_slow_p[j]).abs() < 1e-5);
        assert!((auto_slow[j] - golden_slow_p[j]).abs() < 1e-5);
        assert!((cpu_fast[j] - golden_fast_p[j]).abs() < 1e-5);
        assert!((auto_fast[j] - golden_fast_p[j]).abs() < 1e-5);
    }

    // 3. SWA & EMA CPU & Auto
    let (cpu_swa, cpu_ema) = t13::step_swa_ema_cpu(&swa_avg, &ema_avg, &curr_param, 4, 0.99);
    let (auto_swa, auto_ema) = t13::step_swa_ema_auto(&kdev, &swa_avg, &ema_avg, &curr_param, 4, 0.99);
    for j in 0..golden_swa.len() {
        assert!((cpu_swa[j] - golden_swa[j]).abs() < 1e-5);
        assert!((auto_swa[j] - golden_swa[j]).abs() < 1e-5);
        assert!((cpu_ema[j] - golden_ema[j]).abs() < 1e-5);
        assert!((auto_ema[j] - golden_ema[j]).abs() < 1e-5);
    }

    // 4. Model Soup CPU & Auto
    let cpu_soup = t13::uniform_model_soup_cpu(&soup_models, 3, 4);
    let auto_soup = t13::uniform_model_soup_auto(&kdev, &soup_models, 3, 4);
    for j in 0..golden_soup.len() {
        assert!((cpu_soup[j] - golden_soup[j]).abs() < 1e-5);
        assert!((auto_soup[j] - golden_soup[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t13::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_14_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t14_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa as t14;

    let kdev = KernelDevice::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/07_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa.safetensors",
    );
    let tensors = load(weights_path, &Device::Cpu)?;

    let matrix_a = tensors.get("matrix_A").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let state_x = tensors.get("state_x").unwrap().to_vec1::<f32>()?;
    let state_y = tensors.get("state_y").unwrap().to_vec1::<f32>()?;
    let state_x_prev = tensors.get("state_x_prev").unwrap().to_vec1::<f32>()?;
    let state_y_prev = tensors.get("state_y_prev").unwrap().to_vec1::<f32>()?;

    let golden_x_eg = tensors.get("golden_x_eg").unwrap().to_vec1::<f32>()?;
    let golden_y_eg = tensors.get("golden_y_eg").unwrap().to_vec1::<f32>()?;
    let golden_x_ogda = tensors.get("golden_x_ogda").unwrap().to_vec1::<f32>()?;
    let golden_y_ogda = tensors.get("golden_y_ogda").unwrap().to_vec1::<f32>()?;
    let golden_x_sga = tensors.get("golden_x_sga").unwrap().to_vec1::<f32>()?;
    let golden_y_sga = tensors.get("golden_y_sga").unwrap().to_vec1::<f32>()?;

    let x_best = tensors.get("x_best").unwrap().to_vec1::<f32>()?;
    let x_second = tensors.get("x_second").unwrap().to_vec1::<f32>()?;
    let x_worst = tensors.get("x_worst").unwrap().to_vec1::<f32>()?;
    let golden_x_reflect = tensors.get("golden_x_reflect").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    // 1. Extragradient CPU & Auto
    let (cpu_x_eg, cpu_y_eg) = t14::step_extragradient_cpu(&state_x, &state_y, &matrix_a, 0.1, 0.1);
    let (auto_x_eg, auto_y_eg) = t14::step_extragradient_auto(&kdev, &state_x, &state_y, &matrix_a, 0.1, 0.1);
    for j in 0..golden_x_eg.len() {
        assert!((cpu_x_eg[j] - golden_x_eg[j]).abs() < 1e-5);
        assert!((auto_x_eg[j] - golden_x_eg[j]).abs() < 1e-5);
        assert!((cpu_y_eg[j] - golden_y_eg[j]).abs() < 1e-5);
        assert!((auto_y_eg[j] - golden_y_eg[j]).abs() < 1e-5);
    }

    // 2. OGDA CPU & Auto
    let (cpu_x_ogda, cpu_y_ogda) = t14::step_ogda_cpu(&state_x, &state_y, &state_x_prev, &state_y_prev, &matrix_a, 0.1, 0.1);
    let (auto_x_ogda, auto_y_ogda) = t14::step_ogda_auto(&kdev, &state_x, &state_y, &state_x_prev, &state_y_prev, &matrix_a, 0.1, 0.1);
    for j in 0..golden_x_ogda.len() {
        assert!((cpu_x_ogda[j] - golden_x_ogda[j]).abs() < 1e-5);
        assert!((auto_x_ogda[j] - golden_x_ogda[j]).abs() < 1e-5);
        assert!((cpu_y_ogda[j] - golden_y_ogda[j]).abs() < 1e-5);
        assert!((auto_y_ogda[j] - golden_y_ogda[j]).abs() < 1e-5);
    }

    // 3. SGA CPU & Auto
    let (cpu_x_sga, cpu_y_sga) = t14::step_sga_cpu(&state_x, &state_y, &matrix_a, 0.5, 0.1, 0.1);
    let (auto_x_sga, auto_y_sga) = t14::step_sga_auto(&kdev, &state_x, &state_y, &matrix_a, 0.5, 0.1, 0.1);
    for j in 0..golden_x_sga.len() {
        assert!((cpu_x_sga[j] - golden_x_sga[j]).abs() < 1e-5);
        assert!((auto_x_sga[j] - golden_x_sga[j]).abs() < 1e-5);
        assert!((cpu_y_sga[j] - golden_y_sga[j]).abs() < 1e-5);
        assert!((auto_y_sga[j] - golden_y_sga[j]).abs() < 1e-5);
    }

    // 4. Nelder-Mead Reflection CPU & Auto
    let cpu_refl = t14::nelder_mead_reflect_cpu(&x_best, &x_second, &x_worst, 1.0);
    let auto_refl = t14::nelder_mead_reflect_auto(&kdev, &x_best, &x_second, &x_worst, 1.0);
    for j in 0..golden_x_reflect.len() {
        assert!((cpu_refl[j] - golden_x_reflect[j]).abs() < 1e-5);
        assert!((auto_refl[j] - golden_x_reflect[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t14::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_15_grokking_phase_transitions_edge_of_stability_and_implicit_bias_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t15_grokking_phase_transitions_edge_of_stability_and_implicit_bias as t15;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/01_grokking_phase_transitions_edge_of_stability_and_implicit_bias.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let g_raw = tensors.get("g_raw").unwrap().to_vec1::<f32>()?;
    let mu_prev = tensors.get("mu_prev").unwrap().to_vec1::<f32>()?;
    let golden_mu = tensors.get("golden_mu").unwrap().to_vec1::<f32>()?;
    let golden_g_grokfast = tensors.get("golden_g_grokfast").unwrap().to_vec1::<f32>()?;

    let x_eos = tensors.get("x_eos").unwrap().to_vec1::<f32>()?;
    let h_eos = tensors.get("h_eos").unwrap().to_vec1::<f32>()?;
    let gamma_eos = tensors.get("gamma_eos").unwrap().to_vec1::<f32>()?;
    let golden_x_eos = tensors.get("golden_x_eos").unwrap().to_vec1::<f32>()?;

    let w_sep = tensors.get("w_sep").unwrap().to_vec1::<f32>()?;
    let x_sep = tensors.get("x_sep").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let y_sep = tensors.get("y_sep").unwrap().to_vec1::<f32>()?;
    let golden_w_sep = tensors.get("golden_w_sep").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. Grokfast CPU & Auto
    let (cpu_gf, cpu_mu) = t15::step_grokfast_cpu(&g_raw, &mu_prev, 0.8, 2.0);
    let (auto_gf, auto_mu) = t15::step_grokfast_auto(&kdev, &g_raw, &mu_prev, 0.8, 2.0);
    for j in 0..golden_mu.len() {
        assert!((cpu_mu[j] - golden_mu[j]).abs() < 1e-5);
        assert!((auto_mu[j] - golden_mu[j]).abs() < 1e-5);
        assert!((cpu_gf[j] - golden_g_grokfast[j]).abs() < 1e-5);
        assert!((auto_gf[j] - golden_g_grokfast[j]).abs() < 1e-5);
    }

    // 2. EOS CPU & Auto
    let cpu_eos = t15::step_edge_of_stability_cpu(&x_eos, &h_eos, &gamma_eos, 0.1);
    let auto_eos = t15::step_edge_of_stability_auto(&kdev, &x_eos, &h_eos, &gamma_eos, 0.1);
    for j in 0..golden_x_eos.len() {
        assert!((cpu_eos[j] - golden_x_eos[j]).abs() < 1e-5);
        assert!((auto_eos[j] - golden_x_eos[j]).abs() < 1e-5);
    }

    // 3. Implicit Bias CPU & Auto
    let cpu_w_sep = t15::step_implicit_bias_cpu(&w_sep, &x_sep, &y_sep, 0.2);
    let auto_w_sep = t15::step_implicit_bias_auto(&kdev, &w_sep, &x_sep, &y_sep, 0.2);
    for j in 0..golden_w_sep.len() {
        assert!((cpu_w_sep[j] - golden_w_sep[j]).abs() < 1e-5);
        assert!((auto_w_sep[j] - golden_w_sep[j]).abs() < 1e-5);
    }

    // 4. Linear Anchor CPU
    let cpu_anchor_out = t15::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_16_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity_cpu_and_auto() -> Result<()> {
    use ai_kernel::c04_training_methods_and_optimisers::t16_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity as t16;

    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/02_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let z_emb = tensors.get("z_emb").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let mrl_w = tensors.get("mrl_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let mrl_b = tensors.get("mrl_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_logits_d2 = tensors.get("golden_logits_d2").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_logits_d4 = tensors.get("golden_logits_d4").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_logits_d8 = tensors.get("golden_logits_d8").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let p_probs = tensors.get("p_probs").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let y_targets = tensors.get("y_targets").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_el2n = tensors.get("golden_el2n").unwrap().to_vec1::<f32>()?;
    let loss_model = tensors.get("loss_model").unwrap().to_vec1::<f32>()?;
    let loss_ref = tensors.get("loss_ref").unwrap().to_vec1::<f32>()?;
    let golden_rho_loss = tensors.get("golden_rho_loss").unwrap().to_vec1::<f32>()?;

    let activations = tensors.get("activations").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_mean_act = tensors.get("mean_act").unwrap().to_vec1::<f32>()?;
    let golden_dormant_mask = tensors.get("dormant_mask").unwrap().to_vec1::<f32>()?;

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;

    // 1. MRL CPU & Auto
    let cpu_d2 = t16::mrl_slice_forward_cpu(&z_emb, &mrl_w, &mrl_b, 4, 8, 4, 2);
    let auto_d2 = t16::mrl_slice_forward_auto(&kdev, &z_emb, &mrl_w, &mrl_b, 4, 8, 4, 2);
    for j in 0..golden_logits_d2.len() {
        assert!((cpu_d2[j] - golden_logits_d2[j]).abs() < 1e-5);
        assert!((auto_d2[j] - golden_logits_d2[j]).abs() < 1e-5);
    }

    let cpu_d4 = t16::mrl_slice_forward_cpu(&z_emb, &mrl_w, &mrl_b, 4, 8, 4, 4);
    let auto_d4 = t16::mrl_slice_forward_auto(&kdev, &z_emb, &mrl_w, &mrl_b, 4, 8, 4, 4);
    for j in 0..golden_logits_d4.len() {
        assert!((cpu_d4[j] - golden_logits_d4[j]).abs() < 1e-5);
        assert!((auto_d4[j] - golden_logits_d4[j]).abs() < 1e-5);
    }

    let cpu_d8 = t16::mrl_slice_forward_cpu(&z_emb, &mrl_w, &mrl_b, 4, 8, 4, 8);
    let auto_d8 = t16::mrl_slice_forward_auto(&kdev, &z_emb, &mrl_w, &mrl_b, 4, 8, 4, 8);
    for j in 0..golden_logits_d8.len() {
        assert!((cpu_d8[j] - golden_logits_d8[j]).abs() < 1e-5);
        assert!((auto_d8[j] - golden_logits_d8[j]).abs() < 1e-5);
    }

    // 2. EL2N CPU & Auto
    let cpu_el2n = t16::compute_el2n_cpu(&p_probs, &y_targets, 4, 3);
    let auto_el2n = t16::compute_el2n_auto(&kdev, &p_probs, &y_targets, 4, 3);
    for j in 0..golden_el2n.len() {
        assert!((cpu_el2n[j] - golden_el2n[j]).abs() < 1e-5);
        assert!((auto_el2n[j] - golden_el2n[j]).abs() < 1e-5);
    }

    // 3. RHO-Loss CPU & Auto
    let cpu_rho = t16::compute_rho_loss_cpu(&loss_model, &loss_ref);
    let auto_rho = t16::compute_rho_loss_auto(&kdev, &loss_model, &loss_ref);
    for j in 0..golden_rho_loss.len() {
        assert!((cpu_rho[j] - golden_rho_loss[j]).abs() < 1e-5);
        assert!((auto_rho[j] - golden_rho_loss[j]).abs() < 1e-5);
    }

    // 4. ReDo Dormant Neurons CPU & Auto
    let (cpu_mean, cpu_mask) = t16::detect_dormant_neurons_cpu(&activations, 4, 6, 0.15);
    let (auto_mean, auto_mask) = t16::detect_dormant_neurons_auto(&kdev, &activations, 4, 6, 0.15);
    for j in 0..golden_mean_act.len() {
        assert!((cpu_mean[j] - golden_mean_act[j]).abs() < 1e-5);
        assert!((auto_mean[j] - golden_mean_act[j]).abs() < 1e-5);
        assert!((cpu_mask[j] - golden_dormant_mask[j]).abs() < 1e-5);
        assert!((auto_mask[j] - golden_dormant_mask[j]).abs() < 1e-5);
    }

    // 5. Linear Anchor CPU
    let cpu_anchor_out = t16::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 8, 4);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}





