//! Category 02 Classical & Probabilistic ML Integration Test Suite for ai-kernel.

use ai_kernel::c02_classical_and_probabilistic_ml::t01_generalized_linear_models_and_proximal_lasso as t01;
use ai_kernel::c02_classical_and_probabilistic_ml::t02_support_vector_machines_and_kernel_methods as t02;
use ai_kernel::KernelDevice;
use candle_core::{safetensors::load, Device, Result};

#[path = "common/mod.rs"]
mod common;

#[test]
fn test_01_generalized_linear_models_and_proximal_lasso_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let x_data = tensors.get("x_data").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let y_data = tensors.get("y_data").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_w_ols = tensors.get("golden_w_ols").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_mse_ols = tensors.get("golden_mse_ols").unwrap().flatten_all()?.to_vec1::<f32>()?[0];
    let golden_w_ridge = tensors.get("golden_w_ridge").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_w_lasso = tensors.get("golden_w_lasso").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let w_poisson_init = tensors.get("w_poisson_init").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let y_counts = tensors.get("y_counts").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_poisson_grad = tensors.get("golden_poisson_grad").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_w_poisson_next = tensors.get("golden_w_poisson_next").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_mean_huber = tensors.get("golden_mean_huber").unwrap().flatten_all()?.to_vec1::<f32>()?[0];

    let anchor_x = tensors.get("anchor_x").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_w = tensors.get("anchor_w").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let anchor_b = tensors.get("anchor_b").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gold_anchor_out = tensors.get("golden_anchor_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;
    let n = 8;
    let d = 4;

    // 1. OLS CPU & Auto
    let (cpu_w_ols, cpu_mse_ols) = t01::solve_ols_cpu(&x_data, &y_data, n, d);
    let (auto_w_ols, auto_mse_ols) = t01::solve_ols_auto(&kdev, &x_data, &y_data, n, d);
    for j in 0..d {
        assert!((cpu_w_ols[j] - golden_w_ols[j]).abs() < 1e-5);
        assert!((auto_w_ols[j] - golden_w_ols[j]).abs() < 1e-5);
    }
    assert!((cpu_mse_ols - golden_mse_ols).abs() < 1e-5);
    assert!((auto_mse_ols - golden_mse_ols).abs() < 1e-5);

    // 2. Ridge CPU & Auto
    let cpu_w_ridge = t01::solve_ridge_cpu(&x_data, &y_data, n, d, 0.5);
    let auto_w_ridge = t01::solve_ridge_auto(&kdev, &x_data, &y_data, n, d, 0.5);
    for j in 0..d {
        assert!((cpu_w_ridge[j] - golden_w_ridge[j]).abs() < 1e-5);
        assert!((auto_w_ridge[j] - golden_w_ridge[j]).abs() < 1e-5);
    }

    // 3. Lasso FISTA CPU & Auto
    let cpu_w_lasso = t01::solve_lasso_fista_cpu(&x_data, &y_data, n, d, 0.2, 10);
    let auto_w_lasso = t01::solve_lasso_fista_auto(&kdev, &x_data, &y_data, n, d, 0.2, 10);
    for j in 0..d {
        assert!((cpu_w_lasso[j] - golden_w_lasso[j]).abs() < 1e-5);
        assert!((auto_w_lasso[j] - golden_w_lasso[j]).abs() < 1e-5);
    }

    // 4. Poisson GLM CPU & Auto
    let (cpu_p_grad, cpu_w_p_next) = t01::step_poisson_glm_cpu(&x_data, &y_counts, &w_poisson_init, n, d, 0.01);
    let (auto_p_grad, auto_w_p_next) = t01::step_poisson_glm_auto(&kdev, &x_data, &y_counts, &w_poisson_init, n, d, 0.01);
    for j in 0..d {
        assert!((cpu_p_grad[j] - golden_poisson_grad[j]).abs() < 1e-5);
        assert!((auto_p_grad[j] - golden_poisson_grad[j]).abs() < 1e-5);
        assert!((cpu_w_p_next[j] - golden_w_poisson_next[j]).abs() < 1e-5);
        assert!((auto_w_p_next[j] - golden_w_poisson_next[j]).abs() < 1e-5);
    }

    // 5. Huber Loss CPU & Auto
    let mut res = vec![0.0f32; n];
    for r in 0..n {
        let mut pred = 0.0f32;
        for j in 0..d {
            pred += x_data[r * d + j] * cpu_w_ols[j];
        }
        res[r] = pred - y_data[r];
    }
    let cpu_huber = t01::compute_huber_mean_loss_cpu(&res, 0.5);
    let auto_huber = t01::compute_huber_mean_loss_auto(&kdev, &res, 0.5);
    assert!((cpu_huber - golden_mean_huber).abs() < 1e-5);
    assert!((auto_huber - golden_mean_huber).abs() < 1e-5);

    // 6. Linear Anchor CPU
    let cpu_anchor_out = t01::anchor_forward_cpu(&anchor_x, &anchor_w, &anchor_b, 2, 4, 2);
    for j in 0..gold_anchor_out.len() {
        assert!((cpu_anchor_out[j] - gold_anchor_out[j]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_02_support_vector_machines_and_kernel_methods_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "02_classical_and_probabilistic_ml/02_support_vector_machines_and_kernel_methods.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let x = tensors.get("X").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let y_cont = tensors.get("y_cont").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let landmarks = tensors.get("X_landmarks").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_k_rbf = tensors.get("K_rbf").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_k_poly = tensors.get("K_poly").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_c_ridge = tensors.get("c_ridge").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_k_nystrom = tensors.get("K_nystrom").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let n = 64;
    let d = 8;
    let m = 32;
    let gamma = 0.125f32;

    // 1. RBF Kernel CPU & Auto
    let kdev = KernelDevice::Cpu;
    let cpu_k_rbf = t02::rbf_kernel_cpu(&x, n, &x, n, d, gamma);
    let auto_k_rbf = t02::compute_rbf_kernel_auto(&kdev, &x, n, &x, n, d, gamma);
    for i in 0..(n * n) {
        assert!((cpu_k_rbf[i] - golden_k_rbf[i]).abs() < 1e-5);
        assert!((auto_k_rbf[i] - golden_k_rbf[i]).abs() < 1e-5);
    }

    // 2. Poly Kernel CPU
    let cpu_k_poly = t02::poly_kernel_cpu(&x, n, &x, n, d, 2.0, 1.0);
    for i in 0..(n * n) {
        assert!((cpu_k_poly[i] - golden_k_poly[i]).abs() < 1e-5);
    }

    // 3. Kernel Ridge Regression Dual Coefficients
    let cpu_c_ridge = t02::solve_kernel_ridge_cpu(&x, &y_cont, n, d, gamma, 0.1);
    for i in 0..n {
        assert!((cpu_c_ridge[i] - golden_c_ridge[i]).abs() < 1e-4);
    }

    // 4. Nyström Low-Rank Approximation
    let cpu_k_nystrom = t02::nystrom_approximate_gram_cpu(&x, n, &landmarks, m, d, gamma, 1e-4);
    for i in 0..(n * n) {
        assert!((cpu_k_nystrom[i] - golden_k_nystrom[i]).abs() < 1e-4);
    }

    Ok(())
}
