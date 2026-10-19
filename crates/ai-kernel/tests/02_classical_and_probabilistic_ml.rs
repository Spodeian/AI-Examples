//! Category 02 Classical & Probabilistic ML Integration Test Suite for ai-kernel.

use ai_kernel::c02_classical_and_probabilistic_ml::t01_generalized_linear_models_and_proximal_lasso as t01;
use ai_kernel::c02_classical_and_probabilistic_ml::t02_support_vector_machines_and_kernel_methods as t02;
use ai_kernel::c02_classical_and_probabilistic_ml::t03_knn_prototype_and_locally_weighted_regression as t03;
use ai_kernel::c02_classical_and_probabilistic_ml::t04_clustering_kmeans_spectral_and_gmm_em as t04;
use ai_kernel::c02_classical_and_probabilistic_ml::t05_dimensionality_reduction_pca_lda_ica_tsne_umap as t05;
use ai_kernel::c02_classical_and_probabilistic_ml::t06_naive_bayes_bayesian_regression_and_gaussian_processes as t06;
use ai_kernel::c02_classical_and_probabilistic_ml::t07_differentiable_trees_node_soft_gbdt_and_gam_nam as t07;
use ai_kernel::c02_classical_and_probabilistic_ml::t08_matrix_factorization_als_nmf_bpr_and_fm as t08;
use ai_kernel::c02_classical_and_probabilistic_ml::t09_hidden_markov_models_kalman_ukf_and_particle_filters as t09;
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

#[test]
fn test_03_knn_prototype_and_locally_weighted_regression_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "02_classical_and_probabilistic_ml/03_knn_prototype_and_locally_weighted_regression.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let x_train = tensors.get("X_train").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let y_train_cont = tensors.get("y_train_cont").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let x_test = tensors.get("X_test").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_dist_sq = tensors.get("dist_sq").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_knn_reg_pred = tensors.get("knn_reg_pred").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_nw_pred = tensors.get("nw_pred").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let n_train = 64;
    let n_test = 16;
    let d = 8;

    // 1. Pairwise Dist Sq CPU & Auto
    let kdev = KernelDevice::Cpu;
    let cpu_dist_sq = t03::pairwise_sq_dist_cpu(&x_train, n_train, &x_test, n_test, d);
    let auto_dist_sq = t03::compute_pairwise_sq_dist_auto(&kdev, &x_train, n_train, &x_test, n_test, d);
    for i in 0..(n_train * n_test) {
        assert!((cpu_dist_sq[i] - golden_dist_sq[i]).abs() < 1e-4);
        assert!((auto_dist_sq[i] - golden_dist_sq[i]).abs() < 1e-4);
    }

    // 2. k-NN Continuous Regression
    let cpu_knn_pred = t03::knn_regress_cpu(&x_train, &y_train_cont, n_train, &x_test, n_test, d, 5, 1e-5);
    for j in 0..n_test {
        assert!((cpu_knn_pred[j] - golden_knn_reg_pred[j]).abs() < 1e-4);
    }

    // 3. Nadaraya-Watson Regression
    let cpu_nw_pred = t03::nadaraya_watson_regress_cpu(&x_train, &y_train_cont, n_train, &x_test, n_test, d, 1.0);
    for j in 0..n_test {
        assert!((cpu_nw_pred[j] - golden_nw_pred[j]).abs() < 1e-4);
    }

    Ok(())
}

#[test]
fn test_04_clustering_kmeans_spectral_and_gmm_em_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "02_classical_and_probabilistic_ml/04_clustering_kmeans_spectral_and_gmm_em.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let x = tensors.get("X").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let centers = tensors.get("centers").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_kmeans_assign = tensors.get("kmeans_assignments").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_resp = tensors.get("resp").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_l_sym = tensors.get("L_sym").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let n = 64;
    let d = 8;
    let k = 3;

    // 1. k-Means Assign CPU & Auto
    let kdev = KernelDevice::Cpu;
    let cpu_assign = t04::kmeans_assign_cpu(&x, n, &centers, k, d);
    let auto_assign = t04::compute_kmeans_assign_auto(&kdev, &x, n, &centers, k, d);
    for i in 0..n {
        assert!((cpu_assign[i] - golden_kmeans_assign[i]).abs() < 1e-4);
        assert!((auto_assign[i] - golden_kmeans_assign[i]).abs() < 1e-4);
    }

    // 2. GMM E-step CPU
    let pi = vec![1.0f32 / (k as f32); k];
    let var = vec![1.0f32; k * d];
    let cpu_resp = t04::gmm_e_step_cpu(&x, n, &centers, &var, &pi, k, d);
    for i in 0..(n * k) {
        assert!((cpu_resp[i] - golden_resp[i]).abs() < 1e-4);
    }

    // 3. Symmetric Normalized Laplacian CPU
    let cpu_l_sym = t04::compute_symmetric_laplacian_cpu(&x, n, d, 0.1);
    for i in 0..(n * n) {
        assert!((cpu_l_sym[i] - golden_l_sym[i]).abs() < 1e-4);
    }

    Ok(())
}

#[test]
fn test_05_dimensionality_reduction_pca_lda_ica_tsne_umap_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "02_classical_and_probabilistic_ml/05_dimensionality_reduction_pca_lda_ica_tsne_umap.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let x = tensors.get("X").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let y_class = tensors.get("y_class").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let mean = tensors.get("mean").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let v_top = tensors.get("V_top").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_pca_proj = tensors.get("pca_proj").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_w_lda = tensors.get("w_lda").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_lda_proj = tensors.get("lda_proj").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let n = 64;
    let d = 8;
    let k = 3;

    // 1. Mean Centering CPU & Auto
    let kdev = KernelDevice::Cpu;
    let cpu_centered = t05::center_matrix_cpu(&x, n, d, &mean);
    let auto_centered = t05::compute_center_matrix_auto(&kdev, &x, n, d, &mean);
    for i in 0..(n * d) {
        assert!((cpu_centered[i] - auto_centered[i]).abs() < 1e-5);
    }

    // 2. PCA Projection CPU
    let cpu_pca_proj = t05::pca_project_cpu(&cpu_centered, n, d, &v_top, k);
    for i in 0..(n * k) {
        assert!((cpu_pca_proj[i] - golden_pca_proj[i]).abs() < 1e-4);
    }

    // 3. Fisher's LDA Direction & Projection CPU
    let cpu_w_lda = t05::solve_lda_direction_cpu(&cpu_centered, &y_class, n, d, 1e-3);
    for j in 0..d {
        assert!((cpu_w_lda[j] - golden_w_lda[j]).abs() < 1e-4);
    }

    let mut cpu_lda_proj = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..d {
            sum += cpu_centered[i * d + j] * cpu_w_lda[j];
        }
        cpu_lda_proj[i] = sum;
    }
    for i in 0..n {
        assert!((cpu_lda_proj[i] - golden_lda_proj[i]).abs() < 1e-4);
    }

    Ok(())
}

#[test]
fn test_06_naive_bayes_bayesian_regression_and_gaussian_processes_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "02_classical_and_probabilistic_ml/06_naive_bayes_bayesian_regression_and_gaussian_processes.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let x = tensors.get("X").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let means = tensors.get("nb_means").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let vars = tensors.get("nb_vars").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let priors = tensors.get("nb_priors").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_nb_log_probs = tensors.get("nb_log_probs").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let m_n = tensors.get("bayes_m_N").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let s_n = tensors.get("bayes_S_N").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_b_mean = tensors.get("bayes_pred_mean").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_b_var = tensors.get("bayes_pred_var").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_gp_k = tensors.get("gp_K").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let gp_alpha = tensors.get("gp_alpha").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_gp_mean = tensors.get("gp_pred_mean").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let n = 32;
    let d = 4;
    let c = 2;

    // 1. RBF Kernel CPU & Auto
    let kdev = KernelDevice::Cpu;
    let cpu_k = t06::rbf_kernel_cpu(&x, n, d, 1.0, 1.0);
    let auto_k = t06::rbf_kernel_auto(Some(&kdev), &x, n, d, 1.0, 1.0);
    for i in 0..(n * n) {
        assert!((cpu_k[i] - golden_gp_k[i]).abs() < 1e-4);
        assert!((auto_k[i] - golden_gp_k[i]).abs() < 1e-4);
    }

    // 2. Gaussian Naive Bayes Log-Probabilities CPU
    let log_priors = vec![priors[0].ln(), priors[1].ln()];
    let cpu_nb_log_probs = t06::gaussian_naive_bayes_log_probs_cpu(&x, n, d, &means, &vars, &log_priors, c);
    for i in 0..(n * c) {
        assert!((cpu_nb_log_probs[i] - golden_nb_log_probs[i]).abs() < 1e-4);
    }

    // 3. Bayesian Linear Regression CPU
    let (cpu_b_mean, cpu_b_var) = t06::bayesian_linear_regression_predict_cpu(&x, n, d, &m_n, &s_n, 2.0);
    for i in 0..n {
        assert!((cpu_b_mean[i] - golden_b_mean[i]).abs() < 1e-4);
        assert!((cpu_b_var[i] - golden_b_var[i]).abs() < 1e-4);
    }

    // 4. GP Predictive Mean CPU
    let cpu_gp_mean = t06::gp_predict_mean_cpu(&cpu_k, &gp_alpha, n, n);
    for i in 0..n {
        assert!((cpu_gp_mean[i] - golden_gp_mean[i]).abs() < 1e-4);
    }

    Ok(())
}

#[test]
fn test_07_differentiable_trees_node_soft_gbdt_and_gam_nam_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let x = tensors.get("X").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let w_split = tensors.get("W_split").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let b_split = tensors.get("b_split").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let leaf_values = tensors.get("leaf_values").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_node_out = tensors.get("node_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let nam_w1 = tensors.get("nam_w1").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let nam_b1 = tensors.get("nam_b1").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let nam_w2 = tensors.get("nam_w2").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let nam_b2 = tensors.get("nam_b2").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let nam_bias0 = tensors.get("nam_bias0").unwrap().flatten_all()?.to_vec1::<f32>()?[0];
    let golden_nam_out = tensors.get("nam_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let y_noisy = tensors.get("y_noisy").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_iso_fit = tensors.get("iso_fit").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let n = 32;
    let d = 4;
    let n_trees = 2;

    // 1. NODE Forward CPU & Auto
    let kdev = KernelDevice::Cpu;
    let cpu_node = t07::node_forward_cpu(&x, n, d, &w_split, &b_split, &leaf_values, n_trees);
    for i in 0..n {
        assert!((cpu_node[i] - golden_node_out[i]).abs() < 1e-4);
    }

    let single_tree_w = &w_split[..8]; // depth=2, d=4 -> 8
    let single_tree_b = &b_split[..2];
    let single_tree_leaves = &leaf_values[..4];
    let auto_single = t07::node_single_tree_auto(Some(&kdev), &x, n, d, single_tree_w, single_tree_b, single_tree_leaves);
    assert_eq!(auto_single.len(), n);

    // 2. NAM Forward CPU
    let cpu_nam = t07::nam_forward_cpu(&x, n, d, &nam_w1, &nam_b1, &nam_w2, &nam_b2, nam_bias0, 8);
    for i in 0..n {
        assert!((cpu_nam[i] - golden_nam_out[i]).abs() < 1e-4);
    }

    // 3. PAV Isotonic Regression CPU
    let cpu_iso = t07::pav_isotonic_regression_cpu(&y_noisy);
    for i in 0..n {
        assert!((cpu_iso[i] - golden_iso_fit[i]).abs() < 1e-5);
    }

    Ok(())
}

#[test]
fn test_08_matrix_factorization_als_nmf_bpr_and_fm_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "02_classical_and_probabilistic_ml/08_matrix_factorization_als_nmf_bpr_and_fm.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let x = tensors.get("X").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let w0 = tensors.get("w0").unwrap().flatten_all()?.to_vec1::<f32>()?[0];
    let w_fm = tensors.get("w_fm").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let v_fm = tensors.get("V_fm").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_fm_out = tensors.get("fm_out").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let p_als = tensors.get("P_als").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let q_als = tensors.get("Q_als").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_r_als = tensors.get("R_als_pred").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let n = 32;
    let d = 4;
    let k = 2;

    // 1. FM Forward CPU & Auto
    let kdev = KernelDevice::Cpu;
    let cpu_fm = t08::factorization_machine_cpu(&x, n, d, k, w0, &w_fm, &v_fm);
    let auto_fm = t08::factorization_machine_auto(Some(&kdev), &x, n, d, k, w0, &w_fm, &v_fm);
    for i in 0..n {
        assert!((cpu_fm[i] - golden_fm_out[i]).abs() < 1e-4);
        assert!((auto_fm[i] - golden_fm_out[i]).abs() < 1e-4);
    }

    // 2. Low-rank Matrix Factorization Prediction CPU
    let cpu_r_als = t08::matrix_factorization_predict_cpu(&p_als, &q_als, 8, 8, k);
    for i in 0..64 {
        assert!((cpu_r_als[i] - golden_r_als[i]).abs() < 1e-4);
    }

    Ok(())
}

#[test]
fn test_09_hidden_markov_models_kalman_ukf_and_particle_filters_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "02_classical_and_probabilistic_ml/09_hidden_markov_models_kalman_ukf_and_particle_filters.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Kalman Filter
    let f = tensors.get("F").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let h = tensors.get("H").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let q = tensors.get("Q").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let r = tensors.get("R").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let y = tensors.get("Y").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_x_filt = tensors.get("x_filt").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_p_diag = tensors.get("P_filt_diag").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let x0 = vec![0.0f32, 0.0f32];
    let p0 = vec![1.0f32, 0.0f32, 0.0f32, 1.0f32];

    let (cpu_x_filt, cpu_p_diag) = t09::kalman_filter_forward_cpu(&y, 8, &f, &h, &q, &r, &x0, &p0, 2, 1);
    for i in 0..16 {
        assert!((cpu_x_filt[i] - golden_x_filt[i]).abs() < 1e-4);
        assert!((cpu_p_diag[i] - golden_p_diag[i]).abs() < 1e-4);
    }

    // 2. HMM Forward & Viterbi
    let log_pi = tensors.get("log_pi").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let log_a = tensors.get("log_A").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let log_b = tensors.get("log_B").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let obs_seq = tensors.get("obs_seq").unwrap().flatten_all()?.to_vec1::<i64>()?;
    let golden_log_prob = tensors.get("hmm_log_prob").unwrap().flatten_all()?.to_vec1::<f32>()?[0];
    let golden_viterbi = tensors.get("viterbi_path").unwrap().flatten_all()?.to_vec1::<i64>()?;

    let cpu_ll = t09::hmm_forward_log_prob_cpu(&obs_seq, 8, &log_pi, &log_a, &log_b, 2, 3);
    assert!((cpu_ll - golden_log_prob).abs() < 1e-4);

    let cpu_viterbi = t09::hmm_viterbi_decode_cpu(&obs_seq, 8, &log_pi, &log_a, &log_b, 2, 3);
    for i in 0..8 {
        assert_eq!(cpu_viterbi[i], golden_viterbi[i]);
    }

    // 3. UKF Sigma Points CPU & Auto
    let mu_ukf = tensors.get("mu_ukf").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let sigma_ukf = tensors.get("Sigma_ukf").unwrap().flatten_all()?.to_vec1::<f32>()?;
    let golden_sigma_pts = tensors.get("sigma_pts").unwrap().flatten_all()?.to_vec1::<f32>()?;

    let kdev = KernelDevice::Cpu;
    let cpu_sigma = t09::generate_ukf_sigma_points_cpu(&mu_ukf, &sigma_ukf, 2, 1e-3, 2.0, 0.0);
    let auto_sigma = t09::generate_ukf_sigma_points_auto(Some(&kdev), &mu_ukf, &sigma_ukf, 2, 1e-3, 2.0, 0.0);

    for i in 0..10 {
        assert!((cpu_sigma[i] - golden_sigma_pts[i]).abs() < 1e-4);
        assert!((auto_sigma[i] - golden_sigma_pts[i]).abs() < 1e-4);
    }

    Ok(())
}



