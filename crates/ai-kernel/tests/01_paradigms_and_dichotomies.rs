use ai_kernel::{
    c01_paradigms_and_dichotomies::{
        t01_generative_vs_discriminative_classifiers as t01,
        t02_tree_vs_kernel_vs_neural_inductive_biases as t02,
        t03_discrete_vs_continuous_latents_and_gradient_estimators as t03,
        t04_deterministic_vs_stochastic_and_mixture_density_networks as t04,
        t05_parametric_vs_nonparametric_and_neural_processes as t05,
        t06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic as t06,
        t07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds as t07,
        t08_computation_spaces_latent_reasoning_and_jepa as t08,
    },
    KernelDevice,
};
use candle_core::{safetensors::load, Device};

mod common;

#[test]
fn test_01_generative_vs_discriminative_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path("01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.safetensors");
    let tensors = load(path, &device)?;

    let x = tensors.get("test_input").expect("test_input").to_vec2::<f32>()?;
    let y_gold = tensors.get("test_output").expect("test_output").to_vec2::<f32>()?;
    let w = tensors.get("layers.weight").expect("weight").to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t01::discriminative_forward(&KernelDevice::Cpu, &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "CPU error at {i}: {calc} vs {gold}");
    }

    let y_auto = t01::discriminative_forward(&KernelDevice::auto(), &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Auto error at {i}: {calc} vs {gold}");
    }

    // GDA logits test against layers.gda_logits
    if let Some(gda_logits) = tensors.get("layers.gda_logits") {
        let gda_gold: Vec<f32> = gda_logits.to_vec2::<f32>()?.into_iter().flatten().collect();
        let w_gda: Vec<f32> = tensors.get("layers.gda_weight").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
        let b_gda = tensors.get("layers.gda_bias").unwrap().to_vec1::<f32>()?;

        let calc_gda = t01::predict_logits_gda(&KernelDevice::Cpu, &x_flat, &w_gda, &b_gda, batch, in_dim, out_dim);
        for (i, (&calc, &gold)) in calc_gda.iter().zip(gda_gold.iter()).enumerate() {
            assert!((calc - gold).abs() < 1e-5, "GDA error at {i}: {calc} vs {gold}");
        }
    }

    Ok(())
}

#[test]
fn test_02_tree_vs_kernel_vs_neural_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path("01_paradigms_and_dichotomies/02_tree_vs_kernel_vs_neural_inductive_biases.safetensors");
    let tensors = load(path, &device)?;

    let x = tensors.get("test_input").expect("test_input").to_vec2::<f32>()?;
    let y_gold = tensors.get("test_output").expect("test_output").to_vec2::<f32>()?;
    let w = tensors.get("layers.weight").expect("weight").to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t02::linear_forward(&KernelDevice::Cpu, &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "CPU error at {i}: {calc} vs {gold}");
    }

    let y_auto = t02::linear_forward(&KernelDevice::auto(), &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Auto error at {i}: {calc} vs {gold}");
    }

    // Verify soft decision tree
    let w_route: Vec<f32> = tensors.get("layers.tree_routing_weights").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let thresh = tensors.get("layers.tree_thresholds").unwrap().to_vec1::<f32>()?;
    let temp = tensors.get("layers.tree_temperature").unwrap().to_vec1::<f32>()?[0];
    let w_leaf: Vec<f32> = tensors.get("layers.tree_leaf_weights").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let tree_gold: Vec<f32> = tensors.get("layers.tree_output").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();

    let tree_calc = t02::soft_decision_tree_cpu(&x_flat, &w_route, &thresh, temp, &w_leaf, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in tree_calc.iter().zip(tree_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Tree error at {i}: {calc} vs {gold}");
    }

    // Verify RBF kernel ridge
    let support: Vec<f32> = tensors.get("layers.kernel_support").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let dual_coeffs: Vec<f32> = tensors.get("layers.kernel_dual_coeffs").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let gamma = tensors.get("layers.kernel_gamma").unwrap().to_vec1::<f32>()?[0];
    let k_bias = tensors.get("layers.kernel_bias").unwrap().to_vec1::<f32>()?;
    let k_gold: Vec<f32> = tensors.get("layers.kernel_output").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();

    let k_calc = t02::rbf_kernel_ridge_cpu(&x_flat, &support, &dual_coeffs, gamma, &k_bias, batch, support.len() / in_dim, in_dim, out_dim);
    for (i, (&calc, &gold)) in k_calc.iter().zip(k_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Kernel ridge error at {i}: {calc} vs {gold}");
    }

    // Verify MLP classifier
    let mlp_w1: Vec<f32> = tensors.get("layers.mlp_w1").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let mlp_b1 = tensors.get("layers.mlp_b1").unwrap().to_vec1::<f32>()?;
    let mlp_w2: Vec<f32> = tensors.get("layers.mlp_w2").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let mlp_b2 = tensors.get("layers.mlp_b2").unwrap().to_vec1::<f32>()?;
    let mlp_gold: Vec<f32> = tensors.get("layers.mlp_output").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();

    let mlp_calc = t02::mlp_classifier_cpu(&x_flat, &mlp_w1, &mlp_b1, &mlp_w2, &mlp_b2, batch, in_dim, mlp_b1.len(), out_dim);
    for (i, (&calc, &gold)) in mlp_calc.iter().zip(mlp_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "MLP error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

#[test]
fn test_03_discrete_vs_continuous_latents_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path("01_paradigms_and_dichotomies/03_discrete_vs_continuous_latents_and_gradient_estimators.safetensors");
    let tensors = load(path, &device)?;

    let x = tensors.get("test_input").expect("test_input").to_vec2::<f32>()?;
    let y_gold = tensors.get("test_output").expect("test_output").to_vec2::<f32>()?;
    let w = tensors.get("layers.weight").expect("weight").to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t03::linear_forward(&KernelDevice::Cpu, &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "CPU error at {i}: {calc} vs {gold}");
    }

    let y_auto = t03::linear_forward(&KernelDevice::auto(), &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Auto error at {i}: {calc} vs {gold}");
    }

    // Verify Gaussian Reparameterization and KL
    let w_mu: Vec<f32> = tensors.get("layers.w_mu").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_mu = tensors.get("layers.b_mu").unwrap().to_vec1::<f32>()?;
    let w_logvar: Vec<f32> = tensors.get("layers.w_logvar").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_logvar = tensors.get("layers.b_logvar").unwrap().to_vec1::<f32>()?;
    let eps: Vec<f32> = tensors.get("layers.eps").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let z_gold: Vec<f32> = tensors.get("layers.continuous_z").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let kl_gold = tensors.get("layers.kl_divergence").unwrap().to_vec1::<f32>()?;

    let latent_dim = b_mu.len();
    let (mu, logvar) = t03::gaussian_encode_cpu(&x_flat, &w_mu, &b_mu, &w_logvar, &b_logvar, batch, in_dim, latent_dim);
    let z_calc = t03::gaussian_reparameterize_cpu(&mu, &logvar, &eps, batch, latent_dim);
    for (i, (&calc, &gold)) in z_calc.iter().zip(z_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "z error at {i}: {calc} vs {gold}");
    }

    let kl_calc = t03::gaussian_kl_divergence_cpu(&mu, &logvar, batch, latent_dim);
    for (i, (&calc, &gold)) in kl_calc.iter().zip(kl_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "KL error at {i}: {calc} vs {gold}");
    }

    // Verify Gumbel Softmax
    let w_logits: Vec<f32> = tensors.get("layers.w_logits").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_logits = tensors.get("layers.b_logits").unwrap().to_vec1::<f32>()?;
    let gumbel_noise: Vec<f32> = tensors.get("layers.gumbel_noise").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let temp = tensors.get("layers.temperature").unwrap().to_vec1::<f32>()?[0];
    let gumbel_gold: Vec<f32> = tensors.get("layers.gumbel_soft").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();

    let gumbel_calc = t03::gumbel_softmax_sample_cpu(&x_flat, &w_logits, &b_logits, &gumbel_noise, temp, batch, in_dim, b_logits.len());
    for (i, (&calc, &gold)) in gumbel_calc.iter().zip(gumbel_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Gumbel error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

#[test]
fn test_04_deterministic_vs_stochastic_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path("01_paradigms_and_dichotomies/04_deterministic_vs_stochastic_and_mixture_density_networks.safetensors");
    let tensors = load(path, &device)?;

    let x = tensors.get("test_input").expect("test_input").to_vec2::<f32>()?;
    let y_gold = tensors.get("test_output").expect("test_output").to_vec2::<f32>()?;
    let w = tensors.get("layers.weight").expect("weight").to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t04::linear_forward(&KernelDevice::Cpu, &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "CPU error at {i}: {calc} vs {gold}");
    }

    let y_auto = t04::linear_forward(&KernelDevice::auto(), &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Auto error at {i}: {calc} vs {gold}");
    }

    // Verify deterministic forward
    let w_h: Vec<f32> = tensors.get("layers.w_h").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_h = tensors.get("layers.b_h").unwrap().to_vec1::<f32>()?;
    let w_det: Vec<f32> = tensors.get("layers.w_det").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_det = tensors.get("layers.b_det").unwrap().to_vec1::<f32>()?;
    let y_det_gold: Vec<f32> = tensors.get("layers.y_det").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();

    let y_det_calc = t04::deterministic_forward_cpu(&x_flat, &w_h, &b_h, &w_det, &b_det, batch, in_dim, b_h.len(), b_det.len());
    for (i, (&calc, &gold)) in y_det_calc.iter().zip(y_det_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Det forward error at {i}: {calc} vs {gold}");
    }

    // Verify MDN log probs
    let w_pi: Vec<f32> = tensors.get("layers.w_pi").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_pi = tensors.get("layers.b_pi").unwrap().to_vec1::<f32>()?;
    let w_mu: Vec<f32> = tensors.get("layers.w_mu").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_mu = tensors.get("layers.b_mu").unwrap().to_vec1::<f32>()?;
    let w_sigma: Vec<f32> = tensors.get("layers.w_sigma").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_sigma = tensors.get("layers.b_sigma").unwrap().to_vec1::<f32>()?;
    let sample_y = tensors.get("layers.sample_y").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let log_probs_gold = tensors.get("layers.mdn_log_probs").unwrap().to_vec1::<f32>()?;

    let (pi, mu, sigma) = t04::mdn_forward_cpu(&x_flat, &w_h, &b_h, &w_pi, &b_pi, &w_mu, &b_mu, &w_sigma, &b_sigma, batch, in_dim, b_h.len(), b_pi.len());
    let log_probs_calc = t04::mdn_log_prob_cpu(&pi, &mu, &sigma, &sample_y, batch, b_pi.len());
    for (i, (&calc, &gold)) in log_probs_calc.iter().zip(log_probs_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-4, "MDN log prob error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

#[test]
fn test_05_parametric_vs_nonparametric_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path("01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.safetensors");
    let tensors = load(path, &device)?;

    let x = tensors.get("test_input").expect("test_input").to_vec2::<f32>()?;
    let y_gold = tensors.get("test_output").expect("test_output").to_vec2::<f32>()?;
    let w = tensors.get("layers.weight").expect("weight").to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t05::linear_forward(&KernelDevice::Cpu, &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "CPU error at {i}: {calc} vs {gold}");
    }

    let y_auto = t05::linear_forward(&KernelDevice::auto(), &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Auto error at {i}: {calc} vs {gold}");
    }

    // Verify parametric MLP
    let w_mlp1: Vec<f32> = tensors.get("layers.w_mlp1").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_mlp1 = tensors.get("layers.b_mlp1").unwrap().to_vec1::<f32>()?;
    let w_mlp2: Vec<f32> = tensors.get("layers.w_mlp2").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_mlp2 = tensors.get("layers.b_mlp2").unwrap().to_vec1::<f32>()?;
    let x_target = tensors.get("layers.x_target").unwrap().to_vec3::<f32>()?;
    let y_param_gold: Vec<f32> = tensors.get("layers.y_parametric").unwrap().to_vec3::<f32>()?.into_iter().flatten().flatten().collect();

    let x_target_flat: Vec<f32> = x_target.into_iter().flatten().flatten().collect();
    let y_param_calc = t05::parametric_forward_cpu(&x_target_flat, &w_mlp1, &b_mlp1, &w_mlp2, &b_mlp2, batch * 4, 2, b_mlp1.len(), 1);
    for (i, (&calc, &gold)) in y_param_calc.iter().zip(y_param_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Parametric error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

#[test]
fn test_06_ai_by_purpose_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path("01_paradigms_and_dichotomies/06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.safetensors");
    let tensors = load(path, &device)?;

    let x = tensors.get("test_input").expect("test_input").to_vec2::<f32>()?;
    let y_gold = tensors.get("test_output").expect("test_output").to_vec2::<f32>()?;
    let w = tensors.get("layers.weight").expect("weight").to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t06::linear_forward(&KernelDevice::Cpu, &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "CPU error at {i}: {calc} vs {gold}");
    }

    let y_auto = t06::linear_forward(&KernelDevice::auto(), &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Auto error at {i}: {calc} vs {gold}");
    }

    // Verify Cost predictor, regret and SPO+ loss
    let w_cost1: Vec<f32> = tensors.get("layers.w_cost1").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_cost1 = tensors.get("layers.b_cost1").unwrap().to_vec1::<f32>()?;
    let w_cost2: Vec<f32> = tensors.get("layers.w_cost2").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_cost2 = tensors.get("layers.b_cost2").unwrap().to_vec1::<f32>()?;
    let c_pred_gold: Vec<f32> = tensors.get("layers.c_pred").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let c_star: Vec<f32> = tensors.get("layers.c_star").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let regret_gold = tensors.get("layers.decision_regret").unwrap().to_vec1::<f32>()?;
    let spo_gold = tensors.get("layers.spo_plus_loss").unwrap().to_vec1::<f32>()?;

    let c_pred_calc = t06::cost_predictor_forward_cpu(&x_flat, &w_cost1, &b_cost1, &w_cost2, &b_cost2, batch, in_dim, b_cost1.len(), b_cost2.len());
    for (i, (&calc, &gold)) in c_pred_calc.iter().zip(c_pred_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "c_pred error at {i}: {calc} vs {gold}");
    }

    let regret_calc = t06::decision_regret_cpu(&c_pred_calc, &c_star, batch, b_cost2.len());
    for (i, (&calc, &gold)) in regret_calc.iter().zip(regret_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "regret error at {i}: {calc} vs {gold}");
    }

    let spo_calc = t06::spo_plus_loss_cpu(&c_pred_calc, &c_star, batch, b_cost2.len());
    for (i, (&calc, &gold)) in spo_calc.iter().zip(spo_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "SPO+ error at {i}: {calc} vs {gold}");
    }

    // Verify Causal
    let t_vec: Vec<f32> = tensors.get("layers.treatment_t").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let w_tau: Vec<f32> = tensors.get("layers.w_tau").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_tau = tensors.get("layers.b_tau").unwrap().to_vec1::<f32>()?;
    let w_mu0: Vec<f32> = tensors.get("layers.w_mu0").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_mu0 = tensors.get("layers.b_mu0").unwrap().to_vec1::<f32>()?;
    let y_fac_gold: Vec<f32> = tensors.get("layers.y_factual").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();

    let y_fac_calc = t06::causal_forward_cpu(&x_flat, &t_vec, &w_tau, &b_tau, &w_mu0, &b_mu0, batch, in_dim);
    for (i, (&calc, &gold)) in y_fac_calc.iter().zip(y_fac_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Causal error at {i}: {calc} vs {gold}");
    }

    // Verify Symbolic EQL
    let w_eql: Vec<f32> = tensors.get("layers.w_eql").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_eql = tensors.get("layers.b_eql").unwrap().to_vec1::<f32>()?;
    let y_sym_gold: Vec<f32> = tensors.get("layers.y_symbolic").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();

    let y_sym_calc = t06::symbolic_forward_cpu(&x_flat, &w_eql, &b_eql, batch);
    for (i, (&calc, &gold)) in y_sym_calc.iter().zip(y_sym_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Symbolic error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

#[test]
fn test_07_input_output_spaces_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path("01_paradigms_and_dichotomies/07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.safetensors");
    let tensors = load(path, &device)?;

    let x = tensors.get("test_input").expect("test_input").to_vec2::<f32>()?;
    let y_gold = tensors.get("test_output").expect("test_output").to_vec2::<f32>()?;
    let w = tensors.get("layers.weight").expect("weight").to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    let y_cpu = t07::linear_forward(&KernelDevice::Cpu, &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "CPU error at {i}: {calc} vs {gold}");
    }

    let y_auto = t07::linear_forward(&KernelDevice::auto(), &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Auto error at {i}: {calc} vs {gold}");
    }

    // Verify 6D rotation matrix
    let rot_gold: Vec<f32> = tensors.get("layers.rot_matrices").unwrap().to_vec3::<f32>()?.into_iter().flatten().flatten().collect();
    let rot_calc = t07::rotation_6d_to_matrix_cpu(&x_flat, batch);
    for (i, (&calc, &gold)) in rot_calc.iter().zip(rot_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "6D rotation error at {i}: {calc} vs {gold}");
    }

    // Verify Fourier coordinate encoding
    let coords: Vec<f32> = tensors.get("layers.coords_2d").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let fourier_gold: Vec<f32> = tensors.get("layers.coords_fourier").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let fourier_calc = t07::fourier_coordinate_encode_cpu(&coords, batch, 2, 4);
    for (i, (&calc, &gold)) in fourier_calc.iter().zip(fourier_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Fourier coords error at {i}: {calc} vs {gold}");
    }

    // Verify ViT patch embeddings
    let patches: Vec<f32> = tensors.get("layers.patches").unwrap().to_vec3::<f32>()?.into_iter().flatten().flatten().collect();
    let w_patch: Vec<f32> = tensors.get("layers.w_patch").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_patch = tensors.get("layers.b_patch").unwrap().to_vec1::<f32>()?;
    let patch_embed_gold: Vec<f32> = tensors.get("layers.patch_embeddings").unwrap().to_vec3::<f32>()?.into_iter().flatten().flatten().collect();

    let patch_embed_calc = t07::patch_embedding_forward_cpu(&patches, &w_patch, &b_patch, batch, 4, 4, b_patch.len());
    for (i, (&calc, &gold)) in patch_embed_calc.iter().zip(patch_embed_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Patch embedding error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

#[test]
fn test_08_computation_spaces_cpu_and_auto() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let path = common::get_weights_path("01_paradigms_and_dichotomies/08_computation_spaces_latent_reasoning_and_jepa.safetensors");
    let tensors = load(path, &device)?;

    let x = tensors.get("test_input").expect("test_input").to_vec2::<f32>()?;
    let y_gold = tensors.get("test_output").expect("test_output").to_vec2::<f32>()?;
    let w = tensors.get("layers.weight").expect("weight").to_vec2::<f32>()?;
    let b = tensors.get("layers.bias").expect("bias").to_vec1::<f32>()?;

    let batch = x.len();
    let in_dim = w.len();
    let out_dim = w[0].len();

    let x_flat: Vec<f32> = x.into_iter().flatten().collect();
    let w_flat: Vec<f32> = w.into_iter().flatten().collect();
    let y_gold_flat: Vec<f32> = y_gold.into_iter().flatten().collect();

    // 1. Canonical Linear Forward
    let y_cpu = t08::linear_forward(&KernelDevice::Cpu, &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_cpu.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "CPU error at {i}: {calc} vs {gold}");
    }

    let y_auto = t08::linear_forward(&KernelDevice::auto(), &x_flat, &w_flat, &b, batch, in_dim, out_dim);
    for (i, (&calc, &gold)) in y_auto.iter().zip(y_gold_flat.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Auto error at {i}: {calc} vs {gold}");
    }

    // 2. Continuous Latent Reasoning
    let w_proj: Vec<f32> = tensors.get("layers.w_proj").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_proj = tensors.get("layers.b_proj").unwrap().to_vec1::<f32>()?;
    let w_z: Vec<f32> = tensors.get("layers.w_z").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let w_anc: Vec<f32> = tensors.get("layers.w_anc").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_z = tensors.get("layers.b_z").unwrap().to_vec1::<f32>()?;
    let traj_gold: Vec<f32> = tensors.get("layers.latent_trajectories").unwrap().to_vec3::<f32>()?.into_iter().flatten().flatten().collect();
    let final_gold: Vec<f32> = tensors.get("layers.final_thought").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();

    let dim = b_z.len();
    let steps = 3;
    let (traj_calc, final_calc) = t08::continuous_latent_reasoning_cpu(
        &x_flat, &w_proj, &b_proj, &w_z, &w_anc, &b_z, batch, in_dim, dim, steps,
    );

    for (i, (&calc, &gold)) in traj_calc.iter().zip(traj_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Traj error at {i}: {calc} vs {gold}");
    }
    for (i, (&calc, &gold)) in final_calc.iter().zip(final_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Final thought error at {i}: {calc} vs {gold}");
    }

    // 3. PonderNet Halting Hazards
    let w_h = tensors.get("layers.w_h").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
    let b_h = tensors.get("layers.b_h").unwrap().to_vec1::<f32>()?;
    let halting_gold: Vec<f32> = tensors.get("layers.halting_probs").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();

    let halting_calc = t08::ponder_halting_cpu(&traj_calc, &w_h, &b_h, batch, steps, dim);
    for (i, (&calc, &gold)) in halting_calc.iter().zip(halting_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "Halting error at {i}: {calc} vs {gold}");
    }

    // 4. JEPA Latent-Space Predictor & Loss
    let w_p1: Vec<f32> = tensors.get("layers.w_p1").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_p1 = tensors.get("layers.b_p1").unwrap().to_vec1::<f32>()?;
    let w_p2: Vec<f32> = tensors.get("layers.w_p2").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let b_p2 = tensors.get("layers.b_p2").unwrap().to_vec1::<f32>()?;
    let jepa_pred_gold: Vec<f32> = tensors.get("layers.jepa_pred").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let target_sy: Vec<f32> = tensors.get("layers.target_sy").unwrap().to_vec2::<f32>()?.into_iter().flatten().collect();
    let jepa_loss_gold = tensors.get("layers.jepa_error").unwrap().to_vec1::<f32>()?;

    let jepa_pred_calc = t08::jepa_predict_cpu(&final_calc, &w_p1, &b_p1, &w_p2, &b_p2, batch, dim);
    for (i, (&calc, &gold)) in jepa_pred_calc.iter().zip(jepa_pred_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "JEPA pred error at {i}: {calc} vs {gold}");
    }

    let jepa_loss_calc = t08::jepa_loss_cpu(&jepa_pred_calc, &target_sy, batch, dim);
    for (i, (&calc, &gold)) in jepa_loss_calc.iter().zip(jepa_loss_gold.iter()).enumerate() {
        assert!((calc - gold).abs() < 1e-5, "JEPA loss error at {i}: {calc} vs {gold}");
    }

    Ok(())
}

