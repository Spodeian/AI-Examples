//! # AI-Kernel Crate (`ai-kernel`)
//!
//! Dual-engine hardware kernel track providing both native CPU SIMD/vectorized kernels
//! and WebGPU WGSL compute shader pipelines with automatic CPU fallback.

pub mod context;
#[cfg(feature = "wgpu")]
pub mod wgpu_util;

pub use context::{KernelDevice, WgpuContext};

#[path = "../../../kernels/00_foundations"]
pub mod c00_foundations {
    #[path = "01_linear_neuron_and_invertible_transforms.rs"]
    pub mod t01_linear_neuron_and_invertible_transforms;

    #[path = "02_tensors_views_strides_and_memory_layouts.rs"]
    pub mod t02_tensors_views_strides_and_memory_layouts;

    #[path = "03_autograd_mechanics_complex_wirtinger_and_custom_functions.rs"]
    pub mod t03_autograd_mechanics_complex_wirtinger_and_custom_functions;

    #[path = "04_forward_ad_and_functional_transforms_torch_func.rs"]
    pub mod t04_forward_ad_and_functional_transforms_torch_func;

    #[path = "05_hooks_and_saved_tensor_hooks.rs"]
    pub mod t05_hooks_and_saved_tensor_hooks;

    #[path = "06_nn_module_internals_parametrizations_and_meta_device.rs"]
    pub mod t06_nn_module_internals_parametrizations_and_meta_device;

    #[path = "07_feature_engineering_scalers_and_tabular_preprocessing.rs"]
    pub mod t07_feature_engineering_scalers_and_tabular_preprocessing;

    #[path = "08_reproducible_preprocessing_pipelines_and_determinism.rs"]
    pub mod t08_reproducible_preprocessing_pipelines_and_determinism;

    #[path = "09_nested_ragged_and_specialized_sparse_tensors.rs"]
    pub mod t09_nested_ragged_and_specialized_sparse_tensors;

    #[path = "10_floating_point_arithmetic_conditioning_and_numerical_stability.rs"]
    pub mod t10_floating_point_arithmetic_conditioning_and_numerical_stability;

    #[path = "11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.rs"]
    pub mod t11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp;
}

#[path = "../../../kernels/01_paradigms_and_dichotomies"]
pub mod c01_paradigms_and_dichotomies {
    #[path = "01_generative_vs_discriminative_classifiers.rs"]
    pub mod t01_generative_vs_discriminative_classifiers;

    #[path = "02_tree_vs_kernel_vs_neural_inductive_biases.rs"]
    pub mod t02_tree_vs_kernel_vs_neural_inductive_biases;

    #[path = "03_discrete_vs_continuous_latents_and_gradient_estimators.rs"]
    pub mod t03_discrete_vs_continuous_latents_and_gradient_estimators;

    #[path = "04_deterministic_vs_stochastic_and_mixture_density_networks.rs"]
    pub mod t04_deterministic_vs_stochastic_and_mixture_density_networks;

    #[path = "05_parametric_vs_nonparametric_and_neural_processes.rs"]
    pub mod t05_parametric_vs_nonparametric_and_neural_processes;

    #[path = "06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.rs"]
    pub mod t06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic;

    #[path = "07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.rs"]
    pub mod t07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds;

    #[path = "08_computation_spaces_latent_reasoning_and_jepa.rs"]
    pub mod t08_computation_spaces_latent_reasoning_and_jepa;

    #[path = "09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd.rs"]
    pub mod t09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd;

    #[path = "10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.rs"]
    pub mod t10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games;

    #[path = "11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.rs"]
    pub mod t11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes;

    #[path = "12_inductive_vs_transductive_and_online_regret_vs_batch_iid.rs"]
    pub mod t12_inductive_vs_transductive_and_online_regret_vs_batch_iid;

    #[path = "13_information_geometry_solomonoff_mdl_and_maxent_free_energy.rs"]
    pub mod t13_information_geometry_solomonoff_mdl_and_maxent_free_energy;
}

#[path = "../../../kernels/03_building_blocks_and_losses"]
pub mod c03_building_blocks_and_losses {
    #[path = "01_activation_functions_zoo.rs"]
    pub mod t01_activation_functions_zoo;

    #[path = "02_normalization_layers_complete.rs"]
    pub mod t02_normalization_layers_complete;

    #[path = "03_weight_initialization_and_mup_scaling.rs"]
    pub mod t03_weight_initialization_and_mup_scaling;

    #[path = "04_positional_and_structural_encodings.rs"]
    pub mod t04_positional_and_structural_encodings;
}

