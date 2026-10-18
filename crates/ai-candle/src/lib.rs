//! # AI-Examples Candle Crate (`ai-candle`)
//!
//! Systems-level Candle Rust track for the 18-category AI-Examples knowledge catalog.
//! Provides zero-Python, zero-GC, memory-safe implementations of machine learning operators
//! and architectures with full numerical parity against PyTorch and JAX references.

#[path = "../../../candle/00_foundations"]
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

#[path = "../../../candle/01_paradigms_and_dichotomies"]
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
}



