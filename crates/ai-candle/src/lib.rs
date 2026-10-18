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
}

