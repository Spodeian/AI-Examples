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

#[path = "../../../candle/02_classical_and_probabilistic_ml"]
pub mod c02_classical_and_probabilistic_ml {
    #[path = "01_generalized_linear_models_and_proximal_lasso.rs"]
    pub mod t01_generalized_linear_models_and_proximal_lasso;

    #[path = "02_support_vector_machines_and_kernel_methods.rs"]
    pub mod t02_support_vector_machines_and_kernel_methods;

    #[path = "03_knn_prototype_and_locally_weighted_regression.rs"]
    pub mod t03_knn_prototype_and_locally_weighted_regression;

    #[path = "04_clustering_kmeans_spectral_and_gmm_em.rs"]
    pub mod t04_clustering_kmeans_spectral_and_gmm_em;

    #[path = "05_dimensionality_reduction_pca_lda_ica_tsne_umap.rs"]
    pub mod t05_dimensionality_reduction_pca_lda_ica_tsne_umap;

    #[path = "06_naive_bayes_bayesian_regression_and_gaussian_processes.rs"]
    pub mod t06_naive_bayes_bayesian_regression_and_gaussian_processes;

    #[path = "07_differentiable_trees_node_soft_gbdt_and_gam_nam.rs"]
    pub mod t07_differentiable_trees_node_soft_gbdt_and_gam_nam;

    #[path = "08_matrix_factorization_als_nmf_bpr_and_fm.rs"]
    pub mod t08_matrix_factorization_als_nmf_bpr_and_fm;

    #[path = "09_hidden_markov_models_kalman_ukf_and_particle_filters.rs"]
    pub mod t09_hidden_markov_models_kalman_ukf_and_particle_filters;

    #[path = "10_survival_analysis_and_contextual_bandits.rs"]
    pub mod t10_survival_analysis_and_contextual_bandits;
}

#[path = "../../../candle/03_building_blocks_and_losses"]
pub mod c03_building_blocks_and_losses {
    #[path = "01_activation_functions_zoo.rs"]
    pub mod t01_activation_functions_zoo;

    #[path = "02_normalization_layers_complete.rs"]
    pub mod t02_normalization_layers_complete;

    #[path = "03_weight_initialization_and_mup_scaling.rs"]
    pub mod t03_weight_initialization_and_mup_scaling;

    #[path = "04_positional_and_structural_encodings.rs"]
    pub mod t04_positional_and_structural_encodings;

    #[path = "05_pooling_and_downsampling_operators.rs"]
    pub mod t05_pooling_and_downsampling_operators;

    #[path = "06_regularization_and_training_stabilizers.rs"]
    pub mod t06_regularization_and_training_stabilizers;

    #[path = "07_losses_regression_classification_and_barron_robust.rs"]
    pub mod t07_losses_regression_classification_and_barron_robust;

    #[path = "08_losses_metric_contrastive_and_structured_vision.rs"]
    pub mod t08_losses_metric_contrastive_and_structured_vision;

    #[path = "09_losses_divergences_information_bottleneck_and_coding_rate.rs"]
    pub mod t09_losses_divergences_information_bottleneck_and_coding_rate;

    #[path = "10_losses_optimal_transport_wasserstein_sinkhorn_and_mmd.rs"]
    pub mod t10_losses_optimal_transport_wasserstein_sinkhorn_and_mmd;

    #[path = "11_losses_proper_scoring_evidential_and_differentiable_ranking.rs"]
    pub mod t11_losses_proper_scoring_evidential_and_differentiable_ranking;

    #[path = "12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting.rs"]
    pub mod t12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting;

    #[path = "13_differentiable_sorting_ranking_topk_permutahedron_and_imle.rs"]
    pub mod t13_differentiable_sorting_ranking_topk_permutahedron_and_imle;
}

#[path = "../../../candle/04_training_methods_and_optimisers"]
pub mod c04_training_methods_and_optimisers {
    #[path = "01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.rs"]
    pub mod t01_first_order_and_adaptive_optimisers;

    #[path = "01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.rs"]
    pub mod t02_modern_sign_and_factored_optimisers;

    #[path = "01_gradient_and_curvature_optimisers/03_large_batch_and_layerwise_optimisers.rs"]
    pub mod t03_large_batch_and_layerwise_optimisers;

    #[path = "01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.rs"]
    pub mod t04_orthogonal_and_preconditioned_matrix_optimisers;

    #[path = "01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.rs"]
    pub mod t05_second_order_and_curvature_optimisers;

    #[path = "01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.rs"]
    pub mod t06_schedule_free_and_parameter_free_optimisers;

    #[path = "01_gradient_and_curvature_optimisers/07_learning_rate_schedulers_and_lr_finder.rs"]
    pub mod t07_learning_rate_schedulers_and_lr_finder;

    #[path = "02_constrained_pareto_and_blackbox_optimisation/01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.rs"]
    pub mod t08_constrained_proximal_frank_wolfe_admm_and_mirror_descent;

    #[path = "02_constrained_pareto_and_blackbox_optimisation/02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery.rs"]
    pub mod t09_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery;

    #[path = "02_constrained_pareto_and_blackbox_optimisation/03_coordinate_alternating_projections_and_majorization_minimization.rs"]
    pub mod t10_coordinate_alternating_projections_and_majorization_minimization;

    #[path = "02_constrained_pareto_and_blackbox_optimisation/04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn.rs"]
    pub mod t11_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn;

    #[path = "02_constrained_pareto_and_blackbox_optimisation/05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed.rs"]
    pub mod t12_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed;

    #[path = "02_constrained_pareto_and_blackbox_optimisation/06_sharpness_aware_swad_and_weight_averaging_dynamics.rs"]
    pub mod t13_sharpness_aware_swad_and_weight_averaging_dynamics;

    #[path = "02_constrained_pareto_and_blackbox_optimisation/07_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa.rs"]
    pub mod t14_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa;

    #[path = "03_training_dynamics_self_optimising_non_backprop/01_grokking_phase_transitions_edge_of_stability_and_implicit_bias.rs"]
    pub mod t15_grokking_phase_transitions_edge_of_stability_and_implicit_bias;

    #[path = "03_training_dynamics_self_optimising_non_backprop/02_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity.rs"]
    pub mod t16_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity;

    #[path = "03_training_dynamics_self_optimising_non_backprop/03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting.rs"]
    pub mod t17_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting;

    #[path = "03_training_dynamics_self_optimising_non_backprop/04_self_optimising_test_time_adaptation_tent_memo_star_and_spin.rs"]
    pub mod t18_self_optimising_test_time_adaptation_tent_memo_star_and_spin;

    #[path = "03_training_dynamics_self_optimising_non_backprop/05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian.rs"]
    pub mod t19_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian;

    #[path = "03_training_dynamics_self_optimising_non_backprop/06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck.rs"]
    pub mod t20_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck;

    #[path = "03_training_dynamics_self_optimising_non_backprop/07_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse.rs"]
    pub mod t21_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse;

    #[path = "03_training_dynamics_self_optimising_non_backprop/08_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr.rs"]
    pub mod t22_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr;

    #[path = "03_training_dynamics_self_optimising_non_backprop/09_curriculum_self_paced_learning_and_dataset_distillation_mtt.rs"]
    pub mod t23_curriculum_self_paced_learning_and_dataset_distillation_mtt;
}



