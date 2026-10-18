# Index by Goal: Decision Matrices & Diagnostic Troubleshooting Guide

[![Catalog Progress](https://img.shields.io/badge/Progress-3%2F263%20Topics-blue.svg)](README.md)
[![Parity Invariant](https://img.shields.io/badge/Parity-4--Way%20Transitive%20%3C1e--5-success.svg)](tests/test_cross_track_parity.py)
[![Rust Engine](https://img.shields.io/badge/Candle%20Rust-Passing-success.svg)](crates/ai-candle)

This guide routes you directly to the correct notebook and architectural pattern based on **what you are trying to achieve** or **what specific failure mode you are experiencing**.

---

## Part 1: 12 Task-Oriented Decision Matrices

### Matrix 1: Supervised vs Self-Supervised vs RL vs Meta-Learning
| Data / Feedback Regime | Primary Objective | Failure Mode if Chosen Incorrectly | Recommended Notebook |
| :--- | :--- | :--- | :--- |
| Abundant labels $(x, y)$ | Empirical Risk Minimization | Overfitting / Spurious correlation | `01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.ipynb` |
| Massive unlabelled data $x$ (with negatives) | Contrastive Representation Learning | Contrastive false-negative repulsion | `08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/01_contrastive_self_supervised_simclr_byol.ipynb` |
| Massive unlabelled data $x$ (no negatives) | Non-Contrastive Decorrelation & Masking | Representation collapse to constant ($z \to c$) | `08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa.ipynb` |
| Complex verifiable reasoning tasks | Test-Time Compute Scaling & Search | Goodhart's Law reward hacking as $N \to \infty$ | `08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.ipynb` |
| Interactive scalar reward $r_t$ | Long-horizon dynamic control | High variance / Reward hacking | `08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_policy_gradients_and_ppo.ipynb` |
| Few shots per task ($k \le 5$) | Rapid task adaptation | Catastrophic forgetting / Negative transfer | `08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/03_meta_learning_maml.ipynb` |
| Streaming non-stationary data | Continual lifelong learning | Plasticity loss / Catastrophic interference | `01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.ipynb` |

### Matrix 2: Sequence Models: Dense Attention vs Linear RNN vs State Space (Mamba)
| Sequence Length ($T$) | Hardware Bottleneck | Required Inductive Bias | Recommended Architecture / Notebook |
| :--- | :--- | :--- | :--- |
| $T < 4\text{k}$ | Compute bound ($\mathcal{O}(T^2)$ feasible) | Unrestricted associative recall | `05_architectures/04_transformers_and_heads/01_flash_attention_2_and_online_softmax.ipynb` |
| $4\text{k} \le T < 32\text{k}$ | Memory bandwidth bound | Local sliding window + global tokens | `05_architectures/04_transformers_and_heads/04_sparse_attention.ipynb` |
| $T > 64\text{k}$ (Real-time audio/sensor) | KV Cache memory exhaustion ($\mathcal{O}(T)$ inference) | Continuous time state compression | `05_architectures/03_recurrent_ssm_and_memory/03_mamba_and_mamba_2_ssd.ipynb` |
| Ultra-low latency edge RNN | Parallel training + $\mathcal{O}(1)$ step state | Recurrent linear decay | `05_architectures/03_recurrent_ssm_and_memory/04_rwkv_and_linear_attention.ipynb` |

### Matrix 3: Generative Modeling: Diffusion vs Flow Matching vs VAE vs AR
| Generation Requirement | Training Stability | Sampling Steps | Recommended Notebook |
| :--- | :--- | :--- | :--- |
| Deterministic straight trajectories | Stable MSE regression | 1–10 steps (ODE/Predictor-Corrector) | `05_architectures/05_generative_models/06_rectified_flow_matching.ipynb` |
| High sample diversity | Stable score matching | 20–50 steps (DDIM / SDE) | `05_architectures/05_generative_models/05_continuous_score_sdes.ipynb` |
| Instant single-pass latent encoding | Susceptible to posterior collapse | 1 forward pass | `05_architectures/05_generative_models/01_vae_and_beta_vae.ipynb` |
| Discrete token sequences | Standard cross-entropy | $T$ autoregressive passes | `05_architectures/04_transformers_and_heads/01_flash_attention_2_and_online_softmax.ipynb` |

### Matrix 4: Numerical Precision & Quantisation Selection
| Target Environment | Available VRAM / Memory | Allowable SQNR Degradation | Optimal Format & Notebook |
| :--- | :--- | :--- | :--- |
| Training on modern GPUs (H100/B200) | Standard FP16/BF16 capacity | $< 0.1\text{ dB}$ (near zero loss) | `13_precision_and_numerics/04_fp8_formats_e4m3_vs_e5m2.ipynb` |
| Enterprise Server Serving | $50\%$ memory reduction | $< 0.5\text{ dB}$ | `14_quantisation/03_llm_ptq_algorithms/03_smoothquant_migration.ipynb` |
| Consumer GPU Inference (RTX 4090/3090) | $75\%$ memory reduction (4-bit) | $< 1.5\text{ dB}$ | `14_quantisation/03_llm_ptq_algorithms/01_gptq_second_order.ipynb` |
| Edge CPU / Microcontroller | Ultra-low power, integer arithmetic | Modest degradation acceptable | `14_quantisation/01_foundations_formats_and_observers/01_uniform_affine_int8.ipynb` |
| Matrix-Multiplication Free CPU | $90\%$ memory reduction (1.58-bit) | Re-trained weights | `14_quantisation/04_vector_lattice_qat_and_lowbit/05_ternary_bitnet_b158.ipynb` |

### Matrix 5: Optimizer Selection & Loss Surface Topology
| Parameter Scale & Model Family | Gradient Noise / Conditioning | Memory Budget | Recommended Optimizer & Notebook |
| :--- | :--- | :--- | :--- |
| Convex / Classical regression ($D < 1000$) | Ill-conditioned Hessian | Minimal | `04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_lbfgs_and_sophia.ipynb` |
| Standard Deep Networks ($< 1\text{B}$) | Non-convex, noisy minibatches | $2\times$ parameter states (first/second moments) | `04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.ipynb` |
| Large Transformers ($> 1\text{B}$) | Matrix-shaped coordinate coupling | Moderate | `04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.ipynb` |
| Memory-constrained fine-tuning | High throughput requirements | $1\times$ parameter state (sign momentum) | `04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/02_modern_sign_and_factored_optimisers.ipynb` |
| Flat minima generalization | Poor out-of-distribution robustness | $2\times$ forward passes per step | `04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/07_lookahead_and_sam.ipynb` |

### Matrix 6: Fine-Tuning & Parameter-Efficient Adaptation (PEFT)
| Downstream Hardware Constraint | Rank / Subspace Structure | Mergeability into Base Weights | Recommended Notebook |
| :--- | :--- | :--- | :--- |
| Minimal VRAM ($< 16\text{ GB}$) | Low intrinsic dimensionality ($r=8\text{--}16$) | $100\%$ zero-latency merge $W + B A$ | `15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_low_rank_adaptation_lora.ipynb` |
| Quantized base model (4-bit NF4) | Low-rank on dequantized weights | Zero-overhead edge serving | `14_quantisation/04_vector_lattice_qat_and_lowbit/06_quantized_finetuning_qlora_loftq.ipynb` |
| Decoupled Magnitude & Direction | Full rank directional steering | Exact linear merge | `15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_weight_decomposed_lora_dora.ipynb` |
| Multi-task routing without conflict | Sparse modular activation | Sparse conditional routing | `05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_switch_deepseekmoe.ipynb` |
| Multi-checkpoint ensemble | Zero-cost post-training merge | Linear / spherical interpolation | `05_architectures/01_mlp_moe_soups_and_kan/03_model_soups_and_merging.ipynb` |

### Matrix 7: Uncertainty Quantification & Calibration
| Safety-Critical Setting | Compute Budget | Statistical Guarantee | Recommended Technique & Notebook |
| :--- | :--- | :--- | :--- |
| Medical / Legal classification | 1 forward pass | Finite-sample distribution-free coverage ($1 - \alpha$) | `01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.ipynb` |
| Bayesian posterior exploration | Moderate ($5\text{--}10$ stochastic passes) | Epistemic parameter uncertainty | `02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.ipynb` |
| Single-pass epistemic priors | Minimal ($1\times$ forward pass) | Dirichlet / Normal-Inverse-Gamma | `03_building_blocks_and_losses/11_losses_proper_scoring_evidential_and_differentiable_ranking.ipynb` |
| Post-hoc confidence calibration | Negligible (single scalar optimization) | Minimum Expected Calibration Error | `notebooks/00_foundations/01_linear_neuron_and_invertible_transforms.ipynb` |

### Matrix 8: Graph & Geometric Deep Learning (5Gs)
| Domain Geometry | Symmetry Group | Invariant vs Equivariant | Recommended Notebook |
| :--- | :--- | :--- | :--- |
| Irregular relational network | Permutation group $S_n$ | Node/Graph permutation invariance | `05_architectures/06_graphs_geometric_and_inr/01_spatial_spectral_gnns.ipynb` |
| Over-squashing bottleneck | Permutation group $S_n$ | Discrete Ollivier-Ricci curvature rewiring | `05_architectures/06_graphs_geometric_and_inr/03_curvature_rewiring_sdrf.ipynb` |
| 3D Molecular coordinates | Euclidean group $SE(3)$ | Vector direction equivariance | `05_architectures/06_graphs_geometric_and_inr/02_equivariant_gnns_egnn_so3.ipynb` |
| Continuous manifold / Sphere | Fiber bundle / Sheaf | Sheaf restriction maps $\mathcal{F}_{v \unlhd e}$ | `05_architectures/06_graphs_geometric_and_inr/04_cellular_sheaf_networks.ipynb` |

### Matrix 9: Scientific Machine Learning & Differential Equations
| Physical Prior Known? | Data Availability | Constraint Enforcement | Recommended Notebook |
| :--- | :--- | :--- | :--- |
| Exact PDE differential operator | Sparse boundary points only | Soft loss penalty $\lambda \|\mathcal{N}[u]\|^2$ | `06_smooth_and_physics_informed_ai/03_physics_informed_neural_networks_pinn.ipynb` |
| Exact PDE + Conservation laws | Boundary & initial data | Exact hard-coded symplectic structure | `06_smooth_and_physics_informed_ai/04_hamiltonian_and_lagrangian_symplectic_networks.ipynb` |
| Unknown PDE operator | Rich sensor field observations | Mesh-independent continuous operator mapping | `06_smooth_and_physics_informed_ai/05_fourier_neural_operators_fno.ipynb` |
| Stiff multi-scale kinetics | Time series trajectories | Predict-Correct-Evaluate-Correct (PCEC) | `06_smooth_and_physics_informed_ai/07_predictor_corrector_ode_pde_solvers_pcec_rk45_bdf.ipynb` |

### Matrix 10: Model Surgery, Safety & Interpretability Interventions
| Intervention Objective | Intervention Stage | Mechanistic Mechanism | Recommended Notebook |
| :--- | :--- | :--- | :--- |
| Eradicate refusal / jailbreak behavior | Post-training inference | Residual stream direction subtraction | `11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents.ipynb` |
| Factual knowledge update (e.g. Paris $\to$ Rome) | Zero-retraining model edit | Rank-1 MLP second-layer projection (ROME / MEMIT) | `10_xai_and_ai_analysis_methods/06_ai_brain_surgery_knowledge_editing_rome_memit_and_stitching.ipynb` |
| Machine unlearning / hazardous knowledge erasure | Fine-tuning stage | Negative Preference Optimization (`NPO`) / RMU | `11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.ipynb` |
| Safety counter-defense against abliteration & sleeper agents | Training stage | Safety Circuit Breakers & representation entanglement | `11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents.ipynb` |
| Clean-label poisoning & trojan backdoor defense | Pre-training & Post-training | Neural Cleanse $L_1$ trigger inversion, STRIP entropy & Spectral Signatures | `11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.ipynb` |
| Provable provenance & synthetic watermarking across modalities | Generation time | Green/Red-list logit bias, Gumbel, Tree-Ring FFT & AudioSeal | `11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.ipynb` |
| Zero-shot AI-generated text/image/audio detection | Test time | Likelihood curvature (`Fast-DetectGPT`), `Binoculars` & Spectral Artifacts | `11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.ipynb` |
| Model weight protection & anti-distillation | Model distribution | Passport Layers, Honeypot Weights & Proof-of-Weights | `11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.ipynb` |
| Provable deterministic safety verification | Post-training certificate | Interval Bound Propagation (`IBP`) & CROWN linear relaxation | `11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation.ipynb` |
| Forward-invariant safety guarantees in robotics/control | Control loop | Differentiable Control Barrier Functions (`CBF-QP`) | `11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.ipynb` |
| Cryptographic verifiable inference without revealing weights | Inference verification | Finite-field arithmetic circuits & Freivalds' $O(n^2)$ check (`ZKML`) | `11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.ipynb` |
| Zero-trust private inference on encrypted data | Encrypted inference | Ring-LWE polynomial evaluation (`CKKS` Homomorphic Encryption) | `11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.ipynb` |

### Matrix 11: Mechanistic Interpretability & Circuit Discovery
| Interpretability Question | Analysis Level | Diagnostic Tool | Recommended Notebook |
| :--- | :--- | :--- | :--- |
| How much does each input feature contribute to the prediction? | Local feature level | LIME, Exact Shapley & KernelSHAP | `10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.ipynb` |
| Do explanations reflect actual model decision rules vs cosmetic heuristics? | Explanation evaluation | Faithfulness Correlation, Monotonicity & Remove-and-Retrain (`ROAR`) | `10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.ipynb` |
| Which input tokens/pixels propagate relevance without gradient shattering? | Axiomatic gradient flow | Integrated Gradients (`IG`), LRP ($\epsilon/\gamma/\alpha\beta$) & Grad-CAM++ | `10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention.ipynb` |
| Are saliency maps sensitive to model weights vs. class-agnostic edge detectors? | Model sanity checks | Adebayo Cascading Model Parameter & Data Randomization Tests | `10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention.ipynb` |
| Is a high-level human concept used in latent representations? | Concept level | Testing with Concept Activation Vectors (`TCAV`) & Concept Bottlenecks | `10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale.ipynb` |
| Which training samples caused this specific test error or hallucination? | Training data attribution | Influence Functions via LiSSA, TracIn & TRAK | `10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak.ipynb` |
| How to disentangle polysemantic superposition across layers? | Feature dictionary | Sparse Autoencoders (`L1`, `Top-K`, `JumpReLU`), Transcoders & Cross-Coders | `10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae.ipynb` |
| When does the model commit to an output token across depth? | Token-by-token layer progression | Direct Logit Attribution (`DLA`), Logit Lens & Tuned Lens | `10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae.ipynb` |
| Which attention heads and circuits mediate factual recall? | Circuit mechanism | Fast Attribution Patching ($\Delta \mathcal{L} \approx \Delta a^T \nabla_a \mathcal{L}$) & Induction Heads | `10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae.ipynb` |
| How to diagnose layer training health with zero test data? | Weight spectrum | Heavy-Tailed RMT (`WeightWatcher` power-law $\alpha$) & Neural Collapse ETF | `10_xai_and_ai_analysis_methods/08_ai_analysis_neural_collapse_superposition_and_weightwatcher_rmt.ipynb` |
| How to track internal circuit emergence across training phase transitions? | Algebraic topology & geometry | Singular Learning Theory Local Learning Coefficient (`LLC`) & TDA Betti curves | `10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc.ipynb` |

### Matrix 12: Production Serving & System Topologies
| Deployment Target | Primary Constraint | Optimization Pattern | Recommended Notebook |
| :--- | :--- | :--- | :--- |
| Single GPU LLM Serving | Memory bandwidth / KV Cache | PagedAttention + FP8 KV Cache | `14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi.ipynb` |
| Edge CPU / Mobile App | Zero Python runtime dependency | AOTInductor C++ export / ONNX | `17_deployment_low_level_and_capstones/03_standalone_cpp_deployment_aotinductor.ipynb` |
| Low-latency autoregressive generation | Arithmetic intensity of decode | Speculative Decoding with draft model | `15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_speculative_decoding_and_verification.ipynb` |
| Safe model weight exchange | Security (no pickle execution) | Zero-copy `safetensors` memory mapping | `17_deployment_low_level_and_capstones/01_safetensors_serialization_and_zero_copy_mmap.ipynb` |

---

## Part 2: 40-Symptom Diagnostic & Troubleshooting Guide

### Training Stability & Gradient Diagnostics
1. **Loss produces `NaN` or `Inf` immediately:**
   - *Cause:* Unnormalized inputs, learning rate too high, or unclipped numerical operations (`log(0)`, `exp(x)` overflow).
   - *Remedy:* Inspect `notebooks/00_foundations/01_linear_neuron_and_invertible_transforms.ipynb` for bounded transforms and `00_foundations/10_reproducibility_determinism_and_numerical_stability.ipynb`.
2. **Vanishing gradients in deep recurrent or deep residual networks:**
   - *Cause:* Repeated matrix multiplications with spectral radius $\rho(W) < 1.0$.
   - *Remedy:* See `05_architectures/03_recurrent_ssm_and_memory/01_modern_xlstm_mlstm.ipynb` and `03_building_blocks_and_losses/02_normalization_layers_complete.ipynb`.
3. **Exploding gradients during backpropagation through time:**
   - *Cause:* Unbounded Jacobians across long sequences.
   - *Remedy:* Implement norm-based gradient clipping $\nabla_\theta \leftarrow \nabla_\theta \cdot \min(1, \tau / \|\nabla_\theta\|)$; see `04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.ipynb`.
4. **Dead ReLUs (neurons never fire and have zero gradient everywhere):**
   - *Cause:* Negative bias drift pushing activations permanently into zero-slope regime.
   - *Remedy:* Switch to LeakyReLU, GELU, or SwiGLU; see `03_building_blocks_and_losses/01_activation_functions_zoo.ipynb`.
5. **Loss plateaus early at $\ln(C)$ for classification with $C$ classes:**
   - *Cause:* Random guessing baseline due to severe gradient vanishing in the classifier head or poor initialization.
   - *Remedy:* Verify initialization scaling ($\text{std} = 1/\sqrt{D}$) in `03_building_blocks_and_losses/03_weight_initialization_and_mup_scaling.ipynb`.
6. **Loss oscillates wildly with AdamW:**
   - *Cause:* Decoupled weight decay applied inappropriately to bias or LayerNorm parameters, or learning rate too high for second-moment accumulator.
   - *Remedy:* Exclude 1D tensors from weight decay; see `04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/01_first_order_and_adaptive_optimisers.ipynb`.
7. **Severe gradient noise stall in batch size 1 streaming:**
   - *Cause:* SGD variance dominates directional drift.
   - *Remedy:* Implement heavy-ball momentum and Polyak averaging; see `01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.ipynb`.
8. **Ill-conditioned Hessian causing zigzagging convergence:**
   - *Cause:* Condition number $\kappa(H) \gg 10^4$.
   - *Remedy:* Apply matrix-orthogonal updates (Muon) or K-FAC natural gradient; see `04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.ipynb`.
9. **Saddle point stagnation:**
   - *Cause:* Zero-gradient plateaus with mixed positive/negative curvature eigenvalues.
   - *Remedy:* Perturbed SGD or second-order curvature escaping; see `04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_lbfgs_and_sophia.ipynb`.
10. **Catastrophic gradient explosion when fine-tuning with LoRA:**
    - *Cause:* Unscaled LoRA delta ($\Delta W = \frac{\alpha}{r} B A$) with $\alpha \gg r$.
    - *Remedy:* Standardize $\alpha/r = 1.0$ or $2.0$; see `15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_low_rank_adaptation_lora.ipynb`.

### Generative, Latent Space & Diffusion Diagnostics
11. **VAE Posterior Collapse ($D_{KL}(q(z|x) \parallel p(z)) \to 0$):**
    - *Cause:* Autoregressive or powerful decoder ignores latent code $z$ early in training.
    - *Remedy:* Apply KL annealing schedules or free-bits $\max(\tau, D_{KL})$; see `05_architectures/05_generative_models/01_vae_and_beta_vae.ipynb`.
12. **Vector Quantized Codebook Collapse (index starvation in VQ-VAE):**
    - *Cause:* Sub-cluster of codebook embeddings monopolizes all argmin assignments.
    - *Remedy:* Apply EMA codebook updates and random codebook restart; see `05_architectures/05_generative_models/02_vq_vae_2.ipynb`.
13. **Mode collapse in Generative Adversarial Networks:**
    - *Cause:* Discriminator gradient vanishing or generator cycling between discrete modes.
    - *Remedy:* Switch to Wasserstein distance with gradient penalty or Flow Matching; see `05_architectures/05_generative_models/06_rectified_flow_matching.ipynb`.
14. **Blurry or noisy diffusion generation at step 0:**
    - *Cause:* Incorrect variance schedule $\beta_t$ or mismatch in velocity/score prediction formulation.
    - *Remedy:* Inspect $\epsilon$-prediction vs $v$-prediction parameterizations in `05_architectures/05_generative_models/04_ddpm.ipynb`.
15. **DDIM sampling error at low step counts ($N < 10$):**
    - *Cause:* Discretization truncation error in first-order Euler solver.
    - *Remedy:* Apply second-order Runge-Kutta or Predict-Correct-Evaluate-Correct (PCEC) sampling; see `05_architectures/05_generative_models/07_ode_sde_predictor_corrector_samplers_pcec_rk45.ipynb`.
16. **Classifier-Free Guidance (CFG) over-saturation / unnatural artifacts:**
    - *Cause:* Guidance scale $w > 7.5$ pushes latents outside training distribution manifold.
    - *Remedy:* Apply dynamic thresholding or mimic CFG scaling; see `05_architectures/05_generative_models/08_cfg_dynamics.ipynb`.

### Quantization & Precision Diagnostics
17. **SQNR drops below $20\text{ dB}$ after 8-bit quantization:**
    - *Cause:* Asymmetric distribution with zero-point clamping or extreme outliers.
    - *Remedy:* Use per-channel asymmetric quantization or SmoothQuant; see `14_quantisation/01_foundations_formats_and_observers/01_uniform_affine_int8.ipynb` and `14_quantisation/03_llm_ptq_algorithms/03_smoothquant_migration.ipynb`.
18. **Massive perplexity spike when quantizing to 4-bit INT4:**
    - *Cause:* Salient activation outlier channels corrupted by round-to-nearest.
    - *Remedy:* Apply AWQ outlier protection or second-order GPTQ; see `14_quantisation/03_llm_ptq_algorithms/01_gptq_second_order.ipynb` and `02_awq_outlier_protection.ipynb`.
19. **BFloat16 underflow in softmax denominators:**
    - *Cause:* BFloat16 only has 7 mantissa bits (equivalent to FP32 dynamic range but FP7 precision).
    - *Remedy:* Maintain accumulator and softmax in FP32; see `13_precision_and_numerics/01_ieee754_floating_point_and_subnormals_ftz_daz.ipynb`.
20. **FP8 training diverges during backward pass:**
    - *Cause:* E4M3 dynamic range ($[-448, 448]$) clips backward gradient scale.
    - *Remedy:* Use E5M2 for backward gradients and E4M3 for forward activations; see `13_precision_and_numerics/04_fp8_formats_e4m3_vs_e5m2.ipynb`.
21. **BitNet 1.58-bit gradient oscillation:**
    - *Cause:* STE (Straight-Through Estimator) gradient mismatch with ternary clamp.
    - *Remedy:* Inspect Absmean scaling and latent float master weights in `14_quantisation/04_vector_lattice_qat_and_lowbit/05_ternary_bitnet_b158.ipynb`.

### Architecture & Attention Diagnostics
22. **Attention memory out of memory (OOM) at sequence length $T > 8\text{k}$:**
    - *Cause:* Materialization of the full $[B, H, T, T]$ attention matrix in VRAM.
    - *Remedy:* Use tiled online softmax (FlashAttention-2); see `05_architectures/04_transformers_and_heads/01_flash_attention_2_and_online_softmax.ipynb`.
23. **Positional extrapolation failure beyond training context length:**
    - *Cause:* Absolute positional embeddings lack periodic or relative shift invariance.
    - *Remedy:* Switch to Rotary Position Embeddings (RoPE) or ALiBi; see `03_building_blocks_and_losses/04_positional_and_structural_encodings.ipynb`.
24. **KV Cache exhausts GPU memory during multi-turn chat:**
    - *Cause:* Linear growth of uncompressed key-value activations over sequence turns.
    - *Remedy:* Implement PagedAttention and FP8 KV quantization; see `14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi.ipynb`.
25. **MoE Expert routing collapse (all tokens route to Expert 0):**
    - *Cause:* Winner-takes-all positive feedback loop in softmax router.
    - *Remedy:* Introduce auxiliary load balancing loss $\alpha \cdot N \sum f_i P_i$; see `05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_switch_deepseekmoe.ipynb`.

### Safety, Model Surgery & Interpretability Diagnostics
26. **Refusal direction subtraction destroys general reasoning capabilities:**
    - *Cause:* Non-orthogonal refusal vector contaminated with core semantic features.
    - *Remedy:* Project refusal direction orthogonal to benign activation manifold; see `11_ai_safety_privacy_and_verification/04_abliteration_and_refusal_surgery.ipynb`.
27. **ROME factual editing corrupts unrelated knowledge associations:**
    - *Cause:* Inversion of unregularized covariance matrix $C = \sum k k^\top$.
    - *Remedy:* Increase sample covariance regularization $\lambda I$; see `10_xai_and_ai_analysis_methods/06_ai_brain_surgery_and_direct_knowledge_editing_rome_memit.ipynb`.
28. **Sparse Autoencoder (SAE) dead latents ($> 80\%$ latents never activate):**
    - *Cause:* $L_1$ sparsity penalty $\lambda$ set too high, starving dictionary atoms.
    - *Remedy:* Implement Top-K activation SAE or ghost gradients; see `10_xai_and_ai_analysis_methods/05_sparse_autoencoders_sae.ipynb`.
29. **Probe achieves $99\%$ accuracy on random representations:**
    - *Cause:* Probe capacity too high (overparameterized non-linear MLP memorizing training set).
    - *Remedy:* Use strictly linear probes with cross-validated $L_2$ regularization; see `10_xai_and_ai_analysis_methods/07_probing_and_representation_geometry.ipynb`.

### Reinforcement Learning & Control Diagnostics
30. **PPO policy entropy collapses to zero immediately:**
    - *Cause:* Policy becomes deterministic before exploring reward space.
    - *Remedy:* Increase entropy bonus coefficient $\beta_{\text{ent}}$; see `08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_policy_gradients_and_ppo.ipynb`.
31. **Q-learning values diverge to infinity:**
    - *Cause:* Overestimation bias due to maximization operator $\max_a Q(s', a)$.
    - *Remedy:* Implement Double DQN and Target Network Polyak updates; see `08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_soft_actor_critic_sac.ipynb`.
32. **Actor-critic policy updates cause sudden performance crashes:**
    - *Cause:* Step size exceeds trusted local policy manifold.
    - *Remedy:* Enforce trust region clipping $r_t(\theta) \in [1-\epsilon, 1+\epsilon]$; see `08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_policy_gradients_and_ppo.ipynb`.
33. **Imitation learning agent drifts off track after short rollout:**
    - *Cause:* Covariate shift ($p_{\text{test}}(s) \ne p_{\text{train}}(s)$) compounding over time.
    - *Remedy:* Use Action Chunking with Transformers (ACT) or Diffusion Policy; see `07_domain_tasks/13_robotics_diffusion_policy_act_and_mppi.ipynb`.

### Scientific ML & Numerical Solvers Diagnostics
34. **PINN PDE residual converges to zero while boundary error remains huge:**
    - *Cause:* Multi-objective loss imbalance between PDE residual and boundary conditions.
    - *Remedy:* Apply adaptive loss weighting (e.g. GradNorm / learning rate annealing); see `06_smooth_and_physics_informed_ai/03_physics_informed_neural_networks_pinn.ipynb`.
35. **Energy drifts upward over long integration in Hamiltonian system:**
    - *Cause:* Standard explicit Runge-Kutta fails to preserve phase space volume (symplectic form $dp \wedge dq$).
    - *Remedy:* Switch to symplectic leapfrog / Verlet integration; see `06_smooth_and_physics_informed_ai/04_hamiltonian_and_lagrangian_symplectic_networks.ipynb`.
36. **Stiff ODE solver step size shrinks to zero:**
    - *Cause:* Explicit solver stability domain exceeded by rapid eigenvalue decay.
    - *Remedy:* Use implicit Backward Differentiation Formulas (BDF) or PCEC; see `06_smooth_and_physics_informed_ai/07_predictor_corrector_ode_pde_solvers_pcec_rk45_bdf.ipynb`.

### Systems, Compilation & Memory Diagnostics
37. **`torch.compile` produces repeated recompilations (graph breaks):**
    - *Cause:* Dynamic Python control flow or data-dependent tensor shapes.
    - *Remedy:* Inspect Dynamo graph breaks with `torch._dynamo.explain()`; see `12_compilation_and_performance/01_torch_compile_and_torchdynamo.ipynb`.
38. **Pickle deserialization security alert in CI/CD pipeline:**
    - *Cause:* Loading untrusted `.pt` or `.bin` weights with `torch.load()`.
    - *Remedy:* Standardize on `safetensors.torch.load_file()`; see `17_deployment_low_level_and_capstones/01_safetensors_serialization_and_zero_copy_mmap.ipynb`.
39. **CPU inference latency bottlenecked by memory bandwidth:**
    - *Cause:* Memory-bound weight transfers dominate low-arithmetic-intensity operations.
    - *Remedy:* Fuse layers into single kernel via TorchInductor; see `12_compilation_and_performance/02_torchinductor_cpp_and_triton_codegen.ipynb`.
40. **Speculative Decoding acceptance rate falls below $20\%$:**
    - *Cause:* Draft model distribution diverges from target model or temperature is set too high.
    - *Remedy:* Calibrate draft model on target token prefix; see `15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_speculative_decoding_and_verification.ipynb`.
