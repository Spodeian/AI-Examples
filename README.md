# AI-Examples: The Exhaustive, Multi-Track AI & Deep Learning Catalog

[![Catalog Progress](https://img.shields.io/badge/Progress-17%2F263%20Topics-blue.svg)](README.md)
[![Parity Invariant](https://img.shields.io/badge/Parity-4--Way%20Transitive%20%3C1e--5-success.svg)](tests/test_cross_track_parity.py)
[![Rust Engine](https://img.shields.io/badge/Candle%20Rust-Passing-success.svg)](crates/ai-candle)

A mathematically rigorous, completely deterministic, zero-external-download catalog of **263 Topics** across **18 categories**, mirrored identically across 7 cohesive tracks:
1. `markdown/` (Textbook-Grade GFM + KaTeX + Mermaid Architectural Blueprints)
2. `notebooks/` (Strict 8-Cell Educational Concept Explorers with Inline SVG & Failure Contrasts)
3. `pytorch2/` (Production-Grade PyTorch 2.x Modules with `torch.compile` & `jaxtyping`)
4. `jax/` (Stateless Functional Equinox / JAX Implementations with XLA JIT)
5. `candle/` & `crates/ai-candle/` (Native Rust Candle Implementations with Integration Tests & Doc-Tests)
6. `kernels/` (Tiled Hardware Kernels with Pure-CPU Fallbacks)
7. `weights/` (Deterministic Reference Checkpoints enforcing Transitive $<10^{-5}$ Parity)

---

## Repository Invariants & Design Principles

1. **Pure PyTorch 2 & Native CPU Execution:** Every notebook runs deterministically on CPU in under 3 seconds using `torch>=2.2.0` (tested on PyTorch 2.10). No GPUs or cloud credits required.
2. **Zero External Network Calls:** All micro-datasets (tabular, Lorenz attractor, synthetic tokens, Burgers' PDE, particle dynamics, Erdős-Rényi graphs, SCM DAGs, control rollouts) are generated procedurally on-the-fly via the lightweight `core` package (`pip install -e .`).
3. **Strict 8-Cell Notebook Blueprint:**
   - **Cell 1 (Markdown):** Title, metadata, historical lineage, papers/citations, purpose, and an explicit **Decision Boundary: When NOT to use it**.
   - **Cell 2 (Markdown):** Rigorous mathematical formulation (LaTeX), theoretical invariants, and asymptotic $\mathcal{O}(\cdot)$ time/memory complexity.
   - **Cell 3 (Code):** Deterministic setup (`core.env.setup_env(42)`) and micro-dataset generation.
   - **Cell 4 (Code):** Production PyTorch 2 implementation (first-principles tensor math + modern PyTorch equivalents e.g., `torch.compile`, `torch.func`).
   - **Cell 5 (Code):** Quantitative verification (`torch.testing.assert_close`, `gradcheck`, numerical conservation, SQNR dB, Simplex ETF collapse).
   - **Cell 6 (Code):** Standardized benchmark table (median latency, peak RAM/VRAM, FLOPs, arithmetic intensity) and inline SVG visualization.
   - **Cell 7 (Code):** Anti-pattern & runnable failure contrast with `[!WARNING]` and programmatic failure assertion.
   - **Cell 8 (Markdown):** Key takeaways checklist, Cross-Domain Generalisation Matrix, and bidirectional navigation footer.
4. **Multi-Axis Indexing:**
   - [INDEX_BY_GOAL.md](INDEX_BY_GOAL.md): 12 Task-Oriented Decision Matrices + 40-Symptom Diagnostic & Troubleshooting Guide.
   - [GLOSSARY_AND_NOTATION.md](GLOSSARY_AND_NOTATION.md): Unified mathematical symbols, 27 numeric bit layouts, and asymptotic complexity reference.
5. **Transitive 4-Way Cross-Framework Parity:**
   - Direct reference parity against `weights/<cat>/<topic>.safetensors`:
     $$\|Y_{\text{torch}} - Y_{\text{golden}}\|_\infty < 10^{-5}, \quad \|Y_{\text{jax}} - Y_{\text{golden}}\|_\infty < 10^{-5}, \quad \|Y_{\text{candle}} - Y_{\text{golden}}\|_\infty < 10^{-5}$$
   - Verified via `pytest tests/test_cross_track_parity.py` and `cargo test -p ai-candle`.

---

## 8 Unified Generalisation Frameworks

Every notebook in this catalog connects to one or more of eight unified foundational axes:

1. **Subspace Intervention & Representation Algebra:** Activation steering, SAE sparse dictionary features, linear representation hypothesis, and geometric subspaces.
2. **Geometric Deep Learning 5Gs:** Domain-specific symmetries under Group, Grid, Groupoid, Graph, and Geodesic structures ($SE(3)$, $SO(3)$, permutation equivariance).
3. **Variational, Score, OT & Self-Supervised Divergence Geometry:** Unification of KL, Jensen-Shannon, Wasserstein $\mathcal{W}_2$, Score/Flow Matching, and the 6 Anti-Collapse SSL geometries (Contrastive repulsion, Covariance decorrelation VICReg/Barlow, Asymmetric stop-gradient SimSiam, Masked reconstruction MAE, Latent prediction JEPA, and Maximal Coding Rate MCR²).
4. **Hierarchical Decision-Making Continuum (L0–L5):** Progressive spectrum from fixed inductive bias (L0) to autonomous evolutionary meta-learning (L5).
5. **Continuous 27-Format Quantisation Lattice:** Complete precision spectrum from 1-bit binary/ternary to 16/32-bit floats, posit/logarithmic representations, and outlier preservation.
6. **Optimisation Dynamics, Energy Landscapes & Training Physics:** First/second-order preconditioning (Muon, Sophia, K-FAC), non-convex training phenomena (Edge of Stability EOS, Grokking & Grokfast, Implicit Margin Bias), and architecture/data training stabilizers ($\mu$P, DyT, WSD, NEFTune, SemDeDup).
7. **Inference Economics, Deliberation DAGs & Infinite Test-Time Scaling:** Compute-accuracy Pareto frontiers, KV prefix-tree caching (RadixAttention), Speculative decoding, and Test-Time Compute Scaling Laws (Best-of-$N$ Goodhart limits vs Verifier-Guided Tree Search & Test-Time Training TTT).
8. **Numerical Integration, Solvers & Differential Calculus:** Symplectic preservation, stiff ODE/BDF methods, adaptive RK45, and Predict-Evaluate-Correct-Evaluate (PECE/PCEC) multi-step integrators.

---

## Master Taxonomic Index (263 Topics across 18 Categories)

### [00. Foundations & Classical Machine Learning](notebooks/00_foundations/) (11 Topics)
- [x] **01. Linear Neuron and Invertible Transforms** (`.ipynb` [Notebook](notebooks/00_foundations/01_linear_neuron_and_invertible_transforms.ipynb) | [Markdown](markdown/00_foundations/01_linear_neuron_and_invertible_transforms.md) | [PyTorch 2](pytorch2/00_foundations/01_linear_neuron_and_invertible_transforms.py) | [JAX](jax/00_foundations/01_linear_neuron_and_invertible_transforms.py) | [Candle](candle/00_foundations/01_linear_neuron_and_invertible_transforms.rs) | [Kernel](kernels/00_foundations/01_linear_neuron_and_invertible_transforms_cpu.py) | [Weights](weights/00_foundations/01_linear_neuron_and_invertible_transforms.safetensors))
- [x] **02. Tensors, Views, Strides & Memory Layouts** (`.ipynb` [Notebook](notebooks/00_foundations/02_tensors_views_strides_and_memory_layouts.ipynb) | [Markdown](markdown/00_foundations/02_tensors_views_strides_and_memory_layouts.md) | [PyTorch 2](pytorch2/00_foundations/02_tensors_views_strides_and_memory_layouts.py) | [JAX](jax/00_foundations/02_tensors_views_strides_and_memory_layouts.py) | [Candle](candle/00_foundations/02_tensors_views_strides_and_memory_layouts.rs) | [Kernel](kernels/00_foundations/02_tensors_views_strides_and_memory_layouts_cpu.py) | [Weights](weights/00_foundations/02_tensors_views_strides_and_memory_layouts.safetensors))
- [x] **03. Autograd Mechanics, Complex Wirtinger & Custom Functions** (`.ipynb` [Notebook](notebooks/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.ipynb) | [Markdown](markdown/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.md) | [PyTorch 2](pytorch2/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.py) | [JAX](jax/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.py) | [Candle](candle/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.rs) | [Kernel](kernels/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions_cpu.py) | [Weights](weights/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.safetensors))
- [x] **04. Forward AD & Functional Transforms (`torch.func`)** (`.ipynb` [Notebook](notebooks/00_foundations/04_forward_ad_and_functional_transforms_torch_func.ipynb) | [Markdown](markdown/00_foundations/04_forward_ad_and_functional_transforms_torch_func.md) | [PyTorch 2](pytorch2/00_foundations/04_forward_ad_and_functional_transforms_torch_func.py) | [JAX](jax/00_foundations/04_forward_ad_and_functional_transforms_torch_func.py) | [Candle](candle/00_foundations/04_forward_ad_and_functional_transforms_torch_func.rs) | [Kernel](kernels/00_foundations/04_forward_ad_and_functional_transforms_torch_func_cpu.py) | [Weights](weights/00_foundations/04_forward_ad_and_functional_transforms_torch_func.safetensors))
- [x] **05. Hooks, Saved Tensor Offloading & Autograd Engine Internals** (`.ipynb` [Notebook](notebooks/00_foundations/05_hooks_and_saved_tensor_hooks.ipynb) | [Markdown](markdown/00_foundations/05_hooks_and_saved_tensor_hooks.md) | [PyTorch 2](pytorch2/00_foundations/05_hooks_and_saved_tensor_hooks.py) | [JAX](jax/00_foundations/05_hooks_and_saved_tensor_hooks.py) | [Candle](candle/00_foundations/05_hooks_and_saved_tensor_hooks.rs) | [Kernel](kernels/00_foundations/05_hooks_and_saved_tensor_hooks_cpu.py) | [Weights](weights/00_foundations/05_hooks_and_saved_tensor_hooks.safetensors))
- [x] **06. nn.Module Internals, Parametrizations & Meta Devices** (`.ipynb` [Notebook](notebooks/00_foundations/06_nn_module_internals_parametrizations_and_meta_device.ipynb) | [Markdown](markdown/00_foundations/06_nn_module_internals_parametrizations_and_meta_device.md) | [PyTorch 2](pytorch2/00_foundations/06_nn_module_internals_parametrizations_and_meta_device.py) | [JAX](jax/00_foundations/06_nn_module_internals_parametrizations_and_meta_device.py) | [Candle](candle/00_foundations/06_nn_module_internals_parametrizations_and_meta_device.rs) | [Kernel](kernels/00_foundations/06_nn_module_internals_parametrizations_and_meta_device_cpu.py) | [Weights](weights/00_foundations/06_nn_module_internals_parametrizations_and_meta_device.safetensors))
- [x] **07. Feature Engineering, Scalers & Tabular Preprocessing** (`.ipynb` [Notebook](notebooks/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.ipynb) | [Markdown](markdown/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.md) | [PyTorch 2](pytorch2/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.py) | [JAX](jax/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.py) | [Candle](candle/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.rs) | [Kernel](kernels/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing_cpu.py) | [Weights](weights/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.safetensors))
- [x] **08. Reproducible Preprocessing Pipelines & Determinism** (`.ipynb` [Notebook](notebooks/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.ipynb) | [Markdown](markdown/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.md) | [PyTorch 2](pytorch2/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.py) | [JAX](jax/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.py) | [Candle](candle/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.rs) | [Kernel](kernels/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism_cpu.py) | [Weights](weights/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.safetensors))
- [x] **09. Nested, Ragged & Specialized Sparse Tensors** (`.ipynb` [Notebook](notebooks/00_foundations/09_nested_ragged_and_specialized_sparse_tensors.ipynb) | [Markdown](markdown/00_foundations/09_nested_ragged_and_specialized_sparse_tensors.md) | [PyTorch 2](pytorch2/00_foundations/09_nested_ragged_and_specialized_sparse_tensors.py) | [JAX](jax/00_foundations/09_nested_ragged_and_specialized_sparse_tensors.py) | [Candle](candle/00_foundations/09_nested_ragged_and_specialized_sparse_tensors.rs) | [Kernel](kernels/00_foundations/09_nested_ragged_and_specialized_sparse_tensors_cpu.py) | [Weights](weights/00_foundations/09_nested_ragged_and_specialized_sparse_tensors.safetensors))
- [x] **10. Floating-Point Arithmetic, Conditioning & Numerical Stability** (`.ipynb` [Notebook](notebooks/00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability.ipynb) | [Markdown](markdown/00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability.md) | [PyTorch 2](pytorch2/00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability.py) | [JAX](jax/00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability.py) | [Candle](candle/00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability.rs) | [Kernel](kernels/00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability_cpu.py) | [Weights](weights/00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability.safetensors))
- [x] **11. Randomized Numerical Linear Algebra (RandNLA), Spectral Autograd & Hutch++** (`.ipynb` [Notebook](notebooks/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.ipynb) | [Markdown](markdown/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.md) | [PyTorch 2](pytorch2/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.py) | [JAX](jax/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.py) | [Candle](candle/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.rs) | [Kernel](kernels/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp_cpu.py) | [Weights](weights/00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.safetensors))



### [01. Learning Paradigms & Taxonomic Dichotomies](notebooks/01_paradigms_and_dichotomies/) (12 Notebooks)
- [x] **01. Generative vs Discriminative Classifiers** (`.ipynb` [Notebook](notebooks/01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.ipynb) | [Markdown](markdown/01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.md) | [PyTorch 2](pytorch2/01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.py) | [JAX](jax/01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.py) | [Candle](candle/01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.rs) | [Kernel](kernels/01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers_cpu.py) | [Weights](weights/01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.safetensors))
- [x] **02. Tree vs. Kernel vs. Neural Inductive Biases** (`.ipynb` [Notebook](notebooks/01_paradigms_and_dichotomies/02_tree_vs_kernel_vs_neural_inductive_biases.ipynb) | [Markdown](markdown/01_paradigms_and_dichotomies/02_tree_vs_kernel_vs_neural_inductive_biases.md) | [PyTorch 2](pytorch2/01_paradigms_and_dichotomies/02_tree_vs_kernel_vs_neural_inductive_biases.py) | [JAX](jax/01_paradigms_and_dichotomies/02_tree_vs_kernel_vs_neural_inductive_biases.py) | [Candle](candle/01_paradigms_and_dichotomies/02_tree_vs_kernel_vs_neural_inductive_biases.rs) | [Kernel](kernels/01_paradigms_and_dichotomies/02_tree_vs_kernel_vs_neural_inductive_biases_cpu.py) | [Weights](weights/01_paradigms_and_dichotomies/02_tree_vs_kernel_vs_neural_inductive_biases.safetensors))
- [x] **03. Discrete vs. Continuous Latents & Gradient Estimators** (`.ipynb` [Notebook](notebooks/01_paradigms_and_dichotomies/03_discrete_vs_continuous_latents_and_gradient_estimators.ipynb) | [Markdown](markdown/01_paradigms_and_dichotomies/03_discrete_vs_continuous_latents_and_gradient_estimators.md) | [PyTorch 2](pytorch2/01_paradigms_and_dichotomies/03_discrete_vs_continuous_latents_and_gradient_estimators.py) | [JAX](jax/01_paradigms_and_dichotomies/03_discrete_vs_continuous_latents_and_gradient_estimators.py) | [Candle](candle/01_paradigms_and_dichotomies/03_discrete_vs_continuous_latents_and_gradient_estimators.rs) | [Kernel](kernels/01_paradigms_and_dichotomies/03_discrete_vs_continuous_latents_and_gradient_estimators_cpu.py) | [Weights](weights/01_paradigms_and_dichotomies/03_discrete_vs_continuous_latents_and_gradient_estimators.safetensors))
- [x] **04. Deterministic vs. Stochastic & Mixture Density Networks** (`.ipynb` [Notebook](notebooks/01_paradigms_and_dichotomies/04_deterministic_vs_stochastic_and_mixture_density_networks.ipynb) | [Markdown](markdown/01_paradigms_and_dichotomies/04_deterministic_vs_stochastic_and_mixture_density_networks.md) | [PyTorch 2](pytorch2/01_paradigms_and_dichotomies/04_deterministic_vs_stochastic_and_mixture_density_networks.py) | [JAX](jax/01_paradigms_and_dichotomies/04_deterministic_vs_stochastic_and_mixture_density_networks.py) | [Candle](candle/01_paradigms_and_dichotomies/04_deterministic_vs_stochastic_and_mixture_density_networks.rs) | [Kernel](kernels/01_paradigms_and_dichotomies/04_deterministic_vs_stochastic_and_mixture_density_networks_cpu.py) | [Weights](weights/01_paradigms_and_dichotomies/04_deterministic_vs_stochastic_and_mixture_density_networks.safetensors))
- [x] **05. Parametric vs. Nonparametric & Neural Processes** (`.ipynb` [Notebook](notebooks/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.ipynb) | [Markdown](markdown/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.md) | [PyTorch 2](pytorch2/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.py) | [JAX](jax/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.py) | [Candle](candle/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.rs) | [Kernel](kernels/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes_cpu.py) | [Weights](weights/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.safetensors))
- [x] **06. AI by Purpose: Prediction, Prescription, Control, Combinatorial, Symbolic** (`.ipynb` [Notebook](notebooks/01_paradigms_and_dichotomies/06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.ipynb) | [Markdown](markdown/01_paradigms_and_dichotomies/06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.md) | [PyTorch 2](pytorch2/01_paradigms_and_dichotomies/06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.py) | [JAX](jax/01_paradigms_and_dichotomies/06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.py) | [Candle](candle/01_paradigms_and_dichotomies/06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.rs) | [Kernel](kernels/01_paradigms_and_dichotomies/06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic_cpu.py) | [Weights](weights/01_paradigms_and_dichotomies/06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.safetensors))
- Input/Output Spaces (BLT/6D Rotations), Latent Computation (Coconut/JEPA), Non-Euclidean Geometries (Poincaré/SPD), L0–L5 Decision Hierarchy, 5 Statistical Schools (Bayesian/Frequentist/MDL/PAC-Bayes/E-Values), and Online Regret vs Batch IID.

### [02. Classical & Probabilistic Machine Learning](notebooks/02_classical_and_probabilistic_ml/) (12 Notebooks)
- Generalized Linear Models & FISTA Lasso, Support Vector Machines & Kernels, KNN & Locally Weighted Regression, Clustering & GMM EM, PCA/LDA/t-SNE/UMAP, Gaussian Processes & SVGP, Differentiable Trees (NODE/NAM), Matrix Factorization & FM, HMM & Kalman/Particle Filters, Survival Analysis & Bandits, Factor Graphs & Variational Inference, and MCMC/HMC/SGLD/SVGD/QMC.

### [03. Foundational Building Blocks, Layers & Losses](notebooks/03_building_blocks_and_losses/) (11 Notebooks)
- Activation Zoo (SwiGLU/Snake/Rational), Normalization Layers (LayerNorm/RMSNorm/DyT), Weight Initialization & $\mu$P, Positional & Structural Encodings (RoPE/ALiBi/RWPE), Pooling Operators, Regularization & Stabilizers, Regression/Classification/Barron Losses, Metric & Contrastive Losses (InfoNCE/SigLIP/CIoU), Divergences & Information Bottleneck (VIB/MCR²), Optimal Transport (Sinkhorn/SWD), and Proper Scoring & Evidential Losses.

### [04. Training Methods & Optimizers](notebooks/04_training_methods_and_optimisers/) (20 Notebooks across 3 Subfolders)
- **01. Gradient & Curvature Optimizers (7):** First-Order/AdamW, Modern Sign/Lion/AdEMAMix, Large-Batch LARS/LAMB, Orthogonal Muon/Shampoo, Second-Order L-BFGS/Sophia, Stochastic Line Search, and Lookahead/SAM.
- **02. Constrained, Pareto & Blackbox (6):** Projected Gradient/Frank-Wolfe, Augmented Lagrangian, Multi-Objective Pareto MGDA, Evolutionary CMA-ES, Bayesian Optimization BOED, and Riemannian Optimization (Stiefel/Poincaré).
- **03. Training Dynamics, Self-Optimizing & Non-Backprop (7):** Grokking & Grokfast, Edge of Stability (EOS), Implicit Margin Bias, Hypergradient Meta-Learning, Test-Time Adaptation (TENT/MEMO), Forward-Forward Algorithm, and Zeroth-Order MeZO.

### [05. Architectures & Sequence Models](notebooks/05_architectures/) (37 Notebooks across 7 Subfolders)
- **01. MLP, MoE, Soups & KAN (5):** ResNet/MLP-Mixer, Mixture of Experts (Switch/DeepSeekMoE), Model Soups & Merging, Soup of Experts (SMEAR/BTX), and Kolmogorov-Arnold Networks (KAN).
- **02. Convolutional & Spatiotemporal (4):** Modern ConvNeXt, Depthwise Separable, Deformable Convolutions, and 3D Video Convolutions.
- **03. Recurrent, SSM & Memory (6):** Modern xLSTM/mLSTM, HiPPO/S4, Mamba & Mamba-2 (SSD), RWKV & Linear Attention, Differential Attention / RetNet, and Persistent Neural Memory (Titans).
- **04. Transformers & Heads (5):** FlashAttention-2 & Online Softmax, Multi-Head Latent Attention (MLA), KV Cache Compression, Sparse Attention, and Specialized Transformer Heads (Induction/MTP/Registers).
- **05. Generative Models (9):** VAE/Beta-VAE, VQ-VAE-2, Normalizing Flows, DDPM, Continuous Score SDEs, Rectified Flow Matching, ODE/SDE Predictor-Corrector Samplers (PCEC/RK45), CFG Dynamics, and Visual Autoregressive Models (VAR).
- **06. Graphs, Geometric & INR (5):** Spatial/Spectral GNNs, Equivariant GNNs (EGNN/SO(3)), Curvature Rewiring (SDRF), Cellular Sheaf Networks, and Implicit Neural Representations (SIREN/NeRF).
- **07. Multimodal, Neuro-Symbolic & Neuromorphic (3):** Vision-Language CLIP/Cross-Attention, Spiking Neural Networks (SNN/LIF), and Capsule Networks.

### [06. Smooth & Physics-Informed AI](notebooks/06_smooth_and_physics_informed_ai/) (10 Notebooks)
- 1-Lipschitz Networks, Input-Convex Neural Networks (ICNN), Physics-Informed Neural Networks (PINN), Hamiltonian & Lagrangian Symplectic Networks, Fourier Neural Operators (FNO), DeepONet, Predictor-Corrector ODE/PDE Solvers (PCEC/PECE/RK45/BDF), Neural ODEs & SDEs, Deep Equilibrium Models (DEQ), and OptNet Constrained Optimization Layers.

### [07. Domain Tasks, Vector Indexing & Embodied AI](notebooks/07_domain_tasks/) (13 Notebooks)
- Vision Object Detection (DETR), Semantic Segmentation, Masked Autoencoding (MAE), Byte-Level Language Modeling (BLT), Time Series Forecasting, Tabular Deep Learning (FT-Transformer), Audio Transformers, Graph Link Prediction, Information Retrieval & Bi-Encoders, Vector Indexing (HNSW/IVF-PQ), Differentiable Search (TIGER), Recommendation DLRM, and Robotics Control (Diffusion Policy, ACT, MPPI).

### [08. Learning Paradigms, RL & Alignment](notebooks/08_learning_paradigms_rl_and_alignment/) (19 Notebooks across 3 Subfolders)
- **01. Self, Semi, Meta, Continual, Federated (7):** Contrastive Self-Supervised (SimCLR/BYOL), Semi-Supervised FixMatch, Meta-Learning MAML, Continual EWC, Federated Learning FedAvg/FedProx, Self-Training & Pseudo-Labeling, and Open-World Out-of-Distribution Detection.
- **02. Reinforcement Learning & World Models (7):** Policy Gradients & PPO, Soft Actor-Critic (SAC), Offline RL (IQL/CQL), Model-Based RL (Dreamer RSSM), GFlowNets, Hierarchical RL & POMDP, and Multi-Agent RL (QMIX/CFR).
- **03. LLM Alignment & Reasoning (5):** Direct Preference Optimization (DPO/KTO), Process-Supervised Reward Models (PRM), Tree of Thoughts (ToT), Direct Nash Optimization (DNO), and Constitutional AI.

### [09. Causal AI & Evolutionary Search](notebooks/09_causal_ai_and_evolutionary_search/) (6 Notebooks)
- Structural Causal Models & NOTEARS, Double Machine Learning & R-Learner, Invariant Risk Minimization (IRM), Evolutionary Strategies & CMA-ES, Quality Diversity (MAP-Elites/DQD), and Neural Architecture Search (DARTS/PBT).

### [10. XAI & AI Analysis Methods](notebooks/10_xai_and_ai_analysis_methods/) (11 Notebooks)
- Local Explainability (LIME/SHAP), Integrated Gradients & DeepLIFT, Concept Attribution (TCAV/CBM), Influence Functions & TRAK, Sparse Autoencoders (SAE), AI Brain Surgery & Direct Knowledge Editing (ROME/MEMIT/Model Stitching), Probing & Representation Geometry, Layer Dynamics (Logit Lens/Tuned Lens), CKA Representation Alignment, Neural Collapse & Superposition, and Bradley-Terry-Luce & IRT Evaluation.

### [11. AI Safety, Privacy & Verification](notebooks/11_ai_safety_privacy_and_verification/) (9 Notebooks)
- Adversarial Robustness (PGD/AutoAttack), Certified Neural Verification (IBP/CROWN), Jailbreak Red-Teaming (GCG/PAIR), Abliteration & Refusal Surgery, Safety Circuit Breakers & Representation Entanglement, Machine Unlearning & Privacy Scrubbing, Safe Reinforcement Learning (CBF), Differential Privacy (DP-SGD/PATE), and Model Watermarking & Fingerprinting.

### [12. Compilation & Performance](notebooks/12_compilation_and_performance/) (7 Notebooks)
- `torch.compile` & TorchDynamo, TorchInductor C++ / Triton Codegen, AOTInductor Standalone Export, Custom Autograd Kernel Fusion, GPU Memory Management & Caching Allocator, Model Profiling (Kineto/Roofline), and Operator Benchmarking & Micro-benchmarks.

### [13. Precision & Numerics](notebooks/13_precision_and_numerics/) (6 Notebooks)
- IEEE 754 Floating Point & Subnormals (FTZ/DAZ), Stochastic Computing & Bitstream Arithmetic, Posit Number Systems & Tapered Arithmetic, FP8 Formats (E4M3 vs E5M2), Sub-Byte Floating Point (FP4/FP6), and Microscaling Formats (OCP MXFP).

### [14. Quantisation](notebooks/14_quantisation/) (26 Notebooks across 5 Subfolders)
- **01. Foundations & Formats (4):** Uniform Affine INT8, Logarithmic/PoT/Posit, Non-Uniform NF4/Lloyd-Max, and Dynamic Observers/Calibrators.
- **02. PT2E & Bitpacking (3):** PyTorch 2 Export Quantization (PT2E), TorchAO & Tensor Subclasses, and Bitpacking / SIMD LUT Kernels.
- **03. LLM PTQ Algorithms (7):** GPTQ Second-Order, AWQ Outlier Protection, SmoothQuant Migration, LLM.int8() Outlier Decomposition, QuaRot / SpinQuant, OmniQuant / HQQ, and Sparse-Quantized SpQR / SqueezeLLM.
- **04. Vector, Lattice, QAT & Low-Bit (6):** Vector Quantization & Product Quantization, Lattice E8 & QTIP, Quantization-Aware Training (QAT/LSQ+), Extreme Binary Networks (XNOR-Net/Bi-LLM), Ternary BitNet 1.58-bit, and Quantized Fine-Tuning (QLoRA/LoftQ).
- **05. KV Cache, Attention & Synthesis (6):** KV Cache Quantization (KIVI/KVQuant), Quantized Attention Kernels (SageAttention), Low-Precision Optimizers (8-bit Adam), Distributed Gradient Quantization (QSGD/1-Bit Adam), Mixed-Precision Integer Linear Programming (HAWQ), and the 27-Format Head-to-Head Benchmark.

### [15. Compression, PEFT & Speculative Inference](notebooks/15_compression_peft_and_speculative_inference/) (15 Notebooks across 3 Subfolders)
- **01. Pruning, Sparsity & Distillation (5):** Magnitude & Structured Pruning, Dynamic Sparse Attention, Low-Rank Decomposition (SVD/Tensor Trains), Knowledge Distillation, and Data Pruning.
- **02. Adaptive KV, Speculative & Merging (5):** Adaptive KV Eviction (H2O/Scissorhands), Speculative Decoding & Verification, Medusa & Multi-Token Speculation, Model Cascades & Frugal Routers (FrugalGPT), and Deliberation Graphs (Graph-of-Thoughts).
- **03. Parameter-Efficient Fine-Tuning (5):** Low-Rank Adaptation (LoRA), Weight-Decomposed LoRA (DoRA), Principal Singular Adaptation (PiSSA), Orthonormal Fine-Tuning (BOFT), and Prefix/Prompt Tuning.

### [16. Distributed & Scalable](notebooks/16_distributed_and_scalable/) (8 Notebooks)
- Distributed Data Parallel (DDP), Distributed Tensor (DTensor), Fully Sharded Data Parallel (FSDP2), Tensor Parallelism (TP), Context Parallelism (Ring/Ulysses), Pipeline Parallelism (1F1B/DualPipe), Sequence Packing & Dynamic Batching, and Fault-Tolerant Checkpointing.

### [17. Deployment, Low-Level & Capstones](notebooks/17_deployment_low_level_and_capstones/) (11 Notebooks)
- Safetensors Serialization & Zero-Copy Mmap, ONNX Runtime & Graph Optimization, Standalone C++ Deployment (AOTInductor), Structured Generation (JSON FSM/Min-P), Prefill-Decode Disaggregation, Custom Triton Kernels, Liger Kernel Integration, End-to-End Edge Capstone, End-to-End Server Capstone, Scientific Computing Capstone, and Autonomous Agent Capstone.

---

## Tooling & Architecture Workbench (`tools/`)

### AI Architect Workbench (`tools/ai_architect_workbench/`)
An interactive visual and programmatic workbench for designing custom AI architectures:
- **Mix & Match:** Compose arbitrary combinations of Input/Output modalities (Text, Image, Video, Audio, Graph, Tabular, 3D, Robot Actions), backbones (MLA, Mamba-2 SSD, ConvNeXt, KAN), normalization, precision formats (FP8, BitNet b1.58, MXFP), and training/alignment setups (Muon, WSD, GRPO, PRM MCTS).
- **Hardware-Aware Diagnostic & Recommender Engine:** Automatically cross-references the 12 Decision Matrices in `INDEX_BY_GOAL.md` to flag invalid/pathological configurations and recommend Pareto-optimal FLOP/VRAM improvements across target hardware (MCU/Edge, Mobile NPU, Consumer GPU, Data Center Cluster).
- **Code Synthesis:** Exports ready-to-run, verified PyTorch 2 code with built-in `torch.compile` and quantization support.

---

## License & Citation
Licensed under the Apache 2.0 License. Designed for education, research, and production AI engineering.
