# AI-Examples: The Exhaustive, Production-Grade PyTorch 2 Catalog

A mathematically rigorous, completely deterministic, zero-external-download catalog of **243 Jupyter Notebooks** across **18 categories**, covering the entire landscape of modern artificial intelligence, deep learning, optimization, numerical integration, quantization, safety, brain surgery, interpretability, and systems engineering.

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
   - **Cell 6 (Code):** Standardized benchmark table (median latency, peak RAM/VRAM, FLOPs, arithmetic intensity).
   - **Cell 7 (Code):** Anti-pattern & runnable failure contrast (exposing a critical failure mode and its fix).
   - **Cell 8 (Markdown):** Key takeaways checklist and bidirectional navigation footer.
4. **Multi-Axis Indexing:**
   - [INDEX_BY_GOAL.md](INDEX_BY_GOAL.md): 12 Task-Oriented Decision Matrices + 40-Symptom Diagnostic & Troubleshooting Guide.
   - [GLOSSARY_AND_NOTATION.md](GLOSSARY_AND_NOTATION.md): Unified mathematical symbols, 27 numeric bit layouts, and asymptotic complexity reference.
5. **Decoupled Cross-Framework Parity (Stretch Goals):**
   - Pure JAX/Flax/Equinox implementations in `stretch/jax/`.
   - Pure Candle (Rust) implementations in `stretch/candle/`.
   - Cross-framework numerical parity verified to within $10^{-5}$ via `safetensors`.

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

## Master Taxonomic Index (243 Notebooks across 18 Categories)

### [00. Foundations & Classical Machine Learning](notebooks/00_foundations/) (10 Notebooks)
- Core PyTorch 2, Views/Strides/Memory Layouts, Wirtinger Complex Autograd, Functional `torch.func`, Hooks, Meta Device, Feature Engineering, Pipelines, Specialized Tensors, and Numerical Stability.

### [01. Learning Paradigms & Taxonomic Dichotomies](notebooks/01_paradigms_and_dichotomies/) (12 Notebooks)
- Generative vs Discriminative, Tree vs Kernel vs Neural, Discrete vs Continuous Latents, Deterministic vs Stochastic, Parametric vs Nonparametric, AI by Purpose (SPO+), Input/Output Spaces (BLT/6D Rotations), Latent Computation (Coconut/JEPA), Non-Euclidean Geometries (Poincaré/SPD), L0–L5 Decision Hierarchy, 5 Statistical Schools (Bayesian/Frequentist/MDL/PAC-Bayes/E-Values), and Online Regret vs Batch IID.

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
