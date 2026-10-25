# Robotics and Embodied AI: Diffusion Policy, ACT, and Action Chunking

## 1. The Embodied AI Revolution: Physical Robot Manipulation

Modern imitation learning for robotic manipulation (e.g., bi-manual manipulation, mobile manipulation, peg-insertion, and folding) has evolved from single-step Markovian policies $a_t = \pi(o_t)$ to trajectory-level generative modeling. Single-step policies suffer from compounding errors: minor deviations from demonstration distributions cascade into fatal task failures. Moreover, human teleoperated demonstrations exhibit high multimodality: an operator may navigate around an obstacle to the left or right with equal validity, causing mean-seeking regression models to output physically invalid averages through obstacles.

## 2. Action Chunking with Ensembling (ACT)

Action Chunking with Ensembling (ACT, Zhao et al.) predicts a sequence of future actions simultaneously:
- Given observation history $o_t$ (multi-camera RGB streams and joint positions), the policy outputs an action chunk of length $H$:
  $$A_t = [a_t, a_{t+1}, \dots, a_{t+H-1}] \in \mathbb{R}^{H \times D_a}$$
- **Temporal Ensembling:** To prevent jerky transitions between adjacent chunks and enforce smooth execution, the robot continuously queries new chunks at every timestep $t$. Actions predicted for time $t$ from past overlapping chunks $k \in \{0, \dots, K-1\}$ are blended using exponential weighting:
  $$w_k = \exp(-m \cdot k), \quad \bar{a}_t = \frac{\sum_{k=0}^{K-1} w_k a_{t, k}}{\sum_{k=0}^{K-1} w_k}$$
  where $m$ controls the recency bias. This temporal filter produces compliant, highly stable end-effector trajectories.

## 3. Conditional VAEs (CVAE) for Multimodal Demonstrations

To capture distinct execution styles and multimodal behaviors without mode collapse, ACT trains as a Conditional VAE:
- **Encoder:** During training, an encoder network maps the ground-truth action trajectory $A$ and observation $o$ to latent parameters $\mu_z, \log \sigma_z^2 \in \mathbb{R}^{D_z}$.
- **Prior Regularization:** The latent space is regularized toward standard normal $\mathcal{N}(0, I)$ via Kullback-Leibler divergence:
  $$\mathcal{L}_{\text{KL}} = D_{\text{KL}}\left(\mathcal{N}(\mu, \operatorname{diag}(\sigma^2)) \parallel \mathcal{N}(0, I)\right) = -\frac{1}{2} \sum_{j=1}^{D_z} \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$
- **Decoder:** A Transformer decoder cross-attends between observation tokens and latent style vector $z$ to reconstruct the trajectory chunk $A$ via L1/Huber loss.

## 4. Diffusion Policy: Visuomotor Control as Denoising Diffusion

Diffusion Policy (Chi et al.) reformulates robot policy learning as conditional score-based diffusion over action trajectories:
- Rather than predicting actions directly via regression, a continuous noise-prediction network $\epsilon_\theta(A^k, s, k)$ iteratively denoises Gaussian noise $A^K \sim \mathcal{N}(0, I)$ into a valid action chunk $A^0$.
- By representing the policy as the gradient field of energy distributions, Diffusion Policy natively models:
  - Multimodal action distributions without mode collapse.
  - Arbitrary complex joint bounds and discontinuous manifolds.
  - High-dimensional output spaces (e.g., dual 7-DoF arms plus grippers: $D_a = 16, H = 16$).

## 5. DDPM and DDIM Reverse Denoising Steps

The reverse Denoising Diffusion Probabilistic Model (DDPM) transition from step $k$ to $k-1$ is computed analytically:

$$A^{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( A^k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}} \epsilon_\theta(A^k, s, k) \right) + \sigma_k z$$

where $\alpha_k = 1 - \beta_k$, $\bar{\alpha}_k = \prod_{s=1}^k \alpha_s$, and $z \sim \mathcal{N}(0, I)$. For real-time low-latency execution, Denoising Diffusion Implicit Models (DDIM) skip steps, executing full action generation in 4 to 10 iterations.

## 6. Robot Control Envelopes and Safety Clamping

Raw policy outputs must never violate physical actuator limits. Commanded joint velocities or end-effector deltas $\Delta x = [\Delta x, \Delta y, \Delta z, \Delta r, \Delta p, \Delta y]$ are clipped against safety boundaries:

$$a_{\text{safe}} = \operatorname{clamp}(a, a_{\min}, a_{\max})$$

Velocity and acceleration clamping prevents motor burnout, cable snap, and workspace boundary breaches.

## 7. Real-Time Hardware Deployment and Latency Constraints

Robotic control loops strictly require deterministic update rates (typically 20 Hz to 50 Hz):
- **Asynchronous Execution:** Inference runs on an isolated GPU worker thread producing action chunks, while a real-time CPU thread executes temporal ensembling and sends joint commands via EtherCAT or CAN bus.
- Kernel fusion in temporal ensembling and DDPM denoising minimizes host-to-device synchronization latency.

## 8. Cross-Track Verification and Parity

All robotics operators are validated across the 7-track curriculum against seed-42 golden safetensors:
- **PyTorch 2.x:** Reference implementations of exponential temporal ensembling, DDPM action denoising step, CVAE KL divergence regularization, and physical safety velocity clamping.
- **JAX:** Functional implementation using `jnp.arange`, vectorized DDPM reverse step algebra, and `jnp.clip`, verified to $< 10^{-5}$ error.
- **Candle:** Native Rust implementation with broadcast tensor scaling, analytical Gaussian KL reduction, and explicit safety bound clipping, passing `cargo test -p ai-candle --test 07_domain_tasks test_13`.
- **WGSL Kernel:** Fused action chunk temporal reduction compute shader computing blended joint commands without intermediate buffer allocations.
