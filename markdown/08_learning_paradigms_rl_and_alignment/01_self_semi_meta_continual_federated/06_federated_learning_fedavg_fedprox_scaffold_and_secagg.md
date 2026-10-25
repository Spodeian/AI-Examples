# Federated Learning: FedAvg, FedProx, SCAFFOLD, and Secure Aggregation

## 1. Mathematical Foundations & Lineage

Federated Learning (FL) enables decentralized edge devices (smartphones, hospitals, IoT sensors) to collaboratively train a shared global model without centralizing private data. The central objective is:
$$\min_{w \in \mathbb{R}^D} f(w) = \sum_{k=1}^K \frac{n_k}{N} F_k(w)$$
where $F_k(w) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(w; x_{k,i}, y_{k,i})$ represents the local loss on client $k$'s private partition.

Key mathematical breakthroughs:
1. **FedAvg (McMahan et al., 2017)**: Iteratively runs $E$ steps of local SGD on client devices and performs weighted parameter averaging on the coordination server.
2. **FedProx (Li et al., 2020)**: Tackles statistical non-IID data distribution and systems heterogeneity by augmenting local objectives with a proximal regularization term $\frac{\mu}{2}\|w - w^t\|_2^2$, providing provable convergence under partial work completion.
3. **SCAFFOLD (Karimireddy et al., 2020)**: Demonstrates that under non-IID partitions, client gradients point in disparate directions ("client drift"), causing FedAvg to diverge. SCAFFOLD utilizes control variates to estimate and cancel client drift.
4. **SecAgg (Bonawitz et al., 2017)**: Cryptographically protects client updates from an honest-but-curious server using zero-sum pairwise additive masks ($s_{i,j} = -s_{j,i}$) that cancel out upon aggregation.

---

## 2. Federated Averaging (FedAvg) & Statistical Non-IID Drift

In round $t$:
1. The server broadcasts global model $w_t$ to selected client subset $S_t \subseteq \{1, \dots, K\}$.
2. Each client $k \in S_t$ initializes $w_k^{(0)} = w_t$ and runs $E$ local epochs of SGD:
   $$w_k^{(\tau+1)} = w_k^{(\tau)} - \eta \nabla F_k(w_k^{(\tau)})$$
3. The server aggregates client model parameters weighted by sample counts:
   $$w_{t+1} = \sum_{k \in S_t} \frac{n_k}{N_{S_t}} w_k^{(E)}$$

When client data partitions are non-IID ($P_i(x, y) \neq P_j(x, y)$), local optima $w_k^* = \arg\min F_k(w)$ drift far away from the global optimum $w^* = \arg\min f(w)$, leading to catastrophic convergence slowdown.

---

## 3. Heterogeneous Proximal Optimization (FedProx)

To prevent local models from drifting excessively far from the global broadcast under heterogeneous local computation budgets, FedProx introduces a proximal constraint to local objectives:
$$\min_w h_k(w; w^t) = F_k(w) + \frac{\mu}{2} \|w - w^t\|_2^2$$

The corresponding gradient step is:
$$w \leftarrow w - \eta \left( \nabla F_k(w) + \mu (w - w^t) \right)$$
The proximal operator dampens aggressive local updates in noisy directions, stabilizing convergence on non-IID partitions.

---

## 4. Variance-Reduced Control Variates (SCAFFOLD)

SCAFFOLD explicitly models client drift through client-specific control variates $c_i$ and a global control variate $c = \frac{1}{K} \sum_{i=1}^K c_i$:
- Local client update step:
  $$y \leftarrow y - \eta_l \left( g_i(y) - c_i + c \right)$$
  where $-c_i + c$ acts as a drift-correction vector shifting local gradient $g_i(y)$ toward the global gradient direction.
- Client control variate update:
  $$c_i^+ = c_i - c + \frac{1}{K \eta_l} (w^t - y_K)$$
  $$\Delta c_i = c_i^+ - c_i$$
- Server aggregation:
  $$w^{t+1} = w^t + \frac{\eta_g}{|S|} \sum_{i \in S} (y_K^i - w^t), \quad c \leftarrow c + \frac{1}{K} \sum_{i \in S} \Delta c_i$$

---

## 5. Cryptographic Privacy & Secure Aggregation (SecAgg)

To prevent the central server or an eavesdropper from reconstructing private client training data from individual gradient updates, SecAgg introduces pairwise random masking.

For each pair of participating clients $i < j$:
- Clients agree on a shared random mask $s_{i,j} \sim \mathcal{U}(\mathbb{Z}_q^D)$ using Diffie-Hellman key exchange.
- Client $i$ adds $s_{i,j}$; client $j$ subtracts $s_{i,j}$:
  $$y_i = w_i + \sum_{j > i} s_{i,j} - \sum_{j < i} s_{j,i} \pmod q$$
- Upon receiving all masked vectors $y_1, \dots, y_K$, the server computes:
  $$\sum_{i=1}^K y_i = \sum_{i=1}^K w_i + \sum_{i=1}^K \left( \sum_{j > i} s_{i,j} - \sum_{j < i} s_{j,i} \right) = \sum_{i=1}^K w_i$$
The masks cancel identically, revealing the exact sum $\sum w_i$ without exposing any individual $w_i$.

---

## 6. Cross-Framework Implementations & Numerical Parity

Every component in this topic is validated across PyTorch 2.x, JAX, and Candle against seed-42 golden tensors in `06_federated_learning_fedavg_fedprox_scaffold_and_secagg.safetensors`:
1. `linear_anchor`: System-wide parity check.
2. `fedavg_aggregate`: Weighted parameter aggregation.
3. `fedprox_gradient_step`: Proximal gradient update step.
4. `scaffold_correct_gradient`: Control variate drift subtraction and global correction.
5. `secagg_aggregate_sum`: Exact mask cancellation upon aggregation.

All implementations achieve transitive numerical parity $< 10^{-5}$.

---

## 7. Hardware Acceleration & WGSL Compute Shaders

The FedProx proximal step is accelerated on edge devices using WebGPU compute shader `FEDPROX_PROXIMAL_STEP_WGSL`:
```wgsl
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= cfg.d) {
        return;
    }
    let diff = w_local[i] - w_global[i];
    let prox_g = grad_local[i] + cfg.mu * diff;
    w_next[i] = w_local[i] - cfg.eta * prox_g;
}
```
This fused kernel computes differences, applies scaling, and writes updated weights directly in local GPU memory.

---

## 8. Performance Profiling, Anti-Patterns & Misuse Guardrails

### Performance Profiling
1. **Communication vs. Computation Tradeoff**: Running more local epochs $E$ reduces expensive communication rounds between edge devices and the server, but exacerbates client drift on non-IID datasets.
2. **SCAFFOLD Memory Footprint**: SCAFFOLD doubles client state storage by maintaining local control variate $c_i \in \mathbb{R}^D$ alongside model weights $w \in \mathbb{R}^D$.

### Misuse Guardrails & Anti-Patterns
1. **BatchNorm Divergence in Non-IID FL**: Using Batch Normalization across heterogeneous clients causes catastrophic divergence because running mean and variance statistics diverge across clients. Always substitute GroupNorm or LayerNorm in federated architectures.
2. **Dropping Clients in SecAgg**: If client $j$ drops out before sending $y_j$, the pairwise masks $s_{i,j}$ sent by participating clients do not cancel out, corrupting the aggregate sum. Production SecAgg uses Shamir's Secret Sharing to reconstruct dropped masks.
3. **Unweighted Aggregation**: Averaging models uniformly ($\frac{1}{K} \sum w_k$) rather than weighting by sample size ($\frac{n_k}{N}$) introduces significant bias toward small, noisy edge nodes.
