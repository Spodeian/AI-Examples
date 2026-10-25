# Cryptographic Verifiable AI: ZKML, Freivalds' Algorithm, and Homomorphic Encryption (CKKS)

## 1. Mathematical Foundations & Lineage

As deep learning models are outsourced to untrusted cloud servers or decentralized networks, two fundamental cryptographic challenges arise: **Integrity** (verifying that the server evaluated the correct model on the client's input without tampering) and **Confidentiality** (performing inference on encrypted data without ever decrypting it):
1. **Probabilistic Matrix Verification (Freivalds 1977)**: Established that verifying a matrix product $A \times B \stackrel{?}{=} C$ can be completed in randomized $\mathcal{O}(n^2)$ time with bounded soundness error $\le 1/2$, drastically reducing the cost of proving matrix operations.
2. **Interactive Proofs & Sumcheck Protocol (Lund et al. 1992; Thaler 2013)**: Provided the algebraic backbone for verifying multivariate polynomial summations over finite fields in $\mathcal{O}(\text{poly}(d, \log N))$ verifier time.
3. **CryptoNets & Neural Homomorphic Encryption (Gilad-Bachrach et al. 2016)**: First demonstrated neural network inference over encrypted data using Ring Learning With Errors (RLWE) homomorphic encryption with low-degree polynomial activations.
4. **CKKS Homomorphic Encryption (Cheon, Kim, Kim & Song 2017)**: Introduced approximate homomorphic arithmetic over complex and real vectors with SIMD slot packing and rescaling, forming the foundation of modern privacy-preserving deep learning.
5. **Zero-Knowledge Machine Learning (ZKML — Kang et al. 2022, 2024)**: Enabled non-interactive zero-knowledge proofs (zk-SNARKs / Halo2 / PlonK) of private neural network execution, compiling quantized fixed-point layers into finite-field arithmetic circuits $\mathbb{F}_p$.

---

## 2. Freivalds' Randomized Matrix Verification

Given matrices $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, and claimed product $C \in \mathbb{R}^{M \times N}$:
Standard matrix multiplication requires $\mathcal{O}(M K N) \approx \mathcal{O}(n^3)$ operations. A resource-constrained verifier cannot afford $\mathcal{O}(n^3)$ compute.

### Algorithm
1. Sample a random binary vector $r \in \{0, 1\}^N$ (or uniformly from finite field $\mathbb{F}_p^N$).
2. Compute associative vector products right-to-left:
   $$v_1 = B \cdot r \in \mathbb{R}^K, \quad v_2 = A \cdot v_1 \in \mathbb{R}^M, \quad v_3 = C \cdot r \in \mathbb{R}^M$$
3. Evaluate the residual vector:
   $$v = v_2 - v_3 = A(Br) - Cr$$
4. **Decision Rule**:
   - If $v = \mathbf{0}$, accept $C = AB$.
   - If $v \ne \mathbf{0}$, reject $C \ne AB$ with certainty.

### Complexity & Soundness
- Time Complexity: $\mathcal{O}(M K + K N + M N) = \mathcal{O}(n^2)$ arithmetic operations.
- Soundness Error: If $C \ne AB$, then $\mathbb{P}(A(Br) = Cr) \le \frac{1}{2}$ for $r \in \{0, 1\}^N$ (and $\le \frac{1}{|\mathbb{F}_p|}$ over finite fields). Repeating across $k$ independent random vectors bounds error below $2^{-k}$.

---

## 3. ZKML: Arithmetic Circuits and Fixed-Point Arithmetic

Zero-knowledge proof systems operate over finite fields $\mathbb{F}_p$ (e.g. the BN254 scalar field where $p \approx 2^{254}$). Continuous IEEE-754 floating-point operations cannot be represented natively:
1. **Fixed-Point Quantization**: Continuous real values $x \in \mathbb{R}$ are scaled by $S = 2^B$ and rounded to integers:
   $$x_{\text{int}} = \text{round}(x \cdot S), \quad W_{\text{int}} = \text{round}(W \cdot S)$$
2. **Linear Accumulation**:
   $$z_{\text{int}} = x_{\text{int}} \cdot W_{\text{int}} + b_{\text{int}} \cdot S$$
3. **Rescaling / Bit-Shift Lookup**:
   $$y_{\text{int}} = \left\lfloor \frac{z_{\text{int}}}{S} \right\rfloor$$
Because floor division is non-algebraic in $\mathbb{F}_p$, ZKML systems prove range constraints via polynomial lookup arguments (`plookup`).

---

## 4. Homomorphic Encryption: CKKS and Polynomial Activations

In CKKS homomorphic encryption, ciphertexts represent packed polynomial rings $R_q = \mathbb{Z}_q[X]/(X^N + 1)$:
- Homomorphic Addition: $\text{Enc}(m_1) \oplus \text{Enc}(m_2) = \text{Enc}(m_1 + m_2)$ (essentially free).
- Homomorphic Multiplication: $\text{Enc}(m_1) \otimes \text{Enc}(m_2) = \text{Enc}(m_1 \cdot m_2)$ (consumes one level of ciphertext modulus $q$).
- **Non-Polynomial Operations are Forbidden**: Standard operations like $\text{ReLU}(x) = \max(0, x)$, $\text{Sign}(x)$, and $\text{Softmax}$ cannot be evaluated homomorphically without bootstrapping or interactive protocols.

### Polynomial Activation Approximations
To bypass this limitation, non-linear activations are replaced with low-degree polynomial approximations:
1. **Degree-2 Approximation (CryptoNets)**:
   $$y = a x^2 + b x + c$$
   e.g. $a = 0.125, b = 0.5, c = 0.25$ provides a smooth quadratic approximation to $\text{ReLU}(x)$ over $[-2, 2]$.
2. **Degree-4 Minimax Polynomial**:
   $$y = c_4 x^4 + c_2 x^2 + c_0$$
   Provides tighter approximation bounds across wider dynamic ranges while maintaining multiplicative depth budget $L \le 4$.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`freivalds_verify`** | Associative `@` chaining | Right-to-left `@` | Matrix-vector `.matmul` | Fused Freivalds residual |
| **`ckks_degree2`** | Quadratic polynomial | `a * x**2 + b*x + c` | Vector loop polynomial | `CKKS_DEGREE2_ACTIVATION_WGSL` |
| **`ckks_degree4`** | Quartic polynomial | `c4 * x**4 + c2 * x**2` | Vector loop quartic | Fused degree-4 evaluator |
| **`zkml_fixed_point`** | Scaled round & unscale | `jnp.round` scaling | Scaled round & unscale | Fixed-point integer GEMM |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Non-Polynomial ReLU in CKKS Homomorphic Evaluators**:
   - *Symptom*: Fatal compiler abort or runtime failure when running pre-trained networks under CKKS encryption.
   - *Cause*: Attempting to execute conditional branching (`if x > 0`) or step comparisons inside encrypted ciphertext domains.
   - *Fix*: Replace ReLUs with degree-2 or degree-4 polynomial activations prior to model encryption.
2. **Ciphertext Noise Overflow from Multiplicative Depth Exhaustion**:
   - *Symptom*: Decrypted inference outputs produce garbage or complete numerical noise.
   - *Cause*: Exceeding the modulus budget $L$ through successive non-linear multiplications without intermediate bootstrapping.
   - *Fix*: Minimize polynomial degree and rescale ciphertexts after each homomorphic multiplication.
3. **ZKML Fixed-Point Scale Overflow in Finite Fields**:
   - *Symptom*: ZK-SNARK verifier rejects proof on large batches.
   - *Cause*: Fixed-point accumulator $z_{\text{int}}$ overflows the field characteristic $p$, causing wraparound modular reduction.
   - *Fix*: Calibrate fixed-point scaling factor $S = 2^8$ or $2^{12}$ and verify intermediate bit widths.

---

## 7. Real-World Engineering Applications

1. **Decentralized AI Inference (Bittensor, Modulus Labs)**: Verifying neural network inference computations on untrusted edge nodes via ZKML proofs and Freivalds checks.
2. **Confidential Healthcare Diagnostics (CryptoNets)**: Running diagnostic image classifiers on encrypted patient MRIs in the cloud without exposing raw medical records.
3. **Financial Fraud Auditing**: Proving to regulatory auditors that an anti-money laundering neural network was executed faithfully without revealing proprietary model weights.

---

## 8. Verification, Parity Protocols & Synchronized Track Navigation

Parity across PyTorch, JAX, and Candle is verified against `weights/11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.safetensors`:
1. `freivalds_valid_ref`: Freivalds zero-residual vector on valid matrix multiplication.
2. `freivalds_tampered_ref`: Freivalds non-zero residual vector detecting tampered matrix entries.
3. `poly2_out_ref`: CKKS degree-2 polynomial activation outputs.
4. `poly4_out_ref`: CKKS degree-4 polynomial activation outputs.
5. `zk_out_ref`: ZKML quantized fixed-point linear layer representations.

### Synchronized Ecosystem Artifacts
- **Notebook:** [10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.ipynb](../../notebooks/11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.ipynb)
- **PyTorch 2 Module:** [10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.py](../../pytorch2/11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.py)
- **JAX Module:** [10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.py](../../jax/11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.py)
- **Candle Module:** [10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.rs](../../candle/11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.rs)
- **Hardware Kernel:** [10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.rs](../../kernels/11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.rs)
