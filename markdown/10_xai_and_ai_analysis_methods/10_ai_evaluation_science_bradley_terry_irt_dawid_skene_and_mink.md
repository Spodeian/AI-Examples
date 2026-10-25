# AI Evaluation Science: Bradley-Terry, Item Response Theory, Dawid-Skene, and Min-K%

## 1. Mathematical Foundations & Lineage

Evaluating modern Large Language Models and Foundation Models requires rigorous statistical psychometrics and contamination detection:
1. **Bradley-Terry-Luce Model (BTL — Bradley & Terry 1952; Luce 1959)**: The foundational statistical model behind LMSYS Chatbot Arena. Models pairwise human or automated head-to-head preference judgments by parameterizing latent skill ratings $s_i$, converting logits to logistic Elo scales.
2. **Item Response Theory (2PL IRT — Lord 1980; Birnbaum 1968)**: Standard evaluation frameworks (e.g., HELM, Open LLM Leaderboard v2) suffer from benchmark saturation. 2PL IRT disentangles model latent ability ($\theta$) from item difficulty ($b$) and item discrimination capability ($a$).
3. **Dawid-Skene Expectation-Maximization (Dawid & Skene 1979)**: When gathering ground-truth labels from multiple noisy LLM-as-a-judge or crowdsourced annotators, Dawid-Skene estimates latent true labels and individual judge error/confusion matrices simultaneously without requiring verified ground truth.
4. **Min-K% Prob Contamination Detection (Shi et al. 2023, NeurIPS)**: Audits benchmark leakage and pretraining data extraction by measuring the mean log-likelihood of the lowest $k\%$ token probabilities in a sequence, detecting memorized texts via anomalous outlier tail statistics.

---

## 2. Bradley-Terry-Luce (BTL) Preference Modeling

Let each evaluated model $i \in \{1, \dots, M\}$ possess a latent scalar skill $s_i \in \mathbb{R}$.
The probability that model $i$ wins over model $j$ in a pairwise comparison is:
$$P(i \succ j) = \sigma(s_i - s_j) = \frac{1}{1 + e^{-(s_i - s_j)}} = \frac{e^{s_i}}{e^{s_i} + e^{s_j}}$$
The pairwise win probability matrix $P \in (0, 1)^{M \times M}$ satisfies anti-symmetry $P_{ij} + P_{ji} = 1$ and $P_{ii} = 0.5$.
Elo ratings are calibrated via standard logistic scaling:
$$\text{Elo}_i = 1000 + \frac{400}{\ln 10} \cdot s_i$$

---

## 3. Two-Parameter Logistic Item Response Theory (2PL IRT)

When evaluating $M$ models across $I$ distinct benchmark questions, raw percentage accuracy fails to account for varying item difficulty.
Under 2PL IRT, the binary correctness $Y_{mi} \in \{0, 1\}$ follows:
$$P(Y_{mi} = 1 \mid \theta_m, a_i, b_i) = \sigma(a_i (\theta_m - b_i)) = \frac{1}{1 + e^{-a_i (\theta_m - b_i)}}$$
where:
- $\theta_m \in \mathbb{R}$: Latent capability / ability of model $m$.
- $b_i \in \mathbb{R}$: Difficulty parameter of question $i$ (ability level required for 50% success).
- $a_i > 0$: Discrimination slope of question $i$ (how sharply question $i$ separates high-ability from low-ability models).

---

## 4. Dawid-Skene Consensus & Min-K% Prob

### Dawid-Skene Expectation Step (E-Step)
For $N$ instances annotated by $J$ judges into $K$ classes with priors $p_k$ and confusion matrices $\Pi^{(j)} \in \mathbb{R}^{K \times K}$ ($\Pi_{k, l}^{(j)} = P(Y_{ij} = l \mid Z_i = k)$):
The latent posterior class probability $T_{ik} = P(Z_i = k \mid Y_i, \Pi, p)$ is:
$$T_{ik} = \frac{p_k \prod_{j=1}^J \Pi_{k, Y_{ij}}^{(j)}}{\sum_{k'=1}^K p_{k'} \prod_{j=1}^J \Pi_{k', Y_{ij}}^{(j)}}$$

### Min-K% Prob Detection
For a sequence of tokens with model log-probabilities $\ell = (\ln P(x_1), \dots, \ln P(x_T))$, let $\ell_{(1)} \le \ell_{(2)} \le \dots \le \ell_{(T)}$ be the sorted log-probabilities.
For a cutoff fraction $k \in (0, 1]$ (typically $k = 0.20$ or $0.25$) with $K_{\text{cut}} = \lceil k \cdot T \rceil$:
$$\text{Min-K\% Prob} = \frac{1}{K_{\text{cut}}} \sum_{t=1}^{K_{\text{cut}}} \ell_{(t)}$$
Unseen non-member text contains several high-surprise tokens ($\ell_{(t)} \ll 0$), lowering the Min-K% score. Pretrained / memorized text exhibits elevated log-probabilities even on the hardest tokens, producing significantly higher Min-K% scores.

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`btl_win_probability`** | Outer difference + sigmoid | Broadcasting difference + logistic | 2D vector difference loop | `BTL_WIN_PROBABILITY_WGSL` |
| **`irt_2pl_probability`** | Scaled discrimination difference + sigmoid | Broadcast multiply + logistic | 2D parameter matrix loop | Vectorized IRT logistic kernel |
| **`dawid_skene_e_step`** | Log-posterior accumulation + softmax | Log-likelihood sum + softmax | Numerically stable log-sum-exp | Parallel judge likelihood reducer |
| **`min_k_percent_prob`** | `torch.sort` + prefix slice mean | `jnp.sort` + slice mean | Vector sort + slice mean | Parallel bitonic sort + mean |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Intransitivity Violations in BTL Ranking**:
   - *Symptom*: Model A beats B, B beats C, but C beats A (rock-paper-scissors loop), causing BTL likelihood optimization to oscillate.
   - *Cause*: Unidimensional scalar skill $s$ cannot represent multidimensional trade-offs (e.g. coding vs. creative writing).
   - *Fix*: Cluster evaluation queries into distinct domains before fitting separate domain-specific BTL models.
2. **IRT Scale / Translation Invariance Indeterminacy**:
   - *Symptom*: Abilities $\theta_m$ and difficulties $b_i$ drift unboundedly during joint estimation.
   - *Cause*: Shifting $\theta \to \theta + c$ and $b \to b + c$ leaves probabilities unchanged.
   - *Fix*: Anchor the item difficulty mean to zero: $\frac{1}{I}\sum_{i=1}^I b_i = 0$.
3. **Min-K% False Positives on Repetitive Prompts**:
   - *Symptom*: Synthetic repetitive phrases or common code boilerplate are misclassified as pretraining memorization.
   - *Cause*: Degenerate low-entropy sequences naturally lack surprise tokens.
   - *Fix*: Normalize Min-K% scores against a reference small language model (Min-K%++ ratio).

---

## 7. Real-World Engineering Applications

1. **Crowdsourced LLM Leaderboard Scoring**: Powering LMSYS Chatbot Arena leaderboards using BTL maximum likelihood skill estimation.
2. **Benchmark Saturation Resistance**: Using 2PL IRT to compute model capability estimates that remain stable as new, harder benchmarks are introduced.
3. **Data Contamination Auditing**: Flagging benchmark leakage in open-weight models prior to publishing evaluation results.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `btl_win_prob_ref`: Pairwise win probability matrix $P \in (0, 1)^{M \times M}$.
3. `irt_prob_ref`: 2PL item response probability matrix $P \in (0, 1)^{M \times I}$.
4. `ds_posterior_ref`: Dawid-Skene instance class posterior matrix $T \in [0, 1]^{N \times K}$.
5. `mink_score_ref`: Min-K% mean tail log-probability scalar.
