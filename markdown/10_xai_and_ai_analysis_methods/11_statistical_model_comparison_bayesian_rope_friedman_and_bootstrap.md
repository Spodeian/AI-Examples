# Statistical Model Comparison: Bayesian ROPE, Friedman Ranks, and Corrected Resampling

## 1. Mathematical Foundations & Lineage

Claims of model superiority in AI research and production deployment require rigorous hypothesis testing:
1. **McNemar's Test (Dietterich 1998)**: For a single fixed test set, classical two-sample tests incorrectly assume independence between model errors. McNemar's non-parametric test conditions strictly on discordant test instances where one model succeeds and the other fails.
2. **Nadeau-Bengio Corrected Resampled $t$-Test (Nadeau & Bengio 2003)**: Standard paired $t$-tests on $K$-fold cross-validation or repeated random splits severely underestimate error variance because training sets overlap, inflating Type I error (false discovery rates up to 50%). Nadeau and Bengio derived an exact analytical variance correction factor $c = \frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}}$.
3. **Friedman Non-Parametric Rank Comparison (Demšar 2006)**: When comparing $K$ algorithms across $N$ diverse benchmarks, parametric tests (e.g., ANOVA) violate commensurability and normality assumptions. The Friedman test ranks algorithms per benchmark to evaluate overall significant differences.
4. **Bayesian Correlated $t$-Test with ROPE (Benavoli et al. 2017; Kruschke 2018)**: Replaces misleading null-hypothesis $p$-values with full posterior probabilities over three mutually exclusive hypotheses: $P(\text{Model 1 better})$, $P(\text{Practically Equivalent})$ within Region of Practical Equivalence $[-r, r]$, and $P(\text{Model 2 better})$.

---

## 2. McNemar's Test for Paired Classifiers

Given two models $M_1$ and $M_2$ evaluated on $N$ test samples, let:
- $b$: Number of samples where $M_1$ is correct and $M_2$ is incorrect.
- $c$: Number of samples where $M_1$ is incorrect and $M_2$ is correct.

Under the null hypothesis $H_0: P(M_1 \text{ err}) = P(M_2 \text{ err})$, concordant predictions ($a, d$) provide no information about relative superiority.
With Edwards' continuity correction:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \sim \chi_1^2$$

---

## 3. Nadeau-Bengio Corrected Resampled $t$-Test

Let $d_k = \text{Loss}(M_1, \mathcal{D}_k^{\text{test}}) - \text{Loss}(M_2, \mathcal{D}_k^{\text{test}})$ across $K$ cross-validation folds.
The sample mean $\bar{d} = \frac{1}{K}\sum_{k=1}^K d_k$ and sample variance $S^2 = \frac{1}{K-1}\sum_{k=1}^K (d_k - \bar{d})^2$.

Because each training set of size $n_{\text{train}}$ shares $(K-2)/(K-1)$ data with other folds, differences $d_k$ have positive correlation $\rho \approx \frac{n_{\text{test}}}{n_{\text{train}}}$.
The corrected variance and test statistic are:
$$c = \frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}}, \quad \hat{\sigma}^2 = c \cdot S^2, \quad t_{\text{corrected}} = \frac{\bar{d}}{\hat{\sigma}}$$
This correction restores nominal Type I error rates (5%) across cross-validation experiments.

---

## 4. Friedman Rank Test & Bayesian ROPE

### Friedman Non-Parametric Multi-Model Test
Let $r_i^j$ be the rank of algorithm $j \in \{1, \dots, K\}$ on benchmark $i \in \{1, \dots, N\}$ (rank 1 = best).
The average rank is $R_j = \frac{1}{N}\sum_{i=1}^N r_i^j$.
The Friedman test statistic is:
$$\chi_F^2 = \frac{12 N}{K(K+1)} \left( \sum_{j=1}^K R_j^2 - \frac{K(K+1)^2}{4} \right) \sim \chi_{K-1}^2$$

### Bayesian Region of Practical Equivalence (ROPE)
Given practical threshold $r > 0$ (e.g., $1\%$ accuracy gain), differences in $[-r, r]$ are deemed practically equivalent.
Under the corrected posterior distribution $\mathcal{N}(\bar{d}, \hat{\sigma}^2)$:
$$z_1 = \frac{r - \bar{d}}{\hat{\sigma}}, \quad z_2 = \frac{-r - \bar{d}}{\hat{\sigma}}$$
$$P(\text{ROPE}) = \Phi(z_1) - \Phi(z_2), \quad P(M_1 > M_2) = 1 - \Phi(z_1), \quad P(M_2 > M_1) = \Phi(z_2)$$

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`mcnemar_test_statistic`** | Absolute difference + squared quotient | Pure scalar array operations | Scalar extraction + math | Scalar contingency kernel |
| **`nadeau_bengio_corrected_t`** | Fold variance + ratio scaling | Array reductions + sqrt | Vector variance + ratio scale | Corrected variance reducer |
| **`friedman_rank_statistic`** | Mean rank column reduction + sum sq | `jnp.mean` + rank formula | 2D matrix column mean | `AVERAGE_RANKS_WGSL` |
| **`bayesian_rope_probabilities`** | Erf-based normal CDF stack | `jax.scipy.special.erf` stack | Abramowitz & Stegun erf | Vectorized normal CDF |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Uncorrected Paired $t$-Test False Positives**:
   - *Symptom*: Standard $t$-test reports $p < 0.001$ superiority between two identical models evaluated on 10 random train/test splits.
   - *Cause*: Shared training instances induce strong positive covariance between splits, deflating uncorrected standard errors.
   - *Fix*: Always use the Nadeau-Bengio correction factor $c = 1/K + n_{\text{test}}/n_{\text{train}}$.
2. **Arbitrary ROPE Radius Selection**:
   - *Symptom*: $P(\text{ROPE})$ fluctuates wildly based on user-selected threshold $r$.
   - *Cause*: Treating statistical significance as practical significance without domain business requirements.
   - *Fix*: Pre-specify $r$ based on deployment cost trade-offs (e.g., minimum latency or revenue impact).
3. **Commensurability Violations in Cross-Benchmark Averaging**:
   - *Symptom*: A model with 99% accuracy on an easy dataset masks a 20% degradation on a hard benchmark.
   - *Cause*: Averages across datasets with wildly different baseline difficulty distributions.
   - *Fix*: Use Friedman rank transformations or 2PL IRT to eliminate scale dependencies.

---

## 7. Real-World Engineering Applications

1. **A/B Testing Model Releases**: Deciding whether a candidate LLM genuinely outperforms production or falls within the Region of Practical Equivalence.
2. **Benchmark Leaderboard Meta-Analysis**: Ranking foundation models across diverse evaluation suites (MMLU, HumanEval, GSM8K) using Friedman ranks.
3. **Cross-Validation Model Selection**: Accurately distinguishing between two hyperparameter configurations without overestimating significance.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `mcnemar_stat_ref`: Corrected McNemar $\chi^2$ statistic.
3. `nb_t_ref`, `nb_sigma_ref`: Nadeau-Bengio corrected $t$-statistic and standard deviation.
4. `friedman_stat_ref`, `avg_ranks_ref`: Friedman $\chi_F^2$ and average algorithm ranks.
5. `rope_probs_ref`: Bayesian ROPE probability vector $[P(\text{better}), P(\text{ROPE}), P(\text{worse})]$.
