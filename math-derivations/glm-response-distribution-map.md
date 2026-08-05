# GLM Response and Distribution Map

Cross-link: see the main Lecture 4 note, especially [Conceptual Interlude A](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-a-from-response-space-to-probability-distribution), [Conceptual Interlude C](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-c-why-glm-components-form-a-statistical-model), [GLM Components](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#8-glm-components), [GLM Workflow](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#9-the-complete-glm-modeling-workflow), and [Hypothesis Function](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#10-deep-meaning-of-the-hypothesis-function). For deeper derivations, see [Exponential Family Anatomy](exponential-family-anatomy.md), [GLM Construction Recipe](glm-construction-recipe.md), and [Log-Partition Mean, Variance, and Convexity](log-partition-mean-variance-convexity.md), and [Sufficient Statistics, Likelihood, and Moments](sufficient-statistics-likelihood-and-moments.md).

## 1. Why distribution choice comes before loss choice

The first modeling decision is what the response random variable $`Y_i`$ means. Once a conditional distribution for $`Y_i\mid x_i`$ is chosen, the likelihood and negative log likelihood follow. Squared loss, binary cross-entropy, multiclass cross-entropy, and Poisson NLL are consequences of different response spaces and uncertainty assumptions.

## 2. Support sets and response semantics

```math
\mathbb R=(-\infty,\infty)
```

```math
\mathbb R_{>0}=(0,\infty)
```

```math
\mathbb R_{\geq0}=[0,\infty)
```

```math
\mathbb N_0=\{0,1,2,\ldots\}
```

```math
\Delta^{K-1}=\left\{p\in\mathbb R^K:p_k\geq0,\ \sum_{k=1}^{K}p_k=1\right\}
```

Support is necessary because the model should not assign predictions or random outcomes to impossible values. It is not sufficient because variance, skew, tail behavior, zero inflation, dependence, and measurement mechanism still matter.

## 3. Distribution-selection map

| Response type | Support | Task type | Candidate distribution | GLM response | Lecture 4 status |
| ------------- | ------- | --------- | ---------------------- | ------------ | ---------------- |
| Real-valued continuous | $`\mathbb R`$ | regression | Gaussian | identity | official core derivation |
| Binary event | $`\{0,1\}`$ | binary classification | Bernoulli | sigmoid | official core derivation |
| Single multiclass label | $`\{1,\dots,K\}`$ | multiclass classification | categorical / multinomial one-trial | softmax | official core derivation |
| Count-valued | $`\mathbb N_0`$ | count regression | Poisson | exponential | problem-set-level extension |
| Positive continuous | $`\mathbb R_{>0}`$ | durations, costs, waiting times | Exponential / Gamma | inverse or log-linked | mentioned examples |
| Scalar probability | $`(0,1)`$ | probability or proportion target | Beta | mean in $`(0,1)`$ | extension |
| Probability vector | $`\Delta^{K-1}`$ | composition target | Dirichlet | simplex-valued mean | extension |

## 4. Core canonical GLM comparison

Use the shared linear predictor:

```math
\xi_i=s_\theta(x_i)=x_i^T\theta
```

Only under a canonical link do we also have:

```math
\xi_i=\eta_i
```

The table keeps formulas short; the family sections below give the derivations.

| Family | Support | Ordinary $`\psi_i`$ | Natural $`\eta_i`$ | Statistic $`T(y_i)`$ | Mean $`\mu_i`$ | Variance | Link | Meaning of $`x_i^T\theta`$ | Residual | Common misspecification |
|---|---|---|---|---|---|---|---|---|---|---|
| Gaussian fixed variance | $`\mathbb R`$ | $`\mu_i`$ | mean coord. | $`y_i`$ | $`\mu_i`$ | $`\sigma^2`$ | identity | mean | $`y_i-\mu_i`$ | heavy tails, heteroscedasticity |
| Bernoulli | $`\{0,1\}`$ | $`p_i`$ | log-odds | $`y_i`$ | $`p_i`$ | $`p_i(1-p_i)`$ | logit | log-odds | $`y_i-p_i`$ | imbalance, separation, calibration |
| Poisson | $`\mathbb N_0`$ | $`\lambda_i`$ | log-rate | $`y_i`$ | $`\lambda_i`$ | $`\lambda_i`$ | log | log-rate | $`y_i-\lambda_i`$ | overdispersion, zero inflation |

The residual column is a conditional-mean residual. It does not imply fixed Gaussian noise except in the Gaussian identity-link model with an explicit homoscedastic Gaussian assumption. Bernoulli residuals have support $`\{-p_i,1-p_i\}`$; Poisson residual variance grows with $`\lambda_i`$.

The global parameter is not an ordinary distribution parameter for a single sample. It parameterizes the mapping from $`x_i`$ to the local distribution scale.
## 5. Gaussian

**Support.** $`Y_i\in\mathbb R`$.

**Density.**

```math
p(y_i;\mu_i,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i-\mu_i)^2}{2\sigma^2}\right)
```

**Ordinary parameter.** The familiar mean is $`\psi_i=\mu_i`$. The variance may be fixed, estimated separately, or treated as dispersion depending on the modeling convention.

**Natural parameter.** In the CS229 fixed-variance-one simplification:

```math
\eta_i=\mu_i
```

With fixed non-unit variance in ordinary density form, the natural coordinate can be scaled by the variance. If variance is also unknown, the natural parameter becomes vector-valued and the statistic must include second-order information.

**Canonical GLM meaning.**

```math
x_i^T\theta=\eta_i=\mu_i
```

So:

```math
h_\theta(x_i)=\mu_i=x_i^T\theta
```

**Variance structure.** Fixed-variance Gaussian regression assumes constant conditional variance around the mean.

**Failure modes.** Heavy tails, asymmetric residuals, heteroscedasticity, bounded outcomes, or invalid negative predictions.

## 6. Bernoulli

**Support.** $`Y_i\in\{0,1\}`$.

**PMF.**

```math
p(y_i;p_i)=p_i^{y_i}(1-p_i)^{1-y_i}
```

**Ordinary parameter.**

```math
\psi_i=p_i=P(Y_i=1\mid x_i;\theta)
```

**Natural parameter.**

```math
\eta_i=\log\frac{p_i}{1-p_i}
```

The inverse map is:

```math
p_i=\frac{1}{1+\exp(-\eta_i)}
```

**Canonical GLM meaning.**

```math
x_i^T\theta=\eta_i
```

So the linear predictor is log-odds, not probability:

```math
h_\theta(x_i)=p_i=\frac{1}{1+\exp(-x_i^T\theta)}
```

**Variance structure.**

```math
\mathrm{Var}(Y_i\mid x_i)=p_i(1-p_i)
```

**Failure modes.** Complete separation, class imbalance, label noise, poor calibration, and distribution shift in base rates.

## 7. Poisson

**Support.** $`Y_i\in\mathbb N_0`$.

**PMF.**

```math
p(y_i;\lambda_i)=\frac{\lambda_i^{y_i}e^{-\lambda_i}}{y_i!}
```

**Ordinary parameter.**

```math
\psi_i=\lambda_i
```

**Natural parameter.**

```math
\eta_i=\log\lambda_i
```

The inverse map is:

```math
\lambda_i=\exp(\eta_i)
```

**Canonical GLM meaning.**

```math
x_i^T\theta=\eta_i
```

So the linear predictor is log-rate, not the count itself:

```math
h_\theta(x_i)=\lambda_i=\exp(x_i^T\theta)
```

**Variance structure.**

```math
\mathrm{Var}(Y_i\mid x_i)=\lambda_i
```

**Failure modes.** Overdispersion, excess zeros, changing exposure, event dependence, underdispersion, and bursty processes.

## 8. Categorical and multinomial

**Support.** For one class per sample, $`Y_i\in\{1,\dots,K\}`$. For count vectors over repeated categorical trials, counts are nonnegative and sum to the trial count.

**Categorical PMF.**

```math
p(Y_i=k;\phi_i)=\phi_{ik}
```

with:

```math
\sum_{k=1}^{K}\phi_{ik}=1
```

**Statistic.** The statistic is one-hot or reference-class one-hot. It records category identity; it does not treat the class label as a magnitude.

```math
T_k(Y_i)=\mathbf1\{Y_i=k\}
```

**GLM role.** Softmax maps class scores to a coupled probability vector. All class probabilities share one normalization, so they are not independent one-vs-rest probabilities.

**Failure modes.** Poisson on class IDs, ordinal interpretation of nominal labels, independent probabilities that do not sum to one, or rare classes with unstable estimates.

## 9. Extension families

**Exponential.** Positive waiting-time distribution under a memoryless rate mechanism. It is useful for positive durations but fails with nonconstant hazards, censoring, heavy tails, or mixtures.

**Gamma.** Positive continuous response family for skewed amounts, costs, or durations. Gamma GLMs often encode variance that grows with the mean.

**Beta.** Distribution for probability-valued observations in $`(0,1)`$. It is not the same as Bernoulli binary labels.

**Dirichlet.** Distribution for random probability vectors or compositions on the simplex. Basic softmax classification predicts a probability vector but observes a categorical label, so the response distribution is categorical rather than Dirichlet.

## 10. How distribution choice changes the learning objective

| Distribution | Conditional mean | NLL shape | Practical interpretation |
| ------------ | ---------------- | --------- | ------------------------ |
| Gaussian fixed variance | $`x_i^T\theta`$ | squared error | real-valued residual model |
| Bernoulli | sigmoid of score | binary cross-entropy | event probability model |
| Categorical / softmax | softmax probabilities | multiclass cross-entropy | one mutually exclusive class |
| Poisson | exponential of score | Poisson NLL | nonnegative count-rate model |
| Gamma | positive mean | Gamma likelihood | positive skewed response |
| Beta | mean in $`(0,1)`$ | Beta likelihood | probability-valued observation |
| Dirichlet | simplex mean | Dirichlet likelihood | composition-valued observation |

For a canonical scalar exponential-family GLM, the per-sample NLL ignoring constants is:

```math
J_i(\theta)=a(\eta_i)-\eta_iT(y_i)
```

with:

```math
\eta_i=x_i^T\theta
```

This is why the response distribution controls both the response function and the loss.

## 11. Reliability diagnostics

| Check | What to ask | Example failure |
| ----- | ----------- | --------------- |
| Support | Can the model predict only legal means? | negative count prediction |
| Mean-variance relation | Does variance scale as assumed? | Poisson overdispersion |
| Tail behavior | Are extremes plausible? | Gaussian underestimates heavy tails |
| Calibration | Do predicted probabilities match frequencies? | overconfident probabilities |
| Dependence | Are samples conditionally independent enough? | time-series arrivals treated as iid |
| Zero handling | Are zeros structural or sampling variation? | zero-inflated counts modeled as plain Poisson |
| Identifiability | Do parameters produce unique distributions? | softmax common-shift invariance |
| Shift | Does the conditional mechanism persist? | deployment base-rate change |

## 12. Summary

Distribution choice is the bridge from response semantics to GLM mathematics. Support tells what values are legal; the distribution adds variance, tail, dependence, and mechanism assumptions; exponential-family form exposes $`T(y_i)`$, $`\eta_i`$, $`a(\eta_i)`$, and $`b(y_i)`$; the GLM link turns the systematic component into the relevant distribution coordinate; and the response mean becomes the hypothesis function used for prediction.
