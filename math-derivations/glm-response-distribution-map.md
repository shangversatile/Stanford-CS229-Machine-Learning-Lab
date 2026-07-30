# GLM Response and Distribution Map

Cross-link: see the main Lecture 4 note, especially [Conceptual Interlude A](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-a-from-response-space-to-probability-distribution), [Conceptual Interlude C](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-c-why-exponential-family-and-glm-exist), [Conceptual Interlude D](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-d-how-a-glm-connects-features-to-a-conditional-distribution), [GLM workflow](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#9-the-complete-glm-modeling-workflow), and [hypothesis-function interpretation](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#10-deep-meaning-of-the-hypothesis-function). For the deeper origin story, see [Why Exponential Family and GLM Exist](why-exponential-family-and-glm.md).

## 1. Why distribution choice comes before loss choice

A loss is not the first modeling decision. The first decision is what the response variable $`Y`$ means and what kind of random outcome it is. Once a conditional distribution for $`Y\mid x`$ is chosen, the likelihood and negative log likelihood follow.

This is why squared loss, binary cross-entropy, multiclass cross-entropy, and Poisson NLL are not interchangeable penalties. They correspond to different response spaces and different uncertainty assumptions.

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

## 3. Detailed distribution table

| Response type | Support | Task type | Candidate distribution | Mean | Variance pattern | GLM response | Lecture 4 status |
| ------------- | ------- | --------- | ---------------------- | ---- | ---------------- | ------------ | ---------------- |
| Real-valued continuous | $`\mathbb R`$ | regression | Gaussian | $`\mu`$ | constant if variance fixed | identity | official core derivation |
| Binary event | $`\{0,1\}`$ | binary classification | Bernoulli | $`\phi`$ | $`\phi(1-\phi)`$ | sigmoid | official core derivation |
| Single multiclass label | $`\{1,\dots,K\}`$ | multiclass classification | categorical / multinomial one-trial | $`\phi_k`$ | coupled categorical covariance | softmax | official core derivation |
| Count-valued | $`\mathbb N_0`$ | count regression | Poisson | $`\lambda`$ | variance equals mean | exponential | mentioned / problem-set-level extension |
| Positive continuous | $`\mathbb R_{>0}`$ | durations, costs, waiting times | Exponential / Gamma | positive mean | right-skewed, mean-dependent | inverse or log-linked depending parameterization | mentioned examples |
| Scalar probability | $`(0,1)`$ | probability or proportion target | Beta | $`\alpha/(\alpha+\beta)`$ | bounded, shape-dependent | mean in $`(0,1)`$ with separate link | extension for probability-valued data |
| Probability vector | $`\Delta^{K-1}`$ | composition target | Dirichlet | simplex-valued mean | negative covariance from sum-to-one constraint | simplex-valued mean | extension for probability-vector data |

## Core canonical GLM comparison

Use the shared linear predictor:

```math
s_\theta(x)=\theta^Tx
```

and, for these scalar canonical GLMs:

```math
\eta(x)=s_\theta(x)
```

A common variance-function summary is:

```math
\mathrm{Var}(Y\mid x)=\phi V(\mu(x))
```

Here $`\phi`$ is dispersion, not the Bernoulli success probability.

| Family | Support | Mean domain | Variance function $`V(\mu)`$ | Natural parameter | Canonical link | Response function | Meaning of $`\theta^Tx`$ | Prediction | Main misspecification risks |
| ------ | ------- | ----------- | --------------------------- | ----------------- | -------------- | ----------------- | ------------------------- | ---------- | -------------------------- |
| Gaussian | $`\mathbb R`$ | $`\mathbb R`$ | $`1`$ | $`\eta=\mu`$ in CS229 variance-1 form | identity | identity | conditional mean | fitted mean | heavy tails, heteroscedasticity, bounded outcomes |
| Bernoulli | $`\{0,1\}`$ | $`(0,1)`$ | $`\mu(1-\mu)`$ | $`\eta=\log(p/(1-p))`$ | logit | sigmoid | log-odds | event probability | separation, label noise, imbalance, calibration failure |
| Poisson | $`\mathbb N_0`$ | $`(0,\infty)`$ | $`\mu`$ | $`\eta=\log\lambda`$ | log | exponential | log-rate | expected count | overdispersion, excess zeros, exposure mismatch, dependence |

## 4. Gaussian

**Support.** $`Y\in\mathbb R`$.

**PMF/PDF.**

```math
p(y;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
```

**Parameters.** Mean $`\mu\in\mathbb R`$ and variance $`\sigma^2>0`$.

**Mean and variance.**

```math
\mathbb E[Y]=\mu,\qquad \mathrm{Var}(Y)=\sigma^2
```

**Task meaning.** Real-valued regression with additive measurement noise.

**Lecture 4 role.** Official CS229 Lecture 4 core derivation, typically with fixed variance.

**GLM role.** Fixed-variance canonical Gaussian gives identity response $`h_\theta(x)=\theta^Tx`$ and squared-loss NLL.

**Clarification.** The Gaussian model does not say the target is deterministic. It says the target is random around a conditional mean, and variance controls the spread around that mean.

**Failure modes.** Heavy tails, asymmetric residuals, heteroscedasticity, bounded outcomes, or invalid negative predictions.

## 5. Bernoulli

**Support.** $`Y\in\{0,1\}`$.

**PMF/PDF.**

```math
p(y;\phi)=\phi^y(1-\phi)^{1-y},\qquad y\in\{0,1\}
```

**Parameters.** Success probability $`\phi\in(0,1)`$.

**Mean and variance.**

```math
\mathbb E[Y]=\phi,\qquad \mathrm{Var}(Y)=\phi(1-\phi)
```

**Task meaning.** Binary event prediction.

**Lecture 4 role.** Official CS229 Lecture 4 core derivation.

**GLM role.** Natural parameter is log-odds. Canonical linear predictor produces sigmoid response and binary cross-entropy NLL.

**Failure modes.** Complete separation, class imbalance, label noise, poor calibration, and distribution shift in base rates.

## 6. Categorical and Multinomial

**Support.** For one class per sample, $`Y\in\{1,\dots,K\}`$. For count vectors over $`n`$ categorical trials, $`c_k\in\mathbb N_0`$ and $`\sum_k c_k=n`$.

**PMF/PDF.** Categorical one-trial form:

```math
p(y=k;\phi)=\phi_k,\qquad \sum_{k=1}^{K}\phi_k=1
```

One-hot statistic:

```math
T_k(Y)=\mathbf1\{Y=k\}
```

```math
\mathbb E[T_k(Y)]=\phi_k
```

Multinomial count form:

```math
p(c_1,\ldots,c_K;\phi)=\frac{n!}{\prod_{k=1}^{K}c_k!}\prod_{k=1}^{K}\phi_k^{c_k}
```

**Parameters.** Probability vector $`\phi\in\Delta^{K-1}`$.

**Mean and covariance for one-hot categorical statistic.**

```math
\mathrm{Cov}(T_i(Y),T_j(Y))=
\begin{cases}
\phi_i(1-\phi_i), & i=j\\
-\phi_i\phi_j, & i\neq j
\end{cases}
```

**Task meaning.** Mutually exclusive multiclass classification.

**Lecture 4 role.** Official CS229 Lecture 4 core derivation for multinomial/softmax.

**GLM role.** Softmax maps class scores to a normalized probability vector.

**Failure modes.** Poisson on class IDs, ordinal interpretation of nominal labels, independent one-vs-rest probabilities that do not sum to one, or rare classes with unstable estimates.

## 7. Poisson

**Support.** $`Y\in\mathbb N_0`$.

**PMF/PDF.**

```math
p(y;\lambda)=\frac{\lambda^y e^{-\lambda}}{y!},\qquad y\in\mathbb N_0
```

**Parameters.** Rate or mean count $`\lambda>0`$.

**Mean and variance.**

```math
\mathbb E[Y]=\lambda,\qquad \mathrm{Var}(Y)=\lambda
```

**Task meaning.** Count regression for event counts under a rate/exposure mechanism.

**Lecture 4 role.** Mentioned and natural GLM extension / problem-set-level example.

**GLM role.** Natural parameter is $`\eta=\log\lambda`$. Canonical linear predictor gives $`h_\theta(x)=e^{\theta^Tx}`$.

**Failure modes.** Overdispersion, excess zeros, changing exposure, event dependence, underdispersion, and bursty processes.

## 8. Exponential

**Support.** $`Y\in\mathbb R_{>0}`$.

**PMF/PDF.**

```math
p(y;\lambda)=\lambda e^{-\lambda y},\qquad y>0
```

**Parameters.** Rate $`\lambda>0`$.

**Mean and variance.**

```math
\mathbb E[Y]=\frac{1}{\lambda},\qquad \mathrm{Var}(Y)=\frac{1}{\lambda^2}
```

**Task meaning.** Positive waiting time under a memoryless constant-rate mechanism.

**Lecture 4 role.** Mentioned as an exponential-family example, not fully developed as a CS229 Lecture 4 GLM derivation here.

**GLM role.** Shows how positive response support can be tied to a rate or mean parameter.

**Failure modes.** Nonconstant hazards, censoring, heavy tails, deterministic delays, and mixtures of waiting-time mechanisms.

## 9. Gamma

**Support.** $`Y\in\mathbb R_{>0}`$.

**PMF/PDF.** Shape-rate parameterization:

```math
p(y;\alpha,\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}y^{\alpha-1}e^{-\beta y},\qquad y>0
```

**Parameters.** Shape $`\alpha>0`$ and rate $`\beta>0`$.

**Mean and variance.**

```math
\mathbb E[Y]=\frac{\alpha}{\beta},\qquad \mathrm{Var}(Y)=\frac{\alpha}{\beta^2}
```

**Task meaning.** Positive continuous regression for skewed durations, costs, or amounts.

**Lecture 4 role.** Mentioned as an exponential-family example.

**GLM role.** Gamma GLMs often model variance that grows with the mean; inverse and log links are common depending on convention and goal.

**Failure modes.** Exact zeros, extreme tails, multimodality, censoring, and mixtures of mechanisms.

## 10. Beta

**Support.** $`Y\in(0,1)`$.

**PMF/PDF.**

```math
p(y;\alpha,\beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}y^{\alpha-1}(1-y)^{\beta-1},\qquad 0<y<1
```

**Parameters.** Shape parameters $`\alpha>0`$ and $`\beta>0`$.

**Mean.**

```math
\mathbb E[Y]=\frac{\alpha}{\alpha+\beta}
```

**Variance.**

```math
\mathrm{Var}(Y)=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
```

**Task meaning.** Scalar probability or proportion as the observed response.

**Lecture 4 role.** Mentioned as a distribution over probabilities; extension layer for this note.

**GLM role.** A Beta model is for probability-valued observations, not binary labels. The mean is in $`(0,1)`$ and the link is chosen separately.

**Failure modes.** Exact boundary values, different denominators behind proportions, zero-one inflation, or binomial-count data better modeled with exposure.

## 11. Dirichlet

**Support.** $`p\in\Delta^{K-1}`$.

**PMF/PDF.**

```math
p(p;\alpha)=\frac{\Gamma(\sum_{k=1}^{K}\alpha_k)}{\prod_{k=1}^{K}\Gamma(\alpha_k)}\prod_{k=1}^{K}p_k^{\alpha_k-1}
```

with:

```math
p\in\Delta^{K-1}
```

**Parameters.** Concentration vector $`\alpha_k>0`$.

**Mean.**

```math
\mathbb E[p_k]=\frac{\alpha_k}{\sum_{j=1}^{K}\alpha_j}
```

**Covariance.**

```math
\mathrm{Cov}(p_i,p_j)=
\begin{cases}
\frac{\alpha_i(\alpha_0-\alpha_i)}{\alpha_0^2(\alpha_0+1)}, & i=j\\
-\frac{\alpha_i\alpha_j}{\alpha_0^2(\alpha_0+1)}, & i\neq j
\end{cases}
```

where:

```math
\alpha_0=\sum_{k=1}^{K}\alpha_k
```

**Task meaning.** Composition or probability-vector response.

**Lecture 4 role.** Mentioned as a distribution over probabilities; extension layer for this note.

**GLM role.** Dirichlet is for a random probability vector. Basic softmax classification predicts a probability vector but observes a categorical label, so the response distribution is categorical rather than Dirichlet.

**Failure modes.** Structural zeros, multimodality, richer component dependence, and compositional effects not captured by a single Dirichlet.

## 12. Official CS229 core vs extension

| Distribution | Lecture 4 role |
| ------------ | -------------- |
| Gaussian | Official core derivation |
| Bernoulli | Official core derivation |
| Multinomial / Softmax | Official core derivation |
| Poisson | Mentioned and natural GLM extension / problem-set-level |
| Gamma / Exponential | Mentioned as exponential-family examples |
| Beta / Dirichlet | Mentioned as distributions over probabilities |
| Others | Outside this lecture |

The official core is the construction pattern, not a claim that every useful distribution is fully derived in the lecture. The extension layer is included to make distribution selection usable in real modeling problems.

## 13. How distribution choice changes the learning objective

| Distribution | Conditional mean | NLL shape | Practical interpretation |
| ------------ | ---------------- | --------- | ------------------------ |
| Gaussian fixed variance | $`\theta^Tx`$ | squared error up to constants | penalizes real-valued residuals symmetrically |
| Bernoulli | sigmoid of $`\theta^Tx`$ | binary cross-entropy | fits event probabilities |
| Categorical / softmax | softmax probabilities | multiclass cross-entropy | fits one mutually exclusive class per sample |
| Poisson | $`e^{\theta^Tx}`$ | Poisson NLL | fits nonnegative count rates |
| Gamma | positive mean | Gamma deviance-like objective | fits positive skewed continuous responses |
| Beta | mean in $`(0,1)`$ | Beta likelihood | fits probability-valued observations |
| Dirichlet | simplex mean | Dirichlet likelihood | fits composition-valued observations |

For a canonical scalar exponential-family GLM, the per-sample NLL ignoring constants is:

```math
J_i(\theta)=a(\eta_i)-\eta_iT(y^{(i)})
```

with:

```math
\eta_i=\theta^Tx^{(i)}
```

This is why the response distribution controls both the response function and the loss.

## 14. Reliability diagnostics

| Check | What to ask | Example failure |
| ----- | ----------- | --------------- |
| Support | Can the model predict only legal means? | Gaussian mean used for harmful negative count predictions |
| Mean-variance relation | Does variance scale as assumed? | Poisson data with variance far above mean |
| Tail behavior | Are extreme values plausible under the family? | Gaussian underestimates heavy tails |
| Calibration | Do predicted probabilities match frequencies? | Bernoulli or softmax probabilities overconfident |
| Dependence | Are samples conditionally independent enough? | time series arrivals treated as iid counts |
| Zero handling | Are zeros structural or sampling variation? | zero-inflated count data modeled as plain Poisson |
| Identifiability | Do parameter choices produce unique distributions? | softmax class parameters shift together without changing probabilities |
| Shift | Does the conditional mechanism persist? | class base rates change after deployment |

## 15. Summary

Distribution choice is the bridge from response semantics to GLM mathematics. Support tells what values are legal; the distribution adds variance, tail, dependence, and mechanism assumptions; exponential-family form exposes $`T(y)`$, $`\eta`$, $`a(\eta)`$, and $`b(y)`$; the GLM link turns the natural parameter into a linear predictor; and the response mean becomes the hypothesis function used for prediction.
