# GLM Response Distribution Map

## 1. Why Response Semantics Come First

The response variable determines what predictions must mean. A real-valued measurement, a binary event, a category, a count, and a probability vector have different supports and different statistical mechanisms.

The common mistake is to treat numeric encodings as mathematical meaning. Multiclass labels encoded as $0,1,2$ are not count data. A class label of $2$ is not twice class label $1$；it is a category identifier. Poisson is for counts; multiclass classification uses categorical or multinomial models.

## 2. Distribution Summary

| Response semantics | Support | Candidate distribution | Key modeling meaning |
| ------------------ | ------- | ---------------------- | -------------------- |
| Continuous measurement | $\mathbb R$ | Gaussian | additive symmetric noise |
| Binary event | $\{0,1\}$ | Bernoulli | probability of success |
| Single multiclass label | $\{1,\ldots,K\}$ | categorical | one class per sample |
| Class counts over trials | count vector | multinomial | counts from repeated categorical trials |
| Event count | $\{0,1,2,\ldots\}$ | Poisson | count/rate process |
| Waiting time | $\mathbb R_{>0}$ | Exponential | memoryless duration |
| Positive skewed amount | $\mathbb R_{>0}$ | Gamma | positive continuous with flexible variance |
| Scalar probability | $(0,1)$ | Beta | random proportion |
| Probability vector | simplex | Dirichlet | random categorical probability vector |

## 3. Gaussian

Definition:

```math
p(y;\mu,\sigma^2)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
```

Use when response is continuous, can plausibly take values on a real line, and residuals are roughly symmetric with finite variance. Fixed-variance Gaussian GLM gives identity response and squared-loss MLE.

## 4. Bernoulli

Definition:

```math
p(y;\phi)=\phi^y(1-\phi)^{1-y},
\quad y\in\{0,1\}
```

Use when each observation records whether an event happened. Bernoulli GLM with canonical link gives sigmoid response and binary cross-entropy NLL.

## 5. Categorical and Multinomial

Categorical definition:

```math
P(Y=k;\phi)=\phi_k,
\quad
\sum_{k=1}^{K}\phi_k=1
```

One-hot form:

```math
p(y;\phi)
=
\prod_{k=1}^{K}\phi_k^{\mathbf1\{y=k\}}
```

Multinomial count definition:

```math
p(c_1,\ldots,c_K;\phi)
=
\frac{n!}{\prod_{k=1}^{K}c_k!}
\prod_{k=1}^{K}\phi_k^{c_k}
```

Categorical is for one class per sample. Multinomial is for counts across $n$ categorical trials. Softmax regression is the GLM response for categorical/multinomial class probabilities.

## 6. Poisson

Definition:

```math
p(y;\lambda)=\frac{\lambda^ye^{-\lambda}}{y!},
\quad y\in\{0,1,2,\ldots\}
```

Use when $y$ is a nonnegative integer count from a rate process. Poisson GLM with log link gives:

```math
\mathbb E[Y\mid x;\theta]=e^{\theta^Tx}
```

Core caveat: Poisson implies:

```math
\mathbb E[Y]=\mathrm{Var}(Y)=\lambda
```

If observed variance is much larger than mean, overdispersion diagnostics are needed.

## 7. Exponential

Definition:

```math
p(y;\lambda)=\lambda e^{-\lambda y},
\quad y>0
```

Use for waiting times with a memoryless mechanism. It is too restrictive when hazard changes over time or tails are heavier than exponential.

## 8. Gamma

Definition:

```math
p(y;\alpha,\beta)
=
\frac{\beta^\alpha}{\Gamma(\alpha)}
y^{\alpha-1}e^{-\beta y},
\quad y>0
```

Use for positive continuous outcomes with skewness, such as durations, costs, or intensities. Gamma can model variance increasing with the mean more flexibly than Gaussian.

## 9. Beta

Definition:

```math
p(y;\alpha,\beta)
=
\frac{\Gamma(\alpha+\beta)}
{\Gamma(\alpha)\Gamma(\beta)}
y^{\alpha-1}(1-y)^{\beta-1},
\quad 0<y<1
```

Use for scalar probabilities or proportions inside the open interval. If observations include exact $0$ or $1$, a plain Beta model may need zero-one inflation or boundary handling.

## 10. Dirichlet

Definition for $p_k>0$ and $\sum_k p_k=1$:

```math
p(p_1,\ldots,p_K;\alpha)
=
\frac{\Gamma\left(\sum_{k=1}^{K}\alpha_k\right)}
{\prod_{k=1}^{K}\Gamma(\alpha_k)}
\prod_{k=1}^{K}p_k^{\alpha_k-1}
```

Use when the response itself is a probability vector. The components are dependent because they must sum to one.

## 11. Final Distinction

| Problem | Correct family direction | Incorrect shortcut |
| ------- | ------------------------ | ------------------ |
| Predict one of $K$ classes | categorical / softmax | Poisson on class IDs |
| Predict number of events | Poisson or count model | softmax over arbitrary maximum count without count mechanism |
| Predict a real-valued measurement | Gaussian or robust continuous model | Bernoulli unless thresholded event is the target |
| Predict a probability vector | Dirichlet or compositional model | independent regressions that ignore sum-to-one |

Distribution choice is a modeling assumption, not a datatype conversion.

