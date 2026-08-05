# Exponential Family Anatomy

Cross-link: see [Lecture 4 Section 6](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#6-anatomy-of-the-exponential-family), [Conceptual Interlude B](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-b-why-exponential-family-and-glm-exist), [Conceptual Interlude C](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-c-why-glm-components-form-a-statistical-model), [Sufficient Statistics, Likelihood, and Moments](sufficient-statistics-likelihood-and-moments.md), [GLM Components](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#8-glm-components), and [Why Exponential Family and GLM Exist](why-exponential-family-and-glm.md).

## 1. Canonical form

The canonical exponential-family form is:

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

It separates four jobs: what statistic is read from the observation, which coordinate controls the chosen family member, how normalization happens, and what baseline measure remains.

| Component | Role | Meaning |
| --------- | ---- | ------- |
| $`Y`$ | random response | outcome before observation |
| $`y`$ | realized value | observed value or candidate value being evaluated |
| $`T(y)`$ | per-observation statistic | readout from one realized response |
| $`\eta`$ | natural parameter | canonical coordinate inside the chosen family |
| $`\eta^TT(y)`$ | coupling | compatibility between parameter coordinate and statistic |
| $`a(\eta)`$ | log-partition | normalizer and moment engine |
| $`b(y)`$ | base measure | parameter-independent support and baseline weighting |

Before normalization, the unnormalized log-score is:

```math
s_\eta(y)=\eta^TT(y)+\log b(y)
```

After normalization:

```math
\log p(y;\eta)=s_\eta(y)-a(\eta)
```

The key mental model is: $`T(y)`$ decides what the model reads from $`y`$; $`\eta`$ decides how the distribution values those readings.

The deeper reason $`T(y)`$ exists is parameter comparison. The data are observed; the parameter is unknown. To decide which candidate parameter is more plausible, the likelihood needs only the parts of $`y`$ that change the comparison.

```math
\log p(y;\eta_1)
-
\log p(y;\eta_2)
=
(\eta_1-\eta_2)^TT(y)
-
a(\eta_1)
+
a(\eta_2)
```

The base term $`\log b(y)`$ cancels because it is parameter-independent. Thus $`T(y)`$ is the evidence interface from the sample to the unknown parameter. The natural parameter $`\eta`$ is the parameter-side weight on those evidence directions, and $`\eta^TT(y)`$ is the compatibility score between parameter preference and observed statistic. This is a dual pairing: data contribute the statistic, parameters weight the statistic.

Do not read $`T(y)`$ as an estimator. It is the parameter-relevant representation used by estimators such as MLE.

## 2. Probability model and likelihood

A probability model assigns mass or density over possible values of $`Y`$ before observation. It does not modify the realized value $`y`$.

For a Gaussian model:

```math
Y\sim\mathcal N(\mu,\sigma^2)
```

```math
p(y;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
```

For continuous $`Y`$, this is a density. Probabilities come from integrating over intervals:

```math
P(a\leq Y\leq b)=\int_a^b p(y;\mu,\sigma^2)dy
```

After observing data, the same density becomes a likelihood as a function of unknown parameters. The data are fixed; candidate parameters vary.

```math
L(\eta)=\prod_{i=1}^n p(y_i;\eta)
```

This is inverse reasoning relative to a chosen family, not a posterior probability for $`\eta`$.

## 3. Per-observation statistic and sample sufficient statistic

There are four distinct objects:

```math
T(Y_i)
```

This is the canonical statistic for one random observation.

```math
T(y_i)
```

This is the statistic evaluated on one observed realization.

```math
S(\mathbf Y)=\sum_{i=1}^nT(Y_i)
```

This is the iid sample-level statistic before observation.

```math
S(\mathbf y)=\sum_{i=1}^nT(y_i)
```

This is the observed aggregate statistic that enters the likelihood.

For iid samples:

```math
p(y_1,\dots,y_n;\eta)
=
\left(\prod_{i=1}^n b(y_i)\right)
\exp\left(\eta^T\sum_{i=1}^nT(y_i)-na(\eta)\right)
```

All dependence on $`\eta`$ flows through:

```math
S(\mathbf y)=\sum_{i=1}^nT(y_i)
```

By Fisher-Neyman factorization, $`S(\mathbf Y)`$ is sufficient for $`\eta`$ in this iid model. Sufficient does not mean the statistic preserves every fact about the raw sample. It means it preserves all likelihood information about the current unknown parameter under the current family. If the family or unknown parameter set changes, sufficiency can change.

| Model | Per-observation statistic | Sample sufficient statistic | Parameter information |
| ----- | ------------------------- | --------------------------- | --------------------- |
| Bernoulli | $`T(y)=y`$ | $`\sum_i y_i`$ | success probability |
| Poisson | $`T(y)=y`$ | $`\sum_i y_i`$ | rate |
| Gaussian, known variance | $`T(y)=y`$ | $`\sum_i y_i`$ | common mean |
| Gaussian, unknown mean and variance | $`T(y)=(y,y^2)`$ | $`(\sum_i y_i,\sum_i y_i^2)`$ | location and spread |
| Categorical | one-hot statistic | class-count vector | class probabilities |

## 4. Unknown-parameter dependence

A statistic is sufficient relative to the parameter being estimated. The same raw Gaussian sample has different sufficient summaries depending on which parameter is unknown.

Known variance, unknown mean:

```math
\ell(\mu)
=
-\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-\mu)^2+C
```

Expanding:

```math
\ell(\mu)
=
\frac{\mu}{\sigma^2}\sum_{i=1}^ny_i
-
\frac{n\mu^2}{2\sigma^2}
+
C'
```

The likelihood depends on $`\mu`$ through $`\sum_i y_i`$ or equivalently $`\bar y`$.

Known mean, unknown variance:

```math
\ell(\sigma^2)
=
-\frac n2\log\sigma^2
-
\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-\mu)^2
+
C
```

The parameter-dependent statistic is:

```math
\sum_{i=1}^n(y_i-\mu)^2
```

Unknown mean and variance:

```math
\ell(\mu,\sigma^2)
=
-\frac n2\log\sigma^2
-
\frac{1}{2\sigma^2}
\left(
\sum_{i=1}^ny_i^2
-
2\mu\sum_{i=1}^ny_i
+
n\mu^2
\right)
+
C
```

The sufficient statistic is:

```math
\left(
\sum_{i=1}^ny_i,
\sum_{i=1}^ny_i^2
\right)
```

The $`y`$ term carries first-order location information. The $`y^2`$ term carries second-order magnitude information. Signed deviations can cancel, so spread cannot be recovered by summing raw deviations. The identity:

```math
\mathrm{Var}(Y)=\mathbb E[Y^2]-\mathbb E[Y]^2
```

explains why second moments matter. The reason $`y^2`$ must appear in the sufficient statistic is that its coefficient depends on the unknown $`\sigma^2`$.

## 5. Ordinary parameter versus natural parameter

Ordinary parameters are familiar coordinates used to describe a distribution. Natural parameters are canonical exponential-family coordinates. They are connected by reparameterization within the same local distribution:

```math
\eta=q(\psi)
```

```math
\psi=q^{-1}(\eta)
```

Bernoulli ordinary parameter:

```math
\psi=p
```

Bernoulli natural parameter:

```math
\eta=\log\frac{p}{1-p}
```

Poisson ordinary parameter:

```math
\psi=\lambda
```

Poisson natural parameter:

```math
\eta=\log\lambda
```

Gaussian under the CS229 variance-one simplification:

```math
\psi=\mu,
\qquad
\eta=\mu
```

For a Gaussian with fixed non-unit variance written in ordinary density form, the mean coordinate can be scaled by the variance. If mean and variance are both unknown, the natural parameter is vector-valued and the statistic includes $`y^2`$. Do not treat the CS229 simplification as the whole Gaussian parameterization.

The natural parameter does not necessarily control dispersion, tail behavior, truncation, mixtures, dependence, or censoring. Those can require additional parameters or a different model family.

## 6. Natural parameter as exponential-tilt coordinate

The natural parameter appears linearly against the statistic:

```math
\log p(y;\eta)=\eta^TT(y)-a(\eta)+\log b(y)
```

Relative probabilities under the same family member show the coordinate meaning:

```math
\log
\frac{p_\eta(y_1)}
{p_\eta(y_2)}
=
\eta^T\left(T(y_1)-T(y_2)\right)
+
\log\frac{b(y_1)}{b(y_2)}
```

The log-partition term cancels because both outcomes are normalized under the same $`\eta`$. Changing $`\eta`$ changes the relative mass assigned to outcomes according to their sufficient-statistic coordinates.

A small change $`\Delta\eta`$ gives the local approximation:

```math
\Delta\log p(y;\eta)
\approx
\Delta\eta^T
\left(
T(y)-\mathbb E_\eta[T(Y)]
\right)
```

If an observed statistic is above its current model expectation, increasing the corresponding natural coordinate raises the log probability of that outcome relative to normalization.

## 7. Log-partition as moment engine

The log-partition function normalizes the model:

```math
a(\eta)=\log\int b(y)\exp\left(\eta^TT(y)\right)dy
```

Its first derivative gives the mean statistic:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

Its second derivative gives covariance:

```math
\nabla^2a(\eta)=\mathrm{Cov}_\eta(T(Y))
```

In the scalar case:

```math
a''(\eta)=\mathrm{Var}_\eta(T(Y))
```

Because covariance matrices are positive semidefinite, $`a(\eta)`$ is convex. This is the source of the favorable natural-parameter likelihood geometry.

## 8. MLE as moment matching

For iid data, the log-likelihood is:

```math
\ell(\eta)
=
\eta^T\sum_{i=1}^nT(y_i)
-
na(\eta)
+
\sum_{i=1}^n\log b(y_i)
```

The score is:

```math
\nabla_\eta\ell(\eta)
=
\sum_{i=1}^nT(y_i)
-
n\nabla a(\eta)
```

Using $`\nabla a(\eta)=\mathbb E_\eta[T(Y)]`$, any finite interior MLE satisfies:

```math
\frac1n\sum_{i=1}^nT(y_i)
=
\mathbb E_{\hat\eta}[T(Y)]
```

The left side is empirical sufficient-statistic average. The right side is model-expected statistic under the fitted natural parameter. This is the precise exponential-family version of learning distribution parameters from samples.

## 9. From natural parameter to GLM

In an unconditional exponential-family model, $`\eta`$ is fixed. A GLM makes the local natural coordinate depend on features through a global trainable parameter.

Define the systematic component:

```math
\xi_i=s_\theta(x_i)=x_i^T\theta
```

For the scalar canonical link:

```math
\eta_i=\xi_i=x_i^T\theta
```

The conditional distribution is:

```math
p(y_i\mid x_i;\theta)
=
b(y_i)\exp\left(\eta_iT(y_i)-a(\eta_i)\right)
```

The prediction is the conditional mean statistic:

```math
h_\theta(x_i)
=
\mathbb E[T(Y_i)\mid x_i;\theta]
=
\nabla a(\eta_i)
```

This is the bridge from probability to machine learning. The feature vector does not directly force $`Y_i=x_i^T\theta`$; it chooses a coordinate of the conditional distribution from which $`Y_i`$ is randomly realized.

## 10. Modeling lesson

Exponential family is not the set of all distributions. It is a structured family where sufficient statistics enter linearly, the log-partition function normalizes the model and generates moments, and likelihood estimation compares observed statistics with model expectations. GLMs use that structure to connect response semantics, conditional probability models, global parameter learning, and prediction.
