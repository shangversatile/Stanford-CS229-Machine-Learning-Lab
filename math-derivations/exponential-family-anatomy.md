# Exponential Family Anatomy

Cross-link: see [Lecture 4 Section 6](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#6-anatomy-of-the-exponential-family) and [Lecture 4 Section 10](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#10-deep-meaning-of-the-hypothesis-function).

## 1. Normalized form

The canonical exponential-family form is:

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

The form is useful because it separates four jobs: how data enter the likelihood, which parameter is canonical, how the distribution is normalized, and what support/base weighting remains.

## 2. Normalization

Define the unnormalized normalizer:

```math
Z(\eta)=\int b(y)\exp\left(\eta^TT(y)\right)dy
```

For discrete distributions, replace the integral with a sum. The log-partition function is:

```math
a(\eta)=\log Z(\eta)
```

Then:

```math
\int p(y;\eta)dy
=
\int b(y)\exp\left(\eta^TT(y)-a(\eta)\right)dy
```

```math
=
e^{-a(\eta)}Z(\eta)
```

```math
=1
```

So $a(\eta)$ is not decoration. It is the term that makes the expression a valid probability distribution.

## 3. Natural parameter

The natural parameter $\eta$ is the parameter that appears linearly against the sufficient statistic in canonical form. It is often not the parameter used in ordinary distribution descriptions.

Bernoulli common parameter:

```math
\phi=P(Y=1)
```

Bernoulli natural parameter:

```math
\eta=\log\frac{\phi}{1-\phi}
```

Poisson common parameter:

```math
\lambda=\mathbb E[Y]
```

Poisson natural parameter:

```math
\eta=\log\lambda
```

In a canonical GLM, this natural parameter is set equal to a linear predictor:

```math
\eta=\theta^Tx
```

## 4. Sufficient statistic

The sufficient statistic $T(y)$ is the part of the observation that interacts with the natural parameter. It is not always equal to $y$.

| Model | Common sufficient statistic | Why it matters |
| ----- | --------------------------- | -------------- |
| Bernoulli | $T(y)=y$ | the single binary value is the success count |
| Fixed-variance Gaussian | $T(y)=y$ | the mean is the only unknown response parameter |
| Unknown-variance Gaussian | $T(y)=(y,y^2)$ | mean and second moment both carry parameter information |
| Poisson | $T(y)=y$ | total count summarizes evidence for the rate |
| Categorical / multinomial | one-hot or count vector | class counts summarize evidence for probabilities |

For iid data, the parameter-dependent information aggregates through:

```math
\sum_{i=1}^{m}T(y^{(i)})
```

That is the sufficiency intuition behind exponential-family likelihoods.

## 5. Log-partition function as moment engine

Under regularity conditions allowing differentiation under the integral:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

and:

```math
\nabla^2a(\eta)=\mathrm{Cov}_\eta(T(Y))
```

In the scalar case:

```math
a''(\eta)=\mathrm{Var}_\eta(T(Y))
```

Because covariance matrices are positive semidefinite:

```math
\nabla^2a(\eta)\succeq0
```

Thus $a(\eta)$ is convex. This convexity is the source of the favorable geometry in many GLM negative log likelihoods.

## 6. Base measure

The base measure $b(y)$ collects the part of the mass or density independent of $\eta$. It affects support and the full likelihood value, but it does not affect gradients with respect to $\eta$.

Examples:

```math
b_{\mathrm{Gaussian}}(y)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{y^2}{2}\right)
```

```math
b_{\mathrm{Bernoulli}}(y)=1
```

```math
b_{\mathrm{Poisson}}(y)=\frac{1}{y!}
```

## 7. Connection to GLM response functions

A GLM predicts the conditional mean of the sufficient statistic:

```math
h_\theta(x)=\mathbb E[T(Y)\mid x;\theta]
```

In canonical exponential-family form:

```math
h_\theta(x)=\nabla a(\eta)
```

With the canonical linear predictor:

```math
\eta=\theta^Tx
```

so:

```math
h_\theta(x)=\nabla a(\theta^Tx)
```

This is the distribution-to-response-function bridge:

| Distribution | $a(\eta)$ | $\nabla a(\eta)$ | GLM response |
| ------------ | --------- | ---------------- | ------------ |
| Gaussian fixed variance | $\eta^2/2$ | $\eta$ | identity |
| Bernoulli | $\log(1+e^\eta)$ | $e^\eta/(1+e^\eta)$ | sigmoid |
| Poisson | $e^\eta$ | $e^\eta$ | exponential |
| Multinomial / softmax | $\log\sum_j e^{\eta_j}$ | normalized probabilities | softmax |

The response function is therefore not an arbitrary activation. It is the inverse-link or mean map implied by the chosen distribution and its log-partition function.

## 8. Iid factorization and likelihood geometry

For iid data:

```math
p(y^{(1)},\ldots,y^{(m)};\eta)
=
\prod_{i=1}^{m}b(y^{(i)})\exp\left(\eta^TT(y^{(i)})-a(\eta)\right)
```

Rearrange:

```math
=
\left(\prod_{i=1}^{m}b(y^{(i)})\right)
\exp\left(\eta^T\sum_{i=1}^{m}T(y^{(i)})-ma(\eta)\right)
```

The natural-parameter-dependent part is linear in the sample statistic and subtracts $ma(\eta)$. Since $a(\eta)$ is convex, the log-likelihood is concave in $\eta$ and the NLL is convex in $\eta$ for the regular canonical setting.

## 9. Modeling lesson

Exponential family is not the set of all distributions. It is a family with a special algebraic structure: sufficient statistics enter linearly, the log-partition function normalizes the distribution, and derivatives of the log-partition function produce moments. GLMs use that structure to turn a response distribution into a principled response function and likelihood.