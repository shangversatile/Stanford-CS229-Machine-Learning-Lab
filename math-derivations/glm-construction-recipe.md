# GLM Construction Recipe

Cross-link: see the main Lecture 4 note sections [Conceptual Interlude D](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-d-how-a-glm-connects-features-to-a-conditional-distribution), [GLM Workflow](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#9-the-complete-glm-modeling-workflow), and [Hypothesis Function](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#10-deep-meaning-of-the-hypothesis-function). For related reference maps, see [Exponential Family Anatomy](exponential-family-anatomy.md), [GLM Response and Distribution Map](glm-response-distribution-map.md), and [Log-Partition Mean, Variance, and Convexity](log-partition-mean-variance-convexity.md).

```text
response semantics
-> conditional family
-> sufficient statistic
-> natural parameter
-> linear predictor
-> link / response mapping
-> conditional distribution
-> likelihood
-> parameter estimation
-> prediction
-> diagnostics
```

This order matters because the loss is a consequence of the distribution, not the starting point.

## 1. Random component

The random component chooses the conditional distribution of the response:

```math
Y\mid x;\theta
\sim
\text{an exponential-family conditional distribution}
```

This is a statement about $`Y\mid X=x`$, not necessarily about the marginal distribution of $`Y`$. Mixing different feature values can make the marginal distribution look non-Gaussian even when each conditional distribution is Gaussian, and similarly for Bernoulli or Poisson models.

The random component should be chosen from response semantics:

| Response meaning | Support | Natural starting family |
| ---------------- | ------- | ----------------------- |
| real-valued measurement | $`\mathbb R`$ | Gaussian |
| binary event | $`\{0,1\}`$ | Bernoulli |
| event count | $`\mathbb N_0`$ | Poisson |
| mutually exclusive class | $`\{1,\dots,K\}`$ | categorical / multinomial |

Support is only the first check. The distribution also encodes variance, tail behavior, skew, dependence assumptions, and a data-generating story.

## 2. Systematic component

The systematic component is the feature-side score:

```math
s_\theta(x)=\theta^Tx
```

Here $`x\in\mathbb R^d`$ and $`\theta\in\mathbb R^d`$. The parameter $`\theta`$ is global and learned from the dataset. The value $`s_\theta(x)`$ is sample-specific.

This score is useful because it gives additive feature effects:

```math
\frac{\partial s_\theta(x)}{\partial x_j}=\theta_j
```

The meaning of that derivative depends on the scale where the score is placed. In Gaussian regression it can be a mean-scale effect. In logistic regression it is a log-odds effect. In Poisson regression it is a log-rate effect.

## 3. Link component

A raw linear predictor can be any real number. Many response means cannot. A Bernoulli mean must stay in $`(0,1)`$; a Poisson mean must stay in $`(0,\infty)`$.

The link function resolves this mismatch by making a transformed mean linear:

```math
g(\mu(x))=s_\theta(x)
```

The inverse direction maps the score back to the mean:

```math
\mu(x)=g^{-1}(s_\theta(x))
```

In this file, $`g`$ is reserved for the link direction. The response mapping from natural parameter to mean is written as $`\rho`$.

## 4. General link versus canonical link

For an exponential-family distribution:

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

The natural parameter $`\eta`$ is the coordinate that couples linearly to $`T(y)`$. The conditional mean of the sufficient statistic is:

```math
\mu=\mathbb E_\eta[T(Y)]
```

A canonical link maps this mean back to the natural parameter. For scalar $`T(Y)=Y`$, define:

```math
\rho(\eta)=a'(\eta)
```

Then:

```math
g_{\mathrm{can}}(\mu)=\rho^{-1}(\mu)
```

A canonical-link GLM chooses:

```math
g_{\mathrm{can}}(\mu(x))=s_\theta(x)
```

which is equivalent to:

```math
\eta(x)=s_\theta(x)=\theta^Tx
```

A noncanonical link is possible. It may fit domain knowledge better, but the clean gradient and Hessian formulas below are specific to the canonical scalar construction.

## 5. Mean map from the log-partition function

Let:

```math
Z(\eta)=\int b(y)\exp\left(\eta^TT(y)\right)dy
```

and:

```math
a(\eta)=\log Z(\eta)
```

Differentiate:

```math
\nabla a(\eta)
=
\frac{\nabla Z(\eta)}{Z(\eta)}
```

Because:

```math
\nabla Z(\eta)
=
\int b(y)\exp\left(\eta^TT(y)\right)T(y)dy
```

we get:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

For scalar $`T(Y)=Y`$:

```math
\mu=a'(\eta)
```

This is why the response function is not guessed from data. It is implied by the chosen family through the log-partition function.

## 6. Canonical construction

The scalar canonical GLM combines the previous pieces:

```math
\eta(x)=s_\theta(x)=\theta^Tx
```

```math
p(y\mid x;\theta)
=
b(y)\exp\left(\theta^TxT(y)-a(\theta^Tx)\right)
```

and:

```math
h_\theta(x)
=
\mu(x)
=
\rho(\theta^Tx)
=
a'(\theta^Tx)
```

Read each equality carefully:

* $`h_\theta(x)=\mu(x)`$ is the prediction convention.
* $`\mu(x)=\rho(\eta(x))`$ defines the response mapping.
* $`\rho(\eta)=a'(\eta)`$ is the exponential-family moment identity in the scalar case.
* $`\eta(x)=\theta^Tx`$ is the canonical linear natural-parameter design choice.

## 7. Training likelihood

For a dataset:

```math
D=\{(x^{(i)},y^{(i)})\}_{i=1}^{m}
```

conditional independence gives:

```math
L(\theta)=\prod_{i=1}^{m}p(y^{(i)}\mid x^{(i)};\theta)
```

The MLE is:

```math
\hat\theta=\underset{\theta}{\mathrm{argmax}}\ L(\theta)
```

or equivalently minimizes NLL:

```math
\hat\theta=\underset{\theta}{\mathrm{argmin}}\ J_{\mathrm{NLL}}(\theta)
```

This is inverse reasoning relative to a chosen forward model. Given $`\theta`$, the model predicts conditional distributions. Given observations, learning finds the $`\theta`$ whose conditional distributions make those observations plausible.

## 8. Gradient and Hessian

For one scalar canonical sample:

```math
\log p(y\mid x;\theta)
=
T(y)\theta^Tx
-
a(\theta^Tx)
+
\log b(y)
```

Let:

```math
\eta=\theta^Tx
```

First differentiate with respect to $`\eta`$:

```math
\frac{\partial}{\partial\eta}\log p(y\mid x;\theta)
=
T(y)-a'(\eta)
```

Then use:

```math
\nabla_\theta\eta=x
```

so chain rule gives:

```math
\nabla_\theta\log p(y\mid x;\theta)
=
x\left(T(y)-a'(\eta)\right)
```

Using $`a'(\eta)=\mathbb E[T(Y)\mid x;\theta]`$:

```math
\nabla_\theta\log p(y\mid x;\theta)
=
x\left(T(y)-\mathbb E[T(Y)\mid x;\theta]\right)
```

The update is observed sufficient statistic minus model-expected sufficient statistic, multiplied by the feature direction.

Examples:

| Model | Residual term | Gradient direction |
| ----- | ------------- | ------------------ |
| Gaussian | $`y-\mu`$ | $`x(y-\mu)`$ |
| Bernoulli | $`y-p`$ | $`x(y-p)`$ |
| Poisson | $`y-\lambda`$ | $`x(y-\lambda)`$ |

For the per-sample NLL:

```math
J_i(\theta)
=
a(\eta)-\eta T(y)
-
\log b(y)
```

The Hessian is:

```math
\nabla_\theta^2J_i(\theta)
=
a''(\eta)xx^T
```

and:

```math
a''(\eta)=\mathrm{Var}(T(Y)\mid x;\theta)
```

Therefore canonical GLM curvature is variance-weighted feature geometry. It is positive semidefinite, but strict convexity and finite MLE still require conditions such as full-rank features, identifiability, no complete separation, and a non-boundary optimum. Regularization is often added for stability.

## 9. Prediction

After training, prediction plugs in the fitted parameter:

```math
h_{\hat\theta}(x)
=
\mathbb E[T(Y)\mid x;\hat\theta]
```

For Gaussian regression, this is a fitted conditional mean. For Bernoulli logistic regression, it is an event probability. For Poisson regression, it is an expected count or rate. For softmax, it is a probability vector over mutually exclusive classes.

Prediction is not $`X\theta=Y`$. The model predicts a conditional distribution and reports a mean or probability derived from that distribution.

## 10. Assumptions and falsification

GLM family selection should be checked in this order:

1. support and response semantics;
2. plausible data-generating mechanism;
3. conditional mean-variance relationship;
4. tail, skewness, and dependence;
5. diagnostics and falsification.

Many GLM texts summarize variance as:

```math
\mathrm{Var}(Y\mid x)=\phi V(\mu(x))
```

Here $`\phi`$ is a dispersion constant. Bernoulli has $`V(\mu)=\mu(1-\mu)`$; Poisson has $`V(\mu)=\mu`$; Gaussian with constant variance has $`V(\mu)=1`$ with dispersion carrying the variance scale.

Failure modes are not rare edge cases. They include overdispersion, zero inflation, heteroscedasticity, truncation, censoring, mixture populations, temporal dependence, spatial dependence, distribution shift, and calibration failure. MLE only finds the best parameter inside the chosen family; it does not prove that the family, link, or representation is correct.

## 11. Nonlinear extensions

Linearity in:

```math
\eta(x)=\theta^Tx
```

is a design choice. It can be replaced by richer representation models while keeping the same observation model:

```math
\eta(x)=\theta^T\phi(x)
```

or:

```math
\eta(x)=f_\theta(x)
```

The first form covers polynomial features, interactions, splines, generalized additive models, and kernels. The second covers neural conditional models and structured spatial or temporal representations.

Keep the layers separate:

```text
representation model: x -> eta(x)
observation model: eta(x) -> p(Y|x)
```

Changing the representation does not automatically justify the observation family. A neural Poisson model can still be overdispersed; a flexible logistic model can still be poorly calibrated.

## 12. Summary

A GLM is a disciplined way to connect features to a conditional distribution. The random component chooses what kind of response is plausible. The systematic component creates an additive score. The link decides which distribution scale receives that score. The log-partition function turns the natural parameter into a mean. Likelihood training then compares observed sufficient statistics with model-expected sufficient statistics and updates $`\theta`$ accordingly.
