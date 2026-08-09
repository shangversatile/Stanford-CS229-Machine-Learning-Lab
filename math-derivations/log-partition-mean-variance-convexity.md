# Log-Partition Mean, Variance, and Convexity

Cross-link: see [Lecture 4 Conceptual Interlude A: What Information About a Parameter Is Actually in the Data?](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-a-what-information-about-a-parameter-is-actually-in-the-data), [Conceptual Interlude D: Why GLM Components Form a Statistical Model](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-d-why-glm-components-form-a-statistical-model), [Conceptual Interlude E: What Does a Response Value Mean?](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-e-what-does-a-response-value-mean), [Sufficient Statistics and Likelihood Equivalence](sufficient-statistics-likelihood-equivalence.md), [8. GLM Components](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#8-glm-components), [GLM Construction Recipe](glm-construction-recipe.md), and [Response Spaces, Measures, and Expectations](response-spaces-measures-and-expectations.md).

## 1. Setup

The log-partition function appears first as a normalizer, but it later becomes the mathematical engine for response means, variance, Fisher information, moment matching, and convexity.

Consider an exponential-family distribution:

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

Define:

```math
Z(\eta)=\int b(y)\exp\left(\eta^TT(y)\right)dy
```

and:

```math
a(\eta)=\log Z(\eta)
```

Assume regularity conditions that justify differentiating under the integral sign.

## 2. Normalization

```math
\int p(y;\eta)dy
=
\int b(y)\exp\left(\eta^TT(y)-a(\eta)\right)dy
```

```math
=
e^{-a(\eta)}
\int b(y)\exp\left(\eta^TT(y)\right)dy
```

```math
=
e^{-a(\eta)}Z(\eta)
```

Since $`a(\eta)=\log Z(\eta)`$:

```math
e^{-a(\eta)}Z(\eta)=1
```

Thus $`a(\eta)`$ is exactly the term that normalizes the density or mass function.

## 3. Mean identity

Differentiate $`Z(\eta)`$:

```math
\nabla Z(\eta)
=
\int b(y)
\exp\left(\eta^TT(y)\right)
T(y)dy
```

Then:

```math
\nabla a(\eta)
=
\nabla\log Z(\eta)
=
\frac{\nabla Z(\eta)}{Z(\eta)}
```

Substitute the expression for $`\nabla Z(\eta)`$:

```math
\nabla a(\eta)
=
\int T(y)
\frac{b(y)\exp\left(\eta^TT(y)\right)}
{Z(\eta)}
dy
```

Because:

```math
\frac{b(y)\exp\left(\eta^TT(y)\right)}
{Z(\eta)}
=
b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
=
p(y;\eta)
```

we get:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

This is the expectation of the canonical statistic, not automatically the scalar response mean. Use:

```math
m(\eta)
=
\mathbb E_\eta[T(Y)]
=
\nabla a(\eta)
```

for the expectation parameter on the statistic scale, and reserve:

```math
\mu
=
\mathbb E[Y]
```

for the response mean. If $`T(Y)=Y`$, then $`m(\eta)=\mu`$. If $`T(Y)=(Y,Y^2)`$, then $`m(\eta)`$ contains both $`\mathbb E[Y]`$ and $`\mathbb E[Y^2]`$. For categorical data, $`m(\eta)`$ is a vector of class probabilities because $`T(Y)`$ is one-hot. Thus the log-partition derivative returns the model-expected canonical statistic, and only special cases make that identical to a scalar response mean.

## 4. Covariance identity

Differentiate again:

```math
\nabla^2a(\eta)
=
\nabla\left(\frac{\nabla Z(\eta)}{Z(\eta)}\right)
```

Using the quotient/product rule:

```math
\nabla^2a(\eta)
=
\frac{\nabla^2Z(\eta)}{Z(\eta)}
-
\frac{\nabla Z(\eta)\nabla Z(\eta)^T}{Z(\eta)^2}
```

The second derivative of $`Z`$ is:

```math
\nabla^2Z(\eta)
=
\int b(y)
\exp\left(\eta^TT(y)\right)
T(y)T(y)^Tdy
```

Therefore:

```math
\frac{\nabla^2Z(\eta)}{Z(\eta)}
=
\mathbb E_\eta[T(Y)T(Y)^T]
```

and:

```math
\frac{\nabla Z(\eta)\nabla Z(\eta)^T}{Z(\eta)^2}
=
\mathbb E_\eta[T(Y)]
\mathbb E_\eta[T(Y)]^T
```

Thus:

```math
\nabla^2a(\eta)
=
\mathbb E_\eta[T(Y)T(Y)^T]
-
\mathbb E_\eta[T(Y)]\mathbb E_\eta[T(Y)]^T
```

So:

```math
\nabla^2a(\eta)=\mathrm{Cov}_\eta(T(Y))
```

For scalar $`T(Y)`$:

```math
a''(\eta)=\mathrm{Var}_\eta(T(Y))
```

## 5. Convexity of the log-partition function

For any vector $`v`$:

```math
v^T\nabla^2a(\eta)v
=
v^T\mathrm{Cov}_\eta(T(Y))v
```

```math
=
\mathrm{Var}_\eta(v^TT(Y))
```

```math
\geq0
```

Therefore:

```math
\nabla^2a(\eta)\succeq0
```

So $`a(\eta)`$ is convex. It is strictly convex only when no nonzero direction $`v`$ makes $`v^TT(Y)`$ almost surely constant under the family.

## 6. Concavity of log-likelihood

For iid samples:

```math
\ell(\eta)
=
\eta^T\sum_{i=1}^{n}T(y_i)
-na(\eta)
+
\sum_{i=1}^{n}\log b(y_i)
```

Gradient:

```math
\nabla\ell(\eta)
=
\sum_{i=1}^{n}T(y_i)
-n\nabla a(\eta)
```

Hessian:

```math
\nabla^2\ell(\eta)
=
-n\nabla^2a(\eta)
=
-n\,\mathrm{Cov}_\eta(T(Y))
```

Hence:

```math
\nabla^2\ell(\eta)\preceq0
```

So log-likelihood is concave in $`\eta`$ for regular exponential families.

## 7. Empirical-to-model moment matching

The score equation for a finite interior MLE is:

```math
0
=
\nabla\ell(\hat\eta)
=
\sum_{i=1}^{n}T(y_i)
-
n\nabla a(\hat\eta)
```

Using the mean identity:

```math
\nabla a(\hat\eta)=\mathbb E_{\hat\eta}[T(Y)]
```

we get:

```math
\frac1n\sum_{i=1}^{n}T(y_i)
=
\mathbb E_{\hat\eta}[T(Y)]
```

The left side is the empirical canonical-statistic mean. The right side is the model-expected canonical statistic. This is the statistic-matching interpretation of exponential-family MLE: the parameter is adjusted until the model reproduces the sample statistics that are relevant to likelihood comparison. The result resembles method of moments, but it is specifically the exponential-family score equation and should not be confused with every possible MLE or with general GMM.

Moment matching is different from minimal sufficiency. Minimal sufficiency is a data-compression property defined by factorization or likelihood-ratio equivalence. Moment matching is an optimization first-order condition that appears after a regular finite interior MLE is assumed to exist. The statistic comes from the model representation; MLE uses it.

## 8. Convexity of NLL

Negative log likelihood:

```math
J_{\mathrm{NLL}}(\eta)=-\ell(\eta)
```

Its Hessian is:

```math
\nabla^2J_{\mathrm{NLL}}(\eta)
=
n\,\mathrm{Cov}_\eta(T(Y))
\succeq0
```

Therefore NLL is convex in the natural parameter.

## 9. Canonical GLM training bridge

For one scalar canonical GLM sample, set:

```math
\eta_i=x_i^T\theta
```

The log likelihood is:

```math
\log p(y_i\mid x_i;\theta)
=
T(y_i)x_i^T\theta
-
a(x_i^T\theta)
+
\log b(y_i)
```

Differentiate by chain rule:

```math
\nabla_\theta\log p(y_i\mid x_i;\theta)
=
x_i\left(T(y_i)-a'(\eta_i)\right)
```

Using the mean identity $`a'(\eta_i)=\mathbb E[T(Y_i)\mid x_i;\theta]`$:

```math
\nabla_\theta\log p(y_i\mid x_i;\theta)
=
x_i\left(T(y_i)-\mathbb E[T(Y_i)\mid x_i;\theta]\right)
```

Summing over samples:

```math
\nabla_\theta\ell(\theta)
=
\sum_{i=1}^{n}
x_i
\left(
T(y_i)-\mathbb E[T(Y_i)\mid x_i;\theta]
\right)
```

Thus canonical GLM learning compares the observed sufficient statistic with the model-expected sufficient statistic, then weights that residual by the feature vector. Gaussian, Bernoulli, and Poisson give the familiar residuals $`y_i-\mu_i`$, $`y_i-p_i`$, and $`y_i-\lambda_i`$ on their own response scales.

For per-sample NLL:

```math
J_i(\theta)=-\log p(y_i\mid x_i;\theta)
```

its Hessian is:

```math
\nabla_\theta^2J_i(\theta)
=
a''(\eta_i)x_ix_i^T
```

and the covariance identity gives:

```math
a''(\eta_i)=\mathrm{Var}(T(Y_i)\mid x_i;\theta)
```

This is why canonical GLM curvature can be read as variance-weighted feature geometry. It is convex-friendly, but not automatically strictly convex or guaranteed to have a finite unique MLE.

## 10. Strict-convexity and MLE-existence caveats

Convex-friendly does not mean every MLE is finite, unique, and numerically stable.

Strict convexity may fail when:

* parameters are not identifiable;
* sufficient statistics are linearly dependent;
* feature matrix rank is deficient in a GLM;
* softmax parameters are unconstrained and share common shift invariance.

Finite MLE may fail when:

* logistic or softmax data are completely separable;
* observed sufficient statistic lies on the boundary of the achievable mean set;
* the model support or parameter space excludes the empirical behavior.

Numerical stability may fail when:

* Hessian is ill-conditioned;
* fitted probabilities are extremely close to $`0`$ or $`1`$;
* count rates are extremely large or small;
* features are poorly scaled.

The correct conclusion is precise: exponential-family likelihoods have favorable convex geometry in natural parameters, but existence, uniqueness, and reliable computation require extra conditions.
