# Log-Partition Mean, Variance, and Convexity

## 1. Setup

Consider an exponential-family distribution:

The log-partition function appears first as a normalizer, but it later becomes the mathematical engine for response means, variance, Fisher information, and convexity. The deeper motivation is explained in [Why Exponential Family and GLM Exist](why-exponential-family-and-glm.md).


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

Since $a(\eta)=\log Z(\eta)$:

```math
e^{-a(\eta)}Z(\eta)=1
```

Thus $a(\eta)$ is exactly the term that normalizes the density or mass function.

## 3. Mean Identity

Differentiate $Z(\eta)$:

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

Substitute the expression for $\nabla Z(\eta)$:

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

## 4. Covariance Identity

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

The second derivative of $Z$ is:

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

For scalar $T(Y)$:

```math
a''(\eta)=\mathrm{Var}_\eta(T(Y))
```

## 5. Convexity of the Log-Partition Function

For any vector $v$:

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

So $a(\eta)$ is convex. It is strictly convex only when no nonzero direction $v$ makes $v^TT(Y)$ almost surely constant under the family.

## 6. Concavity of Log-Likelihood

For iid samples:

```math
\ell(\eta)
=
\eta^T\sum_{i=1}^{m}T(y^{(i)})
-ma(\eta)
+\sum_{i=1}^{m}\log b(y^{(i)})
```

Gradient:

```math
\nabla\ell(\eta)
=
\sum_{i=1}^{m}T(y^{(i)})
-m\nabla a(\eta)
```

Hessian:

```math
\nabla^2\ell(\eta)
=
-m\nabla^2a(\eta)
=
-m\,\mathrm{Cov}_\eta(T(Y))
```

Hence:

```math
\nabla^2\ell(\eta)\preceq0
```

So log-likelihood is concave in $\eta$.

## 7. Convexity of NLL

Negative log likelihood:

```math
J_{\mathrm{NLL}}(\eta)=-\ell(\eta)
```

Its Hessian is:

```math
\nabla^2J_{\mathrm{NLL}}(\eta)
=
m\,\mathrm{Cov}_\eta(T(Y))
\succeq0
```

Therefore NLL is convex in the natural parameter.

## 8. Strict-Convexity and MLE-Existence Caveats

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
* fitted probabilities are extremely close to $0$ or $1$;
* count rates are extremely large or small;
* features are poorly scaled.

The correct conclusion is therefore precise: exponential-family likelihoods have favorable convex geometry in natural parameters, but existence, uniqueness, and reliable computation require extra conditions.

