# GLM Construction Recipe

Cross-link: see the main Lecture 4 note sections [Conceptual Interlude C](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-c-from-random-samples-to-statistical-inference), [GLM Components](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#8-glm-components), [GLM Workflow](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#9-the-complete-glm-modeling-workflow), and [Hypothesis Function](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#10-deep-meaning-of-the-hypothesis-function). For related reference maps, see [Exponential Family Anatomy](exponential-family-anatomy.md), [GLM Response and Distribution Map](glm-response-distribution-map.md), and [Log-Partition Mean, Variance, and Convexity](log-partition-mean-variance-convexity.md).

```text
random sampling
-> observed sample
-> probability model
-> likelihood
-> maximum likelihood estimation
-> sufficient statistics
-> ordinary distribution parameters
-> natural parameters
-> global trainable parameters
-> systematic component
-> conditional prediction
```

This order matters because a GLM is not a curve-fitting trick. It is a constrained conditional-distribution model fitted by likelihood.

## 1. Conditional rather than joint modeling

A joint model factors as:

```math
p(x,y)
=
p(y\mid x)p(x)
```

Supervised prediction usually asks for $`Y`$ after $`x`$ is known. A conditional GLM therefore models $`p(Y_i\mid x_i;\theta)`$ directly. It can treat $`X`$ as fixed design, or condition on the observed covariates. This avoids modeling a high-dimensional marginal distribution for $`x`$ when that marginal distribution is not needed for the prediction objective.

If the joint model has independent parameter blocks:

```math
\ell(\theta,\alpha)
=
\sum_{i=1}^n\log p_\theta(y_i\mid x_i)
+
\sum_{i=1}^n\log p_\alpha(x_i)
```

then the optimizer for the conditional parameter $`\theta`$ is unaffected by the $`p_\alpha(x_i)`$ term. This is a modeling simplification, not a denial that a joint distribution exists.

Joint modeling can be necessary for generative sampling of covariates, missing covariates, latent variables, selection effects, and causal structure. Conditional modeling also does not automatically solve covariate shift; a changed deployment distribution or changed conditional mechanism still has to be diagnosed.

This is also the discriminative/generative taxonomy boundary. Logistic regression and conditional GLMs directly model $`p(y\mid x;\theta)`$ and do not model $`p(x)`$, so they are probabilistic discriminative models. Gaussian Discriminant Analysis and Naive Bayes model a joint distribution or class-conditional input distribution, such as $`p(y)p(x\mid y)`$, and are classical generative models. A GLM can sample $`Y\mid x`$ after $`x`$ is given, but that does not make it a model of complete $`(X,Y)`$ pairs.

## 2. Global and local parameter hierarchy

For sample $`i`$, use the hierarchy:

```text
global parameter theta
    +
local input x_i
    ↓
linear predictor xi_i
    ↓
natural parameter eta_i
    ↓
ordinary distribution parameter psi_i
    ↓
conditional distribution p(Y_i | x_i; theta)
    ↓
observed y_i
```

The global trainable parameter is:

```math
\theta\in\mathbb R^p
```

The input and design matrix are:

```math
x_i\in\mathbb R^p,
\qquad
X\in\mathbb R^{n\times p}
```

The systematic component is:

```math
\xi_i=x_i^T\theta
```

The local natural parameter is not the global parameter:

```math
\eta_i\neq\theta
```

In the scalar canonical construction, the relation is:

```math
\eta_i=x_i^T\theta
```

For all samples:

```math
\boldsymbol\eta=X\theta
```

Ordinary and natural parameters are coordinates of the same local conditional distribution:

```math
\eta_i=q(\psi_i)
```

```math
\psi_i=q^{-1}(\eta_i)
```

Examples:

| Family | Ordinary parameter $`\psi_i`$ | Natural parameter $`\eta_i`$ |
| ------ | ----------------------------- | ---------------------------- |
| Gaussian, CS229 variance one | mean $`\mu_i`$ | $`\eta_i=\mu_i`$ |
| Bernoulli | success probability $`p_i`$ | $`\eta_i=\log(p_i/(1-p_i))`$ |
| Poisson | rate $`\lambda_i`$ | $`\eta_i=\log\lambda_i`$ |

The learned object is $`\theta`$, the shared parameter of the mapping from features to distribution coordinates. The local ordinary parameters $`\psi_i`$ and natural parameters $`\eta_i`$ are induced per sample.

Only under a canonical link is:

```math
\xi_i=\eta_i=x_i^T\theta
```

For a general link, the guaranteed relation is:

```math
g(\mu_i)=\xi_i
```

with $`\eta_i=q(\mu_i)`$ determined by the distribution's mean-to-natural map.

## 3. Random component

The random component chooses the conditional response family:

```math
Y_i\mid x_i;\theta
\sim
\text{an exponential-family conditional distribution}
```

This defines the legal response support, probability or density shape, and variance behavior. It also defines the statistic $`T(Y_i)`$ whose mean is the GLM prediction.

In canonical form:

```math
p(y_i;\eta_i)
=
b(y_i)
\exp\left(
\eta_i^TT(y_i)-a(\eta_i)
\right)
```

The conditional model substitutes a feature-dependent local parameter:

```math
p(y_i\mid x_i;\theta)
=
b(y_i)
\exp\left(
\eta_i^TT(y_i)-a(\eta_i)
\right)
```

## 4. Systematic component, general link, and canonical link

The systematic component is the feature-side score:

```math
\xi_i
=
s_\theta(x_i)
=
x_i^T\theta
```

A general GLM links the conditional mean to this score:

```math
g(\mu_i)=\xi_i
```

where:

```math
\mu_i=\mathbb E[T(Y_i)\mid x_i;\theta]
```

The natural parameter is a distribution coordinate determined by the same mean:

```math
\eta_i=q(\mu_i)
```

A canonical link is the special case:

```math
g=q
```

Only in that case:

```math
\xi_i=\eta_i=x_i^T\theta
```

This equality is a condition, not a general GLM fact. With a noncanonical link, the systematic component $`\xi_i`$ and natural parameter $`\eta_i`$ are connected indirectly through $`\mu_i`$.

## 5. Mean map from the log-partition function

The log-partition function is:

```math
a(\eta)=\log\int b(y)\exp\left(\eta^TT(y)\right)dy
```

Under regularity conditions:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

For a scalar statistic:

```math
\mu=a'(\eta)
```

This is the response mapping from natural parameter to conditional mean. If the canonical scalar GLM sets $`\eta_i=x_i^T\theta`$, prediction is:

```math
h_\theta(x_i)
=
\mu_i
=
a'(x_i^T\theta)
```

For vector-valued statistics, the gradient gives the mean vector:

```math
h_\theta(x_i)
=
\mathbb E[T(Y_i)\mid x_i;\theta]
=
\nabla a(\eta_i)
```

## 6. Forward model, inverse learning, and prediction

Forward probability model:

```text
x_i
-> eta_i = x_i^T theta
-> ordinary parameter psi_i
-> conditional distribution p(Y_i | x_i; theta)
-> random sample y_i
```

Inverse learning:

```text
observed (X, y)
-> likelihood as a function of theta
-> maximum likelihood optimization
-> theta_hat
```

Prediction:

```text
x_new
-> eta_hat(x_new)
-> conditional distribution
-> mean / probability / predictive uncertainty
```

The forward direction explains what parameters do before sampling. The inverse direction explains why likelihood varies $`\theta`$ while holding observed $`(X,\mathbf y)`$ fixed. The prediction direction explains why $`h_\theta(x)`$ is a distributional summary rather than the trainable parameter itself.

## 7. Canonical log-likelihood and score equation

For a scalar canonical GLM:

```math
\eta_i=x_i^T\theta
```

Substitute into the exponential-family log density:

```math
\log p(y_i\mid x_i;\theta)
=
T(y_i)x_i^T\theta
-
a(x_i^T\theta)
+
\log b(y_i)
```

The full log-likelihood is:

```math
\ell(\theta)
=
\sum_{i=1}^n
\left(
T(y_i)x_i^T\theta
-
a(x_i^T\theta)
+
\log b(y_i)
\right)
```

Differentiate one sample. Since:

```math
\nabla_\theta(x_i^T\theta)=x_i
```

the gradient is:

```math
\nabla_\theta\log p(y_i\mid x_i;\theta)
=
x_i
\left(
T(y_i)-a'(x_i^T\theta)
\right)
```

Using:

```math
a'(x_i^T\theta)
=
\mathbb E[T(Y_i)\mid x_i;\theta]
```

we get the canonical GLM score equation:

```math
\nabla_\theta\ell(\theta)
=
\sum_{i=1}^n
x_i
\left(
T(y_i)
-
\mathbb E[T(Y_i)\mid x_i;\theta]
\right)
```

A finite interior MLE satisfies:

```math
\sum_{i=1}^n
x_i
\left(
T(y_i)
-
\mathbb E_{\hat\theta}[T(Y_i)\mid x_i]
\right)
=
0
```

The term $`T(y_i)-\mathbb E[T(Y_i)\mid x_i;\theta]`$ is a distributional residual: observed statistic minus model expectation. Each residual pushes the global parameter along the feature direction $`x_i`$. At the optimum, no feature direction has a remaining systematic residual, subject to existence and identifiability conditions. This is the algebraic link between $`X`$, $`Y_i`$, $`T(Y_i)`$, $`\eta_i`$, $`\theta`$, likelihood, and MLE.

This is feature-weighted matching, not shared-mean iid matching. A GLM does not estimate every $`\mu_i`$ by the global label mean. It learns one shared $`\theta`$ so every feature direction balances observed statistic against conditional model expectation.

## 8. Hessian and existence caveats

For the per-sample negative log-likelihood:

```math
J_i(\theta)
=
a(\eta_i)-\eta_iT(y_i)-\log b(y_i)
```

the scalar canonical Hessian is:

```math
\nabla_\theta^2J_i(\theta)
=
a''(\eta_i)x_ix_i^T
```

and:

```math
a''(\eta_i)=\mathrm{Var}(T(Y_i)\mid x_i;\theta)
```

So canonical GLM curvature is variance-weighted feature geometry.

Convex-friendly does not mean every MLE is finite, unique, or numerically stable. Important caveats include:

* finite MLE may not exist under logistic or softmax perfect separation;
* rank deficiency in $`X`$ can make parameters non-identifiable;
* softmax parameters need an identifiability convention such as a reference class or equivalent constraint;
* boundary solutions can occur when observed statistics lie on the boundary of the achievable mean set;
* regularization is often added to stabilize estimation and select among weakly identified directions;
* noncanonical links can lose the clean canonical Hessian form.

## 9. Column-space meaning of the systematic component

In the canonical scalar model:

```math
\boldsymbol\eta=X\theta
```

The $`n`$ local natural parameters are constrained to lie in the column space of $`X`$. When $`p\ll n`$, this is a strong low-dimensional restriction: the model cannot choose an arbitrary natural parameter for each sample.

This restriction is statistical sharing. Every sample contributes to the same $`\theta`$, and the fitted $`\theta`$ can be reused on a new feature vector $`x_{\mathrm{new}}`$. The price is possible underfit when the true natural-parameter vector is not well approximated by the column-space constraint.

This is not:

```math
Y=X\theta
```

It is:

```math
\text{natural coordinate of }p(Y_i\mid x_i)
=
x_i^T\theta
```

The realized response remains random.

## 10. Gaussian regression sufficient statistics

For fixed-design Gaussian regression:

```math
\mathbf y\mid X;\theta
\sim
\mathcal N(X\theta,\sigma^2I)
```

If $`\sigma^2`$ is known, the log-likelihood is:

```math
\ell(\theta)
=
-\frac{1}{2\sigma^2}
(\mathbf y-X\theta)^T(\mathbf y-X\theta)
+
C
```

Expand:

```math
\ell(\theta)
=
\frac{1}{\sigma^2}
\theta^TX^T\mathbf y
-
\frac{1}{2\sigma^2}
\theta^TX^TX\theta
+
C(X,\mathbf y)
```

Given $`X`$, the response-dependent statistic for $`\theta`$ is:

```math
X^T\mathbf y
=
\sum_i x_i y_i
```

If $`\sigma^2`$ is unknown too, the likelihood also depends on:

```math
\mathbf y^T\mathbf y
```

This differs from the iid shared-mean Gaussian model. There, the common mean uses $`\sum_i y_i`$. In regression, different means are tied together by the features, so $`X^T\mathbf y`$ is the feature-weighted sufficient statistic for the global slope vector.

## 11. Linearity is a design choice

CS229 uses:

```math
\eta_i=x_i^T\theta
```

because it gives interpretable additive effects, a compact shared parameter, and clean likelihood geometry for canonical links. It is not a theorem that the true data-generating mechanism must be linear on the natural-parameter scale.

The representation can be expanded while preserving the same observation model:

```math
\eta_i=\phi(x_i)^T\theta
```

or, in a more flexible conditional model:

```math
\eta_i=f_\theta(x_i)
```

The separation remains important. The representation model maps features to a distribution coordinate. The observation model maps that coordinate to $`p(Y_i\mid x_i)`$. A richer representation can reduce underfit, but it does not automatically fix the wrong response family, wrong variance structure, missing exposure, dependence, or calibration failure.

## 12. Canonical examples

Gaussian with CS229 fixed variance one:

```math
\eta_i=\mu_i=x_i^T\theta
```

```math
h_\theta(x_i)=\mu_i=x_i^T\theta
```

Bernoulli logistic regression:

```math
\eta_i=\log\frac{p_i}{1-p_i}=x_i^T\theta
```

```math
h_\theta(x_i)=p_i=\frac{1}{1+\exp(-x_i^T\theta)}
```

Poisson regression:

```math
\eta_i=\log\lambda_i=x_i^T\theta
```

```math
h_\theta(x_i)=\lambda_i=\exp(x_i^T\theta)
```

The same linear score has different statistical meanings because it is placed on different distribution scales: mean, log-odds, or log-rate.

## 13. Conditional residual interpretation

For scalar response GLMs, define the conditional mean residual:

```math
\epsilon_i=Y_i-\mu_i
```

where:

```math
\mu_i=\mathbb E[Y_i\mid x_i;\theta]
```

Then:

```math
Y_i=\mu_i+\epsilon_i
```

and:

```math
\mathbb E[\epsilon_i\mid x_i]=0
```

This is a conditional-mean decomposition. It does not impose Gaussianity, independence, or constant variance. Gaussian identity-link regression is the special case where $`\mu_i=x_i^T\theta`$ and the model may be equivalently written as linear signal plus conditional Gaussian noise. Bernoulli and Poisson GLMs instead have distribution-specific residual supports and variance structures.

## 14. Summary

A GLM is a disciplined way to learn a conditional distribution. The random component selects the response family. The ordinary parameter names the familiar local distribution member. The natural parameter is the canonical local coordinate. The systematic component maps features and the global trainable parameter into a score. The link decides when that score is also the natural parameter. Likelihood training then matches observed sufficient statistics to model expectations through feature-weighted score equations, and residual diagnostics interpret deviations around the conditional mean.
