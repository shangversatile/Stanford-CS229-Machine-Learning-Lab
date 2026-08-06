# Sufficient Statistics, Likelihood, and Moments

Cross-link: see [Lecture 4 Conceptual Interlude A](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-a-what-information-about-a-parameter-is-actually-in-the-data), [Lecture 4 Conceptual Interlude D](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-d-why-glm-components-form-a-statistical-model), [Sufficient Statistics and Likelihood Equivalence](sufficient-statistics-likelihood-equivalence.md), [Exponential Family Anatomy](exponential-family-anatomy.md), [Log-Partition Mean, Variance, and Convexity](log-partition-mean-variance-convexity.md), and [GLM Construction Recipe](glm-construction-recipe.md).

## 1. The Inference Problem

The unknown parameter is not observed. The sample is observed. Statistical inference therefore asks which aspects of the sample can change likelihood comparisons among candidate parameters.

For an exponential family:

```math
\log p(y;\eta)
=
\log b(y)
+
\eta^TT(y)
-
a(\eta)
```

Comparing two candidate natural parameters gives:

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

The parameter-independent base term cancels. Thus $`T(y)`$ is the interface through which one observation affects parameter comparison. It is not an estimator. It is the parameter-relevant representation that estimators use.

## 2. Observation Statistic Versus Sample Statistic

The canonical statistic for one random observation is:

```math
T(Y_i)
```

After observation, the realized readout is:

```math
T(y_i)
```

For an iid sample, the random aggregate statistic is:

```math
S(\mathbf Y)
=
\sum_{i=1}^nT(Y_i)
```

and the observed aggregate statistic is:

```math
S(\mathbf y)
=
\sum_{i=1}^nT(y_i)
```

Sufficiency is a sample-level property. The phrase "the sufficient statistic is $`T(y)`$" is usually shorthand for "the iid sample sufficient statistic is the sum of the per-observation canonical statistics."

## 3. Formal Sufficiency

A statistic $`S(\mathbf Y)`$ is sufficient for $`\theta`$ when the conditional distribution of the full sample given $`S`$ does not depend on $`\theta`$:

```math
p_\theta(\mathbf Y=\mathbf y\mid S(\mathbf Y)=s)
```

is independent of $`\theta`$.

Before conditioning on $`S`$, the full sample can contain information about $`\theta`$. After fixing $`S=s`$, the remaining raw-sample details do not add likelihood information about that particular parameter under that particular model. Those details can still contain arrangement, residual shape, outliers, dependence, or information about a different parameter or a different model family.

## 4. Fisher-Neyman Factorization

A sufficient statistic can often be recognized by factorization:

```math
p_\theta(\mathbf y)
=
h(\mathbf y)
g_\theta^{\mathrm{fac}}(S(\mathbf y))
```

Here $`h(\mathbf y)`$ may depend on the full sample but not on $`\theta`$. The factor $`g_\theta^{\mathrm{fac}}`$ carries all parameter dependence through $`S(\mathbf y)`$. The superscript keeps this factorization term separate from the GLM link function $`g`$.

For a discrete sample space:

```math
P_\theta(\mathbf Y=\mathbf y\mid S=s)
=
\frac{
h(\mathbf y)g_\theta^{\mathrm{fac}}(s)
}{
\sum_{\mathbf y':S(\mathbf y')=s}
h(\mathbf y')g_\theta^{\mathrm{fac}}(s)
}
```

The common parameter factor cancels:

```math
P_\theta(\mathbf Y=\mathbf y\mid S=s)
=
\frac{
h(\mathbf y)
}{
\sum_{\mathbf y':S(\mathbf y')=s}
h(\mathbf y')
}
```

The resulting conditional probability contains no $`\theta`$. For continuous sample spaces, exact events may have probability zero, so the same argument is expressed through conditional densities.

## 5. Bernoulli Exact Example

Let:

```math
Y_i\sim\mathrm{Bernoulli}(p)
```

for iid samples indexed by $`i=1,2,...,n`$. The sample likelihood is:

```math
L(p)
=
\prod_{i=1}^n p^{y_i}(1-p)^{1-y_i}
```

Rearranging:

```math
L(p)
=
p^{\sum_i y_i}
(1-p)^{n-\sum_i y_i}
```

Define:

```math
S=\sum_iY_i
```

The total number of successes carries all likelihood information about $`p`$. Given $`S=k`$, $`p`$ no longer determines the arrangement of successes. Every binary sequence with $`k`$ ones has conditional probability:

```math
P(\mathbf Y=\mathbf y\mid S=k)
=
\frac{1}{\binom{n}{k}}
```

For instance:

```text
(1,0,1,0,1)
(0,1,1,1,0)
```

both have three successes in five observations. Their likelihoods as functions of $`p`$ are identical:

```math
p^3(1-p)^2
```

They differ in arrangement, but the arrangement does not add information about $`p`$ in the iid Bernoulli model.

## 6. Gaussian Known Variance, Unknown Mean

Let:

```math
Y_i\sim\mathcal N(\mu,\sigma^2)
```

with known $`\sigma^2`$. Up to constants independent of $`\mu`$:

```math
\ell(\mu)
=
-
\frac{1}{2\sigma^2}
\sum_{i=1}^n(y_i-\mu)^2
+
C
```

Expanding:

```math
\ell(\mu)
=
\frac{\mu}{\sigma^2}\sum_i y_i
-
\frac{n\mu^2}{2\sigma^2}
+
C'
```

The parameter-dependent part uses the sample only through:

```math
\sum_i y_i
```

The per-observation statistic is $`T(y)=y`$, and the sample sufficient statistic is $`\sum_i y_i`$, equivalently $`\bar y`$. Given the sample mean, the residual configuration around the mean does not add information about $`\mu`$ in this model. It may still matter for variance estimation, outlier detection, heteroscedasticity, or model diagnostics.

## 7. Gaussian Unknown Mean and Variance

When both $`\mu`$ and $`\sigma^2`$ are unknown:

```math
\ell(\mu,\sigma^2)
=
-
\frac n2\log\sigma^2
-
\frac{1}{2\sigma^2}
\sum_i(y_i-\mu)^2
+
C
```

Expanding the square:

```math
\ell(\mu,\sigma^2)
=
-
\frac n2\log\sigma^2
-
\frac{1}{2\sigma^2}
\left(
\sum_i y_i^2
-
2\mu\sum_i y_i
+
n\mu^2
\right)
+
C
```

The per-observation statistic is:

```math
T(y)
=
\begin{bmatrix}
y\\
y^2
\end{bmatrix}
```

and the sample sufficient statistic is:

```math
S(\mathbf y)
=
\begin{bmatrix}
\sum_i y_i\\
\sum_i y_i^2
\end{bmatrix}
```

The first raw moment provides location information. The second raw moment provides magnitude information. The variance is the central second moment:

```math
\mathrm{Var}(Y)
=
\mathbb E[Y^2]
-
\mathbb E[Y]^2
```

This uses both first and second raw moments. A regression residual square, such as $`(y_i-x_i^T\theta)^2`$, is different again: it is a fitted-model diagnostic or likelihood term after a mean model has been specified. The reason $`y^2`$ appears in the Gaussian sufficient statistic is that its coefficient depends on unknown $`\sigma^2`$, so it affects relative likelihood among candidate parameters.

## 8. Moment Matching for Iid Exponential Families

For iid data from:

```math
p(y;\eta)
=
b(y)
\exp\left(
\eta^TT(y)-a(\eta)
\right)
```

we have:

```math
\ell(\eta)
=
\eta^T\sum_iT(y_i)
-
na(\eta)
+
\sum_i\log b(y_i)
```

The score is:

```math
\nabla_\eta\ell(\eta)
=
\sum_iT(y_i)
-
n\nabla a(\eta)
```

Using:

```math
\nabla a(\eta)
=
\mathbb E_\eta[T(Y)]
```

an interior MLE satisfies:

```math
\frac1n\sum_iT(y_i)
=
\mathbb E_{\hat\eta}[T(Y)]
```

The empirical statistic mean equals the fitted model statistic mean. This is the likelihood-based statistic matching view of exponential-family MLE.

## 9. Response Mean Versus Statistic Mean

The response mean is:

```math
\mu=\mathbb E[Y]
```

The statistic expectation parameter is:

```math
m(\eta)
=
\mathbb E_\eta[T(Y)]
=
\nabla a(\eta)
```

If $`T(Y)=Y`$, then $`m(\eta)=\mu`$. If:

```math
T(Y)
=
\begin{bmatrix}
Y\\
Y^2
\end{bmatrix}
```

then:

```math
m(\eta)
=
\begin{bmatrix}
\mathbb E[Y]\\
\mathbb E[Y^2]
\end{bmatrix}
```

which is not a scalar response mean.

## 10. GLM Feature-Weighted Matching

In a shared-mean iid model, every sample uses the same $`\eta`$. In a GLM:

```math
\eta_i=x_i^T\theta
```

so the score equation becomes feature weighted. For a scalar canonical GLM:

```math
\nabla_\theta\ell(\theta)
=
\sum_i
x_i
\left(
T(y_i)
-
\mathbb E[T(Y_i)\mid x_i;\theta]
\right)
```

At an interior MLE:

```math
\sum_i x_iT(y_i)
=
\sum_i x_i\mathbb E[T(Y_i)\mid x_i;\hat\theta]
```

The fitted parameter does not estimate a separate $`\mu_i`$ from each $`y_i`$. It learns one shared $`\theta`$ so that, along each feature direction, observed statistics and model-expected statistics are balanced.

## 11. Model-Relative and Minimal-Sufficiency Caveat

Sufficiency is always relative to a chosen family and a chosen unknown parameter. Change the family, allow new parameters, introduce dependence, or model a different target, and the sufficient statistic may change.

A sufficient statistic need not be minimal. Minimal sufficiency asks for the coarsest statistic that still preserves all likelihood information for the parameter. Lecture 4 mainly needs the more basic fact: exponential-family likelihoods expose the parameter-relevant sample information through canonical statistics and their sums.