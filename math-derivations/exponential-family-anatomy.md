# Exponential Family Anatomy

Cross-link: see [Lecture 4 Section 6: Anatomy of the Exponential Family](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#6-anatomy-of-the-exponential-family), [Lecture 4 Conceptual Interlude A: What Information About a Parameter Is Actually in the Data?](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-a-what-information-about-a-parameter-is-actually-in-the-data), [Sufficient Statistics and Likelihood Equivalence](sufficient-statistics-likelihood-equivalence.md), [Log-Partition Mean, Variance, and Convexity](log-partition-mean-variance-convexity.md), [GLM Construction Recipe](glm-construction-recipe.md), and [Why Exponential Family and GLM Exist](why-exponential-family-and-glm.md).

## 1. Canonical form

The canonical exponential-family form is:

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

It separates four jobs: what is read from the observation, which parameter coordinate controls the family member, how normalization happens, and what baseline measure remains.

| Component | Role | Meaning |
| --------- | ---- | ------- |
| $`Y`$ | random response | outcome before observation |
| $`y`$ | possible or realized value | fixed value evaluated by mass or density |
| $`T(y)`$ | one-observation canonical statistic | observation-side coordinate readout |
| $`\eta`$ | natural parameter | parameter-side coordinate inside the family |
| $`\eta^TT(y)`$ | coupling | score matching parameter weights with observation coordinates |
| $`a(\eta)`$ | log-partition | normalizer, moment generator, and curvature engine |
| $`b(y)`$ | base measure | parameter-independent support and baseline weighting |

Do not treat $`T(y)`$ as a statistic inserted by an estimator. It is part of the probability model's representation before estimation begins.

## 2. Why a function of $`y`$ appears

Start from a distribution written in ordinary parameters:

```math
p_\psi(y)
```

When this family can be expressed in canonical exponential-family form, the parameter-dependent outcome terms are reorganized into:

```math
\log p_\eta(y)
=
\log b(y)+\eta^TT(y)-a(\eta)
```

Thus $`T(y)`$ is found by algebraically identifying which functions of $`y`$ are coupled to unknown parameters. It is not an extra feature added after the density is known.

Example: Bernoulli.

```math
p(y;p)=p^y(1-p)^{1-y}
```

```math
\log p(y;p)
=
y\log\frac{p}{1-p}+\log(1-p)
```

The parameter-dependent outcome term is $`y`$, so the natural canonical statistic is $`T(y)=y`$ and the natural parameter is:

```math
\eta=\log\frac{p}{1-p}
```

Example: Gaussian with unknown mean and variance.

```math
\log p(y;\mu,\sigma^2)
=
-\frac{y^2}{2\sigma^2}
+
\frac{\mu}{\sigma^2}y
-
\frac{\mu^2}{2\sigma^2}
-
\log(\sqrt{2\pi}\sigma)
```

The parameter-dependent outcome functions are $`y`$ and $`y^2`$, so a natural statistic is:

```math
T(y)=
\begin{bmatrix}
y\\
y^2
\end{bmatrix}
```

## 3. Coupling as statistical coordinates

The term:

```math
\eta^TT(y)
=
\sum_j\eta_jT_j(y)
```

is a bilinear coupling between parameter-side and observation-side coordinates. If $`T_j(y)`$ is large for an outcome and $`\eta_j`$ increases, that outcome receives a larger unnormalized log-score in that direction.

For two candidate natural parameters:

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

The base term cancels because it is parameter-independent. This is why $`T(y)`$ is the parameter-relevant readout from one observation.

## 4. Random variable, realization, and sample statistic

For one observation:

```math
T(Y_i)
```

is random before sample $`i`$ is observed, while:

```math
T(y_i)
```

is fixed after observation. The sample-level statistic is separate. In the iid exponential-family case:

```math
S(\mathbf Y)=\sum_{i=1}^nT(Y_i)
```

and:

```math
S(\mathbf y)=\sum_{i=1}^nT(y_i)
```

Sufficiency is a statement about the sample statistic $`S(\mathbf Y)`$, not about an isolated $`T(Y_i)`$ unless the sample has only one observation.

## 5. Why the iid sum appears

For iid data:

```math
p(y_1,\ldots,y_n;\eta)
=
\prod_{i=1}^n b(y_i)\exp\left(\eta^TT(y_i)-a(\eta)\right)
```

Collect terms:

```math
p(y_1,\ldots,y_n;\eta)
=
\left(\prod_{i=1}^n b(y_i)\right)
\exp\left(
\eta^T\sum_{i=1}^nT(y_i)-na(\eta)
\right)
```

The same $`\eta`$ multiplies every observation, so the one-observation readouts add. This is why:

```math
\sum_{i=1}^nT(Y_i)
```

appears naturally as the iid sample statistic.

By Fisher-Neyman factorization, this statistic is sufficient for $`\eta`$ in the regular iid model. For minimality, one must additionally check likelihood-ratio equivalence and redundancy.

## 6. Ordinary parameter versus natural parameter

Ordinary parameters are familiar coordinates for naming a distribution member. Natural parameters are canonical exponential-family coordinates. They are related by reparameterization within the same family:

```math
\eta=q(\psi)
```

```math
\psi=q^{-1}(\eta)
```

Examples:

| Family | Ordinary parameter | Natural parameter |
| ------ | ------------------ | ----------------- |
| Bernoulli | $`p`$ | $`\log(p/(1-p))`$ |
| Poisson | $`\lambda`$ | $`\log\lambda`$ |
| Gaussian, CS229 variance one | $`\mu`$ | $`\mu`$ |
| Gaussian, unknown mean and variance | $`(\mu,\sigma^2)`$ | $`(\mu/\sigma^2,-1/(2\sigma^2))`$ |
| Categorical, reference class | $`\phi_1,\ldots,\phi_K`$ | $`\log(\phi_k/\phi_K)`$ for $`k<K`$ |

Do not treat the CS229 variance-one Gaussian simplification as the whole Gaussian parameterization.

## 7. Non-uniqueness and redundancy

The canonical statistic is not unique as a literal expression. What is determined by the family, after separating parameter-independent terms, is the parameter-relevant function space on the outcome domain and the dimension needed for an identifiable nonredundant representation. Equivalently, look at log-likelihood ratios across candidate parameters: after parameter-independent terms are removed, the remaining functions of the observation span the directions that can change parameter comparisons. The components $`T_1(Y),\ldots,T_m(Y)`$ are a coordinate basis for that space, not objects generated by the realized data.

For Bernoulli, the nonconstant parameter-relevant space is one-dimensional. The usual basis is:

```math
T(Y)=Y
```

but $`1-Y`$ or $`2Y+5`$ preserve the same distinction between outcomes $`0`$ and $`1`$.

For Gaussian with unknown mean and variance, the relevant space is spanned by:

```math
Y
\quad\text{and}\quad
Y^2
```

Any invertible basis transformation gives an equivalent representation. A redundant statistic may still represent the same family but can make the natural parameter non-identifiable. Minimal sufficient statistics are unique only up to one-to-one transformations and are best understood through likelihood-equivalence classes.

## 8. Canonical statistic is not always a polynomial basis

The vector:

```math
T(Y)=
\begin{bmatrix}
T_1(Y)\\
\vdots\\
T_m(Y)
\end{bmatrix}
```

lists statistical coordinates. These coordinates need not be powers of $`Y`$. They can be indicators, squared values, absolute values, or other functions determined by the family and target parameter.

| Family | Why the statistic has this form |
| ------ | ------------------------------- |
| Bernoulli | two outcomes differ by event occurrence, so $`Y`$ records success |
| Gaussian, known variance | only the mean is unknown, so $`Y`$ carries location information |
| Gaussian, unknown mean and variance | $`Y`$ carries location and $`Y^2`$ carries raw second-moment information |
| Categorical | labels are nominal, so indicator functions record class identity |

## 9. Sufficient, minimal sufficient, and parameter-relevant

Parameter-relevant does not imply sufficient:

```math
\text{parameter-relevant}
\;\nRightarrow\;
\text{sufficient}
```

A statistic can affect one parameter coordinate while omitting another. For Gaussian data with both mean and variance unknown, $`\sum_iY_i`$ is relevant but not sufficient because the likelihood also needs $`\sum_iY_i^2`$.

Minimal sufficiency is stricter. Under dominated-family and regularity conditions, $`S`$ is minimal sufficient when:

```math
\frac{p_\theta(\mathbf y)}
{p_\theta(\mathbf y')}
```

is independent of $`\theta`$ if and only if:

```math
S(\mathbf y)=S(\mathbf y')
```

This criterion checks whether the level sets of $`S`$ are exactly the likelihood-equivalence classes.

## 10. Log-partition as moment engine

The log-partition function is:

```math
a(\eta)=\log\int b(y)\exp\left(\eta^TT(y)\right)dy
```

Under standard differentiability conditions:

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

The first derivative gives the expected canonical statistic, not necessarily the scalar response mean. They coincide only when $`T(Y)=Y`$.

## 11. MLE as moment matching, not structure creation

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

A finite interior MLE satisfies:

```math
\frac1n\sum_{i=1}^nT(y_i)
=
\mathbb E_{\hat\eta}[T(Y)]
```

The left side is the empirical canonical-statistic average. The right side is the model-expected canonical statistic. MLE uses the exponential-family coupling and sample compression, but it does not define $`T`$ or create the term $`\eta^TT(y)`$.

## 12. From natural parameter to GLM

In an unconditional exponential-family model, $`\eta`$ is fixed. A canonical GLM makes the local natural coordinate depend on features:

```math
\eta_i=x_i^T\theta
```

The conditional distribution is:

```math
p(y_i\mid x_i;\theta)
=
b(y_i)\exp\left(\eta_i^TT(y_i)-a(\eta_i)\right)
```

The prediction on the statistic scale is:

```math
\mathbb E[T(Y_i)\mid x_i;\theta]
=
\nabla a(\eta_i)
```

The feature vector does not directly force $`Y_i=x_i^T\theta`$. It selects a coordinate of the conditional distribution from which $`Y_i`$ is randomly realized.

## 13. Caveats

* A sufficient statistic is always relative to the model family and target parameter.
* A canonical statistic is not automatically minimal.
* Parameter-dependent support can break simple factorization intuition.
* Full exponential families, minimal exponential families, and curved exponential families should not be conflated.
* One-to-one transformations preserve information; redundant coordinates can damage identifiability.
* Moment matching is an MLE first-order condition in regular exponential families, not the origin of the exponential-family representation.
