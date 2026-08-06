# Sufficient Statistics and Likelihood Equivalence

Cross-link: see [Lecture 4 Conceptual Interlude A](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-a-what-information-about-a-parameter-is-actually-in-the-data), [Exponential Family Anatomy](exponential-family-anatomy.md), [Log-Partition Mean, Variance, and Convexity](log-partition-mean-variance-convexity.md), and [GLM Construction Recipe](glm-construction-recipe.md).

This note separates five objects that are often collapsed in informal discussion:

```text
random variable -> realized observation -> one-observation statistic -> sample statistic -> likelihood equivalence class
```

The goal is to explain why canonical statistics appear in exponential-family probability models, when their aggregates are sufficient, and when a statistic is minimal sufficient.

## 1. Outcome space and random variables

Let the one-observation response space be:

```math
\mathcal Y
```

A response random variable is a measurable function:

```math
Y:\Omega\rightarrow\mathcal Y
```

Here $`\Omega`$ is the underlying sample space, $`\omega\in\Omega`$ is one underlying random outcome, and $`Y(\omega)`$ is the response value produced by that outcome. Usually we write the random variable simply as $`Y`$.

For repeated observations, the $`i`$th response random variable is:

```math
Y_i
```

The complete random sample is:

```math
\mathbf Y=(Y_1,\ldots,Y_n)
```

After observation, the realized value of $`Y_i`$ is:

```math
y_i
```

and the realized dataset is:

```math
\mathbf y=(y_1,\ldots,y_n)
```

Use this status rule:

```text
Y_i is random before observation.
y_i is fixed after observation.
```

The event:

```math
\{Y_i=y_i\}
```

is not the random variable $`Y_i`$ and is not a statistic. It is the event that the random variable takes the realized value.

## 2. Statistic as a function

A one-observation statistic is a function:

```math
T:\mathcal Y\rightarrow\mathbb R^m
```

Applied to a random variable, it gives another random variable:

```math
T(Y_i)
```

Applied to an observed value, it gives a fixed value:

```math
T(y_i)
```

Observation implies statistic equality:

```math
Y_i=y_i
\quad\Longrightarrow\quad
T(Y_i)=T(y_i)
```

The converse is generally false:

```math
T(Y_i)=T(y_i)
\quad\nRightarrow\quad
Y_i=y_i
```

because $`T`$ may be many-to-one. If:

```math
T(Y)=Y^2
```

then:

```math
\{T(Y)=4\}
=
\{Y=2\}\cup\{Y=-2\}
```

Thus $`T(Y)=T(y)`$ may identify a coarser event than $`Y=y`$.

## 3. Sample-level statistic

A sample statistic is a function of the whole sample:

```math
S:\mathcal Y^n\rightarrow\mathcal S
```

Before observation:

```math
S(\mathbf Y)
```

is random. After observation:

```math
S(\mathbf y)
```

is fixed.

For iid exponential-family data, the usual sample statistic is:

```math
S(\mathbf Y)=\sum_{i=1}^nT(Y_i)
```

and after observation:

```math
S(\mathbf y)=\sum_{i=1}^nT(y_i)
```

The distinction matters:

* $`T(Y_i)`$ is the one-observation canonical statistic;
* $`T(y_i)`$ is its realized value for sample $`i`$;
* $`S(\mathbf Y)`$ is a statistic of the whole random sample;
* $`S(\mathbf y)`$ is the observed compressed data value;
* sufficiency for iid exponential families is a sample-level statement about $`S(\mathbf Y)`$.

## 4. Why $`T(y)`$ appears in a probability model

Start with a general parametric family:

```math
p_\theta(y)
```

The density or mass function is already a function of the candidate parameter and the possible outcome. The exponential-family representation does not add a statistic after the probability model is built. It rewrites the original parameter-dependent outcome terms as:

```math
p_\eta(y)
=
b(y)
\exp\left(
\eta^TT(y)-a(\eta)
\right)
```

Taking logs:

```math
\log p_\eta(y)
=
\log b(y)
+
\eta^TT(y)
-
a(\eta)
```

The term $`T(y)`$ is the observation-side coordinate readout. The term $`\eta`$ is the parameter-side natural coordinate. Their product:

```math
\eta^TT(y)=\sum_j\eta_jT_j(y)
```

is the coupling score between parameter weights and observed statistical coordinates. The base measure $`b(y)`$ contains support and baseline weighting independent of the unknown parameter. The log-partition $`a(\eta)`$ normalizes the distribution.

This structure is part of the model representation. It is not created by MLE. MLE only uses this structure after the distribution family has been specified.

## 5. Parameter-relevant information through likelihood comparison

Likelihood reads the probability model in the inverse direction. The data are fixed and candidate parameters vary.

For a fixed observed dataset, compare:

```math
L(\theta_1;\mathbf y)
\quad\text{and}\quad
L(\theta_2;\mathbf y)
```

This asks which features of the fixed data change the ranking of candidate parameters.

A second comparison fixes the model family and compares possible data outcomes:

```math
\mathbf y
\quad\text{and}\quad
\mathbf y'
```

Define the likelihood ratio:

```math
R_\theta(\mathbf y,\mathbf y')
=
\frac{p_\theta(\mathbf y)}
{p_\theta(\mathbf y')}
```

If $`R_\theta(\mathbf y,\mathbf y')`$ is independent of $`\theta`$, then the relative likelihood of the two possible datasets is the same for every candidate parameter. They differ only by parameter-independent weighting, so they carry the same parameter-relevant evidence.

This defines a likelihood-induced equivalence relation:

```math
\mathbf y\sim\mathbf y'
\quad\Longleftrightarrow\quad
\frac{p_\theta(\mathbf y)}
{p_\theta(\mathbf y')}
\text{ is independent of }\theta
```

Minimal sufficient statistics encode these equivalence classes.

## 6. Parameter-relevant versus sufficient

A statistic is parameter-relevant if it changes the likelihood for an unknown parameter. It may contain only part of the information.

A statistic is sufficient if it contains all sample information about the target parameter under the specified model family.

```math
\text{parameter-relevant}
\;\nRightarrow\;
\text{sufficient}
```

Example:

```math
Y_i\sim\mathcal N(\mu,\sigma^2)
```

with both $`\mu`$ and $`\sigma^2`$ unknown. The statistic:

```math
\sum_iY_i
```

is relevant to $`\mu`$, but not sufficient for $`(\mu,\sigma^2)`$. The likelihood also depends on:

```math
\sum_iY_i^2
```

so the joint-parameter statistic is:

```math
S(\mathbf Y)
=
\left(
\sum_iY_i,
\sum_iY_i^2
\right)
```

Sufficiency is always relative to the chosen family and the parameter being estimated.

## 7. Formal sufficiency definition

For a statistic $`S(\mathbf Y)`$, sufficiency for $`\theta`$ means that conditioning on:

```math
S(\mathbf Y)=s
```

makes the conditional distribution of the full sample independent of $`\theta`$:

```math
p_\theta(\mathbf Y=\mathbf y\mid S(\mathbf Y)=s)
```

The statistic does not need to reconstruct the original data. It only needs to preserve all information about the specified parameter. Remaining sample details can still contain order, signs, residual configurations, or other facts, but those facts no longer affect inference for the target parameter once $`S`$ is known.

For continuous samples, exact sample-point probabilities are usually zero. The rigorous version uses conditional densities or regular conditional distributions.

## 8. Fisher-Neyman factorization proof

If the joint density or mass function factors as:

```math
p_\theta(\mathbf y)
=
h(\mathbf y)
g_\theta(S(\mathbf y))
```

then all dependence on $`\theta`$ passes through $`S(\mathbf y)`$.

For a discrete sample space, condition on $`S=s`$:

```math
P_\theta(\mathbf Y=\mathbf y\mid S=s)
=
\frac{
P_\theta(\mathbf Y=\mathbf y)
}{
P_\theta(S=s)
}
```

The denominator is:

```math
P_\theta(S=s)
=
\sum_{\mathbf y':S(\mathbf y')=s}
p_\theta(\mathbf y')
```

Using the factorization:

```math
P_\theta(\mathbf Y=\mathbf y\mid S=s)
=
\frac{
h(\mathbf y)g_\theta(s)
}{
\sum_{\mathbf y':S(\mathbf y')=s}
h(\mathbf y')g_\theta(s)
}
```

Cancel the common parameter factor:

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

The right side is independent of $`\theta`$, so $`S`$ is sufficient.

The factorization theorem does not add a mysterious conclusion. It verifies that after the algebraic split, no other data feature carries parameter dependence.

## 9. Minimal sufficiency

The complete-data statistic:

```math
S(\mathbf Y)=\mathbf Y
```

is always sufficient, but it does not compress.

A minimal sufficient statistic is the coarsest statistic that is still sufficient: it merges as many possible datasets as possible without losing parameter information. Under appropriate dominated-family and regularity conditions, $`S`$ is minimal sufficient if:

```math
\frac{p_\theta(\mathbf y)}
{p_\theta(\mathbf y')}
```

is independent of $`\theta`$ if and only if:

```math
S(\mathbf y)=S(\mathbf y')
```

Sufficient statistics are not unique. Minimal sufficient statistics are unique only up to one-to-one transformations. If:

```math
U=f(S)
```

and $`f`$ is one-to-one on the range of $`S`$, then $`U`$ is an equivalent minimal sufficient statistic. If $`f`$ is many-to-one, it may merge likelihood-distinct data outcomes and break sufficiency.

## 10. Bernoulli single observation

Let:

```math
Y\sim\mathrm{Bernoulli}(p)
```

with:

```math
\mathcal Y=\{0,1\}
```

The PMF is:

```math
P_p(Y=y)=p^y(1-p)^{1-y}
```

Rewrite:

```math
P_p(Y=y)
=
\exp\left(
y\log p+(1-y)\log(1-p)
\right)
```

```math
=
\exp\left(
y\log\frac{p}{1-p}
+
\log(1-p)
\right)
```

Define:

```math
\eta=\log\frac{p}{1-p}
```

Since:

```math
p=\frac{e^\eta}{1+e^\eta}
```

we have:

```math
\log(1-p)=-\log(1+e^\eta)
```

Therefore:

```math
P_\eta(Y=y)
=
\exp\left(
\eta y-\log(1+e^\eta)
\right)
```

The natural canonical statistic is:

```math
T(Y)=Y
```

This is not an arbitrary rule that $`Y`$ must be sufficient for itself. It is the coordinate that appears when the Bernoulli PMF is rewritten in canonical form.

The two outcomes cannot be merged because:

```math
\frac{P_p(Y=1)}
{P_p(Y=0)}
=
\frac{p}{1-p}
```

depends on $`p`$. Hence:

```math
0\nsim1
```

The minimal sufficient partition for one Bernoulli observation is:

```math
\{0\},
\qquad
\{1\}
```

The expression $`T(Y)=Y`$ is not uniquely necessary. The statistics:

```math
Y,
\qquad
1-Y,
\qquad
2Y+5
```

are equivalent because they are one-to-one transformations on $`\{0,1\}`$. What is necessary is preserving the distinction between the outcomes $`0`$ and $`1`$.

## 11. Multiple Bernoulli observations

Let:

```math
Y_1,\ldots,Y_n
\overset{\mathrm{iid}}{\sim}
\mathrm{Bernoulli}(p)
```

The random sample is:

```math
\mathbf Y=(Y_1,\ldots,Y_n)
```

and the observed sample is:

```math
\mathbf y=(y_1,\ldots,y_n)
```

The joint likelihood is:

```math
P_p(\mathbf Y=\mathbf y)
=
\prod_{i=1}^n p^{y_i}(1-p)^{1-y_i}
```

```math
=
p^{\sum_i y_i}
(1-p)^{n-\sum_i y_i}
```

Define the success count:

```math
K(\mathbf Y)=\sum_iY_i
```

Then $`T(Y_i)=Y_i`$ is the one-observation canonical statistic, while $`K(\mathbf Y)`$ is the sample-level sufficient statistic. It forgets order but keeps all likelihood information about $`p`$.

For two possible samples:

```math
\frac{
P_p(\mathbf Y=\mathbf y)
}{
P_p(\mathbf Y=\mathbf y')
}
=
\left(
\frac{p}{1-p}
\right)^{
\sum_i y_i-
\sum_i y_i'
}
```

This ratio is independent of $`p`$ if and only if:

```math
\sum_i y_i=\sum_i y_i'
```

Therefore $`K(\mathbf Y)`$ is minimal sufficient. For example:

```text
10101
01110
```

both have success count $`3`$, so they give the same likelihood:

```math
p^3(1-p)^2
```

and belong to the same likelihood-equivalence class.

## 12. Gaussian: known variance, unknown mean

Let:

```math
Y_i\sim\mathcal N(\mu,\sigma^2)
```

where $`\sigma^2`$ is known. The joint density is proportional to:

```math
\exp\left(
-\frac{1}{2\sigma^2}
\sum_i(y_i-\mu)^2
\right)
```

Expand the square:

```math
\sum_i(y_i-\mu)^2
=
\sum_i y_i^2
-
2\mu\sum_i y_i
+
n\mu^2
```

Thus:

```math
f_\mu(\mathbf y)
=
h(\mathbf y)
\exp\left(
\frac{\mu}{\sigma^2}\sum_i y_i
-
\frac{n\mu^2}{2\sigma^2}
\right)
```

where $`h(\mathbf y)`$ absorbs terms that do not depend on $`\mu`$. Therefore:

```math
S(\mathbf Y)=\sum_iY_i
```

is sufficient.

For minimality, compare two samples:

```math
\frac{f_\mu(\mathbf y)}
{f_\mu(\mathbf y')}
=
C(\mathbf y,\mathbf y')
\exp\left[
\frac{\mu}{\sigma^2}
\left(
\sum_i y_i-
\sum_i y_i'
\right)
\right]
```

The ratio is independent of $`\mu`$ if and only if:

```math
\sum_i y_i=\sum_i y_i'
```

so the sample sum is minimal sufficient under the usual regularity conditions.

## 13. Gaussian: mean zero, unknown variance

Let:

```math
Y_i\sim\mathcal N(0,\sigma^2)
```

The likelihood contains:

```math
\sum_iY_i^2
```

For one observation:

```math
T(Y)=Y^2
```

The likelihood ratio is:

```math
\frac{f_\sigma(y)}
{f_\sigma(y')}
=
\exp\left[
-\frac{y^2-y'^2}{2\sigma^2}
\right]
```

This is independent of $`\sigma^2`$ if and only if:

```math
y^2=y'^2
```

Thus $`y`$ and $`-y`$ are likelihood-equivalent for variance. Given $`Y^2`$, the sign contains no additional information about $`\sigma^2`$.

For iid samples, the sample statistic is:

```math
S(\mathbf Y)=\sum_iY_i^2
```

## 14. Gaussian: unknown mean and variance

Let:

```math
Y_i\sim\mathcal N(\mu,\sigma^2)
```

with both $`\mu`$ and $`\sigma^2`$ unknown. The joint log density is, up to constants:

```math
\ell(\mu,\sigma^2)
=
-\frac n2\log\sigma^2
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

The parameter-dependent sample statistic is:

```math
S(\mathbf Y)
=
\begin{bmatrix}
\sum_iY_i\\
\sum_iY_i^2
\end{bmatrix}
```

The corresponding one-observation canonical statistic is:

```math
T(Y)
=
\begin{bmatrix}
Y\\
Y^2
\end{bmatrix}
```

The natural parameters in a standard full-rank representation are:

```math
\eta_1=\frac{\mu}{\sigma^2},
\qquad
\eta_2=-\frac{1}{2\sigma^2}
```

The first coordinate reads location information. The second reads raw second-moment information. Together they determine mean and spread.

Minimality requires the model to be identifiable and the regularity assumptions behind the likelihood-ratio criterion to hold. Do not infer minimality merely from seeing two algebraic terms in an exponent.

## 15. Categorical example and redundancy

Let:

```math
Y\in\{1,\ldots,K\}
```

Class labels are nominal. They usually do not carry distance or order information. The natural readout is an indicator statistic:

```math
T_k(Y)=\mathbf1\{Y=k\}
```

A full one-hot vector has $`K`$ components, but:

```math
\sum_{k=1}^{K}T_k(Y)=1
```

so one component is linearly redundant. A reference-class representation keeps $`K-1`$ indicators and treats class $`K`$ as baseline. The natural parameters are log odds against the reference class:

```math
\eta_k=\log\frac{\phi_k}{\phi_K},
\qquad
k=1,\ldots,K-1
```

For iid categorical samples, the sample statistic is the vector of class counts. A complete one-hot count vector and a reference-class count vector can be information-equivalent, but the full vector carries a redundant sum-to-$`n`$ constraint. Redundancy affects identifiability even when it does not change the likelihood-equivalence partition.

## 16. Canonical statistics as coordinate bases

The vector:

```math
T(Y)
=
\begin{bmatrix}
T_1(Y)\\
\vdots\\
T_m(Y)
\end{bmatrix}
```

is best read as one chosen coordinate basis for the parameter-relevant functions of the observation. It is not always a polynomial basis. The more intrinsic object is the observation-function space spanned by log-likelihood ratios, after parameter-independent terms are removed:

```math
\log p_{\eta_1}(Y)-\log p_{\eta_2}(Y)
```

Those ratios identify the functions of $`Y`$ that can change how candidate parameters compare. The components $`T_1(Y),\ldots,T_m(Y)`$ choose coordinates for that space.

The natural parameter vector:

```math
\eta
=
\begin{bmatrix}
\eta_1\\
\vdots\\
\eta_m
\end{bmatrix}
```

weights those coordinates through:

```math
\eta^TT(Y)
=
\sum_j\eta_jT_j(Y)
```

For Bernoulli, after removing constants, the nontrivial parameter-relevant function space is one-dimensional, and $`Y`$ is the conventional basis. For Gaussian with unknown mean and variance, the relevant function space is spanned by $`Y`$ and $`Y^2`$. For categorical variables, indicator functions are the natural coordinates because category IDs are labels, not numeric magnitudes.

Any invertible basis transformation gives an equivalent exponential-family representation. Redundant coordinates give non-minimal representations and may create non-identifiability. Minimal sufficiency does not identify a literal formula; it identifies the data partition induced by likelihood equivalence.

## 17. MLE uses the structure; it does not create it

The model representation is:

```math
p_\eta(y)
=
b(y)\exp\left(
\eta^TT(y)-a(\eta)
\right)
```

For iid data, the likelihood is:

```math
L(\eta;\mathbf y)
=
\left[
\prod_i b(y_i)
\right]
\exp\left(
\eta^T\sum_iT(y_i)-na(\eta)
\right)
```

Maximum likelihood solves:

```math
\hat\eta
=
\underset{\eta}{\text{arg max}}
\,
L(\eta;\mathbf y)
```

MLE uses the sample statistic $`\sum_iT(y_i)`$, but it does not define $`T`$. The statistic comes from the model representation.

The log-likelihood is:

```math
\ell(\eta)
=
\eta^T\sum_iT(y_i)-na(\eta)+C
```

Differentiate:

```math
\nabla_\eta\ell(\eta)
=
\sum_iT(y_i)-n\nabla a(\eta)
```

Using:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

a finite interior MLE satisfies:

```math
\frac1n\sum_iT(y_i)
=
\mathbb E_{\hat\eta}[T(Y)]
```

The left side is the observed canonical-statistic average. The right side is the model-expected canonical statistic. This is the exponential-family moment-matching structure.

## 18. $`\mathbb E[Y]`$ versus $`\mathbb E[T(Y)]`$

The response mean is:

```math
\mathbb E[Y]
```

The statistic expectation is:

```math
\mathbb E[T(Y)]
```

They are the same only when $`T(Y)=Y`$. For Bernoulli and Poisson, this coincidence is convenient. For Gaussian with unknown mean and variance:

```math
T(Y)
=
\begin{bmatrix}
Y\\
Y^2
\end{bmatrix}
```

so:

```math
\mathbb E[T(Y)]
=
\begin{bmatrix}
\mathbb E[Y]\\
\mathbb E[Y^2]
\end{bmatrix}
```

Moment matching matches the expected canonical statistic, not necessarily only the scalar response mean. Minimal sufficiency and moment matching are also different concepts: minimal sufficiency is about preserving likelihood information in the data; moment matching is the first-order condition for a regular finite interior exponential-family MLE.

## 19. Caveats

* A sufficient statistic is always relative to a specified model family and target parameter.
* A canonical statistic is not automatically minimal.
* Minimal sufficiency requires dominated-family and regularity assumptions for the likelihood-ratio criterion to apply cleanly.
* Parameter-dependent support can break simple factorization intuition; Uniform $`(0,\theta)`$ is the standard warning case.
* Full exponential families, minimal exponential families, and curved exponential families are different objects.
* One-to-one transformations preserve statistical information, while redundant coordinates can affect identifiability.
* A statistic can be parameter-relevant without containing all parameter information.
