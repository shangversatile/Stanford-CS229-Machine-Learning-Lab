# MLE, MAP, and Beta-Bernoulli

Cross-link: see [Module 01 sections 5-12](../README.md#5-mle-as-a-general-framework).

CS229 bridge: [Lecture 4 sufficient statistics](../../../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-a-what-information-about-a-parameter-is-actually-in-the-data) explains why count statistics can preserve parameter-relevant likelihood information.

Source boundary: independent derivation guided by CMU 10-601 Spring 2023 Lecture 16 selected MLE/MAP segment, Lecture 17 MAP opening, and Tom Mitchell's MLE/MAP reading.

## 1. Likelihood as a Function of Parameters

Let $\mathcal D=\{z^{(i)}\}_{i=1}^{m}$ be observed and fixed. For a parametric model $p(z\mid\theta)$:

```math
L(\theta;\mathcal D)
=
p(\mathcal D\mid\theta).
```

The candidate parameter is $\theta$. The likelihood is evaluated as $\theta$ varies; it is not a posterior distribution over $\theta$.

Under iid sampling:

```math
L(\theta;\mathcal D)
=
\prod_{i=1}^{m}
p(z^{(i)}\mid\theta).
```

Taking logs gives:

```math
\ell(\theta)
=
\log L(\theta;\mathcal D)
=
\sum_{i=1}^{m}
\log p(z^{(i)}\mid\theta).
```

Thus:

```math
\hat\theta_{\mathrm{MLE}}
=
\underset{\theta}{\mathrm{argmax}}
\,
\ell(\theta).
```

## 2. Sufficient-Statistic Bridge

If:

```math
L(\theta;\mathcal D)
=
h(\mathcal D)
g_{\theta}(S(\mathcal D)),
```

then $h(\mathcal D)$ cannot affect the maximizing $\theta$:

```math
\underset{\theta}{\mathrm{argmax}}
\,
L(\theta;\mathcal D)
=
\underset{\theta}{\mathrm{argmax}}
\,
g_{\theta}(S(\mathcal D)).
```

So the parameter-relevant evidence in the data is compressed by $S(\mathcal D)$. This is the CS229 Lecture 4 sufficient-statistics bridge into CMU's closed-form MLE/MAP recipe.

## 3. MAP Objective

MAP starts from:

```math
\hat\theta_{\mathrm{MAP}}
=
\underset{\theta}{\mathrm{argmax}}
\,
p(\theta\mid\mathcal D).
```

Bayes rule gives:

```math
p(\theta\mid\mathcal D)
=
\frac{
p(\mathcal D\mid\theta)p(\theta)
}{
p(\mathcal D)
}.
```

Because $p(\mathcal D)$ is constant with respect to $\theta$:

```math
\hat\theta_{\mathrm{MAP}}
=
\underset{\theta}{\mathrm{argmax}}
\,
p(\mathcal D\mid\theta)p(\theta).
```

The log objective is:

```math
\ell_{\mathrm{MAP}}(\theta)
=
\log p(\mathcal D\mid\theta)
+
\log p(\theta).
```

MLE is recovered when the prior is constant over the relevant parameter region.

## 4. Bernoulli MLE

Assume $Y_i\sim\mathrm{Bernoulli}(\phi)$, with $y_i\in\{0,1\}$. Define:

```math
N_1
=
\sum_{i=1}^{m}
y_i,
```

and:

```math
N_0
=
m-N_1.
```

The likelihood is:

```math
p(\mathcal D\mid\phi)
=
\prod_{i=1}^{m}
\phi^{y_i}(1-\phi)^{1-y_i}
=
\phi^{N_1}(1-\phi)^{N_0}.
```

The log-likelihood is:

```math
\ell(\phi)
=
N_1\log\phi
+
N_0\log(1-\phi).
```

Differentiate:

```math
\frac{d\ell}{d\phi}
=
\frac{N_1}{\phi}
-
\frac{N_0}{1-\phi}.
```

Set equal to zero:

```math
\frac{N_1}{\phi}
=
\frac{N_0}{1-\phi}.
```

Then:

```math
N_1(1-\phi)
=
N_0\phi.
```

So:

```math
N_1
=
(N_1+N_0)\phi
=
m\phi.
```

Therefore:

```math
\hat\phi_{\mathrm{MLE}}
=
\frac{N_1}{m}.
```

## 5. Beta Prior and Posterior

Let:

```math
\phi
\sim
\mathrm{Beta}(\alpha,\beta).
```

The density has proportional form:

```math
p(\phi)
\propto
\phi^{\alpha-1}(1-\phi)^{\beta-1}.
```

The posterior is proportional to likelihood times prior:

```math
p(\phi\mid\mathcal D)
\propto
\phi^{N_1}(1-\phi)^{N_0}
\phi^{\alpha-1}(1-\phi)^{\beta-1}.
```

Combine powers:

```math
p(\phi\mid\mathcal D)
\propto
\phi^{N_1+\alpha-1}
(1-\phi)^{N_0+\beta-1}.
```

This is a Beta density:

```math
\phi\mid\mathcal D
\sim
\mathrm{Beta}
(
N_1+\alpha,
N_0+\beta
).
```

## 6. Beta-Bernoulli MAP

The posterior log-density is:

```math
\ell(\phi)
=
(
N_1+\alpha-1
)
\log\phi
+
(
N_0+\beta-1
)
\log(1-\phi)
+
C.
```

Differentiate:

```math
\frac{d\ell}{d\phi}
=
\frac{
N_1+\alpha-1
}{
\phi
}
-
\frac{
N_0+\beta-1
}{
1-\phi
}.
```

Set the derivative to zero:

```math
\frac{
N_1+\alpha-1
}{
\phi
}
=
\frac{
N_0+\beta-1
}{
1-\phi
}.
```

Cross-multiply:

```math
(N_1+\alpha-1)(1-\phi)
=
(N_0+\beta-1)\phi.
```

Collect terms:

```math
N_1+\alpha-1
=
(N_1+N_0+\alpha+\beta-2)\phi.
```

Since $m=N_1+N_0$:

```math
\hat\phi_{\mathrm{MAP}}
=
\frac{
N_1+\alpha-1
}{
m+\alpha+\beta-2
}.
```

The derivation assumes an interior maximum. The posterior mode is interior if $N_1+\alpha>1$ and $N_0+\beta>1$. Otherwise a boundary mode can occur.

## 7. Posterior Mean versus MAP

For:

```math
\phi\mid\mathcal D
\sim
\mathrm{Beta}
(
N_1+\alpha,
N_0+\beta
),
```

the posterior mean is:

```math
E[\phi\mid\mathcal D]
=
\frac{N_1+\alpha}{m+\alpha+\beta}.
```

The posterior mode is:

```math
\hat\phi_{\mathrm{MAP}}
=
\frac{N_1+\alpha-1}{m+\alpha+\beta-2}.
```

These are different estimators. The posterior mean uses the first moment of the posterior; MAP uses the maximizing value.

## 8. Pseudo-Count Interpretation

The MAP formula can be written as:

```math
\hat\phi_{\mathrm{MAP}}
=
\frac{
N_1+(\alpha-1)
}{
N_1+N_0+(\alpha-1)+(\beta-1)
}.
```

This looks like the empirical-frequency estimator after adding $\alpha-1$ prior ones and $\beta-1$ prior zeros. That is an interpretation of the prior's effect, not a claim that the data literally contain those observations.
