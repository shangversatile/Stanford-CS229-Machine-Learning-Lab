# Bernoulli Naive Bayes Estimation and Logistic Posterior

Cross-link: see [Module 01 sections 13-21](../README.md#13-bernoulli-naive-bayes).

CS229 bridge: [Lecture 5 Naive Bayes](../../../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#15-naive-bayes-for-discrete-features) gives the CS229 mainline factorization; this supplement expands the count-based estimation and logistic posterior connection.

Source boundary: independent derivation guided by CMU 10-601 Spring 2023 Lecture 17 and Tom Mitchell's Naive Bayes / Logistic Regression reading.

## 1. Model

Let $Y\in\{0,1,\ldots,K-1\}$ and $X\in\{0,1\}^d$. The class prior is:

```math
P(Y=k)
=
\pi_k.
```

For a Bernoulli feature in class $k$:

```math
\phi_{jk}
=
P(X_j=1\mid Y=k).
```

The Naive Bayes assumption is:

```math
P(X=x\mid Y=k)
=
\prod_{j=1}^{d}
P(X_j=x_j\mid Y=k).
```

Therefore:

```math
P(X=x\mid Y=k)
=
\prod_{j=1}^{d}
\phi_{jk}^{x_j}
(1-\phi_{jk})^{1-x_j}.
```

The joint model is:

```math
P(X=x,Y=k)
=
\pi_k
\prod_{j=1}^{d}
\phi_{jk}^{x_j}
(1-\phi_{jk})^{1-x_j}.
```

## 2. Parameter Counting

Without conditional independence, $P(X\mid Y=k)$ is a categorical distribution over $2^d$ binary configurations. Because probabilities sum to $1$, this requires $2^d-1$ free parameters per class.

With Bernoulli Naive Bayes, each class needs $d$ Bernoulli parameters $\phi_{1k},\ldots,\phi_{dk}$. The class prior adds $K-1$ independent parameters.

The independence assumption changes the scaling from exponential to linear in $d$.

## 3. MLE for Class Prior

The class-prior likelihood factor is:

```math
\prod_{i=1}^{m}
\pi_{y^{(i)}}.
```

Define:

```math
N_k
=
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}.
```

The MLE is:

```math
\hat\pi_k
=
\frac{N_k}{m}.
```

For multiple classes, this follows from maximizing $\sum_k N_k\log\pi_k$ subject to $\sum_k\pi_k=1$.

## 4. MLE for Feature Parameters

For a fixed feature $j$ and class $k$, only examples with $y^{(i)}=k$ affect $\phi_{jk}$. Define:

```math
N_{jk,1}
=
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}
x_j^{(i)}.
```

Also:

```math
N_{jk,0}
=
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}
(1-x_j^{(i)}).
```

Then $N_k=N_{jk,1}+N_{jk,0}$ for that feature and class. The relevant log-likelihood is:

```math
\ell(\phi_{jk})
=
N_{jk,1}\log\phi_{jk}
+
N_{jk,0}\log(1-\phi_{jk}).
```

By the Bernoulli MLE derivation:

```math
\hat\phi_{jk}
=
\frac{
N_{jk,1}
}{
N_k
}.
```

Expanded over examples:

```math
\hat\phi_{jk}
=
\frac{
\sum_{i=1}^{m}
\mathbf{1}
\{
y^{(i)}=k
\}
x_j^{(i)}
}{
\sum_{i=1}^{m}
\mathbf{1}
\{
y^{(i)}=k
\}
}.
```

## 5. MAP for Feature Parameters

Assume independent Beta priors:

```math
\phi_{jk}
\sim
\mathrm{Beta}(\alpha,\beta).
```

The posterior has shape parameters:

```math
N_{jk,1}+\alpha,
\quad
N_{jk,0}+\beta.
```

The interior MAP estimate is:

```math
\hat\phi_{jk,\mathrm{MAP}}
=
\frac{
N_{jk,1}+\alpha-1
}{
N_k+\alpha+\beta-2
}.
```

This reduces extreme estimates when prior shape parameters pull mass away from $0$ or $1$. Boundary cases must be handled when a posterior shape parameter is at or below $1$.

## 6. Prediction

For any class $k$:

```math
P(Y=k\mid X=x)
=
\frac{
\pi_k p(x\mid Y=k)
}{
\sum_{\ell}
\pi_{\ell}p(x\mid Y=\ell)
}.
```

The denominator is common across candidate classes, so:

```math
\hat y
=
\underset{k}{\mathrm{argmax}}
\,
\pi_k p(x\mid Y=k).
```

Using logs:

```math
\hat y
=
\underset{k}{\mathrm{argmax}}
\,
\left[
\log\pi_k
+
\sum_{j=1}^{d}
x_j\log\phi_{jk}
+
(1-x_j)\log(1-\phi_{jk})
\right].
```

## 7. Logistic Posterior for Binary Bernoulli NB

For $Y\in\{0,1\}$:

```math
\log
\frac{
P(Y=1\mid x)
}{
P(Y=0\mid x)
}
=
\log
\frac{
\pi_1p(x\mid Y=1)
}{
\pi_0p(x\mid Y=0)
}.
```

Substitute the Bernoulli NB likelihood:

```math
=
\log\frac{\pi_1}{\pi_0}
+
\sum_{j=1}^{d}
\left[
x_j\log\frac{\phi_{j1}}{\phi_{j0}}
+
(1-x_j)\log\frac{1-\phi_{j1}}{1-\phi_{j0}}
\right].
```

Group all terms that do not multiply $x_j$:

```math
=
\log\frac{\pi_1}{\pi_0}
+
\sum_{j=1}^{d}
\log\frac{1-\phi_{j1}}{1-\phi_{j0}}
+
\sum_{j=1}^{d}
x_j
\left[
\log\frac{\phi_{j1}}{\phi_{j0}}
-
\log\frac{1-\phi_{j1}}{1-\phi_{j0}}
\right].
```

Define:

```math
b
=
\log\frac{\pi_1}{\pi_0}
+
\sum_{j=1}^{d}
\log\frac{1-\phi_{j1}}{1-\phi_{j0}}.
```

Define:

```math
w_j
=
\log
\frac{
\phi_{j1}(1-\phi_{j0})
}{
\phi_{j0}(1-\phi_{j1})
}.
```

Then:

```math
\log
\frac{
P(Y=1\mid x)
}{
P(Y=0\mid x)
}
=
w^Tx+b.
```

Therefore:

```math
P(Y=1\mid x)
=
\frac{1}{1+\exp(-(w^Tx+b))}
=
\sigma(w^Tx+b).
```

Bernoulli Naive Bayes and logistic regression can share a posterior functional form while having different training objectives and different finite-sample estimators.
