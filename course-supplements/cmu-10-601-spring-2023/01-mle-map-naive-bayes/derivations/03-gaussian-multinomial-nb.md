# Gaussian and Multinomial Naive Bayes

Cross-link: see [Module 01 sections 16-19](../README.md#16-gaussian-naive-bayes).

CS229 bridge: [Lecture 5 GDA/QDA](../../../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#13-qda-unequal-covariance-and-quadratic-boundary) already connects Gaussian covariance assumptions to boundary shape. This supplement adds the Gaussian NB diagonal-covariance case and the Multinomial NB count-event model.

Source boundary: independent derivation guided by CMU 10-601 Spring 2023 Lecture 17 and Tom Mitchell's Naive Bayes / Logistic Regression reading.

## 1. Gaussian Naive Bayes Model

Let $X\in\mathbb R^d$ and $Y\in\{0,\ldots,K-1\}$. Gaussian Naive Bayes assumes:

```math
X_j\mid Y=k
\sim
\mathcal N(\mu_{jk},\sigma_{jk}^2).
```

Conditional independence gives:

```math
p(x\mid Y=k)
=
\prod_{j=1}^{d}
\mathcal N(x_j;\mu_{jk},\sigma_{jk}^2).
```

This is equivalent to a multivariate Gaussian with diagonal covariance:

```math
X\mid Y=k
\sim
\mathcal N(\mu_k,\Sigma_k),
```

where:

```math
\mu_k
=
\begin{bmatrix}
\mu_{1k}\\
\vdots\\
\mu_{dk}
\end{bmatrix},
```

and:

```math
\Sigma_k
=
\mathrm{diag}
(
\sigma_{1k}^2,\ldots,\sigma_{dk}^2
).
```

## 2. Gaussian NB MLE

Let:

```math
N_k
=
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}.
```

The class-specific feature mean estimator is:

```math
\hat\mu_{jk}
=
\frac{1}{N_k}
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}
x_j^{(i)}.
```

The MLE variance estimator is:

```math
\hat\sigma_{jk}^2
=
\frac{1}{N_k}
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}
(
x_j^{(i)}-\hat\mu_{jk}
)^2.
```

The class prior MLE remains:

```math
\hat\pi_k
=
\frac{N_k}{m}.
```

## 3. Covariance Constraints

GDA / LDA-style shared-covariance model:

```math
X\mid Y=k
\sim
\mathcal N(\mu_k,\Sigma).
```

QDA class-specific full-covariance model:

```math
X\mid Y=k
\sim
\mathcal N(\mu_k,\Sigma_k).
```

Gaussian NB class-specific diagonal-covariance model:

```math
X\mid Y=k
\sim
\mathcal N
(
\mu_k,
\mathrm{diag}
(
\sigma_{1k}^2,\ldots,\sigma_{dk}^2
)
).
```

The models are all Gaussian class-conditional generative classifiers, but their covariance constraints differ.

| Model | Covariance | Independent covariance parameters for $K$ classes |
| --- | --- | --- |
| GDA / LDA-style | shared full $\Sigma$ | $d(d+1)/2$ |
| QDA | class-specific full $\Sigma_k$ | $K d(d+1)/2$ |
| Gaussian NB | class-specific diagonal $\Sigma_k$ | $K d$ |

Do not call Gaussian NB a strict submodel of GDA without qualification: classical CS229 GDA uses a shared full covariance, while Gaussian NB usually uses class-specific diagonal covariance. The more accurate statement is that both are constrained versions of Gaussian class-conditional modeling.

## 4. Gaussian NB Prediction Score

The log score for class $k$ is:

```math
s_k(x)
=
\log\pi_k
+
\sum_{j=1}^{d}
\log
\mathcal N(x_j;\mu_{jk},\sigma_{jk}^2).
```

Expanding the univariate Gaussian log-density:

```math
s_k(x)
=
\log\pi_k
-
\frac{1}{2}
\sum_{j=1}^{d}
\left[
\log(2\pi\sigma_{jk}^2)
+
\frac{(x_j-\mu_{jk})^2}{\sigma_{jk}^2}
\right].
```

If variances differ by class, the decision boundary can contain quadratic feature terms. If all classes share the same diagonal covariance, those quadratic terms cancel class-to-class and the boundary becomes linear in $x$.

## 5. Multinomial Naive Bayes Model

For text counts, let $X_j$ be the count of vocabulary item $j$ in a document and let $N=\sum_j X_j$ be the document length. For class $k$:

```math
X\mid Y=k
\sim
\mathrm{Multinomial}(N,\theta_k),
```

where:

```math
\sum_{j=1}^{d}
\theta_{jk}
=
1.
```

The probability mass is:

```math
p(x\mid Y=k)
=
\frac{N!}{\prod_{j=1}^{d}x_j!}
\prod_{j=1}^{d}
\theta_{jk}^{x_j}.
```

For prediction on the same document $x$, the coefficient $N!/\prod_j x_j!$ is independent of $k$.

## 6. Multinomial NB MLE

Let:

```math
C_{jk}
=
\sum_{i:y^{(i)}=k}
x_j^{(i)}.
```

Let:

```math
C_k
=
\sum_{j=1}^{d}
C_{jk}.
```

The class-specific log-likelihood term is:

```math
\ell(\theta_k)
=
\sum_{j=1}^{d}
C_{jk}\log\theta_{jk}
+ C,
```

subject to:

```math
\sum_{j=1}^{d}
\theta_{jk}
=
1.
```

Use Lagrange multiplier $\lambda$:

```math
\mathcal L(\theta_k,\lambda)
=
\sum_{j=1}^{d}
C_{jk}\log\theta_{jk}
+
\lambda
\left(
\sum_{j=1}^{d}
\theta_{jk}
-
1
\right).
```

Stationarity gives:

```math
\frac{\partial \mathcal L}{\partial \theta_{jk}}
=
\frac{C_{jk}}{\theta_{jk}}
+
\lambda
=
0.
```

Thus:

```math
\theta_{jk}
=
-
\frac{C_{jk}}{\lambda}.
```

Sum over $j$:

```math
1
=
\sum_{j=1}^{d}
\theta_{jk}
=
-
\frac{1}{\lambda}
\sum_{j=1}^{d}
C_{jk}
=
-
\frac{C_k}{\lambda}.
```

So $\lambda=-C_k$ and:

```math
\hat\theta_{jk}
=
\frac{C_{jk}}{C_k}.
```

## 7. Dirichlet MAP for Multinomial NB

With:

```math
\theta_k
\sim
\mathrm{Dirichlet}(\alpha_1,\ldots,\alpha_d),
```

the posterior is:

```math
\theta_k\mid\mathcal D
\sim
\mathrm{Dirichlet}
(
C_{1k}+\alpha_1,
\ldots,
C_{dk}+\alpha_d
).
```

The interior MAP estimate is:

```math
\hat\theta_{jk,\mathrm{MAP}}
=
\frac{
C_{jk}+\alpha_j-1
}{
C_k+\sum_{\ell=1}^{d}\alpha_{\ell}-d
}.
```

The posterior mean would instead be:

```math
E[\theta_{jk}\mid\mathcal D]
=
\frac{
C_{jk}+\alpha_j
}{
C_k+\sum_{\ell=1}^{d}\alpha_{\ell}
}.
```

As in Beta-Bernoulli, MAP and posterior mean should not be conflated.
