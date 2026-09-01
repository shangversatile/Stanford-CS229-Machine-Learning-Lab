# Module 01: MLE, MAP, and Naive Bayes

Primary course:
CMU 10-601 Introduction to Machine Learning, Spring 2023

Selected lecture sources:

* Lecture 16 - PAC Learning + MLE/MAP; selected MLE-related segment only.
* Lecture 17 - MLE/MAP + Naive Bayes.

Supporting course reading:

* Tom Mitchell, "Estimating Probabilities: MLE and MAP."
* Tom Mitchell, "Generative and Discriminative Classifiers: Naive Bayes and Logistic Regression."

Source URLs:

* CMU schedule: <https://www.cs.cmu.edu/~mgormley/courses/10601-s23/schedule.html>
* Lecture 16 slides: <https://www.cs.cmu.edu/~mgormley/courses/10601-s23/slides/lecture16-mle-map.pdf>
* Lecture 17 slides: <https://www.cs.cmu.edu/~mgormley/courses/10601-s23/slides/lecture17-nb.pdf>
* Mitchell MLE/MAP reading: <https://www.cs.cmu.edu/~tom/mlbook/Joint_MLE_MAP.pdf>
* Mitchell Naive Bayes / Logistic Regression reading: <https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf>
* CMU coursework page: <https://www.cs.cmu.edu/~mgormley/courses/10601-s23/coursework.html>
* CMU HW6 handout link: <https://www.cs.cmu.edu/~mgormley/courses/10601-s23/homework/hw6.zip>

Local archival source: no official CMU handout, slide deck, or Mitchell chapter is committed to this repository. The URLs above are retained as the archival source of record.

## 1. Module Boundary

This module is not a restart of CMU 10-601 from Lecture 1, and it is not a replacement for Stanford CS229 Lecture 5. It is a topic-based supplement attached to the CS229 generative-learning node.

CMU Lecture 16 begins with PAC learning and sample-complexity material. That PAC component is intentionally deferred to a later statistical-learning supplement. This module only uses the MLE/MAP segment.

CMU Lecture 17 covers MAP continuation and Naive Bayes variants. This module uses the parts that deepen CS229's generative-learning sequence: Bernoulli Naive Bayes, Gaussian Naive Bayes, Multinomial Naive Bayes, generative/discriminative comparison, and efficient parameter estimation.

## 2. What CS229 Already Established

The CS229 mainline already established:

* generative versus discriminative modelling;
* GDA;
* joint likelihood;
* GDA MLE;
* Naive Bayes core factorization;
* GDA -> logistic posterior;
* sufficient statistics as likelihood compression in Lecture 4.

Relevant CS229 anchors:

| Anchor | Repository location | What it already gives |
| --- | --- | --- |
| CS229 Lecture 4 sufficient statistics | [Lecture 4 note](../../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-a-what-information-about-a-parameter-is-actually-in-the-data) | why data can be compressed into parameter-relevant statistics |
| CS229 Lecture 4 exponential-family likelihoods | [Lecture 4 derivation](../../../math-derivations/lecture-04-perceptron-exponential-family-glm/03-sufficient-statistics-likelihood-equivalence.md) | likelihood factorization and sufficient-statistic logic |
| CS229 Lecture 5 GDA | [Lecture 5 note](../../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#9-gda-model-and-generative-story) | Gaussian class-conditional generative modelling |
| CS229 Lecture 5 GDA -> logistic posterior | [GDA derivation](../../../math-derivations/lecture-05-generative-learning-gda-naive-bayes/02-gda-mle-and-logistic-connection.md) | generative assumptions can induce a discriminative posterior form |
| CS229 Lecture 5 Bernoulli NB | [NB derivation](../../../math-derivations/lecture-05-generative-learning-gda-naive-bayes/03-naive-bayes-factorization-and-mle.md) | Bernoulli factorization, MLE, and prediction |

## 3. What CMU Adds

This supplement adds:

1. MLE as a general parameter-estimation principle.
2. MAP as posterior optimization.
3. Role of priors.
4. Beta-Bernoulli example.
5. Pseudo-count interpretation.
6. Parameter estimation in Naive Bayes.
7. Gaussian Naive Bayes.
8. Multinomial Naive Bayes.
9. Implementation-oriented parameter counting.
10. Generative versus discriminative comparison from another perspective.

The central contribution is not more isolated formulas. It is a clearer chain:

```text
CS229 L4 sufficient statistics
-> likelihood compression
-> CMU MLE as a general estimation recipe
-> MAP as likelihood plus prior
-> Beta-Bernoulli as the smallest complete worked example
-> Naive Bayes parameter estimates as repeated count-based Bernoulli/Multinomial estimation
-> Gaussian NB as a covariance-constrained Gaussian generative classifier
-> NB posterior as a logistic-form discriminative score
-> sharper comparison with logistic regression
```

## 4. Module Structure

The supplement keeps its own derivations inside the module rather than adding CMU files to the global CS229 `math-derivations/` tree:

* [derivations/01-mle-map-beta-bernoulli.md](derivations/01-mle-map-beta-bernoulli.md)
* [derivations/02-bernoulli-nb-estimation-logistic-posterior.md](derivations/02-bernoulli-nb-estimation-logistic-posterior.md)
* [derivations/03-gaussian-multinomial-nb.md](derivations/03-gaussian-multinomial-nb.md)
* [figures/](figures/)
* [scripts/generate_figures.py](scripts/generate_figures.py)

This local structure keeps the supplement self-contained while the tables above preserve explicit links back to CS229.

## 5. MLE as a General Framework

Let $\theta$ denote a candidate parameter and let $\mathcal D$ denote the observed, fixed dataset. Maximum likelihood estimation chooses the parameter value that makes the observed data most likely under the assumed model:

```math
\hat\theta_{\mathrm{MLE}}
=
\underset{\theta}{\mathrm{argmax}}
\,
p(\mathcal D\mid\theta).
```

The log form is equivalent because $\log$ is monotone:

```math
\hat\theta_{\mathrm{MLE}}
=
\underset{\theta}{\mathrm{argmax}}
\,
\log p(\mathcal D\mid\theta).
```

Interpretation:

* $\theta$ is the candidate parameter.
* $\mathcal D$ is already observed and fixed.
* The likelihood is a function of $\theta$.
* The likelihood is not $P(\theta\mid\mathcal D)$.
* MLE does not use a parameter prior.

If the data are iid examples $z^{(1)},\ldots,z^{(m)}$, then:

```math
p(\mathcal D\mid\theta)
=
\prod_{i=1}^{m}
p(z^{(i)}\mid\theta).
```

Therefore:

```math
\log L(\theta)
=
\sum_{i=1}^{m}
\log p(z^{(i)}\mid\theta).
```

The CMU recipe is operational: write the generative story, write the log-likelihood, differentiate, solve stationary equations, and check that the solution is a maximum.

## 6. Bridge to Sufficient Statistics

CS229 Lecture 4 introduced sufficient statistics as likelihood compression. If the likelihood factors as:

```math
L(\theta;\mathcal D)
=
h(\mathcal D)
g_{\theta}(S(\mathcal D)),
```

then all information about $\theta$ that is relevant for parameter estimation enters through:

```math
S(\mathcal D).
```

The bridge is:

```text
sufficient statistics
-> likelihood compression
-> parameter estimation
```

This is why Bernoulli counts, Gaussian sums, and class-conditional feature counts keep reappearing in closed-form estimators. CMU makes the estimation recipe explicit; CS229 explains why the compressed statistics are enough.

## 7. MAP from Bayes Rule

Maximum a posteriori estimation chooses the posterior mode:

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

Since $p(\mathcal D)$ does not depend on the candidate $\theta$:

```math
\hat\theta_{\mathrm{MAP}}
=
\underset{\theta}{\mathrm{argmax}}
\,
p(\mathcal D\mid\theta)p(\theta).
```

Equivalently:

```math
\hat\theta_{\mathrm{MAP}}
=
\underset{\theta}{\mathrm{argmax}}
\,
\left[
\log p(\mathcal D\mid\theta)
+
\log p(\theta)
\right].
```

MLE uses data likelihood only. MAP uses data likelihood plus prior preference.

## 8. MLE, MAP, and Full Bayesian Inference

MLE returns a point estimate:

```math
\hat\theta_{\mathrm{MLE}}.
```

MAP returns a posterior mode:

```math
\hat\theta_{\mathrm{MAP}}.
```

Full Bayesian inference keeps the posterior distribution:

```math
p(\theta\mid\mathcal D).
```

For prediction, full Bayesian inference can integrate over parameter uncertainty:

```math
p(y_*\mid x_*,\mathcal D)
=
\int
p(y_*\mid x_*,\theta)
p(\theta\mid\mathcal D)
d\theta.
```

This module uses the distinction conceptually; it does not expand into a Bayesian inference course.

## 9. Beta-Bernoulli: Likelihood and MLE

Assume:

```math
Y_i
\sim
\mathrm{Bernoulli}(\phi).
```

Let:

```math
N_1
=
\sum_i y_i,
```

and:

```math
N_0
=
m-N_1.
```

The Bernoulli likelihood is:

```math
p(\mathcal D\mid\phi)
=
\phi^{N_1}
(1-\phi)^{N_0}.
```

The log-likelihood is:

```math
\ell(\phi)
=
N_1\log\phi
+
N_0\log(1-\phi).
```

Differentiating:

```math
\frac{d\ell}{d\phi}
=
\frac{N_1}{\phi}
-
\frac{N_0}{1-\phi}.
```

Setting the derivative to zero gives:

```math
N_1(1-\phi)
=
N_0\phi.
```

Thus:

```math
\hat\phi_{\mathrm{MLE}}
=
\frac{N_1}{m}.
```

The MLE is the empirical frequency of ones.

## 10. Beta-Bernoulli Posterior and MAP

Now assume the prior:

```math
\phi
\sim
\mathrm{Beta}(\alpha,\beta).
```

The density is:

```math
p(\phi)
\propto
\phi^{\alpha-1}
(1-\phi)^{\beta-1}.
```

Multiplying likelihood and prior:

```math
p(\phi\mid\mathcal D)
\propto
\phi^{N_1+\alpha-1}
(1-\phi)^{N_0+\beta-1}.
```

Therefore:

```math
\phi\mid\mathcal D
\sim
\mathrm{Beta}
(
N_1+\alpha,
N_0+\beta
).
```

The posterior log-density, up to a constant, is:

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

The derivative is:

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

Set it to zero:

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

Expand and collect:

```math
N_1+\alpha-1
=
(N_1+N_0+\alpha+\beta-2)\phi.
```

Because $m=N_1+N_0$:

```math
\hat\phi_{\mathrm{MAP}}
=
\frac{
N_1+\alpha-1
}{
m+\alpha+\beta-2
}.
```

This formula is the interior mode. It requires $N_1+\alpha>1$ and $N_0+\beta>1$. If $\alpha \le 1$ or $\beta \le 1$, or if the data counts leave one posterior shape parameter at or below $1$, the posterior mode may lie on the boundary at $0$ or $1$.

## 11. Pseudo-Counts

In the MAP formula, $\alpha-1$ and $\beta-1$ behave like prior pseudo-counts:

```math
\hat\phi_{\mathrm{MAP}}
=
\frac{
N_1+(\alpha-1)
}{
N_1+N_0+(\alpha-1)+(\beta-1)
}.
```

This interpretation explains how a prior can smooth estimates away from extreme $0/1$ values when data are scarce.

The pseudo-count is an interpretation, not literal previously observed data. The actual Bayesian object is a prior density over $\phi$. The count analogy is useful because the Beta prior is conjugate to the Bernoulli likelihood.

This gives the conceptual bridge:

```text
MAP
-> prior-induced smoothing
-> later Laplace smoothing
```

Formal Laplace smoothing belongs to the later CS229 L6 / CMU supplement boundary.

## 12. Posterior Mean Is Not MAP

For the Beta posterior:

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
\frac{
N_1+\alpha
}{
m+\alpha+\beta
}.
```

The MAP estimate is:

```math
\hat\phi_{\mathrm{MAP}}
=
\frac{
N_1+\alpha-1
}{
m+\alpha+\beta-2
}.
```

The posterior mean and posterior mode are generally different. Smoothing formulas should not be mixed together without specifying whether the estimator is a posterior mean, a posterior mode, or a separate regularized estimator.

![MLE versus MAP Beta-Bernoulli](figures/cmu10601-beta-bernoulli-mle-map.png)

## 13. Bernoulli Naive Bayes

Start with the class prior:

```math
P(Y=k)=\pi_k.
```

For binary features:

```math
X_j\in\{0,1\}.
```

The Naive Bayes conditional-independence assumption is:

```math
P(X=x\mid Y=k)
=
\prod_{j=1}^{d}
P(X_j=x_j\mid Y=k).
```

Define:

```math
\phi_{jk}
=
P(X_j=1\mid Y=k).
```

Then:

```math
P(X=x\mid Y=k)
=
\prod_{j=1}^{d}
\phi_{jk}^{x_j}
(
1-\phi_{jk}
)^{1-x_j}.
```

## 14. Parameter Counting

Without conditional independence, a binary vector $X\in\{0,1\}^d$ requires roughly $2^d-1$ free probabilities per class to model $P(X\mid Y=k)$.

With Bernoulli Naive Bayes, each class needs approximately $d$ feature parameters plus the class prior. Naive Bayes turns exponential parameter growth into linear parameter growth in feature dimension.

That tractability is bought by the conditional-independence assumption.

![NB parameter reduction](figures/cmu10601-nb-parameter-reduction.png)

## 15. Bernoulli NB MLE and MAP

Let:

```math
N_k
=
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}.
```

Let:

```math
N_{jk,1}
=
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}
x_j^{(i)}.
```

The MLE for a Bernoulli feature parameter is:

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

The numerator is the number of class $k$ emails in which word $j$ appears. The denominator is the number of class $k$ emails. The estimate is an empirical conditional frequency.

If:

```math
\phi_{jk}
\sim
\mathrm{Beta}
(
\alpha,\beta
),
```

then the interior MAP estimate is:

```math
\hat\phi_{jk,\mathrm{MAP}}
=
\frac{
N_{jk,1}+\alpha-1
}{
N_k+\alpha+\beta-2
}.
```

The prior can avoid extreme $0/1$ parameter estimates in sparse class-feature cells. This is prior-induced smoothing, not automatically Laplace smoothing unless the prior parameters are chosen to match that rule.

## 16. Gaussian Naive Bayes

Gaussian Naive Bayes applies the same conditional-independence idea to continuous features:

```math
X_j
\mid
Y=k
\sim
\mathcal N
(
\mu_{jk},
\sigma_{jk}^2
).
```

Assuming $X_1,\ldots,X_d$ are conditionally independent given $Y$:

```math
p(x\mid Y=k)
=
\prod_{j=1}^{d}
\mathcal N
(
x_j;
\mu_{jk},
\sigma_{jk}^2
).
```

This is a Gaussian class-conditional generative classifier. Its covariance restriction is:

```math
\Sigma_k
=
\mathrm{diag}
(
\sigma_{1k}^2,\ldots,\sigma_{dk}^2
).
```

Comparison:

| Model | Class-conditional family | Covariance assumption | Boundary implication |
| --- | --- | --- | --- |
| GDA / LDA-style | Gaussian | shared full covariance | linear log-odds |
| QDA | Gaussian | class-specific full covariance | quadratic log-odds |
| Gaussian Naive Bayes | Gaussian | class-specific diagonal covariance | additive feature log-densities; often quadratic if variances differ |

All three are Gaussian class-conditional generative classifiers. They differ in parameter constraints, especially whether covariance is full or diagonal and whether it is shared across classes.

![GDA, QDA, and Gaussian NB covariance assumptions](figures/cmu10601-gda-qda-gnb-covariance.png)

## 17. Gaussian NB and GDA/QDA

The clean hierarchy is:

```text
GDA / LDA-style:
shared full covariance

QDA:
class-specific full covariance

Gaussian Naive Bayes:
class-specific diagonal covariance
```

The relationship is about constraints, not about one single model being the unqualified submodel of another. Classical CS229 GDA uses a shared full covariance. Gaussian NB usually permits class-specific variances but forbids cross-feature covariance. QDA permits class-specific full covariance. These are different ways to restrict Gaussian class-conditionals.

## 18. Multinomial Naive Bayes

CMU Lecture 17 explicitly lists Multinomial Naive Bayes as a Naive Bayes model for integer features. It is especially natural for text when features represent word occurrence counts rather than word presence.

Bernoulli event model:

```text
word present / absent
```

Multinomial event model:

```text
word occurrence counts
```

Let:

```math
X_j
=
\text{count of vocabulary item }j.
```

For class $k$:

```math
X
\mid
Y=k
\sim
\mathrm{Multinomial}
(
N,\theta_k
),
```

where:

```math
\sum_j
\theta_{jk}
=
1.
```

The class-conditional likelihood for a fixed document count vector is:

```math
p(x\mid Y=k)
=
\frac{N!}{\prod_j x_j!}
\prod_{j=1}^{d}
\theta_{jk}^{x_j}.
```

For prediction over $k$, the multinomial coefficient is independent of $k$ for the same $x$, so the score uses:

```math
\log p(x\mid Y=k)
=
C(x)
+
\sum_{j=1}^{d}
x_j\log\theta_{jk}.
```

The MLE is:

```math
\hat\theta_{jk}
=
\frac{
\sum_{i:y^{(i)}=k}
x_j^{(i)}
}{
\sum_{i:y^{(i)}=k}
\sum_{\ell=1}^{d}
x_{\ell}^{(i)}
}.
```

The numerator is the number of occurrences of vocabulary item $j$ in class $k$ documents. The denominator is the total number of vocabulary-item occurrences in class $k$ documents.

With a Dirichlet prior $\theta_k\sim\mathrm{Dirichlet}(\alpha_1,\ldots,\alpha_d)$, the interior posterior mode is:

```math
\hat\theta_{jk,\mathrm{MAP}}
=
\frac{
C_{jk}+\alpha_j-1
}{
C_k+\sum_{\ell=1}^{d}\alpha_{\ell}-d
},
```

where $C_{jk}$ is the class-$k$ count of word $j$ and $C_k=\sum_j C_{jk}$. Boundary conditions again matter when posterior shape parameters are at or below $1$.

## 19. Naive Bayes Prediction

Prediction chooses:

```math
\hat y
=
\underset{k}{\mathrm{argmax}}
\,
P(Y=k\mid X=x).
```

Because:

```math
P(Y=k\mid x)
\propto
P(Y=k)p(x\mid Y=k),
```

we compute:

```math
\hat y
=
\underset{k}{\mathrm{argmax}}
\,
\left[
\log\pi_k
+
\log p(x\mid Y=k)
\right].
```

For Bernoulli NB:

```math
\hat y
=
\underset{k}{\mathrm{argmax}}
\,
\left[
\log\pi_k
+
\sum_j
x_j\log\phi_{jk}
+
(1-x_j)
\log(1-\phi_{jk})
\right].
```

The log form prevents numerical underflow and makes the model an additive scorer over features.

## 20. Bernoulli NB Induces a Logistic Posterior

For binary labels $Y\in\{0,1\}$:

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
\pi_1 p(x\mid Y=1)
}{
\pi_0 p(x\mid Y=0)
}.
```

Insert Bernoulli NB class-conditionals:

```math
=
\log\frac{\pi_1}{\pi_0}
+
\sum_j
\log
\frac{
\phi_{j1}^{x_j}(1-\phi_{j1})^{1-x_j}
}{
\phi_{j0}^{x_j}(1-\phi_{j0})^{1-x_j}
}.
```

Split the terms:

```math
=
\log\frac{\pi_1}{\pi_0}
+
\sum_j
\left[
x_j\log\frac{\phi_{j1}}{\phi_{j0}}
+
(1-x_j)
\log\frac{1-\phi_{j1}}{1-\phi_{j0}}
\right].
```

Separate constant and $x_j$ terms:

```math
=
\left[
\log\frac{\pi_1}{\pi_0}
+
\sum_j
\log\frac{1-\phi_{j1}}{1-\phi_{j0}}
\right]
+
\sum_j
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
\sum_j
\log\frac{1-\phi_{j1}}{1-\phi_{j0}},
```

and:

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
b
+
\sum_j
w_jx_j.
```

So:

```math
P(Y=1\mid x)
=
\sigma
(
w^Tx+b
).
```

Bernoulli Naive Bayes can also induce a logistic-form posterior. This directly connects to the CS229 Lecture 5 pattern:

```text
generative assumptions
-> discriminative posterior form
```

## 21. But Naive Bayes Is Not Logistic Regression

Naive Bayes estimates class priors and feature conditional probabilities, then derives the posterior.

Logistic regression directly estimates conditional log-odds parameters.

Even when both produce a logistic-form posterior, their finite-sample estimators are different. Naive Bayes constrains the log-odds weights through a joint generative model; logistic regression optimizes conditional likelihood without modeling $p(x\mid y)$.

## 22. Generative versus Discriminative Synthesis

| Axis | Naive Bayes | Logistic regression |
| --- | --- | --- |
| Modeling target | joint model $p(x,y)$ via $p(y)p(x\mid y)$ | conditional model $p(y\mid x)$ |
| Main assumption | conditional independence of features given class | linear conditional log-odds |
| Parameter estimation | counting / closed-form MLE or MAP for many event models | iterative optimization of conditional likelihood |
| Data efficiency | often strong in small data if assumptions are roughly useful | often needs more data but can be less biased under dependence |
| Misspecification | feature dependence can distort probabilities | ignores $p(x)$ and can fit decision boundary more directly |
| Parameter count | linear in feature dimension under NB | linear in feature dimension for binary logistic regression |
| Decision boundary | linear for Bernoulli NB log-odds; Gaussian NB can be quadratic when variances differ | linear in the chosen feature representation |
| Probability calibration | can be overconfident when independence is false | often better calibrated after suitable regularization, but not guaranteed |
| Computation | one-pass counting for Bernoulli/Multinomial NB | gradient/Newton/SGD training loop |

CMU's implementation-oriented view is that Naive Bayes is valuable because the conditional-independence assumption converts an intractable joint-distribution estimation problem into efficient counting.

## 23. Complete Logic

The knowledge architecture after this supplement is:

| Step | CS229 mainline role | CMU supplement role | Resulting understanding |
| --- | --- | --- | --- |
| Likelihood | CS229 uses likelihood in linear regression, logistic regression, GLM, GDA, and NB | abstracts MLE as a reusable estimation principle | MLE becomes a general recipe, not a one-off derivation trick |
| Sufficient statistics | CS229 L4 explains parameter-relevant data compression | CMU count-based estimators instantiate that compression | counts are not shortcuts; they are sufficient summaries for the assumed model |
| Prior | CS229 has earlier MAP intuition in linear models | CMU makes MAP a posterior optimization objective | prior preference is separated cleanly from likelihood evidence |
| Beta-Bernoulli | CS229 mentions Bernoulli models across GLM/NB | CMU gives the smallest complete MLE/MAP example | posterior shape, mode, mean, and pseudo-counts become distinguishable |
| Naive Bayes | CS229 introduces factorization and prediction | CMU emphasizes parameter counting and one-pass learning | the "naive" assumption is seen as the tractability engine |
| Gaussian generative models | CS229 develops GDA and QDA boundary logic | CMU adds Gaussian NB as diagonal covariance modelling | covariance constraints unify GDA, QDA, and Gaussian NB |
| Discriminative posterior | CS229 shows GDA can induce logistic posterior | CMU/Mitchell show Bernoulli NB can also induce logistic posterior | logistic form does not imply logistic-regression training |

## 24. Figures

Generated figures:

* [figures/cmu10601-beta-bernoulli-mle-map.png](figures/cmu10601-beta-bernoulli-mle-map.png)
* [figures/cmu10601-nb-parameter-reduction.png](figures/cmu10601-nb-parameter-reduction.png)
* [figures/cmu10601-gda-qda-gnb-covariance.png](figures/cmu10601-gda-qda-gnb-covariance.png)

The figures are generated by [scripts/generate_figures.py](scripts/generate_figures.py).

## 25. Coursework Boundary

CMU HW6 is mapped but not solved. The handout title is "Homework 6: Learning Theory and Generative Models"; the current supplement only covers the MLE/MAP and Naive Bayes subset. PAC learning is deferred, and no answer content is included.

## 26. Status

Module status: ready for interactive study.

PS1 isolation: this module does not modify or depend on `assignments/ps1-supervised-learning/`.
