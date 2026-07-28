# Why Exponential Family and GLM Exist

Cross-links: see [Exponential Family Anatomy](exponential-family-anatomy.md), [Log-Partition Mean, Variance, and Convexity](log-partition-mean-variance-convexity.md), [GLM Construction Recipe](glm-construction-recipe.md), and [GLM Response and Distribution Map](glm-response-distribution-map.md). The main lecture version is [Conceptual Interlude C](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-c-why-exponential-family-and-glm-exist).

## 1. The limitation of ordinary linear models

Ordinary linear regression solves a special but important problem. It keeps the conditional mean linear in features, gives interpretable coefficients, and turns fixed-variance Gaussian maximum likelihood into least squares.

```math
Y|x \approx \theta^Tx+\epsilon
```

```math
\epsilon\sim \mathcal N(0,\sigma^2)
```

This is not a generic supervised-learning template. It assumes a real-valued response, additive noise, symmetric errors, and often constant variance. Those assumptions are unnatural for binary labels, event counts, multiclass labels, positive durations, bounded probabilities, compositional vectors, or data whose variance changes with the mean.

A linear predictor can produce any real number. That is useful for real-valued regression but invalid for probabilities, counts, and simplex-valued objects. The modeling problem is therefore not merely to find a better optimizer. It is to keep useful linear structure while changing the response distribution.

## 2. The three desiderata behind GLMs

The GLM idea can be read as a compromise among three desiderata.

First, keep a linear predictor because it gives interpretable effects and scalable computation.

```math
\theta^Tx
```

Second, choose a response distribution that respects what $`Y`$ means: binary, count, positive, multiclass, real-valued, or compositional.

Third, derive the response function and likelihood from that distribution rather than choosing a loss or activation by visual habit.

In the canonical case, GLM puts linearity on the natural parameter:

```math
\eta=\theta^Tx
```

and predicts the conditional mean of the sufficient statistic:

```math
h_\theta(x)=\mathbb E[T(Y)|x;\theta]
```

The response function is therefore a consequence of the probability model.

## 3. Sufficient statistics and likelihood compression

A statistic is sufficient when it retains all sample information relevant to a parameter. Exponential family makes this compression algebraically explicit.

Start from:

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

For iid observations:

```math
p(y^{(1)},\ldots,y^{(m)};\eta)
=
\prod_{i=1}^{m}b(y^{(i)})\exp\left(\eta^TT(y^{(i)})-a(\eta)\right)
```

Rearrange:

```math
p(y^{(1)},\ldots,y^{(m)};\eta)
=
\left(\prod_{i=1}^{m}b(y^{(i)})\right)
\exp\left(\eta^T\sum_{i=1}^{m}T(y^{(i)})-ma(\eta)\right)
```

The only data-dependent object coupled to $`\eta`$ is:

```math
\sum_{i=1}^{m}T(y^{(i)})
```

Thus the sample can be compressed, for purposes of estimating $`\eta`$, into the aggregate sufficient statistic. Bernoulli data compress into successes; Poisson data compress into total count; categorical data compress into class counts; Gaussian fixed-variance data compress into a sum or mean.

## Why $`T(y)`$ and $`\eta`$ are dual objects

The exponential-family form makes the observation side and the parameter side meet in one term:

```math
\eta^TT(y)
```

$`T(y)`$ lives on the observation side. It is the statistic read from the response value: success/failure for Bernoulli, count for Poisson, one-hot identity for categorical data, or location and squared-magnitude information for a Gaussian with unknown variance.

$`\eta`$ lives on the parameter side. It is the natural coordinate of the distribution: log-odds for Bernoulli, log-rate for Poisson, class log-odds for softmax, or the canonical coordinate attached to Gaussian moments.

Their dot product is a compatibility score. If the current $`\eta`$ gives high weight to the statistic pattern read from $`y`$, the unnormalized probability of that outcome rises. The log-partition function then converts all such scores into valid probabilities by subtracting the log-total unnormalized mass:

```math
\log p(y;\eta)=\eta^TT(y)+\log b(y)-a(\eta)
```

In a GLM, features determine $`\eta`$:

```math
\eta(x)=\theta^Tx
```

So $`x`$ changes the conditional distribution by changing how it weights the sufficient-statistic directions. This is the core reason GLMs connect supervised features to probability models: $`x`$ controls the parameter-side coordinate, $`T(y)`$ defines the observation-side readout, and $`\eta(x)^TT(y)`$ scores their compatibility.

## 4. Pitman-Koopman-Darmois intuition

The Pitman-Koopman-Darmois theorem explains why exponential family is not just a convenient format. Under regularity assumptions such as iid sampling, common support that does not depend on the parameter, and a fixed-dimensional sufficient statistic that works for all sample sizes, the resulting family is essentially an exponential family.

The intuition is that if every sample size can be summarized by a statistic of fixed dimension, the log likelihood must keep adding observations into a stable finite set of summaries. Additive summaries naturally have the form:

```math
\sum_{i=1}^{m}T(y^{(i)})
```

and parameter interaction with those summaries naturally becomes linear in the log likelihood:

```math
\eta^T\sum_{i=1}^{m}T(y^{(i)})
```

This statement has limits. It relies on regularity conditions. Parameter-dependent support breaks the conclusion; Uniform $`(0,\theta)`$ is the standard warning example. The theorem should be used as structural intuition, not as a claim that all useful distributions are exponential families.

## 5. Maximum entropy derivation

There is a second route from information theory. Suppose we choose statistics $`T(y)`$ and know their expected value, but we do not want to impose additional structure. The maximum-entropy principle chooses the least committed distribution satisfying those constraints.

Set up the entropy maximization:

```math
\underset{p}{\mathrm{maximize}}\ -\int p(y)\log p(y)dy
```

subject to:

```math
\int p(y)dy=1
```

and:

```math
\int p(y)T(y)dy=\mu
```

Use Lagrange multipliers $`\lambda_0`$ and $`\eta`$. The Lagrangian can be written as:

```math
\mathcal L(p)
=
-\int p(y)\log p(y)dy
+\lambda_0\left(\int p(y)dy-1\right)
+\eta^T\left(\int p(y)T(y)dy-\mu\right)
```

Stationarity with respect to $`p(y)`$ gives:

```math
-\log p(y)-1+\lambda_0+\eta^TT(y)=0
```

Therefore:

```math
p(y)\propto \exp\left(\eta^TT(y)\right)
```

If the background measure or support weighting is represented by $`b(y)`$, the same calculation gives:

```math
p(y)\propto b(y)\exp\left(\eta^TT(y)\right)
```

Normalize it by defining:

```math
a(\eta)=\log\int b(y)e^{\eta^TT(y)}dy
```

Then:

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

So exponential family also arises as the least-extra-assumption distribution given chosen moment constraints.

## 6. Log-partition as normalizer, moment generator, and curvature engine

The log-partition function first appears because probabilities must sum or integrate to one.

```math
1=\int b(y)\exp\left(\eta^TT(y)-a(\eta)\right)dy
```

Equivalently:

```math
a(\eta)=\log\int b(y)e^{\eta^TT(y)}dy
```

Now differentiate. Under regularity conditions:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

Differentiate again:

```math
\nabla^2a(\eta)=\mathrm{Cov}_\eta(T(Y))
```

This identity explains the apparently magical package of properties. The normalizer generates means. Its second derivative generates covariance. Covariance is positive semidefinite, so $`a(\eta)`$ is convex. In regular canonical models, this gives concave log likelihood and convex negative log likelihood in the natural parameter.

For iid data, the log likelihood is:

```math
\ell(\eta)
=
\eta^T\sum_{i=1}^{m}T(y^{(i)})-ma(\eta)+\sum_{i=1}^{m}\log b(y^{(i)})
```

Therefore:

```math
\nabla^2\ell(\eta)=-m\nabla^2a(\eta)
```

and:

```math
\nabla^2\ell(\eta)=-m\mathrm{Cov}_\eta(T(Y))\preceq0
```

The geometry comes from normalization and covariance, not from a separate optimization trick.

## 7. Why the natural parameter is the linear target

In canonical exponential-family form, the log density is:

```math
\log p(y;\eta)=\eta^TT(y)-a(\eta)+\log b(y)
```

The natural parameter is the coordinate in which the data statistic enters linearly. That is the right place to put the linear predictor if we want to preserve the spirit of linear modeling.

```math
\eta=\theta^Tx
```

Then the response mean follows from the log-partition function:

```math
h_\theta(x)=\mathbb E[T(Y)|x;\theta]=\nabla a(\theta^Tx)
```

This is why a Gaussian GLM has identity response, a Bernoulli GLM has sigmoid response, a Poisson GLM has exponential response, and a multinomial GLM has softmax response. The nonlinear response is not pasted onto the model; it is induced by the distribution.

## 8. From response semantics to distribution choice

The modeling order is semantic before algebraic.

| Response meaning | Support | Natural candidate | Induced response idea |
| ---------------- | ------- | ----------------- | --------------------- |
| real measurement | $`\mathbb R`$ | Gaussian | identity mean |
| binary event | $`\{0,1\}`$ | Bernoulli | sigmoid probability |
| count | $`\mathbb N_0`$ | Poisson | exponential nonnegative mean |
| multiclass label | $`\{1,\ldots,K\}`$ | categorical / multinomial | softmax probabilities |
| positive duration or size | $`\mathbb R_{>0}`$ | Gamma / Exponential | positive mean or rate |
| probability vector | simplex | Dirichlet-type model | simplex-valued mean |

Support is necessary but not sufficient. The modeler must also ask about variance, tails, zero inflation, dependence, exposure, and measurement mechanism.

## 9. Why the response function is derived rather than chosen

A neural-network view might treat identity, sigmoid, exponential, and softmax as activations. In a GLM, they have a different status. They are response maps obtained from the chosen likelihood family.

For Bernoulli:

```math
a(\eta)=\log(1+e^\eta)
```

so:

```math
\nabla a(\eta)=\frac{e^\eta}{1+e^\eta}
```

For Poisson:

```math
a(\eta)=e^\eta
```

so:

```math
\nabla a(\eta)=e^\eta
```

For softmax-style multinomial modeling:

```math
a(\eta)=\log\sum_{j=1}^{K}e^{\eta_j}
```

so:

```math
\frac{\partial a(\eta)}{\partial \eta_k}=\frac{e^{\eta_k}}{\sum_{j=1}^{K}e^{\eta_j}}
```

The response function is therefore a distributional consequence, not merely a convenient nonlinear transformation.

## 10. What GLM solves and what it does not solve

GLM solves a precise problem: it generalizes linear regression to non-Gaussian response types while preserving linear predictors, likelihood estimation, and tractable derivatives.

It does not automatically choose the correct distribution. It does not guarantee iid data. It does not guarantee that the canonical link is empirically right. It does not guarantee that a linear predictor is expressive enough. It does not guarantee that MLE exists or is stable; separation and rank deficiency remain real failures. It also does not replace hierarchical, Bayesian, robust, semiparametric, or nonparametric modeling when the data-generating process demands them.

## 11. Reliability implications

The same structure that makes GLMs elegant also gives diagnostics. Check whether support matches the response. Check mean-variance behavior. Check calibration for probabilities. Check residual structure. Check separation and coefficient growth. Check rank, conditioning, and identifiability. Check train-test and deployment shift.

The reliable conclusion is not “use exponential family because it is mathematically beautiful.” The reliable conclusion is: exponential-family GLMs give a disciplined grammar for connecting response semantics, sufficient statistics, likelihood, response functions, and optimization; the assumptions still have to be tested.
