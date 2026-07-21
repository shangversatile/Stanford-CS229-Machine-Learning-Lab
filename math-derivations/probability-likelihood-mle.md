# Probability, Likelihood, and Maximum Likelihood Estimation

This note focuses on probability, likelihood, and maximum likelihood estimation.

For the full conceptual comparison between ERM, MLE, and Bayesian modeling, see:

`ml-vs-probabilistic-modeling.md`

## 1. Probability vs Likelihood

Probability and likelihood can use the same mathematical expression, but they read it as different functions.

In the probability view, the parameter is fixed and the data are random:

```math
p(y|x;\theta)\quad\text{as a function of }y
```

This asks: if the model parameter is $\theta$, how likely is each possible observation $y$?

In the likelihood view, the observed data are fixed and the parameter varies:

```math
L(\theta)=p(D|\theta)\quad\text{as a function of }\theta
```

This asks: among possible parameters, which $\theta$ makes the already observed dataset $D$ most plausible?

Likelihood is not a probability distribution over $\theta$ in ordinary MLE. It becomes part of a probability distribution over $\theta$ only after a Bayesian prior is introduced. Early CS229 uses this mainly as frequentist MLE: $\theta$ is fixed but unknown, data are random before observation, and the estimator changes when the sampled dataset changes.

## 2. Why Likelihood Is a Function of $\theta$

Suppose the conditional model is:

```math
p(y|x;\theta)
```

Once a dataset has been observed,

```math
D=\{(x^{(i)},y^{(i)})\}_{i=1}^{m}
```

the values $x^{(i)}$ and $y^{(i)}$ are fixed inside the optimization problem. The parameter $\theta$ is the unknown quantity being compared.

Under conditional independence, the likelihood is:

```math
L(\theta)=\prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta)
```

Taking logs gives:

```math
\ell_{\mathrm{log}}(\theta)=\sum_{i=1}^{m}\log p(y^{(i)}|x^{(i)};\theta)
```

MLE is therefore:

```math
\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmax}}\ L(\theta)
```

Equivalently:

```math
\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmin}}\left[-\ell_{\mathrm{log}}(\theta)\right]
```

## 3. Frequentist Interpretation of MLE

Early CS229 MLE derivations are mainly frequentist. The parameter is treated as fixed but unknown, while the data are random samples from a data-generating process.

```math
\theta\ \text{is fixed but unknown}
```

```math
D\ \text{is random because it is sampled from the data-generating process}
```

```math
\hat{\theta}(D)\ \text{is random because it depends on }D
```

The true parameter is not assigned a probability distribution. Instead, the estimator $\hat{\theta}(D)$ has a sampling distribution because different random datasets would produce different estimates.

## 4. Bayesian Contrast

Bayesian modeling treats $\theta$ as a random variable and specifies a prior:

```math
p(\theta)
```

After observing data:

```math
p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}
```

The evidence term is:

```math
p(D)=\int p(D|\theta)p(\theta)d\theta
```

The posterior $p(\theta|D)$ is a probability distribution over parameters. It captures parameter uncertainty after seeing the data.

## 5. MLE vs MAP

MLE ignores the prior and maximizes likelihood:

```math
\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmax}}\ p(D|\theta)
```

MAP uses the posterior and therefore includes the prior:

```math
\hat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\mathrm{argmax}}\ p(D|\theta)p(\theta)
```

Taking logs makes the prior contribution explicit:

```math
\hat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\mathrm{argmax}}\left[\log p(D|\theta)+\log p(\theta)\right]
```

Full Bayesian prediction uses the whole posterior rather than one point estimate:

```math
p(y_*|x_*,D)=\int p(y_*|x_*;\theta)p(\theta|D)d\theta
```

## 6. How Gaussian MLE Produces Squared Loss

Linear regression can be given a probabilistic interpretation by assuming additive Gaussian noise:

```math
y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}
```

```math
\epsilon^{(i)}\sim\mathcal{N}(0,\sigma^2)
```

Then:

```math
p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\left(y^{(i)}-\theta^Tx^{(i)}\right)^2}{2\sigma^2}\right)
```

For independent samples:

```math
L(\theta)=\prod_{i=1}^{m}\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\left(y^{(i)}-\theta^Tx^{(i)}\right)^2}{2\sigma^2}\right)
```

The log likelihood is:

```math
\ell_{\mathrm{log}}(\theta)=-m\log(\sqrt{2\pi}\sigma)-\frac{1}{2\sigma^2}\sum_{i=1}^{m}\left(y^{(i)}-\theta^Tx^{(i)}\right)^2
```

The first term does not depend on $\theta$, and $1/(2\sigma^2)$ is positive. Therefore:

```math
\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmin}}\sum_{i=1}^{m}\left(y^{(i)}-\theta^Tx^{(i)}\right)^2
```

Thus ordinary least squares is MLE under independent homoscedastic Gaussian noise.

## 7. How Bernoulli MLE Produces Cross-entropy

For binary classification, $y\in\{0,1\}$. Logistic regression models:

```math
h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}
```

The Bernoulli conditional model is:

```math
p(y|x;\theta)=h_{\theta}(x)^y\left(1-h_{\theta}(x)\right)^{1-y}
```

For independent examples:

```math
L(\theta)=\prod_{i=1}^{m}h_{\theta}(x^{(i)})^{y^{(i)}}\left(1-h_{\theta}(x^{(i)})\right)^{1-y^{(i)}}
```

The log likelihood is:

```math
\ell_{\mathrm{log}}(\theta)=\sum_{i=1}^{m}\left[y^{(i)}\log h_{\theta}(x^{(i)})+\left(1-y^{(i)}\right)\log\left(1-h_{\theta}(x^{(i)})\right)\right]
```

The negative log likelihood is:

```math
J_{\mathrm{NLL}}(\theta)=-\sum_{i=1}^{m}\left[y^{(i)}\log h_{\theta}(x^{(i)})+\left(1-y^{(i)}\right)\log\left(1-h_{\theta}(x^{(i)})\right)\right]
```

This is binary cross-entropy. It heavily penalizes confidently wrong predictions because log probability goes to negative infinity near zero.

## 8. Reliability Limitations

MLE inherits the assumptions of the likelihood model. Gaussian MLE can be unreliable under heavy-tailed noise, outliers, heteroscedasticity, correlated errors, and misspecified linear structure.

Bernoulli MLE can still fail under label noise, class imbalance, poor calibration, nonlinear boundaries, distribution shift, wrong threshold choice, and complete separation.

Probability modeling clarifies what assumptions justify a loss function, but it does not guarantee those assumptions are true in deployment.