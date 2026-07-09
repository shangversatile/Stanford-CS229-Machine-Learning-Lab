# Probability, Likelihood, and Maximum Likelihood Estimation

## 1. Probability vs Likelihood

Probability and likelihood use the same mathematical object but treat different quantities as variable.

In the probability view, the parameter $\theta$ is fixed and the data are random:

$$p(y|x;\theta).$$

This asks: if the model parameter is $\theta$, how likely is each possible observation $y$?

In the likelihood view, the observed data are fixed and the parameter is the variable:

$$L(\theta)=p(y|X;\theta).$$

This asks: given the data already observed, which parameter $\theta$ makes those observations most plausible?

Likelihood is not a probability distribution over $\theta$ unless a prior and Bayesian model are introduced. In maximum likelihood estimation, it is an objective function over parameters.

## 2. Linear Regression Gaussian Noise Model

Linear regression can be given a probabilistic interpretation by assuming:

$$y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}.$$

Assume independent Gaussian noise:

$$\epsilon^{(i)}\sim\mathcal{N}(0,\sigma^2).$$

Then the conditional distribution of $y^{(i)}$ is:

$$y^{(i)}|x^{(i)};\theta\sim\mathcal{N}(\theta^Tx^{(i)},\sigma^2).$$

The density is:

$$p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{\left(y^{(i)}-\theta^Tx^{(i)}\right)^2}{2\sigma^2}\right).$$

## 3. Full MLE Derivation for Squared Loss

Under conditional independence:

$$L(\theta)=\prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta).$$

Substitute the Gaussian density:

$$L(\theta)=\prod_{i=1}^{m}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{\left(y^{(i)}-\theta^Tx^{(i)}\right)^2}{2\sigma^2}\right).$$

Take log likelihood:

$$\ell(\theta)=\log L(\theta).$$

Products become sums:

$$\ell(\theta)=\sum_{i=1}^{m}\left[-\frac{1}{2}\log(2\pi\sigma^2)-\frac{\left(y^{(i)}-\theta^Tx^{(i)}\right)^2}{2\sigma^2}\right].$$

Collect terms:

$$\ell(\theta)=-\frac{m}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^{m}\left(y^{(i)}-\theta^Tx^{(i)}\right)^2.$$

The first term is constant with respect to $\theta$. Since $\sigma^2>0$, maximizing $\ell(\theta)$ is equivalent to minimizing the squared residual sum:

$$\underset{\theta}{\mathrm{argmax}}\ \ell(\theta)=\underset{\theta}{\mathrm{argmin}}\sum_{i=1}^{m}\left(y^{(i)}-\theta^Tx^{(i)}\right)^2.$$

Thus ordinary least squares is maximum likelihood estimation under independent homoscedastic Gaussian noise.

## 4. CLT Rationale for Gaussian Noise

A common rationale for Gaussian noise is that an observed error may be the sum of many small disturbances:

$$\epsilon=\sum_{k=1}^{K}u_k.$$

If the disturbances are approximately independent, have finite variance, and no single component dominates, then a standardized sum tends toward a Gaussian:

$$\frac{\sum_{k=1}^{K}u_k-\sum_{k=1}^{K}\mathbb{E}[u_k]}{\sqrt{\sum_{k=1}^{K}\mathrm{Var}(u_k)}}\Rightarrow\mathcal{N}(0,1).$$

This Central Limit Theorem view is a rationale, not a proof that real residuals are Gaussian. It can fail with heavy tails, outliers, correlated disturbances, heteroscedasticity, measurement artifacts, or distribution shift.

## 5. Logistic Regression Bernoulli Model

For binary classification, $y\in\{0,1\}$. A Gaussian model is no longer natural because the output space is discrete. Logistic regression instead models:

$$P(y=1|x;\theta)=h_\theta(x).$$

The hypothesis is:

$$h_\theta(x)=g(\theta^Tx).$$

The sigmoid function is:

$$g(z)=\frac{1}{1+e^{-z}}.$$

So the probability of class $0$ is:

$$P(y=0|x;\theta)=1-h_\theta(x).$$

## 6. Bernoulli Likelihood and Cross-entropy

The Bernoulli probability mass function can be written compactly:

$$P(y|x;\theta)=h_\theta(x)^y\left(1-h_\theta(x)\right)^{1-y}.$$

For $m$ independent examples:

$$L(\theta)=\prod_{i=1}^{m}h_\theta(x^{(i)})^{y^{(i)}}\left(1-h_\theta(x^{(i)})\right)^{1-y^{(i)}}.$$

The log likelihood is:

$$\ell(\theta)=\sum_{i=1}^{m}\left[y^{(i)}\log h_\theta(x^{(i)})+\left(1-y^{(i)}\right)\log\left(1-h_\theta(x^{(i)})\right)\right].$$

The negative log likelihood is:

$$J(\theta)=-\ell(\theta).$$

This is binary cross-entropy loss:

$$J(\theta)=-\sum_{i=1}^{m}\left[y^{(i)}\log h_\theta(x^{(i)})+\left(1-y^{(i)}\right)\log\left(1-h_\theta(x^{(i)})\right)\right].$$

Cross-entropy heavily penalizes confident wrong predictions because log probability becomes very negative near zero.

## 7. Why Probability Modeling Defines Loss Functions

The Gaussian noise model leads to squared loss:

$$\text{Gaussian likelihood}\quad\Rightarrow\quad\text{squared-error loss}.$$

The Bernoulli model leads to cross-entropy:

$$\text{Bernoulli likelihood}\quad\Rightarrow\quad\text{cross-entropy loss}.$$

This is the central lesson: a loss function is not just a numerical preference. It often encodes assumptions about how data are generated and what kinds of prediction errors are statistically meaningful.

## 8. Reliability Limitations

Maximum likelihood inherits the assumptions of the likelihood model. For linear regression, squared loss can be unreliable under heavy-tailed noise, outliers, heteroscedasticity, correlated errors, and model misspecification.

For logistic regression, Bernoulli likelihood can still fail under label noise, class imbalance, miscalibration, nonlinearly separable classes, distribution shift, wrong threshold choice, and complete separation.

Probability modeling clarifies the assumptions behind a loss, but it does not guarantee those assumptions hold in deployment.
