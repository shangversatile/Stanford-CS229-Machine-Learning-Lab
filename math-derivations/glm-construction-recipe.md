# GLM Construction Recipe

Cross-link: see the main Lecture 4 note sections [Conceptual Interlude C](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-c-why-exponential-family-and-glm-exist), [GLM Workflow](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#9-the-complete-glm-modeling-workflow), and [Hypothesis Function](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#10-deep-meaning-of-the-hypothesis-function), plus the standalone [GLM response and distribution map](glm-response-distribution-map.md) and [Why Exponential Family and GLM Exist](why-exponential-family-and-glm.md).

## 1. Workflow diagram

```text
response semantics
-> legal support and measurement mechanism
-> conditional distribution for Y|x
-> PMF/PDF
-> exponential-family form
-> T(y), eta, a(eta), b(y)
-> link choice
-> linear predictor
-> response mean
-> likelihood
-> NLL/loss
-> parameter optimization
-> prediction
-> diagnostics
```

This order matters because the loss is a consequence of the distribution, not the starting point.

## Historical and modeling motivation

GLMs were introduced to keep the useful parts of linear regression without pretending every supervised-learning response is a real-valued Gaussian measurement. The historical modeling compromise is:

* keep a linear predictor for interpretability and computation;
* choose a response distribution that matches the support and variance behavior of $`Y`$;
* connect covariates to the distribution through the natural parameter;
* estimate parameters by the likelihood induced by that distribution.

In canonical form, the systematic component is:

```math
\eta=\theta^Tx
```

and the prediction is the mean map induced by the log-partition function:

```math
h_\theta(x)=\nabla a(\theta^Tx)
```

This is why sigmoid, exponential, identity, and softmax responses are derived from Bernoulli, Poisson, Gaussian, and multinomial likelihoods rather than chosen as interchangeable activation functions.
## 2. Official CS229 assumptions and modern GLM terms

| CS229 assumption/design | Modern GLM term | What it means |
| ----------------------- | --------------- | ------------- |
| $`Y\mid x;\theta`$ follows an exponential-family distribution | random component | choose a conditional response distribution |
| $`h_\theta(x)=\mathbb E[T(Y)\mid x;\theta]`$ | prediction target / response mean | prediction is the conditional mean of the sufficient statistic |
| $`\eta=\theta^Tx`$ in the canonical scalar case | systematic component | the natural parameter is linear in features |
| natural parameter equals linear predictor | canonical link | the link maps mean to the natural parameter used in the exponential family |
| sigmoid, identity, exponential, softmax responses | inverse-link / response function | map the linear predictor back to a legal mean or probability |

CS229 sometimes uses $`g`$ for the response function, especially in logistic regression. Many statistics texts use $`g`$ for the link. Always check the direction of the map.

## 3. Full GLM construction workflow

1. Define the response variable $`Y`$ precisely. Decide whether $`Y`$ is a measurement, event, class label, count, positive duration, probability, or probability vector.
2. Identify support and measurement mechanism. Support rules out invalid model families, and mechanism says what kind of randomness is plausible.
3. Choose a candidate conditional distribution for $`Y\mid x`$. This choice encodes support, uncertainty, and mean-variance behavior.
4. Write the PMF/PDF. A concrete probability model is needed before writing the likelihood.
5. Rewrite the distribution in exponential-family form. Put it into the template where the natural parameter and log-partition function are visible.
6. Identify $`T(y)`$, $`\eta`$, $`a(\eta)`$, and $`b(y)`$. These four objects determine what is summarized from data, what is linearized, what normalizes probabilities, and what remains independent of parameters.
7. Decide whether to use canonical link. Canonical links often simplify gradients and curvature, but domain constraints may justify another link.
8. Set the linear predictor. For scalar canonical GLMs, set $`\eta=\theta^Tx`$; for softmax, use class-specific linear scores.
9. Derive the response mean. Use $`h_\theta(x)=\mathbb E[T(Y)\mid x;\theta]=\nabla a(\eta)`$ when in canonical exponential-family form.
10. Write likelihood over all samples. This states how the observations combine under conditional independence.
11. Convert likelihood into NLL. The NLL is the loss implied by the distribution.
12. Optimize parameters. Estimate the shared parameters using an optimization method appropriate for the NLL geometry.
13. Predict using conditional mean or class probability. The fitted parameter is inserted into the response function.
14. Diagnose model assumptions. Check support, variance, calibration, residuals, separation, identifiability, and deployment shift.

## 4. Link function versus response function

Let:

```math
\mu=\mathbb E[T(Y)\mid x;\theta]
```

The link maps the mean to the linear predictor or natural parameter:

```math
g(\mu)=\eta
```

The response function maps the linear predictor back to the mean:

```math
\mu=g^{-1}(\eta)
```

In canonical exponential-family GLMs:

```math
\mu=\nabla a(\eta)
```

If:

```math
\eta=\theta^Tx
```

then:

```math
h_\theta(x)=\nabla a(\theta^Tx)
```

| Distribution | Link direction | Response direction |
| ------------ | -------------- | ------------------ |
| Gaussian | $`g(\mu)=\mu`$ | $`\mu=\eta`$ |
| Bernoulli | $`g(\mu)=\log(\mu/(1-\mu))`$ | $`\mu=1/(1+e^{-\eta})`$ |
| Poisson | $`g(\mu)=\log\mu`$ | $`\mu=e^\eta`$ |
| Multinomial / softmax | log-odds against a reference class | normalized class probabilities |

## 5. Training versus prediction

Training estimates the parameter:

```math
\hat\theta=\underset{\theta}{\mathrm{argmax}}\ p(D\mid\theta)
```

or equivalently minimizes NLL:

```math
\hat\theta=\underset{\theta}{\mathrm{argmin}}\ J_{\mathrm{NLL}}(\theta)
```

Prediction uses the fitted parameter:

```math
h_{\hat\theta}(x)=\mathbb E[T(Y)\mid x;\hat\theta]
```

For classification, this conditional mean is usually a probability or probability vector. For regression and count models, it is a conditional expected value. The hypothesis function is therefore a prediction rule induced by the fitted probabilistic model, not the fitted parameter itself.

## 6. Bernoulli worked mini-example

**Modeling problem.** Predict whether a loan defaults.

```text
Y = 1 if default occurs, Y = 0 otherwise.
```

**Support.**

```math
Y\in\{0,1\}
```

**Distribution.**

```math
p(y;\phi)=\phi^y(1-\phi)^{1-y}
```

**Exponential-family form.**

```math
p(y;\eta)=\exp\left(\eta y-a(\eta)\right)
```

with:

```math
\eta=\log\frac{\phi}{1-\phi}
```

```math
T(y)=y
```

```math
a(\eta)=\log(1+e^\eta)
```

```math
b(y)=1
```

**Canonical linear predictor.**

```math
\eta=\theta^Tx
```

**Response mean.**

```math
h_\theta(x)=\mathbb E[Y\mid x;\theta]=\frac{1}{1+e^{-\theta^Tx}}
```

**NLL.**

```math
J(\theta)=-\sum_{i=1}^{m}\left(y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right)
```

This is binary cross-entropy. The sigmoid appears because Bernoulli natural parameter is log-odds, not because an S-curve was chosen arbitrarily.

## 7. Poisson worked mini-example

**Modeling problem.** Predict number of arrivals per hour.

```text
Y = count of arrivals in one fixed exposure window.
```

**Support.**

```math
Y\in\mathbb N_0
```

**Distribution.**

```math
p(y;\lambda)=\frac{\lambda^y e^{-\lambda}}{y!}
```

**Exponential-family form.**

```math
p(y;\eta)=\frac{1}{y!}\exp\left(\eta y-e^\eta\right)
```

with:

```math
\eta=\log\lambda
```

```math
T(y)=y
```

```math
a(\eta)=e^\eta
```

```math
b(y)=\frac{1}{y!}
```

**Canonical linear predictor.**

```math
\eta=\theta^Tx
```

**Response mean.**

```math
h_\theta(x)=\mathbb E[Y\mid x;\theta]=e^{\theta^Tx}
```

**NLL.** Ignoring constants independent of $`\theta`$:

```math
J(\theta)=\sum_{i=1}^{m}\left(e^{\theta^Tx^{(i)}}-y^{(i)}\theta^Tx^{(i)}\right)
```

This objective is tied to count-rate modeling. It should not be used for multiclass labels merely because labels can be encoded as integers.

## 8. Canonical examples summary

| Model | Response support | Natural parameter | Response function | NLL name |
| ----- | ---------------- | ----------------- | ----------------- | -------- |
| Gaussian GLM | $`\mathbb R`$ | $`\eta=\mu`$ | $`h_\theta(x)=\theta^Tx`$ | squared loss |
| Bernoulli GLM | $`\{0,1\}`$ | $`\eta=\log(\phi/(1-\phi))`$ | sigmoid | binary cross-entropy |
| Poisson GLM | $`\mathbb N_0`$ | $`\eta=\log\lambda`$ | exponential | Poisson NLL |
| Softmax GLM | $`\{1,\dots,K\}`$ | class log-odds/scores | softmax | multiclass cross-entropy |

## 9. Diagnostics after construction

| Check | Why it matters |
| ----- | -------------- |
| Support | invalid support means the model can make impossible predictions |
| Mean-variance relation | NLL curvature and uncertainty interpretation depend on this assumption |
| Link | the transformed mean should be plausibly linear in features |
| Features | a correct distribution can still underfit if the linear predictor is too weak |
| Optimization | separation, rank deficiency, and ill-conditioning can prevent stable estimates |
| Calibration | probability outputs must match empirical frequencies to be reliable |
| Shift | deployment may change the conditional distribution or base rates |

## 10. Summary

The GLM recipe is a disciplined path from response semantics to prediction. Choose the response distribution first, rewrite it in exponential-family form, connect the natural parameter to features, derive the response mean, then let the induced likelihood define the loss. After fitting, validate the assumptions rather than treating the response function as a generic activation.