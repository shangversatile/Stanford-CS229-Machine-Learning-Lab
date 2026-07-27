# GLM Construction Recipe

## 1. Standard Terminology

A generalized linear model connects a response distribution, a linear predictor, and a response mean.

| Term | Meaning |
| ---- | ------- |
| Random component | conditional distribution of $Y$ given features |
| Systematic component | linear predictor, often $\eta=\theta^Tx$ |
| Link function | map from mean $\mu$ to linear predictor $\eta$ |
| Response function | inverse link, map from $\eta$ to mean $\mu$ |
| Canonical link | link where natural parameter equals linear predictor |

CS229 sometimes uses $g$ for the response function, while many statistics texts use $g$ for the link. Always check whether $g$ maps score to mean or mean to score.

## 2. Three Assumptions or Design Choices

GLM construction typically assumes:

1. The conditional response distribution belongs to an exponential family:

```math
p(y\mid x;\theta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

2. The prediction target is the conditional mean of the sufficient statistic:

```math
h_\theta(x)=\mathbb E[T(Y)\mid x;\theta]
```

3. The natural parameter is connected to features by a linear predictor:

```math
\eta=\theta^Tx
```

For vector-valued natural parameters, the linear predictor can be class-specific, as in softmax regression.

## 3. Link Versus Response Function

Let:

```math
\mu=\mathbb E[T(Y)\mid x;\theta]
```

The link function maps mean to natural parameter or linear predictor:

```math
g(\mu)=\eta
```

The response function maps linear predictor to mean:

```math
\mu=g^{-1}(\eta)
```

In canonical exponential-family GLMs:

```math
\mu=\nabla a(\eta)
```

So the response function is often:

```math
g^{-1}(\eta)=\nabla a(\eta)
```

Examples:

| Distribution | Canonical natural parameter | Response mean |
| ------------ | --------------------------- | ------------- |
| Gaussian | $\eta=\mu$ | $\mu=\eta$ |
| Bernoulli | $\eta=\log(\phi/(1-\phi))$ | $\phi=1/(1+e^{-\eta})$ |
| Poisson | $\eta=\log\lambda$ | $\lambda=e^\eta$ |
| Categorical | class log odds | softmax probabilities |

## 4. Full Modeling Workflow

### Step 1: Identify Response Semantics

Ask what $y$ represents. A class label, count, measurement, waiting time, and probability vector should not be modeled with the same distribution merely because they can be stored as numbers.

### Step 2: Choose Conditional Distribution

Choose $p(y\mid x;\theta)$ by considering:

* support;
* variance behavior;
* skewness and tails;
* zero inflation;
* dependence and grouping;
* data-generating mechanism.

### Step 3: Write Exponential-Family Form

Identify:

```math
T(y),\quad \eta,\quad a(\eta),\quad b(y)
```

This step prevents arbitrary activation choices by forcing the distribution to determine the response map.

### Step 4: Choose Link

The canonical link sets:

```math
\eta=\theta^Tx
```

Noncanonical links may be useful, but then the algebra and optimization geometry may be less simple.

### Step 5: Derive Response Mean

Use:

```math
h_\theta(x)
=
\mathbb E[T(Y)\mid x;\theta]
=
\nabla a(\eta)
```

Then substitute the linear predictor.

### Step 6: Write Likelihood

For conditionally independent data:

```math
L(\theta)=\prod_{i=1}^{m}p(y^{(i)}\mid x^{(i)};\theta)
```

and:

```math
J_{\mathrm{NLL}}(\theta)
=
-\sum_{i=1}^{m}
\log p(y^{(i)}\mid x^{(i)};\theta)
```

### Step 7: Estimate Parameters

Frequentist MLE treats $\theta$ as fixed but unknown. The estimator:

```math
\hat\theta(D)
```

is a function of the dataset. Training chooses the parameter that makes the observed sample most plausible under the chosen model family.

### Step 8: Predict and Diagnose

Predictions use conditional means or probabilities, but reliability requires diagnostics:

* residual structure;
* calibration;
* class-specific error;
* overdispersion;
* separation;
* rank and identifiability;
* train/deployment shift.

## 5. Canonical Examples

Gaussian GLM:

```math
h_\theta(x)=\theta^Tx
```

Bernoulli GLM:

```math
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
```

Poisson GLM:

```math
h_\theta(x)=e^{\theta^Tx}
```

Softmax GLM:

```math
p(y=k\mid x;\Theta)
=
\frac{\exp(\theta_k^Tx)}
{\sum_{j=1}^{K}\exp(\theta_j^Tx)}
```

## 6. Reliability Checks

| Check | Question |
| ----- | -------- |
| Support | Can the response function produce only valid means? |
| Distribution | Does the variance/tail behavior match data? |
| Link | Is the transformed mean plausibly linear in features? |
| Features | Is the linear predictor expressive enough? |
| Optimization | Is the objective well-conditioned and finite? |
| Identifiability | Do different parameters produce the same distribution? |
| Calibration | Do predicted probabilities match empirical frequencies? |
| Shift | Does the conditional relationship remain stable out of sample? |

The GLM recipe is a disciplined way to turn response semantics into likelihood, but it does not remove the need for empirical validation.

