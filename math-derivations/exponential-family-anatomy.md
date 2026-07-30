# Exponential Family Anatomy

Cross-link: see [Lecture 4 Section 6](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#6-anatomy-of-the-exponential-family), [Conceptual Interlude C](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-c-why-exponential-family-and-glm-exist), [Conceptual Interlude D](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-d-how-a-glm-connects-features-to-a-conditional-distribution), [Lecture 4 Section 10](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#10-deep-meaning-of-the-hypothesis-function), and [Why Exponential Family and GLM Exist](why-exponential-family-and-glm.md).

## 1. Normalized form

The canonical exponential-family form is:

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

The form is useful because it separates four jobs: how data enter the likelihood, which parameter is canonical, how the distribution is normalized, and what support/base weighting remains.

## Component intuition

The canonical form is:

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

A useful first reading is: exponential family is a normalized scoring model over possible outcomes. It does not let the parameter touch raw observations in arbitrary ways. It first reads selected statistics from $`y`$, then uses the natural parameter to value those readings.

| Component | Formal role | Intuitive role | What happens if it changes |
| --------- | ----------- | -------------- | -------------------------- |
| $`y`$ | observed response value | the outcome being explained | different outcomes receive different probability |
| $`T(y)`$ | sufficient statistic | the information readout extracted from $`y`$ | changes what the model can "see" in $`y`$ |
| $`\eta`$ | natural parameter | coordinate inside the chosen family | tilts mass along sufficient-statistic directions |
| $`\eta^TT(y)`$ | linear coupling | compatibility score between parameter and observation | higher score means higher unnormalized probability |
| $`b(y)`$ | base measure | background geometry or counting/volume rule of outcome space | changes baseline preference over $`y`$ |
| $`a(\eta)`$ | log-partition function | normalization and moment-generating engine | keeps probabilities valid and determines mean/variance |

Before normalization, the unnormalized log-score of outcome $`y`$ is:

```math
s_\eta(y)=\eta^TT(y)+\log b(y)
```

After normalization:

```math
\log p(y;\eta)=s_\eta(y)-a(\eta)
```

So $`a(\eta)`$ is the log-total unnormalized mass. For a discrete outcome space it corresponds to the log of a sum; for a continuous outcome space it corresponds to the log of an integral. Its job is to make the final probabilities sum or integrate to one.

The key mental model is: $`T(y)`$ decides what the model reads from $`y`$; $`\eta`$ decides how the distribution values those readings.

## What a probability density says about y

A probability model separates the random variable from the value being evaluated. Before observation, $`Y`$ is random. A lowercase $`y`$ is a possible value of $`Y`$, or the realized value after measurement. Writing a density such as $`p(y;\mu,\sigma^2)`$ does not change $`y`$; it assigns a density score to that value under the parameters.

For a Gaussian model:

```math
Y\sim\mathcal N(\mu,\sigma^2)
```

```math
p(y;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
```

Because $`Y`$ is continuous, this is a density rather than the exact probability of $`Y=y`$. Probabilities come from intervals:

```math
P(a\leq Y\leq b)=\int_a^b p(y;\mu,\sigma^2)dy
```

The mean $`\mu`$ controls the center of plausible values. The variance $`\sigma^2`$ controls how quickly plausibility decays as $`y`$ moves away from that center. A small variance makes the density narrow and punishes distance heavily; a large variance makes the density wider and punishes the same distance less.

Observed data update parameter estimates through likelihood. Once the family has been chosen, learning asks which parameter makes the observed values most plausible:

```math
D=\{y^{(i)}\}_{i=1}^{m}
```

```math
L(\mu,\sigma^2)=\prod_{i=1}^{m}p(y^{(i)};\mu,\sigma^2)
```

```math
(\hat\mu,\hat\sigma^2)=\underset{\mu,\sigma^2}{\mathrm{argmax}}\ L(\mu,\sigma^2)
```

This is inverse reasoning under a chosen forward model: the forward model maps parameters to a distribution over possible observations, while estimation maps observed data back to plausible parameters.

## Sufficient statistic: single-observation view and dataset view

The phrase "sufficient statistic" has two related levels in this setting.

At the single-observation level, $`T(y)`$ is the transformed representation of $`y`$ that appears in the log probability. It is the model's readout from one observed response.

For Bernoulli:

```math
T(y)=y
```

The model only needs to know whether the event happened. A single Bernoulli observation carries evidence through success versus failure.

For multiclass categorical data with a reference class:

```math
T(y)=\begin{bmatrix}\mathbf1\{y=1\}\\ \cdots\\ \mathbf1\{y=K-1\}\end{bmatrix}
```

The model reads class identity through indicator coordinates. The statistic is not the integer label as a magnitude; class $`3`$ does not mean "three times" class $`1`$. It means one mutually exclusive category was observed.

For Gaussian data with both mean and variance unknown:

```math
T(y)=\begin{bmatrix}y\\ y^2\end{bmatrix}
```

The model must read both location and spread information. The first coordinate tracks where observations lie; the second coordinate tracks squared magnitude, which is needed for variance or second-moment information.

This is why $`T(y)`$ is not always equal to $`y`$. It depends on what aspects of the observation are relevant to the parameters in the chosen distribution family.

At the dataset level, iid factorization shows exactly why the statistic is called sufficient. Starting from one observation and multiplying over $`m`$ iid samples gives:

```math
p(y^{(1)},\dots,y^{(m)};\eta)
=
\left(\prod_{i=1}^{m}b(y^{(i)})\right)
\exp\left(\eta^T\sum_{i=1}^{m}T(y^{(i)})-ma(\eta)\right)
```

All parameter-dependent information in the dataset enters through:

```math
\sum_{i=1}^{m}T(y^{(i)})
```

Once this aggregate is known, the likelihood's dependence on $`\eta`$ is fully determined. The raw sample may contain order, individual identities, or other details, but those details do not change the likelihood as a function of $`\eta`$ within this iid exponential-family model.

| Model | $`T(y)`$ | Dataset sufficient statistic | What information it preserves |
| ----- | ------ | ---------------------------- | ----------------------------- |
| Bernoulli | $`y`$ | $`\sum_i y^{(i)}`$ | number of successes |
| Gaussian, known variance | $`y`$ | $`\sum_i y^{(i)}`$ | mean/location information |
| Gaussian, unknown variance | $`(y,y^2)`$ | $`(\sum_i y^{(i)},\sum_i (y^{(i)})^2)`$ | location and spread |
| Poisson | $`y`$ | $`\sum_i y^{(i)}`$ | total count / rate evidence |
| Categorical | one-hot vector | class-count vector | category frequencies |

This is a model-relative statement. A statistic is sufficient for a parameter inside a specified family. If the family changes, the readout may change too.

## Natural parameter as exponential-tilt coordinate

The natural parameter is called natural because it is the coordinate in which the log density is linear in $`T(y)`$:

```math
\log p(y;\eta)=\eta^TT(y)-a(\eta)+\log b(y)
```

This statement is family-relative. Before $`\eta`$ can mean anything, the modeler has already fixed the support, sufficient statistic $`T(y)`$, base measure $`b(y)`$, log-partition function $`a(\eta)`$, and any dispersion convention. Those choices define the family. The natural parameter then selects one member inside that family.

Relative probabilities show exactly what $`\eta`$ controls. For two possible outcomes under the same family member:

```math
\log
\frac{p_\eta(y_1)}
{p_\eta(y_2)}
=
\eta^T\left(T(y_1)-T(y_2)\right)
+
\log\frac{b(y_1)}{b(y_2)}
```

The log-partition term cancels. The remaining term says that $`\eta`$ rewards or penalizes outcomes according to their sufficient-statistic coordinates, with $`b(y)`$ supplying the baseline geometry of the outcome space.

Changing $`\eta`$ by $`\Delta`$ gives the exponential tilt:

```math
\frac{p_{\eta+\Delta}(y)}
{p_\eta(y)}
=
\exp\left(
\Delta^TT(y)
-
a(\eta+\Delta)
+
a(\eta)
\right)
```

The term $`\Delta^TT(y)`$ raises outcomes whose statistic aligns with $`\Delta`$ and lowers outcomes whose statistic does not. The difference $`a(\eta+\Delta)-a(\eta)`$ re-normalizes the whole distribution. The observed value $`y`$ is not changed; the relative mass over possible outcomes is changed.

Different common parameters become natural coordinates after reparameterization. Bernoulli's success probability $`\phi`$ becomes log-odds:

```math
\eta=\log\frac{\phi}{1-\phi}
```

Poisson's rate $`\lambda`$ becomes log-rate:

```math
\eta=\log\lambda
```

Gaussian notation needs one extra warning. In the CS229 fixed-variance simplification with variance $`1`$, the one-parameter canonical form has:

```math
\eta=\mu
```

If the variance is fixed at a general value and written directly in ordinary density form, a natural coordinate can be written as:

```math
\eta=\frac{\mu}{\sigma^2}
```

In an exponential-dispersion GLM parameterization, the variance scale can be separated as dispersion, so the canonical mean-scale story and the general-variance density are not contradictory. If both mean and variance are unknown, the sufficient statistic expands to include second-moment information such as $`y^2`$.

The natural parameter therefore does not necessarily control every aspect of shape. Dispersion, tail behavior, truncation, censoring, mixtures, and dependence may sit outside $`\eta`$ and need separate modeling.

The local control interpretation follows from the gradient:

```math
\nabla_\eta\log p(y;\eta)=T(y)-\nabla a(\eta)
```

and the moment identity:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

Therefore, for a small change $`\Delta\eta`$:

```math
\Delta\log p(y;\eta)\approx \Delta\eta^T\left(T(y)-\mathbb E_\eta[T(Y)]\right)
```

If $`T_j(y)`$ is larger than its current expectation, increasing $`\eta_j`$ raises the log probability of that outcome. If $`T_j(y)`$ is smaller than expected, increasing $`\eta_j`$ lowers its relative probability.

## The coupling term $`\eta^TT(y)`$

The expression $`\eta^TT(y)`$ is the bridge between the parameter side and the observation side.

$`T(y)`$ lives on the observation side: it says which statistical coordinates were read from the observed response. $`\eta`$ lives on the parameter side: it says how strongly the distribution values each coordinate. Their dot product is a compatibility score:

```math
\eta^TT(y)=\sum_j \eta_jT_j(y)
```

A larger value means the current parameter assigns a higher unnormalized score to outcomes with that statistic pattern. The base measure adds a parameter-independent baseline score, and the log-partition function subtracts the log-total score so the result is a valid probability distribution.

This coupling also explains likelihood fitting. Training increases compatibility between the learned natural parameter and the statistics observed in data, while normalization prevents the model from raising all scores for free.

## From natural parameter to GLM

In an unconditional exponential-family distribution, $`\eta`$ is fixed. A GLM turns it into a supervised-learning model by making the natural coordinate depend on features.

First define the feature-side linear predictor:

```math
s_\theta(x)=\theta^Tx
```

In the scalar canonical construction, set:

```math
\eta(x)=s_\theta(x)
```

For vector-valued natural parameters, as in softmax-style multiclass modeling, each coordinate can have its own linear predictor:

```math
\eta_k(x)=s_k(x)=\theta_k^Tx
```

The conditional distribution becomes:

```math
p(y|x;\theta)=b(y)\exp\left(\eta(x)^TT(y)-a(\eta(x))\right)
```

The prediction is the conditional mean of the sufficient statistic:

```math
h_\theta(x)=\mathbb E[T(Y)\mid x;\theta]=\nabla a(\eta(x))
```

This is the bridge from statistics to machine learning. The feature vector $`x`$ does not directly predict raw $`y`$; it predicts the natural coordinate of the distribution of $`Y\mid x`$. The chosen distribution then determines what the prediction means.

| Model | Statistic readout | Canonical coordinate from features | Prediction meaning |
| ----- | ----------------- | ---------------------------------- | ------------------ |
| Bernoulli | $`T(y)=y`$ | $`\eta(x)=s_\theta(x)`$ controls log-odds | probability of success |
| Poisson | $`T(y)=y`$ | $`\eta(x)=s_\theta(x)`$ controls log-rate | expected count |
| Gaussian, known variance | $`T(y)=y`$ | $`\eta(x)=s_\theta(x)`$ controls location | conditional mean |
| Softmax | one-hot vector | $`\eta_k(x)=s_k(x)`$ controls class log-odds | class-probability vector |

A GLM is therefore not just "choose an activation function." It is a probabilistic construction: choose the response distribution, identify what $`T(y)`$ reads, let features control $`s_\theta(x)`$ and then $`\eta(x)`$ under a link choice, use $`a(\eta)`$ to get the mean response, and fit $`\theta`$ by likelihood.
## 2. Normalization

Define the unnormalized normalizer:

```math
Z(\eta)=\int b(y)\exp\left(\eta^TT(y)\right)dy
```

For discrete distributions, replace the integral with a sum. The log-partition function is:

```math
a(\eta)=\log Z(\eta)
```

Then:

```math
\int p(y;\eta)dy
=
\int b(y)\exp\left(\eta^TT(y)-a(\eta)\right)dy
```

```math
=
e^{-a(\eta)}Z(\eta)
```

```math
=1
```

So $`a(\eta)`$ is not decoration. It is the term that makes the expression a valid probability distribution.

## 3. Natural parameter

The natural parameter $`\eta`$ is the parameter that appears linearly against the sufficient statistic in canonical form. It is often not the parameter used in ordinary distribution descriptions.

Bernoulli common parameter:

```math
\phi=P(Y=1)
```

Bernoulli natural parameter:

```math
\eta=\log\frac{\phi}{1-\phi}
```

Poisson common parameter:

```math
\lambda=\mathbb E[Y]
```

Poisson natural parameter:

```math
\eta=\log\lambda
```

In a canonical scalar GLM, the feature-side linear predictor is first named:

```math
s_\theta(x)=\theta^Tx
```

and the canonical construction sets:

```math
\eta(x)=s_\theta(x)
```

## 4. Sufficient statistic

The sufficient statistic $`T(y)`$ is the part of the observation that interacts with the natural parameter. It is not always equal to $`y`$.

| Model | Common sufficient statistic | Why it matters |
| ----- | --------------------------- | -------------- |
| Bernoulli | $`T(y)=y`$ | the single binary value is the success count |
| Fixed-variance Gaussian | $`T(y)=y`$ | the mean is the only unknown response parameter |
| Unknown-variance Gaussian | $`T(y)=(y,y^2)`$ | mean and second moment both carry parameter information |
| Poisson | $`T(y)=y`$ | total count summarizes evidence for the rate |
| Categorical / multinomial | one-hot or count vector | class counts summarize evidence for probabilities |

For iid data, the parameter-dependent information aggregates through:

```math
\sum_{i=1}^{m}T(y^{(i)})
```

That is the sufficiency intuition behind exponential-family likelihoods.

## 5. Log-partition function as moment engine

Under regularity conditions allowing differentiation under the integral:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

and:

```math
\nabla^2a(\eta)=\mathrm{Cov}_\eta(T(Y))
```

In the scalar case:

```math
a''(\eta)=\mathrm{Var}_\eta(T(Y))
```

Because covariance matrices are positive semidefinite:

```math
\nabla^2a(\eta)\succeq0
```

Thus $`a(\eta)`$ is convex. This convexity is the source of the favorable geometry in many GLM negative log likelihoods.

## 6. Base measure

The base measure $`b(y)`$ collects the part of the mass or density independent of $`\eta`$. It affects support and the full likelihood value, but it does not affect gradients with respect to $`\eta`$.

Examples:

```math
b_{\mathrm{Gaussian}}(y)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{y^2}{2}\right)
```

```math
b_{\mathrm{Bernoulli}}(y)=1
```

```math
b_{\mathrm{Poisson}}(y)=\frac{1}{y!}
```

## 7. Connection to GLM response functions

A GLM predicts the conditional mean of the sufficient statistic:

```math
h_\theta(x)=\mathbb E[T(Y)\mid x;\theta]
```

In canonical exponential-family form:

```math
h_\theta(x)=\nabla a(\eta)
```

With the feature-side linear predictor:

```math
s_\theta(x)=\theta^Tx
```

and the scalar canonical choice:

```math
\eta(x)=s_\theta(x)
```

so:

```math
h_\theta(x)=\nabla a(\eta(x))=\nabla a(\theta^Tx)
```

This is the distribution-to-response-function bridge:

| Distribution | $`a(\eta)`$ | $`\nabla a(\eta)`$ | GLM response |
| ------------ | --------- | ---------------- | ------------ |
| Gaussian fixed variance | $`\eta^2/2`$ | $`\eta`$ | identity |
| Bernoulli | $`\log(1+e^\eta)`$ | $`e^\eta/(1+e^\eta)`$ | sigmoid |
| Poisson | $`e^\eta`$ | $`e^\eta`$ | exponential |
| Multinomial / softmax | $`\log\sum_j e^{\eta_j}`$ | normalized probabilities | softmax |

The response function is therefore not an arbitrary activation. It is the inverse-link or mean map implied by the chosen distribution and its log-partition function.

## 8. Iid factorization and likelihood geometry

For iid data:

```math
p(y^{(1)},\ldots,y^{(m)};\eta)
=
\prod_{i=1}^{m}b(y^{(i)})\exp\left(\eta^TT(y^{(i)})-a(\eta)\right)
```

Rearrange:

```math
=
\left(\prod_{i=1}^{m}b(y^{(i)})\right)
\exp\left(\eta^T\sum_{i=1}^{m}T(y^{(i)})-ma(\eta)\right)
```

The natural-parameter-dependent part is linear in the sample statistic and subtracts $`ma(\eta)`$. Since $`a(\eta)`$ is convex, the log-likelihood is concave in $`\eta`$ and the NLL is convex in $`\eta`$ for the regular canonical setting.

## Why this form is natural rather than arbitrary

The canonical form is not a random algebraic trick. It is natural from several independent directions; see the longer explanation in [Why Exponential Family and GLM Exist](why-exponential-family-and-glm.md).

For iid data, the likelihood aggregates evidence through the sample statistic:

```math
\sum_{i=1}^{m}T(y^{(i)})
```

This is the sufficiency viewpoint: once support and regularity conditions hold, fixed-dimensional sufficient statistics for all sample sizes lead essentially to exponential-family likelihoods. Maximum entropy gives a second route: if only expectations of chosen statistics are constrained, the least committed distribution has density or mass proportional to:

```math
b(y)\exp\left(\eta^TT(y)\right)
```

Normalization then forces the log-partition term $`a(\eta)`$, whose derivatives generate means and covariance. Thus sufficiency explains why $`T(y)`$ appears, maximum entropy explains why the exponential tilt appears, and normalization explains why the same $`a(\eta)`$ controls moments and convexity.

## 9. Modeling lesson

Exponential family is not the set of all distributions. It is a family with a special algebraic structure: sufficient statistics enter linearly, the log-partition function normalizes the distribution, and derivatives of the log-partition function produce moments. GLMs use that structure to turn a response distribution into a principled response function and likelihood.
