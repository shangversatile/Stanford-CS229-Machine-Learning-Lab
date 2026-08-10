# GLM Construction Recipe

Cross-link: see the main Lecture 4 note sections [Conceptual Interlude D: Why GLM Components Form a Statistical Model](../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-d-why-glm-components-form-a-statistical-model), [8. GLM Components](../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#8-glm-components), [9. The Complete GLM Modeling Workflow](../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#9-the-complete-glm-modeling-workflow), and [10. Deep Meaning of the Hypothesis Function](../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#10-deep-meaning-of-the-hypothesis-function). For related reference maps, see [Exponential Family Anatomy](02-exponential-family-anatomy.md), [Sufficient Statistics and Likelihood Equivalence](03-sufficient-statistics-likelihood-equivalence.md), [GLM Response and Distribution Map](08-glm-response-distribution-map.md), and [Log-Partition Mean, Variance, and Convexity](05-log-partition-mean-variance-convexity.md).

```text
random sampling
-> observed sample
-> probability model
-> likelihood
-> maximum likelihood estimation
-> canonical statistics and sample sufficient statistics
-> ordinary distribution parameters
-> natural parameters
-> global trainable parameters
-> systematic component
-> conditional prediction
```

This order matters because a GLM is not a curve-fitting trick. It is a constrained conditional-distribution model fitted by likelihood.

## 1. Conditional rather than joint modeling

A joint model factors as:

```math
p(x,y)
=
p(y\mid x)p(x)
```

Supervised prediction usually asks for $`Y`$ after $`x`$ is known. A conditional GLM therefore models $`p(Y_i\mid x_i;\theta)`$ directly. It can treat $`X`$ as fixed design, or condition on the observed covariates. This avoids modeling a high-dimensional marginal distribution for $`x`$ when that marginal distribution is not needed for the prediction objective.

If the joint model has independent parameter blocks:

```math
\ell(\theta,\alpha)
=
\sum_{i=1}^n\log p_\theta(y_i\mid x_i)
+
\sum_{i=1}^n\log p_\alpha(x_i)
```

then the optimizer for the conditional parameter $`\theta`$ is unaffected by the $`p_\alpha(x_i)`$ term. This is a modeling simplification, not a denial that a joint distribution exists.

Joint modeling can be necessary for generative sampling of covariates, missing covariates, latent variables, selection effects, and causal structure. Conditional modeling also does not automatically solve covariate shift; a changed deployment distribution or changed conditional mechanism still has to be diagnosed.

This is also the discriminative/generative taxonomy boundary. Logistic regression and conditional GLMs directly model $`p(y\mid x;\theta)`$ and do not model $`p(x)`$, so they are probabilistic discriminative models. Gaussian Discriminant Analysis and Naive Bayes model a joint distribution or class-conditional input distribution, such as $`p(y)p(x\mid y)`$, and are classical generative models. A GLM can sample $`Y\mid x`$ after $`x`$ is given, but that does not make it a model of complete $`(X,Y)`$ pairs.

## 2. Global and local parameter hierarchy

For sample $`i`$, use the hierarchy:

```text
global parameter theta
    +
local input x_i
    ↓
linear predictor xi_i
    ↓
natural parameter eta_i
    ↓
ordinary distribution parameter psi_i
    ↓
conditional distribution p(Y_i | x_i; theta)
    ↓
observed y_i
```

The global trainable parameter is:

```math
\theta\in\mathbb R^p
```

The input and design matrix are:

```math
x_i\in\mathbb R^p,
\qquad
X\in\mathbb R^{n\times p}
```

The systematic component is:

```math
\xi_i=x_i^T\theta
```

The local natural parameter is not the global parameter:

```math
\eta_i\neq\theta
```

In the scalar canonical construction, the relation is:

```math
\eta_i=x_i^T\theta
```

For all samples:

```math
\boldsymbol\eta=X\theta
```

Ordinary and natural parameters are coordinates of the same local conditional distribution. Use a separate reparameterization map:

```math
\eta_i=r(\psi_i),
\qquad
r:\Psi\rightarrow\mathcal H
```

When this map is invertible on the relevant domain:

```math
\psi_i=r^{-1}(\eta_i)
```

This means that for every admissible ordinary parameter $`\psi\in\Psi`$, the natural parameter $`\eta=r(\psi)`$ indexes the same probability law:

```math
p(y;\psi)
=
p(y;r(\psi))
\quad
\text{for every admissible }y.
```

The set of probability measures has not changed. Only the coordinates used to index that set have changed. The map $`r`$ is found by rewriting the ordinary family into exponential-family form, so it is the map from an ordinary parameter to the coefficient of the canonical observation statistic in the log density.

Examples:

| Family | Ordinary parameter $`\psi_i`$ | Natural parameter $`\eta_i`$ |
| ------ | ----------------------------- | ---------------------------- |
| Gaussian, CS229 variance one | mean $`\mu_i`$ | $`\eta_i=\mu_i`$ |
| Bernoulli | success probability $`p_i`$ | $`\eta_i=\log(p_i/(1-p_i))`$ |
| Poisson | rate $`\lambda_i`$ | $`\eta_i=\log\lambda_i`$ |

There is also an expectation-parameter coordinate:

```math
\mu_T
:=
\mathbb E_\eta[T(Y)].
```

Define the natural-to-expectation map:

```math
m(\eta)
:=
\mathbb E_\eta[T(Y)].
```

In a regular exponential family:

```math
m(\eta)=\nabla a(\eta).
```

The map $`m`$ is not the same object as $`r`$. The map $`r`$ answers "which natural coefficient represents this ordinary distribution member?" The map $`m`$ answers "given this natural parameter, what canonical statistic does the distribution expect to observe?" If $`m`$ is invertible on the relevant parameter region, the canonical link on the statistic-expectation scale is:

```math
\ell_c=m^{-1}.
```

Then:

```math
\eta=\ell_c(\mu_T).
```

If $`T(Y)=Y`$, then $`\mu_T=\mathbb E[Y]=\mu`$, so the canonical link can be written as a mean-to-natural map. If $`T(Y)\neq Y`$, the statistic expectation and scalar response mean must be kept separate.

The learned object is $`\theta`$, the shared parameter of the mapping from features to distribution coordinates. The local ordinary parameters $`\psi_i`$ and natural parameters $`\eta_i`$ are induced per sample.

Only under a canonical link is:

```math
\xi_i=\eta_i=x_i^T\theta
```

For a general link, the guaranteed relation is:

```math
g(\mu_i)=\xi_i
```

For scalar families with $`T(Y)=Y`$ and identifiable mean map, $`\eta_i=\ell_c(\mu_i)`$ is determined by the distribution family. In vector-statistic cases, first identify the statistic expectation $`\mu_{T,i}`$ or a distribution-specific identity connecting the chosen mean coordinate to $`\mu_{T,i}`$.

Why did the older notation $`\eta=q(\psi)`$ and $`\eta=q(\mu)`$ look plausible? In common scalar examples the ordinary parameter often equals the response mean or statistic expectation:

* Bernoulli has $`\psi=p=\mathbb E[Y]`$, so $`r(p)=\log(p/(1-p))`$ and $`\ell_c(\mu)=\log(\mu/(1-\mu))`$ have the same literal formula.
* Poisson has $`\psi=\lambda=\mathbb E[Y]`$, so $`r(\lambda)=\log\lambda`$ and $`\ell_c(\mu)=\log\mu`$ look identical.
* Gaussian with fixed variance often uses $`\psi=\mu=\mathbb E[Y]`$.
* Categorical probability coordinates $`\phi`$ satisfy $`\mathbb E[T(Y)]=\phi`$ when $`T`$ is one-hot or reference-class one-hot.

These examples collapse different parameter spaces numerically. The coincidence is distribution-specific, not a general definition.

## 3. Random component

The random component chooses the conditional response family:

```math
Y_i\mid x_i;\theta
\sim
\text{an exponential-family conditional distribution}
```

This defines the legal response support, probability or density shape, and variance behavior. It also defines the one-observation canonical statistic $`T(Y_i)`$ whose expectation is the GLM prediction on the statistic scale.

In canonical form:

```math
p(y_i;\eta_i)
=
b(y_i)
\exp\left(
\eta_i^TT(y_i)-a(\eta_i)
\right)
```

The statistic map:

```math
T:\mathcal Y\rightarrow\mathbb R^m
```

is not an arbitrary coordinate chart on observations. It collects the observation-side functions through which the parameter-dependent part of the likelihood is allowed to vary:

```math
\log p_\eta(y)
=
\eta^TT(y)-a(\eta)+\log b(y).
```

Thus, if $`T(y)=T(y')`$, then the parameter-dependent part of:

```math
\frac{p_\eta(y)}
{p_\eta(y')}
```

cancels. Once the canonical statistic is known, distinctions between outcomes in the same $`T`$-fiber do not provide additional likelihood information about $`\eta`$.

This should not be confused with the definition of a general statistic. A statistic $`S(\mathbf Y)`$ is just a function of the sample; it is sufficient only when a model-relative property such as Fisher-Neyman factorization, the conditional-distribution definition, or the likelihood-ratio criterion verifies it. In the iid exponential-family case:

```math
p_\eta(\mathbf y)
=
\prod_i b(y_i)
\exp\left(
\eta^T\sum_iT(y_i)-na(\eta)
\right),
```

so:

```math
S(\mathbf Y)=\sum_iT(Y_i)
```

is sufficient by factorization. Formally, sufficiency is derived from this factorization. Structurally, the exponential-family representation has already forced all parameter-dependent sample information to pass through the canonical statistic.

To identify the necessary $`T`$-structure from an ordinary family $`p(y;\psi)`$, expand $`\log p(y;\psi)`$ and look for functions of $`y`$ whose coefficients vary as the unknown parameter directions vary. For:

```math
Y\sim N(\mu,\sigma^2),
```

```math
\log p(y;\mu,\sigma^2)
=
-\frac{y^2}{2\sigma^2}
+
\frac{\mu}{\sigma^2}y
+
\text{terms not involving }y.
```

If $`\sigma^2`$ is known and only $`\mu`$ varies, $`y^2`$ has a fixed coefficient and $`T(y)=y`$ is enough. If both $`\mu`$ and $`\sigma^2`$ vary, the coefficients of both $`y`$ and $`y^2`$ vary, so:

```math
T(y)
=
\begin{bmatrix}
y\\
y^2
\end{bmatrix}.
```

The statistic map does not change when a particular parameter value changes. The free parameter directions in the model family determine which canonical-statistic functions are needed.

The conditional model substitutes a feature-dependent local parameter:

```math
p(y_i\mid x_i;\theta)
=
b(y_i)
\exp\left(
\eta_i^TT(y_i)-a(\eta_i)
\right)
```

## 4. Systematic component, general link, and canonical link

The systematic component is the feature-side score:

```math
\xi_i
=
s_\theta(x_i)
=
x_i^T\theta
```

A general GLM links a chosen mean coordinate to this score:

```math
g(\mu_i)=\xi_i
```

For scalar-response GLMs, this usually means:

```math
\mu_i=\mathbb E[Y_i\mid x_i;\theta].
```

When the canonical statistic is not the raw response, keep the statistic expectation separate:

```math
\mu_{T,i}
=
\mathbb E[T(Y_i)\mid x_i;\theta].
```

If the expectation map $`m(\eta)=\mathbb E_\eta[T(Y)]`$ is invertible on the relevant region, then:

```math
\eta_i=\ell_c(\mu_{T,i}),
\qquad
\ell_c=m^{-1}.
```

For scalar families with $`T(Y)=Y`$, this reduces to the usual mean-to-natural relation $`\eta_i=\ell_c(\mu_i)`$.

On the statistic-expectation scale, the canonical link is this inverse expectation map:

```math
\ell_c=m^{-1}.
```

For scalar families with $`T(Y)=Y`$, this is the usual GLM link $`g`$, so $`g=\ell_c`$.

Equivalently, if a general construction writes $`\eta_i=f(\xi_i)`$, the canonical construction chooses $`f(\xi)=\xi`$. Only in that case:

```math
\xi_i=\eta_i=x_i^T\theta
```

This equality is a condition, not a general GLM fact. With a noncanonical link, the systematic component $`\xi_i`$ and natural parameter $`\eta_i`$ are connected indirectly through the chosen mean coordinate.

The canonical choice is a modelling choice, but it is structurally motivated. In the exponential-family log density:

```math
\log p(y;\eta)
=
\eta^TT(y)-a(\eta)+\log b(y),
```

the natural parameter is the coefficient coordinate directly coupled to the canonical statistic. Setting $`\eta_i=x_i^T\theta`$ makes the feature-linear coordinate generated by $`x_i`$ directly control the parameter-side coefficients of the sufficient-statistic directions.

For a scalar statistic, write one-sample log likelihood as:

```math
\ell_i
=
T(y_i)\eta_i-a(\eta_i)+\log b(y_i).
```

If:

```math
\eta_i=f(\xi_i),
\qquad
\xi_i=x_i^T\theta,
```

then the chain rule gives:

```math
\nabla_\theta\ell_i
=
x_i f'(\xi_i)
\left[
T(y_i)-a'(\eta_i)
\right].
```

Using $`a'(\eta_i)=\mathbb E[T(Y_i)\mid x_i]`$:

```math
\nabla_\theta\ell_i
=
x_i f'(\xi_i)
\left[
T(y_i)-\mathbb E[T(Y_i)\mid x_i]
\right].
```

The canonical alignment has $`f'(\xi_i)=1`$, so:

```math
\nabla_\theta\ell_i
=
x_i
\left[
T(y_i)-\mathbb E[T(Y_i)\mid x_i]
\right].
```

Setting the summed score to zero gives feature-weighted moment matching:

```math
\sum_i x_iT(y_i)
=
\sum_i x_i\mathbb E[T(Y_i)\mid x_i].
```

The scalar canonical Hessian of the log likelihood is:

```math
\nabla_\theta^2\ell
=
-
\sum_i
x_ix_i^T
\mathrm{Var}(T(Y_i)\mid x_i).
```

So likelihood curvature, canonical-statistic variability, and feature geometry live in the same aligned coordinate system. A noncanonical link generally carries extra chain-rule factors.

The canonical link has prerequisites. The response family must already be selected and written in exponential-family form; the natural parameter must be well defined; the map $`m(\eta)=\mathbb E_\eta[T(Y)]`$ must be regular enough for the intended calculation; writing $`\ell_c=m^{-1}`$ requires invertibility or identifiability on the relevant region; the linear predictor must land in the legal natural-parameter domain; and the modelling assumption is that the useful conditional natural parameter can be represented or approximated by a linear function of the chosen features. "Canonical" means using the response family's own natural coordinate, not being universally correct for the dataset.

### 4.1 Logical status of the relations

The main formulas have different logical status:

* Definition: $`p_\eta(y)=b(y)e^{\eta^TT(y)-a(\eta)}`$ is the exponential-family representation, and $`\mu_T:=\mathbb E_\eta[T(Y)]`$ defines the expectation parameter.
* Algebraically determined relation: $`\eta=r(\psi)`$ comes from rewriting the same ordinary family in canonical form, such as $`r(p)=\log(p/(1-p))`$ for Bernoulli.
* Exponential-family theorem: $`\nabla a(\eta)=\mathbb E_\eta[T(Y)]`$ and $`\nabla^2a(\eta)=\mathrm{Cov}_\eta(T(Y))`$ follow from the representation under regularity conditions.
* Distribution-specific identity: $`\mathbb E[Y]=p`$ for Bernoulli, $`\mathbb E[Y]=\lambda`$ for Poisson, and $`\mathbb E[T(Y)]=\phi`$ for categorical data are facts about those distributions.
* GLM modelling choice: $`\xi_i=x_i^T\theta`$ and, under the canonical construction, $`\eta_i=\xi_i`$, impose a feature-linear conditional natural parameter.

| Status | Example | Consequence |
| ------ | ------- | ----------- |
| Definition | $`p_\eta(y)=b(y)e^{\eta^TT(y)-a(\eta)}`$ | fixes the representation |
| Algebra | $`\eta=r(\psi)`$ | same family, different coordinates |
| Theorem | $`m(\eta)=\nabla a(\eta)`$ | moment map from log partition |
| Distribution identity | Bernoulli $`p=\mathbb E[Y]`$ | explains formula coincidences |
| Modelling choice | $`\eta_i=x_i^T\theta`$ | linear natural-parameter assumption |

The same scalar $`\xi(x)=\theta^Tx`$ has two complementary readings. Geometrically, for any constant $`c`$, the set:

```math
\theta^Tx=c
```

is a hyperplane in input space. Statistically, that hyperplane is a level set of the systematic component: every point on it receives the same $`\xi(x)=c`$. Under a canonical link, this also means the same natural coordinate $`\eta(x)=c`$ and therefore the same local response-distribution parameterization. With a general link, the model instead says:

```math
g(\mu(x))=\theta^Tx
```

so the hyperplane is a level set of the linked conditional mean, not necessarily a level set of the natural parameter itself.

Thus a GLM should be read in this order:

```text
x
-> linear predictor xi(x) = theta^T x
-> distribution coordinate through the link
-> conditional response distribution p(Y | X = x)
-> random response Y
```

The score $`\theta^Tx`$ is not generically "the probability." It is the systematic coordinate produced from $`x`$. With a canonical link, it is the natural parameter of the conditional response distribution; with a noncanonical link, it is the chosen link-scale coordinate for the conditional mean.

For logistic regression:

```math
P(Y=1\mid x)=\sigma(\theta^Tx)
```

so:

```math
\theta^Tx=c
```

is an iso-probability surface:

```math
P(Y=1\mid x)=\sigma(c)
```

The probabilistic model creates a whole family of equal-probability surfaces. The classification decision boundary is only one special level set after a decision rule is imposed. With a $`0.5`$ threshold:

```math
P(Y=1\mid x)=0.5
\quad\Longleftrightarrow\quad
\theta^Tx=0
```

So the correct order is probability model first, decision rule second, decision boundary third. The GLM is not "draw a hyperplane and then attach a probability"; it is a conditional distribution model whose link-scale level sets happen to be hyperplanes when the systematic component is linear.

For softmax with class $`K`$ chosen as the reference class:

```math
\eta_k(x)=\theta_k^Tx,
\quad k=1,\ldots,K-1
```

means:

```math
\eta_k(x)
=
\log\frac{P(Y=k\mid x)}
{P(Y=K\mid x)}
```

In the symmetric $`K`$-score implementation, the same relative statement is:

```math
\log\frac{P(Y=i\mid x)}
{P(Y=j\mid x)}
=
(\theta_i-\theta_j)^Tx
```

The multiple linear score surfaces are the geometry in input space of a conditional categorical distribution; they are not separate independent binary probability models.

## 5. Mean map from the log-partition function

The log-partition function is:

```math
a(\eta)=\log\int b(y)\exp\left(\eta^TT(y)\right)dy
```

Under regularity conditions:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

For a scalar statistic:

```math
\mu_T=a'(\eta)
```

This is the natural-to-expectation map on the statistic scale, $`m(\eta)=\nabla a(\eta)`$. Only when $`T(Y)=Y`$ is it also the usual inverse-link / response mapping from natural parameter to scalar conditional mean. If the canonical scalar GLM sets $`\eta_i=x_i^T\theta`$ and $`T(Y)=Y`$, prediction is:

```math
h_\theta(x_i)
=
\mu_i
=
a'(x_i^T\theta)
```

For vector-valued statistics, the gradient gives the mean vector:

```math
h_\theta(x_i)
=
\mathbb E[T(Y_i)\mid x_i;\theta]
=
\nabla a(\eta_i)
```

## 6. Forward model, inverse learning, and prediction

Forward probability model in the canonical subcase:

```text
x_i
-> xi_i = x_i^T theta
-> eta_i = xi_i
-> ordinary parameter psi_i
-> conditional distribution p(Y_i | x_i; theta)
-> random sample y_i
```

Inverse learning:

```text
observed (X, y)
-> likelihood as a function of theta
-> maximum likelihood optimization
-> theta_hat
```

Prediction:

```text
x_new
-> eta_hat(x_new)
-> conditional distribution
-> mean / probability / predictive uncertainty
```

The forward direction explains what parameters do before sampling. The inverse direction explains why likelihood varies $`\theta`$ while holding observed $`(X,\mathbf y)`$ fixed. The prediction direction explains why $`h_\theta(x)`$ is a distributional summary rather than the trainable parameter itself.

## 7. Canonical log-likelihood and score equation

For a scalar canonical GLM:

```math
\eta_i=x_i^T\theta
```

Substitute into the exponential-family log density:

```math
\log p(y_i\mid x_i;\theta)
=
T(y_i)x_i^T\theta
-
a(x_i^T\theta)
+
\log b(y_i)
```

The full log-likelihood is:

```math
\ell(\theta)
=
\sum_{i=1}^n
\left(
T(y_i)x_i^T\theta
-
a(x_i^T\theta)
+
\log b(y_i)
\right)
```

Differentiate one sample. Since:

```math
\nabla_\theta(x_i^T\theta)=x_i
```

the gradient is:

```math
\nabla_\theta\log p(y_i\mid x_i;\theta)
=
x_i
\left(
T(y_i)-a'(x_i^T\theta)
\right)
```

Using:

```math
a'(x_i^T\theta)
=
\mathbb E[T(Y_i)\mid x_i;\theta]
```

we get the canonical GLM score equation:

```math
\nabla_\theta\ell(\theta)
=
\sum_{i=1}^n
x_i
\left(
T(y_i)
-
\mathbb E[T(Y_i)\mid x_i;\theta]
\right)
```

A finite interior MLE satisfies:

```math
\sum_{i=1}^n
x_i
\left(
T(y_i)
-
\mathbb E_{\hat\theta}[T(Y_i)\mid x_i]
\right)
=
0
```

The term $`T(y_i)-\mathbb E[T(Y_i)\mid x_i;\theta]`$ is a distributional residual: observed statistic minus model expectation. Each residual pushes the global parameter along the feature direction $`x_i`$. At the optimum, no feature direction has a remaining systematic residual, subject to existence and identifiability conditions. This is the algebraic link between $`X`$, $`Y_i`$, $`T(Y_i)`$, $`\eta_i`$, $`\theta`$, likelihood, and MLE.

This is feature-weighted matching, not shared-mean iid matching. A GLM does not estimate every $`\mu_i`$ by the global label mean. It learns one shared $`\theta`$ so every feature direction balances observed statistic against conditional model expectation.

## 8. Hessian and existence caveats

For the per-sample negative log-likelihood:

```math
J_i(\theta)
=
a(\eta_i)-\eta_iT(y_i)-\log b(y_i)
```

the scalar canonical Hessian is:

```math
\nabla_\theta^2J_i(\theta)
=
a''(\eta_i)x_ix_i^T
```

and:

```math
a''(\eta_i)=\mathrm{Var}(T(Y_i)\mid x_i;\theta)
```

So canonical GLM curvature is variance-weighted feature geometry.

Convex-friendly does not mean every MLE is finite, unique, or numerically stable. Important caveats include:

* finite MLE may not exist under logistic or softmax perfect separation;
* rank deficiency in $`X`$ can make parameters non-identifiable;
* softmax parameters need an identifiability convention such as a reference class or equivalent constraint;
* boundary solutions can occur when observed statistics lie on the boundary of the achievable mean set;
* regularization is often added to stabilize estimation and select among weakly identified directions;
* noncanonical links can lose the clean canonical Hessian form.

## 9. Column-space meaning of the systematic component

In the canonical scalar model:

```math
\boldsymbol\eta=X\theta
```

The $`n`$ local natural parameters are constrained to lie in the column space of $`X`$. When $`p\ll n`$, this is a strong low-dimensional restriction: the model cannot choose an arbitrary natural parameter for each sample.

This restriction is statistical sharing. Every sample contributes to the same $`\theta`$, and the fitted $`\theta`$ can be reused on a new feature vector $`x_{\mathrm{new}}`$. The price is possible underfit when the true natural-parameter vector is not well approximated by the column-space constraint.

This is not:

```math
Y=X\theta
```

It is:

```math
\text{natural coordinate of }p(Y_i\mid x_i)
=
x_i^T\theta
```

The realized response remains random.

## 10. Gaussian regression sufficient statistics

For fixed-design Gaussian regression, treat $`X`$ as observed and fixed. The conditional model is:

```math
\mathbf y\mid X;\theta
\sim
\mathcal N(X\theta,\sigma^2I)
```

Assume first that $`\sigma^2`$ is known and the parameter of interest is only $`\theta`$. The conditional density is:

```math
p(\mathbf y\mid X;\theta)
=
(2\pi\sigma^2)^{-n/2}
\exp\left[
-\frac{1}{2\sigma^2}
(\mathbf y-X\theta)^T(\mathbf y-X\theta)
\right]
```

Expand:

```math
(\mathbf y-X\theta)^T(\mathbf y-X\theta)
=
\mathbf y^T\mathbf y
-
2\theta^TX^T\mathbf y
+
\theta^TX^TX\theta
```

Substitute this expansion into the density:

```math
p(\mathbf y\mid X;\theta)
=
(2\pi\sigma^2)^{-n/2}
\exp\left[
-\frac{1}{2\sigma^2}\mathbf y^T\mathbf y
\right]
\exp\left[
\frac{1}{\sigma^2}\theta^TX^T\mathbf y
-
\frac{1}{2\sigma^2}\theta^TX^TX\theta
\right]
```

This is a Fisher-Neyman factorization:

```math
p(\mathbf y\mid X;\theta)
=
h(\mathbf y;X)
g_\theta(S(\mathbf y;X),X)
```

with:

```math
h(\mathbf y;X)
=
(2\pi\sigma^2)^{-n/2}
\exp\left[
-\frac{1}{2\sigma^2}\mathbf y^T\mathbf y
\right]
```

and:

```math
S(\mathbf y;X)
=
X^T\mathbf y
=
\sum_i x_i y_i
```

The key point is conditional on $`X`$ and known $`\sigma^2`$: the term $`\mathbf y^T\mathbf y`$ still affects the numerical density, but it does not contain the unknown $`\theta`$. It therefore belongs to the base factor $`h(\mathbf y;X)`$ for inference about $`\theta`$.

The parameter-dependent part of the likelihood sees the response vector only through:

```math
\theta^TX^T\mathbf y
```

Thus, by the Fisher-Neyman factorization theorem, $`X^T\mathbf Y`$ is sufficient for $`\theta`$ in this fixed-design, known-variance Gaussian regression model.

This is the regression analogue of iid Gaussian mean sufficiency, but it is not the same statistic. In an iid shared-mean model:

```math
Y_i\sim\mathcal N(\mu,\sigma^2)
```

all samples share the same scalar coefficient for $`y_i`$, so the sufficient statistic is $`\sum_iY_i`$. In regression:

```math
\mu_i=x_i^T\theta
```

the same global $`\theta`$ is seen through different feature vectors $`x_i`$, so the sufficient statistic becomes the feature-weighted sum $`\sum_i x_iY_i=X^T\mathbf Y`$.

If $`\sigma^2`$ is unknown too, the parameter of interest is $`(\theta,\sigma^2)`$, and $`\mathbf y^T\mathbf y`$ can no longer be treated merely as a parameter-independent base factor. The conditional likelihood then depends on both:

```math
X^T\mathbf y
```

and:

```math
\mathbf y^T\mathbf y
```

This mirrors the ordinary Gaussian fact that unknown mean and variance require both first-moment and second-moment information.

## 11. Linearity is a design choice

CS229 uses:

```math
\eta_i=x_i^T\theta
```

because it gives interpretable additive effects, a compact shared parameter, and clean likelihood geometry for canonical links. It is not a theorem that the true data-generating mechanism must be linear on the natural-parameter scale.

The representation can be expanded while preserving the same observation model:

```math
\eta_i=\phi(x_i)^T\theta
```

or, in a more flexible conditional model:

```math
\eta_i=f_\theta(x_i)
```

The separation remains important. The representation model maps features to a distribution coordinate. The observation model maps that coordinate to $`p(Y_i\mid x_i)`$. A richer representation can reduce underfit, but it does not automatically fix the wrong response family, wrong variance structure, missing exposure, dependence, or calibration failure.

## 12. Canonical examples

Gaussian with CS229 fixed variance one:

```math
p(y;\mu)
=
\frac{1}{\sqrt{2\pi}}
\exp\left(
-\frac12y^2+\mu y-\frac12\mu^2
\right)
```

```math
\log b(y)=-\frac12y^2-\frac12\log(2\pi),
\qquad
T(y)=y,
\qquad
\eta=r(\mu)=\mu,
\qquad
a(\eta)=\frac{\eta^2}{2}
```

The term $`-\frac12y^2`$ is not part of $`a(\eta)`$ in the mean-only variance-one family, because it does not depend on the unknown mean. The mean map is:

```math
m(\eta)=a'(\eta)=\eta
```

Since $`T(Y)=Y`$, this is also the scalar response mean:

```math
\mu=\mathbb E[Y]=m(\eta)=\eta
```

Thus the canonical link is identity:

```math
\ell_c(\mu)=m^{-1}(\mu)=\mu
```

and the canonical GLM gives:

```math
\eta_i=x_i^T\theta,
\qquad
h_\theta(x_i)=\mu_i=x_i^T\theta
```

With known non-unit variance $`\sigma^2`$, the same calculation gives:

```math
\eta=\frac{\mu}{\sigma^2},
\qquad
a(\eta)=\frac{\sigma^2\eta^2}{2},
\qquad
m(\eta)=\sigma^2\eta=\mu,
\qquad
\ell_c(\mu)=\frac{\mu}{\sigma^2}
```

Bernoulli logistic regression:

```math
p(y;p)
=
\exp\left[
y\log\frac{p}{1-p}
+
\log(1-p)
\right]
```

```math
T(y)=y,
\qquad
\eta=r(p)=\log\frac{p}{1-p},
\qquad
a(\eta)=\log(1+e^\eta)
```

The mean map is:

```math
m(\eta)=a'(\eta)=\frac{e^\eta}{1+e^\eta}
```

Bernoulli has the distribution-specific identity $`\mu=\mathbb E[Y]=p`$, so:

```math
\ell_c(\mu)=m^{-1}(\mu)=\log\frac{\mu}{1-\mu}
```

Canonical logistic regression sets:

```math
\eta_i=x_i^T\theta,
\qquad
h_\theta(x_i)=p_i=\frac{1}{1+\exp(-x_i^T\theta)}
```

Poisson regression:

```math
p(y;\lambda)
=
\frac{1}{y!}
\exp(y\log\lambda-\lambda)
```

```math
T(y)=y,
\qquad
\eta=r(\lambda)=\log\lambda,
\qquad
a(\eta)=e^\eta,
\qquad
b(y)=\frac{1}{y!}
```

The mean map is:

```math
m(\eta)=a'(\eta)=e^\eta
```

Poisson has the distribution-specific identity $`\mu=\mathbb E[Y]=\lambda`$, so:

```math
\ell_c(\mu)=m^{-1}(\mu)=\log\mu
```

Canonical Poisson regression sets:

```math
\eta_i=x_i^T\theta,
\qquad
h_\theta(x_i)=\lambda_i=\exp(x_i^T\theta)
```

Categorical / softmax with class $`K`$ as reference:

```math
\eta_k=r(\phi)_k=\log\frac{\phi_k}{\phi_K},
\quad
k=1,\ldots,K-1
```

and:

```math
a(\eta)=\log\left(1+\sum_{j=1}^{K-1}e^{\eta_j}\right)
```

The expectation parameter is the class-probability vector on the reference statistic:

```math
m_k(\eta)
=
\frac{\partial a}{\partial\eta_k}
=
\frac{e^{\eta_k}}{1+\sum_{j=1}^{K-1}e^{\eta_j}}
=
\phi_k
```

The canonical link from probabilities to natural coordinates is reference logits:

```math
\ell_c(\phi)_k
=
\log\frac{\phi_k}{\phi_K}
```

The inverse link / response map is softmax. With class-specific linear scores:

```math
\eta_k(x)=\theta_k^Tx,
\quad
k<K,
```

we get:

```math
P(Y=k\mid x;\Theta)
=
\frac{\exp(\theta_k^Tx)}
{1+\sum_{j=1}^{K-1}\exp(\theta_j^Tx)}
```

The same linear score has different statistical meanings because it is placed on different distribution scales: Gaussian mean coordinate, Bernoulli log-odds, Poisson log-rate, or categorical reference log-odds.

## 13. Conditional residual interpretation

For scalar response GLMs, define the conditional mean residual:

```math
\epsilon_i=Y_i-\mu_i
```

where:

```math
\mu_i=\mathbb E[Y_i\mid x_i;\theta]
```

Then:

```math
Y_i=\mu_i+\epsilon_i
```

and:

```math
\mathbb E[\epsilon_i\mid x_i]=0
```

This is a conditional-mean decomposition. It does not impose Gaussianity, independence, or constant variance. Gaussian identity-link regression is the special case where $`\mu_i=x_i^T\theta`$ and the model may be equivalently written as linear signal plus conditional Gaussian noise. Bernoulli and Poisson GLMs instead have distribution-specific residual supports and variance structures.

## 14. Summary

A GLM is a disciplined way to learn a conditional distribution. The random component selects the response family. The ordinary parameter names the familiar local distribution member. The natural parameter is the canonical local coordinate. The systematic component maps features and the global trainable parameter into a score. The link decides when that score is also the natural parameter. Likelihood training then matches observed sufficient statistics to model expectations through feature-weighted score equations, and residual diagnostics interpret deviations around the conditional mean.
