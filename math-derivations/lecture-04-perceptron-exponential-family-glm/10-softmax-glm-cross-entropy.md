# Softmax GLM and Cross-Entropy

Cross-link: see [Lecture 4 Section 14: Multinomial Exponential-Family Form](../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#14-multinomial-exponential-family-form), [Lecture 4 Conceptual Interlude E: What Does a Response Value Mean?](../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-e-what-does-a-response-value-mean), and [Response Spaces, Measures, and Expectations](09-response-spaces-measures-and-expectations.md).

## 1. One-Hot Canonical Statistic

For a $K$-class categorical random variable:

```math
Y\in\{1,\ldots,K\}
```

define the one-hot statistic:

```math
T(Y)
=
\begin{bmatrix}
\mathbf1\{Y=1\}\\
\vdots\\
\mathbf1\{Y=K\}
\end{bmatrix}
```

Here $Y$ is the categorical random variable, $y$ is one realized value, $T(Y)$ is a random vector, and $T(y)$ is fixed after $y$ is observed:

```math
(T(y))_k=\mathbf1\{y=k\}
```

Therefore the $i$th component before realization is:

```math
(T(Y))_i=\mathbf1\{Y=i\}
```

This component is itself a $0$-$1$ random variable. Its expectation is not an average over training-sample indices, not an average over the $K$ coordinates of $T(Y)$, and not an average of an already observed fixed vector $T(y)$. It averages over all possible category values of the same categorical random variable $Y$, weighted by the probabilities assigned by the current model.

The general identity is:

```math
\mathbb E[\mathbf1\{A\}]=P(A)
```

So here this is just:

```math
\mathbb E[\mathbf1\{Y=i\}]=P(Y=i)
```

Here $(T(j))_i$ means the $i$th component of the one-hot vector that would be obtained if the realized category were $j$:

```math
(T(j))_i=\mathbf1\{j=i\}
```

Therefore the discrete expectation is:

```math
\mathbb E[(T(Y))_i]
=
\sum_{j=1}^{K}(T(j))_iP(Y=j)
=
\sum_{j=1}^{K}\mathbf1\{j=i\}P(Y=j)
=
P(Y=i)
=
\phi_i
```

The sum collapses because $(T(j))_i=1$ only when $j=i$, and $(T(j))_i=0$ for all other categories.

For a three-class example with $P(Y=1)=0.2$, $P(Y=2)=0.5$, and $P(Y=3)=0.3$:

```math
\mathbb E[(T(Y))_2]=\mathbb E[\mathbf1\{Y=2\}]
```

```math
\mathbb E[(T(Y))_2]=0\cdot0.2+1\cdot0.5+0\cdot0.3=0.5
```

Across repeated draws from this categorical distribution, $(T(Y))_2$ is $1$ about $50\%$ of the time and $0$ otherwise, so its expectation is $0.5$.

This example is only for intuition; the mathematical reason is still the indicator expectation identity above.

For a reference-class parameterization, keep $k=1,\ldots,K-1$ and treat class $K$ as the baseline. In the symmetric softmax implementation, it is common to keep all $K$ class scores and handle identifiability by regularization or by recognizing shift invariance.

For iid categorical samples, aggregating these one-observation readouts gives class counts. The class-count vector is the sample-level sufficient statistic for class probabilities; the one-hot vector is the per-observation canonical statistic.

Expectation of the one-hot canonical statistic gives class probabilities:

```math
\mathbb E[(T(Y))_k]=P(Y=k)=\phi_k
```

Equivalently:

```math
\mathbb E[T(Y)]
=
\begin{bmatrix}
\phi_1\\
\vdots\\
\phi_K
\end{bmatrix}
=
\phi
```

For one realized label $y=i$, $T(y)$ is a fixed one-hot vector. For the random label $Y$ before realization, $T(Y)$ is a random one-hot vector, and its expectation is the probability vector:

```text
Y -> T(Y) -> E[T(Y)]
Y=y -> T(y)
```

Official GLM derivations often first write the shorthand:

```math
\mathbb E[(T(Y))_i]=P(Y=i)=\phi_i
```

But in the actual supervised GLM, each input $x$ defines its own conditional categorical distribution. Therefore the stricter statement is:

```math
\mathbb E[(T(Y))_i\mid X=x;\Theta]
=
P(Y=i\mid X=x;\Theta)
=
\phi_i(x)
```

In vector form:

```math
\mathbb E[T(Y)\mid X=x;\Theta]=\phi(x)
```

These are different levels. For the categorical family, $\mathbb E[T(Y)]=\phi$, so the log-partition gradient $\nabla a(\eta)=\mathbb E[T(Y)]$ gives the class-probability / mean-parameter vector. In the GLM, softmax maps the linear/natural parameters associated with $x$ to the conditional mean of the one-hot sufficient statistic, which is exactly the class-probability vector.

For the response-space reason this is $`\mathbb E[T(Y)]`$ rather than an intrinsic scalar-coded $`\mathbb E[Y]`$, see [Response Spaces, Measures, and Expectations](09-response-spaces-measures-and-expectations.md).

## 2. Reference Class Derivation

Categorical probability mass:

```math
p(y;\phi)
=
\prod_{k=1}^{K}\phi_k^{\mathbf1\{y=k\}}
```

For a one-trial categorical outcome, exactly one class indicator is $`1`$:

```math
\sum_{k=1}^{K}\mathbf1\{y=k\}=1
```

If the reference-class statistic keeps only:

```math
T_k(y)=\mathbf1\{y=k\},
\quad k=1,\ldots,K-1
```

then the missing class indicator is determined by the others:

```math
\mathbf1\{y=K\}
=
1-\sum_{k=1}^{K-1}\mathbf1\{y=k\}
=
1-\sum_{k=1}^{K-1}T_k(y)
```

This identity is the only reason the exponent of $`\phi_K`$ becomes:

```math
1-\sum_{k=1}^{K-1}T_k(y)
```

The PMF can therefore be written as:

```math
p(y;\phi)
=
\left(
\prod_{k=1}^{K-1}\phi_k^{T_k(y)}
\right)
\phi_K^{1-\sum_{k=1}^{K-1}T_k(y)}
```

It cannot be replaced by $`1-T_j(y)`$ for an arbitrary class $`j`$, because:

```math
1-T_j(y)=\mathbf1\{y\neq j\}
```

That is true for every non-$`j`$ class, not only for class $`K`$. For $`K=4`$, if $`T_1(y)=\mathbf1\{y=1\}`$, then $`1-T_1(y)`$ has values:

```text
y = 1 -> 0
y = 2 -> 1
y = 3 -> 1
y = 4 -> 1
```

so it is not $`\mathbf1\{y=4\}`$. By contrast:

```math
1-T_1(y)-T_2(y)-T_3(y)
```

has values $`0,0,0,1`$ across $`y=1,2,3,4`$, exactly matching $`\mathbf1\{y=4\}`$.

The full $`K`$-dimensional one-hot vector:

```math
\begin{bmatrix}
\mathbf1\{Y=1\}\\
\vdots\\
\mathbf1\{Y=K\}
\end{bmatrix}
```

always satisfies:

```math
\sum_{k=1}^{K}T_k(Y)=1
```

so its coordinates have one deterministic affine constraint and only $`K-1`$ degrees of freedom. The reference-class representation keeps:

```math
T(Y)
=
\begin{bmatrix}
\mathbf1\{Y=1\}\\
\vdots\\
\mathbf1\{Y=K-1\}
\end{bmatrix}
```

In this coordinate choice:

```text
Y = 1     -> e_1
...
Y = K - 1 -> e_{K-1}
Y = K     -> 0
```

Choosing class $`K`$ as the zero/reference vector is a coordinate choice, not an intrinsic privilege of class $`K`$. Any class could be selected as the reference class, just as any nonredundant basis can represent the same categorical family.

Use class $K$ as reference:

```math
p(y;\phi)
=
\phi_K
\prod_{k=1}^{K-1}
\left(\frac{\phi_k}{\phi_K}\right)^{\mathbf1\{y=k\}}
```

Taking exponential form:

```math
p(y;\phi)
=
\exp\left(
\sum_{k=1}^{K-1}\mathbf1\{y=k\}
\log\frac{\phi_k}{\phi_K}
+
\log\phi_K
\right)
```

Thus:

```math
\eta_k=\log\frac{\phi_k}{\phi_K}
```

and:

```math
T_k(y)=\mathbf1\{y=k\}
```

Each $`\eta_k`$ is the log-odds coordinate of class $`k`$ relative to the chosen reference class $`K`$.

The log-partition function is:

```math
a(\eta)=\log\left(1+\sum_{j=1}^{K-1}e^{\eta_j}\right)
```

Its gradient is the expectation map for the reference-class one-hot statistic:

```math
m_k(\eta)
=
\frac{\partial a}{\partial\eta_k}
=
\frac{e^{\eta_k}}{1+\sum_{j=1}^{K-1}e^{\eta_j}}
=
\mathbb E[T_k(Y)]
=
\phi_k
```

Thus reference logits are the canonical link:

```math
\ell_c(\phi)_k
=
\log\frac{\phi_k}{\phi_K}
```

and softmax is the inverse link / response map from natural scores to class probabilities.

## 3. Softmax Probabilities

From:

```math
e^{\eta_k}=\frac{\phi_k}{\phi_K}
```

we have:

```math
\phi_k=e^{\eta_k}\phi_K
```

Use normalization:

```math
\phi_K+\sum_{k=1}^{K-1}\phi_k=1
```

Substitute:

```math
\phi_K\left(1+\sum_{k=1}^{K-1}e^{\eta_k}\right)
=
1
```

Thus:

```math
\phi_K
=
\frac{1}{1+\sum_{j=1}^{K-1}e^{\eta_j}}
```

For $k<K$:

```math
\phi_k
=
\frac{e^{\eta_k}}
{1+\sum_{j=1}^{K-1}e^{\eta_j}}
```

If we define $\eta_K=0$, this becomes one formula:

```math
\phi_k
=
\frac{e^{\eta_k}}
{\sum_{j=1}^{K}e^{\eta_j}}
```

With class-specific linear predictors $\eta_k=\theta_k^Tx$:

```math
p(y=k\mid x;\Theta)
=
\frac{\exp(\theta_k^Tx)}
{\sum_{j=1}^{K}\exp(\theta_j^Tx)}
```

## 4. Binary Reduction

For $K=2$:

```math
p(y=1\mid x)
=
\frac{\exp(\theta_1^Tx)}
{\exp(\theta_1^Tx)+\exp(\theta_2^Tx)}
```

Divide by $\exp(\theta_1^Tx)$:

```math
p(y=1\mid x)
=
\frac{1}
{1+\exp(\theta_2^Tx-\theta_1^Tx)}
```

Therefore:

```math
p(y=1\mid x)
=
\frac{1}
{1+\exp\left(-(\theta_1-\theta_2)^Tx\right)}
```

This is the sigmoid form with effective parameter $\theta_1-\theta_2$.

## 5. NLL and Cross-Entropy

For sample $i$, define:

```math
t_{ik}=\mathbf1\{y^{(i)}=k\}
```

and:

```math
p_{ik}=p(y=k\mid x^{(i)};\Theta)
```

Likelihood:

```math
L(\Theta)
=
\prod_{i=1}^{m}
\prod_{k=1}^{K}
p_{ik}^{t_{ik}}
```

Log-likelihood:

```math
\ell(\Theta)
=
\sum_{i=1}^{m}
\sum_{k=1}^{K}
t_{ik}\log p_{ik}
```

NLL:

```math
J(\Theta)
=
-\sum_{i=1}^{m}
\sum_{k=1}^{K}
t_{ik}\log p_{ik}
```

This is multiclass cross-entropy.

## 6. Gradient

For one sample:

```math
\log p_{ik}
=
\theta_k^Tx^{(i)}
-
\log\sum_{j=1}^{K}\exp(\theta_j^Tx^{(i)})
```

So:

```math
J_i
=
-\sum_{k=1}^{K}t_{ik}\theta_k^Tx^{(i)}
+
\log\sum_{j=1}^{K}\exp(\theta_j^Tx^{(i)})
```

because $\sum_k t_{ik}=1$.

Differentiate with respect to $\theta_r$:

```math
\nabla_{\theta_r}J_i
=
-t_{ir}x^{(i)}
+
\frac{\exp(\theta_r^Tx^{(i)})}
{\sum_{j=1}^{K}\exp(\theta_j^Tx^{(i)})}
x^{(i)}
```

Therefore:

```math
\nabla_{\theta_r}J_i=(p_{ir}-t_{ir})x^{(i)}
```

Summing over samples:

```math
\nabla_{\theta_k}J
=
\sum_{i=1}^{m}(p_{ik}-t_{ik})x^{(i)}
```

## 7. Parameter Coupling

The derivative for $\theta_k$ depends on $p_{ik}$, and $p_{ik}$ depends on all class scores through the denominator. Thus all class parameters are jointly trained.

This coupling is exactly what independent one-vs-rest logistic regression lacks. In one-vs-rest, each binary classifier estimates:

```math
q_k(x)=P(Y=k\ \mathrm{versus}\ Y\neq k\mid x)
```

The values $q_1(x),\ldots,q_K(x)$ need not sum to one.

## 8. Identifiability

Softmax probabilities are invariant to adding a common vector $v$ to all class parameters:

```math
\theta_k'=\theta_k+v
```

Then:

```math
\exp(\theta_k'^Tx)
=
\exp(\theta_k^Tx+v^Tx)
=
\exp(v^Tx)\exp(\theta_k^Tx)
```

The common factor cancels in numerator and denominator. Therefore only relative scores matter.

To resolve identifiability, one may:

* fix a reference class parameter such as $\theta_K=0$；
* impose a sum-to-zero constraint；
* use regularization and interpret parameters relatively.

## 9. One-vs-Rest Comparison

Softmax:

```math
\sum_{k=1}^{K}p(y=k\mid x;\Theta)=1
```

Independent one-vs-rest logistic models generally satisfy:

```math
\sum_{k=1}^{K}q_k(x)\neq1
```

For example, three binary classifiers may output $0.7,0.6,0.5$ for one input. These are valid binary probabilities for separate tasks, but they are not a valid categorical distribution over three mutually exclusive classes.

