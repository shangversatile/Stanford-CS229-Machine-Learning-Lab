# Softmax GLM and Cross-Entropy

## 1. One-Hot Sufficient Statistic

For a $K$-class categorical response, define:

```math
(T(y))_k=\mathbf1\{y=k\}
```

For a reference-class parameterization, use $k=1,\ldots,K-1$ and treat class $K$ as the baseline. In the symmetric softmax implementation, it is common to keep all $K$ class scores and handle identifiability by regularization or by recognizing shift invariance.

Expectation of the one-hot sufficient statistic gives class probabilities:

```math
\mathbb E[(T(Y))_k]=P(Y=k)
```

## 2. Reference Class Derivation

Categorical probability mass:

```math
p(y;\phi)
=
\prod_{k=1}^{K}\phi_k^{\mathbf1\{y=k\}}
```

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

The log-partition function is:

```math
a(\eta)=\log\left(1+\sum_{j=1}^{K-1}e^{\eta_j}\right)
```

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
\phi_K
\sum_{k=1}^{K-1}e^{\eta_k}\phi_K
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

