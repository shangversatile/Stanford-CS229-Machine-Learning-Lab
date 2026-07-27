# Lecture 4 Original Exercises: Perceptron, Exponential Family, GLM, and Softmax

These exercises are original practice prompts for the PS1 supervised-learning gate. They are not copied from official problem statements and intentionally contain no solutions.

## Exercise 1: Perceptron Signed-Score Update

Let $y\in\{-1,+1\}$ and let the Perceptron prediction be:

```math
\hat y=\mathrm{sign}(\theta^Tx)
```

Assume the sample $(x,y)$ is misclassified and the update is:

```math
\theta_{\mathrm{new}}=\theta+\alpha yx
```

Prove:

```math
y\theta_{\mathrm{new}}^Tx
=
y\theta^Tx+\alpha\|x\|_2^2
```

Then interpret the update geometrically in terms of the boundary normal vector and boundary rotation.

## Exercise 2: Newton on Quadratic Least Squares

Consider:

```math
J(\theta)=\frac12\|X\theta-y\|_2^2
```

Derive the gradient and Hessian. Show that Newton method reaches the least-squares optimum in one step when the required rank assumptions hold. State those assumptions explicitly.

## Exercise 3: Bernoulli Exponential-Family Form

Starting from:

```math
p(y;\phi)=\phi^y(1-\phi)^{1-y}
```

identify $\eta$, $T(y)$, $a(\eta)$, and $b(y)$ without skipping algebra. Then derive the sigmoid response function.

## Exercise 4: Log-Partition Moments

Let:

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

Prove:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

and:

```math
\nabla^2a(\eta)=\mathrm{Cov}_\eta(T(Y))
```

State the regularity condition needed for the proof.

## Exercise 5: Concavity and Convexity

For iid observations from an exponential family, write the log-likelihood in terms of $\sum_iT(y^{(i)})$. Derive the Hessian of the log-likelihood and the Hessian of the negative log likelihood. Explain why the MLE estimate itself is not described as concave or convex.

## Exercise 6: Distribution Selection

For each response type below, choose one or more candidate distributions and justify your assumptions:

| Response description | Candidate distribution to justify |
| -------------------- | --------------------------------- |
| house price residual after feature adjustment | |
| whether an email is spam | |
| one of five mutually exclusive product categories | |
| number of support tickets per day | |
| time until a device fails | |
| a probability estimated from repeated trials | |
| a vector of topic proportions | |

For at least two rows, describe a plausible misspecification risk.

## Exercise 7: Poisson GLM

Starting from:

```math
p(y;\lambda)=\frac{\lambda^ye^{-\lambda}}{y!}
```

derive the exponential-family form, identify $\eta$, $T(y)$, $a(\eta)$, and $b(y)$, and derive:

```math
\mathbb E[Y\mid x;\theta]=e^{\theta^Tx}
```

Then write the negative log likelihood up to constants independent of $\theta$.

## Exercise 8: Multinomial to Softmax

Use a reference class for a $K$-class categorical response. Define one-hot sufficient statistics for classes $1$ through $K-1$, derive the natural parameters, and recover the normalized class probabilities.

## Exercise 9: Softmax Cross-Entropy Gradient

Let:

```math
p_{ik}
=
\frac{\exp(\theta_k^Tx^{(i)})}
{\sum_{j=1}^{K}\exp(\theta_j^Tx^{(i)})}
```

and:

```math
J(\Theta)
=
-\sum_{i=1}^{m}
\sum_{k=1}^{K}
t_{ik}\log p_{ik}
```

Derive:

```math
\nabla_{\theta_k}J
=
\sum_{i=1}^{m}
(p_{ik}-t_{ik})x^{(i)}
```

Do not skip the derivative of the normalization term.

## Exercise 10: One-vs-Rest vs Softmax

Construct a concrete three-class example where independently trained one-vs-rest logistic classifiers output three probabilities that do not sum to one. Explain why this violates categorical probability semantics but does not contradict the validity of each binary classifier considered separately.

## Completion Checklist

* [ ] I derived the Perceptron signed-score improvement without skipping the $y^2=1$ step.
* [ ] I stated the least-squares Newton rank assumptions.
* [ ] I derived Bernoulli sigmoid from log odds.
* [ ] I proved the log-partition mean and covariance identities.
* [ ] I separated objective convexity from estimator properties.
* [ ] I corrected the multiclass versus Poisson distinction.
* [ ] I derived the Poisson response and NLL.
* [ ] I derived softmax from a reference-class parameterization.
* [ ] I computed the softmax gradient including the denominator derivative.
* [ ] I explained why one-vs-rest probabilities need not form a categorical distribution.

## Private Scratch-Work Reminder

Keep rough algebra, false starts, and self-checks in private scratch work or a local untracked notebook. Commit only the cleaned public-safe derivations and summaries.
