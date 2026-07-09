# Locally Weighted Regression

## 1. Motivation: Why Global Linear Regression Can Fail

Ordinary linear regression assumes one parameter vector $\theta$ can describe the whole input space:

$$h_\theta(x)=\theta^Tx.$$

This is efficient and interpretable, but it can fail when the conditional mean $\mathbb{E}[y|x]$ changes curvature across the domain. A single global line may be too biased: it averages incompatible local trends into one compromise.

Locally weighted regression keeps the linear model locally, but removes the requirement that the same $\theta$ work everywhere. Around each query point $x$, it fits a local linear approximation using nearby training examples more heavily than faraway ones.

## 2. Weighted Least Squares Objective

For a fixed query point $x$, define the local objective:

$$J_x(\theta)=\frac{1}{2}\sum_{i=1}^{m}w^{(i)}(x)\left(y^{(i)}-\theta^Tx^{(i)}\right)^2.$$

The subscript $x$ means the objective depends on the prediction location. The weights satisfy:

$$w^{(i)}(x)\geq0.$$

If $w^{(i)}(x)$ is large, example $i$ strongly influences the local fit. If $w^{(i)}(x)$ is close to $0$, example $i$ contributes little to the local objective.

In matrix form, let $W_x=\mathrm{diag}(w^{(1)}(x),\ldots,w^{(m)}(x))$. Then:

$$J_x(\theta)=\frac{1}{2}(y-X\theta)^TW_x(y-X\theta).$$

The stationary condition is:

$$X^TW_xX\theta=X^TW_xy.$$

If $X^TW_xX$ is invertible, the local weighted least-squares solution is:

$$\theta(x)=(X^TW_xX)^{-1}X^TW_xy.$$

The notation $\theta(x)$ emphasizes that this parameter vector is query-dependent.

## 3. Gaussian-kernel Weight

A common weighting rule is:

$$w^{(i)}(x)=\exp\left(-\frac{\left|x^{(i)}-x\right|_2^2}{2\tau^2}\right).$$

This is Gaussian-kernel-like because it decays exponentially with squared distance. It is a locality weighting function centered at the query point $x$. It is not the same as assuming that all data are globally Gaussian distributed.

At $x^{(i)}=x$, the weight is:

$$w^{(i)}(x)=1.$$

As $\left|x^{(i)}-x\right|_2$ grows, the weight decreases smoothly toward $0$.

## 4. Role of $\tau$

The parameter $\tau>0$ is the bandwidth or locality parameter. It controls how quickly weights decay with distance.

For small $\tau$, the denominator $2\tau^2$ is small, so even moderate distances make the exponent very negative:

$$\tau\downarrow\quad\Rightarrow\quad w^{(i)}(x)\ \text{decays faster with distance}.$$

For large $\tau$, the denominator is larger, so weights decay more slowly:

$$\tau\uparrow\quad\Rightarrow\quad w^{(i)}(x)\ \text{decays slower with distance}.$$

Small $\tau$ produces highly local and flexible fits, but may be unstable. Large $\tau$ averages over broader neighborhoods and approaches the behavior of global linear regression.

## 5. Prediction-time Fitting and Why It Is Non-parametric

LWR does not train once and then discard the data. Instead, prediction at a new point $x$ requires:

1. compute weights $w^{(i)}(x)$ for all training examples;
2. solve the weighted least-squares problem for $\theta(x)$;
3. output $\hat{y}=\theta(x)^Tx$.

The effective model is therefore tied to retained training data. As $m$ grows, the memory requirement and prediction-time computation can grow too. This is why LWR is considered non-parametric in the statistical learning sense, even though each local fit has parameters.

## 6. Bias-variance Interpretation

The bandwidth $\tau$ controls the bias-variance tradeoff.

With small $\tau$, the model focuses on a narrow neighborhood. This can reduce bias if the true function changes rapidly, but it increases variance because the fit depends on few effective samples:

$$\tau\ \text{small}\quad\Rightarrow\quad\text{low bias, high variance}.$$

With large $\tau$, many points receive meaningful weight. This stabilizes the estimate but can smooth away real local structure:

$$\tau\ \text{large}\quad\Rightarrow\quad\text{high bias, low variance}.$$

Good bandwidth selection depends on sample density, noise level, smoothness of the true function, and the distance metric.

## 7. High-dimensional Failure and Curse of Dimensionality

In high-dimensional spaces, local methods face several linked problems. Volume grows rapidly, so a fixed sample size gives sparse coverage. Distances also concentrate, meaning the nearest and farthest points can have similar distances from a query point. Then the kernel weights become less informative.

If all distances look similar, then:

$$w^{(1)}(x)\approx w^{(2)}(x)\approx\cdots\approx w^{(m)}(x).$$

The local method loses its intended locality. If $\tau$ is made very small to recover locality, there may be too few effective samples and the weighted normal equation may become ill-conditioned.

## 8. Reliability Notes

Reliable use of LWR requires checking:

* bandwidth sensitivity;
* feature scaling and distance metric choice;
* local sample density;
* numerical conditioning of $X^TW_xX$;
* outlier influence inside local neighborhoods;
* prediction-time cost;
* high-dimensional distance concentration.

LWR is valuable because it exposes the limitation of one global linear model, but it is not automatically more reliable. It trades global misspecification risk for local data sufficiency and bandwidth-selection risk.
