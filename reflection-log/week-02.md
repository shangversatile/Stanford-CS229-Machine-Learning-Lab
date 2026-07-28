# Week 02 Reflection

## Lecture 2 Completed

Lecture 2: Supervised Learning Setup and Linear Regression 已完成。主笔记现在覆盖 supervised learning setup、linear hypothesis、least-squares objective、LMS / gradient descent、batch GD 与 SGD、normal equation、projection geometry，以及 Gaussian noise 下的 maximum likelihood interpretation。

本周还补充了两个容易被公式压缩掉的基础推导：

* gradient 为什么与 contour tangent 垂直，包括 curve-based proof 和 surface-normal projection proof；
* 为什么 $(AB)^T=B^TA^T$，包括 entry-wise、column-space、row-space 和 linear-map 解释。

同时生成了六张 matplotlib educational figures，用于连接 fitted residual、quadratic loss、gradient geometry、surface normal、optimization path 和 normal-equation projection。

## My Current Understanding

Linear regression 是第一个 supervised learning algorithm，也是第一个完整的 machine-learning paradigm：

```text
data -> hypothesis -> loss -> optimization -> parameters -> prediction
```

Loss minimization 把理论模型与工程计算连接起来。Squared loss 把 prediction error 转换为非负、可聚合、可微的 objective；它避免正负误差抵消，并对较大 deviation 施加更强惩罚。

Gradient descent 是 numerical method：它通过 local first-order information 逐步逼近 optimum。Normal equation 是 analytic method：它通过 stationary condition 和 linear algebra 得到 ordinary least-squares solution。

Residual 定义为 $h_\theta(x)-y$ 或 $y-h_\theta(x)$ 不会改变 squared objective 和 optimum，但会改变一阶 residual 与 update 的 sign convention。因此推导和实现必须固定同一种 convention。

Learning rate 不应被理解为孤立的超参数。它与 objective curvature、feature scaling、conditioning、gradient noise 和 batch construction 共同决定有效 update。

## Remaining Risks

* 在 convex least-squares problem 中混淆 local extrema 与 global minimum；
* 误以为 linear regression 必须对 raw input 呈线性，而不是对 parameters 线性；
* 在未检查 outliers、heavy tails 或 heteroscedasticity 时直接使用 squared loss；
* 推导矩阵公式时不写 dimension checks；
* 在理解 normal-equation projection geometry 之前开始实现；
* 把 closed-form optimization success 误当作 model specification 或 generalization success。

## Next Action

下一步从零实现 Linear Regression，包括：

* gradient descent；
* normal equation / stable linear solve；
* unit tests；
* synthetic data parameter recovery；
* analytic-versus-numerical gradient check；
* Gaussian noise、outlier、heavy-tailed noise、collinearity 和 distribution-shift stress tests。

## Lecture 3 Completed

Lecture 3 was completed. The note now connects global regression, local regression, probability, logistic classification, and Newton method into one modeling sequence rather than treating them as isolated algorithms.

The questions about LWR, $\tau$, non-parametric learning, high-dimensional failure, probability vs likelihood, CLT, logistic vs linear regression, and Newton convergence were integrated into the main note and standalone derivations.

## Current Understanding

* LWR is query-dependent local fitting: each query point creates its own weighted objective and local $\theta(x)$.
* The Gaussian kernel in LWR is a locality weighting function, not necessarily a global data distribution assumption.
* Logistic regression is not linear regression used for classification; it is Bernoulli MLE with a sigmoid link.
* Similar gradient forms do not imply identical models, because the likelihood assumptions and hypothesis functions differ.
* Newton method uses second-order curvature information, which explains both its speed near an optimum and its sensitivity to conditioning.

## Next Action

The next learning target is Lecture 4 or the next official segment on Newton method, GLM, or implementation depending on the course sequence.

Before moving forward, implement logistic regression only after the gradient and Hessian derivations are reviewed.

## PS1 Gate Initialized

After completing Lecture 3, the learning flow now pauses at PS1 Supervised Learning Gate. Lecture 4 should be completed next, but later topics should not be started until PS1-related derivation and implementation tasks are mapped and attempted.

## Lecture 3 Perspective Update

Lecture 3 was updated to separate the machine learning perspective from the probabilistic modeling perspective. The early CS229 derivations are mainly frequentist: parameters are fixed but unknown, data are random, and MLE estimates the parameter. Bayesian modeling was added as a contrast through priors, posteriors, MAP, and posterior predictive distributions.

Formula rendering was audited with the stricter Markdown math checker and kept in repository-compatible form.

## Lecture 4 Conceptual Corrections

Lecture 4 completion corrected several modeling-level misunderstandings:

* Multiclass classification corresponds to categorical or multinomial modeling, not Poisson. Poisson is for count data with a rate interpretation.
* Perceptron and logistic regression may share the same linear score $\theta^Tx$, but Perceptron is mistake-driven while logistic regression is likelihood-driven and probabilistic.
* $T(y)$ is a sufficient statistic, not automatically equal to $y$. For example, unknown-variance Gaussian models need statistics such as $y$ and $y^2$, and multinomial models use indicator vectors.
* The log-partition function $a(\eta)$ is the engine of exponential-family moments: its gradient gives the mean of $T(Y)$ and its Hessian gives the covariance.
* In a GLM, the response function maps the natural linear predictor into the conditional mean $\mathbb E[T(Y)\mid x;\theta]$; it is not the parameter estimate itself.
* Softmax is a joint multinomial model with coupled probabilities that sum to one, not a set of independent one-vs-rest logistic regressions.

The main reliability lesson is that model choice starts from response semantics and support, then checks variance behavior, link choice, identifiability, calibration, and distribution shift.

## Lecture 4 Distribution Map Refinement

Lecture 4 was refined so the GLM logic now starts from response semantics rather than from a preferred loss or activation function. The updated note separates official CS229 core derivations from extension-level distribution intuition, makes the conceptual and mathematical interludes easier to navigate, and connects support, conditional distribution, exponential-family form, response function, NLL, and diagnostics into one modeling workflow.

The key understanding target is now: given a new supervised-learning problem, first define what $Y$ can be and how it is generated, then justify the conditional distribution before deriving the response function and loss.

## Lecture 4 Exponential-Family Origin Update

The Lecture 4 notes now make explicit why exponential family and GLMs arise naturally rather than appearing as a memorized formula. The key connection is that ordinary linear regression is too narrow for binary, count, multiclass, positive, probability-valued, and heteroscedastic responses, while GLMs preserve a linear predictor by placing linearity on the natural parameter.

The deeper understanding target is now: sufficient statistics explain likelihood compression, maximum entropy explains the exponential tilt, and the log-partition function explains normalization, moment generation, Fisher information, and convexity. These properties are consequences of normalization, sufficiency, and convex duality; they still require reliability diagnostics for support, link choice, misspecification, separation, calibration, and shift.
