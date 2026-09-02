# Module 03: Logistic Regression as Regularized Risk

返回 [MIT 9.520 / 6.860 supplement index](../README.md)。

## Source Metadata / 来源元数据

主课程来源：

MIT / CBMM Learning Hub, 9.520 / 6.860, *Statistical Learning Theory and Applications*。

当前选用：

* Class 05: Logistic Regression and Support Vector Machines。
* Class 05 slides: `Class05_LogisticsSVM.pdf`。
* Class 02 slides 中的 logistic loss / ERM notation。
* L. Rosasco, T. Poggio, *Machine Learning: a Regularization Approach*, Chapter 1 and convex optimization appendix。

当前只使用 logistic regression、regularized empirical risk 和 convex optimization 子集。Class 05 的 SVM portion deferred until CS229 L6-L7。

CS229 cross-link：[Lecture 3 logistic regression](../../../lecture-notes/lecture-03-locally-weighted-logistic-regression/note.md#11-logistic-regression-hypothesis)、[logistic gradient derivation](../../../math-derivations/lecture-03-locally-weighted-logistic-regression/02-logistic-regression-gradient-newton.md)、[Lecture 4 GLM derivation](../../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md)。

## Detailed Notes / 深入笔记

| File | 内容 |
| --- | --- |
| [derivations/01-logistic-likelihood-and-risk.md](derivations/01-logistic-likelihood-and-risk.md) | CS229 Bernoulli likelihood 与 MIT empirical logistic risk 的等价、label convention 转换、regularized logistic 的三种读法 |
| [derivations/02-logistic-convexity-gradient-and-separability.md](derivations/02-logistic-convexity-gradient-and-separability.md) | logistic objective 的 gradient、Hessian、convexity、regularized existence / uniqueness、separable-data instability |

## 1. 模块定位

CS229 Lecture 3 和 Lecture 4 已经完成：

```text
Bernoulli exponential family
natural parameter
canonical response
sigmoid
cross entropy
logistic gradient and Hessian
```

本模块不重复 GLM derivation。MIT 这里改用 risk-minimization / regularization viewpoint：

```text
logistic regression
-> linear score class
-> logistic loss
-> empirical logistic risk
-> regularized ERM
-> convex objective
-> stable finite-sample estimator
```

Probability-model derivation 见 CS229 Lecture 4；这里关心的是同一 objective 在 statistical learning 中怎样被解释。

## 2. Label Convention

MIT 9.520 在 binary classification 的 loss notation 中常使用：

```math
y_i\in\{-1,+1\}.
```

CS229 logistic regression 常使用：

```math
y_i\in\{0,1\}.
```

两种写法通过下式转换：

```math
y_{\pm}
=
2y_{01}-1.
```

其中 $y_{\pm}$ 是 MIT margin convention，$y_{01}$ 是 CS229 Bernoulli convention。

## 3. Logistic Empirical Risk

选择 linear score：

```math
f_w(x)
=
w^Tx.
```

对 $y\in\{-1,+1\}$，margin 是：

```math
yf_w(x)
=
yw^Tx.
```

Logistic loss 定义为：

```math
\ell_{\log}(y,f_w(x))
=
\log(1+e^{-y f_w(x)}).
```

给定训练样本：

```math
S
=
\{(x_i,y_i)\}_{i=1}^{n},
```

empirical logistic risk 是：

```math
\widehat R_n(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\log(1+e^{-y_iw^Tx_i}).
```

从 MIT 视角，logistic regression 是在 linear score class 中最小化 empirical logistic risk。

## 4. Negative Log-Likelihood Equals Empirical Logistic Risk

这是 MIT 视角和 CS229 视角最关键的连接。

先用 CS229 的 $y\in\{0,1\}$ notation。定义：

```math
\sigma(z)
=
\frac{1}{1+e^{-z}},
\quad
z=w^Tx.
```

Bernoulli logistic conditional model：

```math
p(y\mid x;w)
=
\sigma(w^Tx)^y
\left(1-\sigma(w^Tx)\right)^{1-y}.
```

Conditional likelihood：

```math
L(w)
=
\prod_{i=1}^{n}
\sigma(w^Tx_i)^{y_i}
\left(1-\sigma(w^Tx_i)\right)^{1-y_i}.
```

取 negative average log-likelihood：

```math
-\frac{1}{n}\log L(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\left[
-y_i\log\sigma(w^Tx_i)
-
(1-y_i)\log(1-\sigma(w^Tx_i))
\right].
```

右侧就是 empirical cross-entropy risk。

若改用 MIT 的 $y_i\in\{-1,+1\}$ notation，logistic model 可写成：

```math
p(y_i\mid x_i;w)
=
\frac{1}{1+e^{-y_iw^Tx_i}}.
```

于是：

```math
-\log p(y_i\mid x_i;w)
=
\log(1+e^{-y_iw^Tx_i}).
```

因此：

```math
-\frac{1}{n}\log L(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\log(1+e^{-y_iw^Tx_i})
=
\widehat R_n(w).
```

结论：

```text
CS229:
maximum conditional likelihood

MIT:
minimizing empirical logistic risk
```

数学上是同一个 objective 的两种解释，只是前者从 probabilistic estimation 进入，后者从 risk minimization 进入。

## 5. Regularized Logistic Regression

Regularized logistic regression 定义为：

```math
\hat w_{\lambda}
=
\underset{w}{\mathrm{argmin}}
\left[
\widehat R_n(w)
+
\lambda
\left\|w\right\|_2^2
\right].
```

展开为：

```math
\hat w_{\lambda}
=
\underset{w}{\mathrm{argmin}}
\left[
\frac{1}{n}
\sum_{i=1}^{n}
\log(1+e^{-y_iw^Tx_i})
+
\lambda
\left\|w\right\|_2^2
\right].
```

同一 objective 有三种读法：

| Viewpoint | 解释 |
| --- | --- |
| Likelihood viewpoint | penalized conditional likelihood |
| Statistical-learning viewpoint | regularized empirical risk minimization |
| Bayesian viewpoint | Gaussian prior on $w$ 对应的 MAP intuition；完整 MAP 细节见 CMU MLE / MAP supplement |

如果 $w$ 含 intercept，很多实现会选择不惩罚 intercept；本模块公式默认 $w$ 是被 regularized 的参数集合。实际代码必须明确这一点。

## 6. Convexity of Logistic Loss

令 scalar margin 为：

```math
z
=
yw^Tx.
```

定义一维 logistic loss：

```math
\ell(z)
=
\log(1+e^{-z}).
```

一阶导数：

```math
\ell'(z)
=
-
\frac{e^{-z}}{1+e^{-z}}
=
-
\frac{1}{1+e^{z}}.
```

二阶导数：

```math
\ell''(z)
=
\frac{e^z}{(1+e^z)^2}.
```

因为 $e^z>0$ 且 denominator 为正：

```math
\ell''(z)
\geq
0.
```

所以 $\ell(z)$ convex。

对 fixed example，$z_i(w)=y_iw^Tx_i$ 是 $w$ 的 affine function。Convex function 与 affine map composition 后仍 convex，因此：

```math
w
\mapsto
\log(1+e^{-y_iw^Tx_i})
```

是 convex function。有限个 convex functions 的平均仍 convex，所以 empirical logistic risk convex。

还可以写 Hessian：

```math
\nabla_w^2
\log(1+e^{-y_iw^Tx_i})
=
\ell''(y_iw^Tx_i)
x_ix_i^T.
```

对任意 $v$：

```math
v^T
\ell''(y_iw^Tx_i)
x_ix_i^T
v
=
\ell''(y_iw^Tx_i)
(x_i^Tv)^2
\geq
0.
```

因此每个 per-example Hessian positive semidefinite，总 Hessian 也 positive semidefinite。

加入 L2 regularization 后，objective Hessian 变为：

```math
\nabla^2
\left[
\widehat R_n(w)
+
\lambda
\left\|w\right\|_2^2
\right]
=
\nabla^2\widehat R_n(w)
+
2\lambda I.
```

若 $\lambda>0$ 且所有被优化的参数都被 L2 penalty 覆盖，则 objective strongly convex，minimizer unique。若 intercept 不被惩罚或 design matrix 有未受惩罚的 degenerate directions，需要单独检查唯一性，不能过度声称完全没有 degeneracy。

## 7. Why Regularization Is Useful in Logistic Regression

Regularization 在 logistic regression 中不只是“后面再防 overfitting”。即使 objective convex，它仍有多个有限样本问题。

### Finite-Sample Estimation

有限样本只提供有限约束。若 feature dimension 相对样本数较大，很多方向在训练数据上几乎不可区分。L2 penalty 对参数 norm 施加 preference，使 learned score 不依赖过大的不稳定系数。

### Collinearity

若 features 近似线性相关，不同参数向量可能产生接近相同的 logits：

```math
Xw
\approx
Xw'.
```

Unregularized objective 对这些方向区分弱，参数可能对样本扰动敏感。L2 regularization 会偏好较小 norm 的代表。

### Separable Data

如果存在 $w_0$ 使得所有样本满足：

```math
y_iw_0^Tx_i
>
0,
\quad
i=1,\ldots,n,
```

则沿着 $aw_0$ 放大 $a$：

```math
\log(1+e^{-y_i(aw_0)^Tx_i})
\to
0
```

当 $a\to\infty$。这意味着 unregularized empirical logistic risk 的 infimum 可以趋近 $0$，但 finite minimizer 不一定存在；参数 norm 会被推向 infinity。

加入：

```math
\lambda\left\|w\right\|_2^2
```

后，过大的 parameter norm 会被惩罚，从而得到稳定的 finite solution。

### Complexity Control

L2 norm 控制 score function 的尺度。过大的 $w$ 会使 logits 极端，概率接近 $0$ 或 $1$，容易产生 overconfident predictions。Regularization 通过限制参数尺度，让经验拟合和模型复杂度之间有可调 trade-off。

## 8. Relationship to GDA and CS229 Lecture 5

CS229 Lecture 5 已经展示 GDA posterior 可以化成 logistic form。这说明：

```text
generative assumptions
-> discriminative posterior form
```

Logistic regression 则直接假设 conditional log-odds 是 linear score：

```math
\log
\frac{
P(Y=1\mid x)
}{
P(Y=0\mid x)
}
=
w^Tx.
```

MIT 视角把这两条线统一为 hypothesis-space restriction：

* GDA restricts joint / class-conditional distribution family；
* Logistic regression restricts conditional log-odds functional family；
* Regularized logistic regression further orders candidate score functions by parameter norm。

这里不重复 generative / discriminative 的完整 CS229 讨论，只保留理论统一。

## 9. What Is Deferred from Class 05

Class 05 标题同时包含 Logistic Regression and Support Vector Machines。本模块当前不展开：

```text
SVM
hinge loss
margin maximization
support vectors
dual optimization
kernels
```

这些属于 CS229 L6-L7 的节点。当前只完成 logistic regression as empirical / regularized risk 的理论补充。
