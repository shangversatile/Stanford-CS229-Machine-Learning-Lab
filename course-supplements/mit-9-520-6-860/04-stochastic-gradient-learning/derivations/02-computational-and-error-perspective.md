# Computational and Error Perspective

返回 [Module 04](../note.md)。

来源边界：本文件覆盖 MIT 9.520 / 6.860 Class 06 中为什么需要 stochastic gradients、online least squares、basic convex optimization baseline，以及 optimization error 与 statistical error 的区分。Early stopping、implicit regularization、stability 和 deep learning theory 不展开。

## 1. Why Optimization Enters Learning

CS229 L1-L5 的模型最终都会产生某个 objective：

```math
\underset{w}{\mathrm{minimize}}
\quad
F(w).
```

例如：

| Model | Objective view |
| --- | --- |
| Linear regression | empirical square risk |
| Ridge regression | empirical square risk plus L2 penalty |
| Logistic regression | empirical logistic risk |
| Regularized logistic regression | empirical logistic risk plus L2 penalty |

MIT Class 06 的重点不是新模型，而是：当 objective 已经写好，如何计算或近似 minimizer，以及这个计算误差和统计误差如何分开。

## 2. Cost of Full Gradient

Finite-sum objective：

```math
F(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
f_i(w).
```

Full gradient：

```math
\nabla F(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\nabla f_i(w).
```

如果每个 $\nabla f_i(w)$ 的成本是 $O(d)$，则一次 full gradient 的成本是：

```math
O(nd).
```

Batch GD 每一步都付出这个成本：

```math
w_{t+1}
=
w_t
-
\gamma_t\nabla F(w_t).
```

当 $n$ 很大时，等待一次 exact full-gradient update 可能不如快速做很多 noisy updates。

## 3. Cost of SGD

One-sample SGD 使用：

```math
g_t
=
\nabla f_{i_t}(w_t).
```

若单样本 gradient 成本为 $O(d)$，一次 update 成本也是：

```math
O(d).
```

因此 SGD 的基本 trade-off 是：

```text
exact but expensive full gradients
vs
cheap but noisy stochastic gradients
```

这不是简单的“大数据更快”口号。它是用每步方向的方差换取更多、更便宜的参数更新。

## 4. Online Least Squares

Class 06 用 least squares 说明 streaming / online 场景。假设第 $t$ 步看到样本 $(x_t,y_t)$，单样本 square loss 为：

```math
f_t(w)
=
\frac{1}{2}
(y_t-x_t^Tw)^2.
```

Gradient：

```math
\nabla f_t(w)
=
-(y_t-x_t^Tw)x_t.
```

SGD update：

```math
w_{t+1}
=
w_t
-
\eta_t\nabla f_t(w_t).
```

代入：

```math
w_{t+1}
=
w_t
+
\eta_t
x_t
(y_t-x_t^Tw_t).
```

这就是 Class 06 中 online least squares 的形式。它不需要每次重新求逆，也不需要保存所有样本才能更新参数。

## 5. Why Not Always Use Closed Form

Least squares 的 closed form 需要处理：

```math
X^TX
```

或 ridge 中的：

```math
X^TX+\lambda I.
```

矩阵求逆或分解在高维时可能昂贵；流式数据下，每来一条样本就重新求解也不现实。Recursive least squares 可以用矩阵更新减少重复计算，但仍需维护矩阵级对象。SGD 只维护 parameter vector，并用当前样本或 mini-batch 做局部更新。

因此 Class 06 的核心工程数学判断是：

```text
when exact batch computation is too expensive,
use stochastic local information while controlling step sizes.
```

## 6. Basic Convex GD Baseline

Class 06 还给出 convex optimization 的基本收敛基线。这里保留最小必要版本，用于理解 SGD 为什么要关心 step-size schedule。

设 $F$ convex，$w^*$ 是 minimizer。GD update：

```math
w_{t+1}
=
w_t
-
\gamma_t\nabla F(w_t).
```

由 convexity：

```math
F(w_t)-F(w^*)
\leq
\nabla F(w_t)^T(w_t-w^*).
```

再看距离平方：

```math
\left\|w_{t+1}-w^*\right\|_2^2
=
\left\|w_t-w^*-\gamma_t\nabla F(w_t)\right\|_2^2.
```

展开：

```math
\left\|w_{t+1}-w^*\right\|_2^2
=
\left\|w_t-w^*\right\|_2^2
-
2\gamma_t
\nabla F(w_t)^T(w_t-w^*)
+
\gamma_t^2
\left\|\nabla F(w_t)\right\|_2^2.
```

移项并结合 convexity：

```math
F(w_t)-F(w^*)
\leq
\frac{
\left\|w_t-w^*\right\|_2^2
-
\left\|w_{t+1}-w^*\right\|_2^2
}{2\gamma_t}
+
\frac{\gamma_t}{2}
\left\|\nabla F(w_t)\right\|_2^2.
```

如果 gradients bounded，取合适 decreasing 或 horizon-aware step size，可以得到平均 iterate 的 suboptimality 随 iteration 增长下降的结论。当前不需要把 theorem 推到最一般形式；重点是：optimization analysis 控制的是 empirical objective suboptimality。

## 7. SGD as Noisy GD

SGD 使用 $g_t$ 替代 $\nabla F(w_t)$：

```math
w_{t+1}
=
w_t
-
\gamma_t g_t.
```

若：

```math
\mathbb E[g_t\mid w_t]
=
\nabla F(w_t),
```

则它是 noisy but unbiased version of GD。分析中会多出 variance 项，因此 step size 不能随便取太大。

这解释了图中的现象：

![Batch GD vs SGD path](../figures/mit9520-batch-vs-sgd-path.png)

SGD path 可以局部上升或摆动，但在合适 step sizes 下，长期趋势仍可能靠近 optimum region。

## 8. Optimization Error

令 empirical objective minimizer 为：

```math
\hat w
\in
\underset{w}{\mathrm{argmin}}
\,
F(w).
```

当前 iterate 为 $w_t$。Optimization error 可写成：

```math
F(w_t)-F(\hat w).
```

它回答：

```text
the algorithm has not solved the empirical optimization problem by how much?
```

影响因素包括 conditioning、step size、iteration count、gradient noise 和 stopping rule。

## 9. Statistical Error

Population risk 为：

```math
R(w)
=
\mathbb E_{(X,Y)\sim\rho}
\left[
\ell(f_w(X),Y)
\right].
```

Population benchmark 可以写为：

```math
w^*
\in
\underset{w}{\mathrm{argmin}}
\,
R(w).
```

Statistical error 或 excess risk 关注：

```math
R(\hat w)-R(w^*).
```

它回答：

```text
even if empirical objective is solved,
how good is the learned predictor on the population?
```

影响因素包括 finite sample、hypothesis space、regularization、noise 和 data distribution。

## 10. Why the Distinction Matters

Optimization error 和 statistical error 可以同时存在，但不是同一个东西。

| Scenario | 解释 |
| --- | --- |
| low optimization error, high statistical error | empirical objective 解得很好，但模型只适应训练样本 |
| high optimization error, low statistical error | 算法未完全收敛，但已经达到足够好的 population performance |
| low training loss, high population risk | empirical proxy 与 expected risk 出现差距 |

这一区分为后续 early stopping / implicit regularization 留出理论入口，但当前不展开这些主题。此处只建立 MIT Class 06 对 CS229 早期 gradient-based learning 的关键补充：优化算法解决的是 empirical objective，学习评价最终要回到 population risk。

## 11. Current Boundary

本文件没有讨论：

```text
early stopping
implicit regularization
stability
uniform convergence
SGD generalization bounds
deep learning theory
```

这些属于后续 CS229 节点。
