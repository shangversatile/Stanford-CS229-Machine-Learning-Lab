# Logistic Convexity, Gradient, and Separability

返回 [Module 03](../note.md)。

来源边界：本文件覆盖 MIT 9.520 / 6.860 Class 05 中 logistic regression objective、smooth convex optimization、regularization 与 separable-data instability 的当前相关部分。SVM 和 hinge loss 不展开。

## 1. Objective

令：

```math
y_i\in\{-1,+1\},
\quad
f_w(x_i)=w^Tx_i.
```

Regularized logistic objective：

```math
F_{\lambda}(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\log(1+e^{-y_iw^Tx_i})
+
\lambda
\left\|w\right\|_2^2.
```

其中 $\lambda\geq0$。当 $\lambda=0$ 时是 unregularized empirical logistic risk。

## 2. Scalar Logistic Loss

定义 margin：

```math
z
=
yw^Tx.
```

Scalar logistic loss：

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
\frac{1}{1+e^z}.
```

二阶导数：

```math
\ell''(z)
=
\frac{e^z}{(1+e^z)^2}.
```

因为：

```math
\frac{e^z}{(1+e^z)^2}
\geq
0,
```

所以 scalar logistic loss convex。

## 3. Per-Sample Gradient

对第 $i$ 个样本：

```math
\phi_i(w)
=
\log(1+e^{-y_iw^Tx_i}).
```

令：

```math
z_i(w)
=
y_iw^Tx_i.
```

链式法则：

```math
\nabla_w\phi_i(w)
=
\ell'(z_i(w))
\nabla_w z_i(w).
```

因为：

```math
\nabla_w z_i(w)
=
y_i x_i,
```

所以：

```math
\nabla_w\phi_i(w)
=
-
\frac{1}{1+e^{y_iw^Tx_i}}
y_i x_i.
```

即：

```math
\nabla_w\phi_i(w)
=
-
\frac{y_i x_i}{1+e^{y_iw^Tx_i}}.
```

因此 regularized objective 的 gradient：

```math
\nabla F_{\lambda}(w)
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\frac{y_i x_i}{1+e^{y_iw^Tx_i}}
+
2\lambda w.
```

Class 05 强调：这个 optimality condition 一般是 nonlinear equation，不像 ridge regression 那样有简单 closed form，因此需要 gradient-based optimization。

## 4. Hessian and Convexity

Per-sample Hessian：

```math
\nabla_w^2\phi_i(w)
=
\ell''(z_i(w))
(y_ix_i)(y_ix_i)^T.
```

由于 $y_i^2=1$：

```math
\nabla_w^2\phi_i(w)
=
\ell''(z_i(w))
x_ix_i^T.
```

对任意 $v\in\mathbb R^d$：

```math
v^T
\nabla_w^2\phi_i(w)
v
=
\ell''(z_i(w))
(x_i^Tv)^2
\geq
0.
```

所以每个 per-sample logistic loss 对 $w$ convex。有限平均保持 convex：

```math
\widehat R_n(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\phi_i(w)
```

也是 convex。

加入 L2 penalty 后：

```math
\nabla^2F_{\lambda}(w)
=
\nabla^2\widehat R_n(w)
+
2\lambda I.
```

若 $\lambda>0$ 且所有被优化参数都被惩罚，则：

```math
v^T\nabla^2F_{\lambda}(w)v
\geq
2\lambda\left\|v\right\|_2^2
```

对所有 $v$ 成立，objective strongly convex，minimizer unique。若 intercept 不惩罚或存在未受惩罚方向，需要另外检查 uniqueness。

## 5. Existence Through Regularization

当 $\lambda>0$ 时：

```math
\lambda\left\|w\right\|_2^2
\to
\infty
```

随着 $\left\|w\right\|_2\to\infty$。Logistic empirical risk 非负，因此：

```math
F_{\lambda}(w)
\to
\infty
```

随着 $\left\|w\right\|_2\to\infty$。这类 objective 称为 coercive。结合 continuity，可以得到 minimizer existence。Class 05 用这个思路说明 regularization 不只是改变泛化，它也让 optimization problem 更稳健。

## 6. Separable Data and Infinite Weights

若训练集 linearly separable，存在 $w_0$ 使得：

```math
y_iw_0^Tx_i
>
0,
\quad
i=1,\ldots,n.
```

考虑 $aw_0$，其中 $a>0$。每个 margin：

```math
y_i(aw_0)^Tx_i
=
a\,y_iw_0^Tx_i.
```

当 $a\to\infty$ 时：

```math
\log(1+e^{-y_i(aw_0)^Tx_i})
\to
0.
```

因此 unregularized empirical logistic risk：

```math
\widehat R_n(aw_0)
=
\frac{1}{n}
\sum_{i=1}^{n}
\log(1+e^{-y_i(aw_0)^Tx_i})
\to
0.
```

但是 finite $a$ 时每一项仍为正，所以 objective 可以逼近 $0$，却不一定在有限 $w$ 处达到 $0$。参数 norm 会沿 separable direction 趋向 infinity。

这说明：convex 不等于 estimator 自动稳定。Unregularized logistic regression 在 separable finite sample 上可能没有 finite minimizer。

## 7. How L2 Regularization Fixes the Instability

加入：

```math
\lambda\left\|w\right\|_2^2,
\quad
\lambda>0,
```

后，沿 $aw_0$：

```math
F_{\lambda}(aw_0)
=
\widehat R_n(aw_0)
+
\lambda a^2
\left\|w_0\right\|_2^2.
```

虽然第一项趋向 $0$，第二项趋向 $\infty$。因此 optimizer 不会通过无限放大 margin 来降低 empirical loss。Regularization 把问题重新变成有 finite solution 的稳定优化问题。

## 8. Complexity Control Interpretation

从 statistical-learning 角度看，large norm 对应更陡、更极端的 score function。它可能让模型在训练点上非常自信，却对样本扰动和未来数据敏感。

L2 regularization 的作用是：

```text
fit logistic loss,
but prefer smaller-norm score functions among competing explanations.
```

这和 ridge 中的稳定化思想一致，只是 loss 从 squared loss 换成 logistic loss，closed form 变成 gradient-based optimization。

## 9. Current Boundary

本文件不使用 hinge loss 证明 margin theory，也不讨论 support vectors 或 dual problem。这里的 separability 只用于解释 logistic regression 为什么仍需要 regularization。
