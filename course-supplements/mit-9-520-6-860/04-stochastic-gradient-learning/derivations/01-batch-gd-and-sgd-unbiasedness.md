# Batch Gradient and SGD Unbiasedness

返回 [Module 04](../note.md)。

来源边界：本文件覆盖 MIT 9.520 / 6.860 Class 06 中 finite-sum objective、batch gradient、stochastic gradient 与 unbiased estimator 的当前相关部分。它不讨论 early stopping 或 SGD generalization theory。

## 1. Finite-Sum Empirical Objective

许多 CS229 早期目标都可以写成：

```math
F(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
f_i(w).
```

其中 $f_i(w)$ 是第 $i$ 个样本贡献的 loss。例子：

| Model | $f_i(w)$ |
| --- | --- |
| Linear regression | $(w^Tx_i-y_i)^2$ |
| Logistic regression | $\log(1+e^{-y_iw^Tx_i})$ |

若有 L2 regularization，可以写成：

```math
F_{\lambda}(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
f_i(w)
+
\lambda\left\|w\right\|_2^2.
```

也可以把 penalty 分摊到每个 $f_i$ 中；实现时必须固定 convention。

## 2. Full Gradient

Full gradient 为：

```math
\nabla F(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\nabla f_i(w).
```

Batch gradient descent：

```math
w_{t+1}
=
w_t
-
\gamma_t
\nabla F(w_t).
```

每步都需要计算所有 $n$ 个 per-sample gradients。这个方向是 empirical objective 的 exact gradient。

## 3. One-Sample SGD

SGD 在第 $t$ 步随机抽一个 index：

```math
i_t\in\{1,\ldots,n\}.
```

然后使用：

```math
g_t
=
\nabla f_{i_t}(w_t).
```

Update：

```math
w_{t+1}
=
w_t
-
\gamma_t g_t.
```

这个 update 使用一个样本的信息，方向是 random variable。

## 4. Unbiasedness Proof

假设 $i_t$ 在 $\{1,\ldots,n\}$ 上 uniform sampling，且在给定 $w_t$ 时抽样：

```math
P(i_t=i\mid w_t)
=
\frac{1}{n}.
```

条件期望：

```math
\mathbb E[g_t\mid w_t]
=
\mathbb E[\nabla f_{i_t}(w_t)\mid w_t].
```

展开：

```math
\mathbb E[g_t\mid w_t]
=
\sum_{i=1}^{n}
P(i_t=i\mid w_t)
\nabla f_i(w_t).
```

代入 uniform probability：

```math
\mathbb E[g_t\mid w_t]
=
\frac{1}{n}
\sum_{i=1}^{n}
\nabla f_i(w_t).
```

因此：

```math
\mathbb E[g_t\mid w_t]
=
\nabla F(w_t).
```

所以 one-sample stochastic gradient 是 full gradient 的 unbiased estimator。

## 5. Mini-Batch Version

若 $B_t$ 是大小为 $b$ 的 mini-batch，定义：

```math
g_t
=
\frac{1}{b}
\sum_{i\in B_t}
\nabla f_i(w_t).
```

在 uniform sampling 且每个 index 被等概率选中的条件下：

```math
\mathbb E[g_t\mid w_t]
=
\nabla F(w_t).
```

Mini-batch 的主要作用是降低 gradient variance 并利用并行计算，但每步成本高于 one-sample SGD。

## 6. Unbiased Does Not Mean Descent

Unbiasedness 只表示：

```math
\mathbb E[g_t\mid w_t]
=
\nabla F(w_t).
```

它不表示：

```math
g_t
=
\nabla F(w_t)
```

也不表示：

```math
F(w_{t+1})
\leq
F(w_t)
```

对每一次随机 update 都成立。

原因是：

```math
g_t
=
\nabla F(w_t)
+
\xi_t,
```

其中 $\xi_t$ 是 zero-mean noise：

```math
\mathbb E[\xi_t\mid w_t]
=
0.
```

但 $\xi_t$ 的 variance 可以很大。某次 update 可能朝着升高 objective 的方向移动；SGD 的理论分析看的是期望、长期平均或高概率行为，而不是每一步单调下降。

## 7. Linear Regression Per-Sample Gradient

对 square loss：

```math
f_i(w)
=
(w^Tx_i-y_i)^2.
```

Gradient：

```math
\nabla f_i(w)
=
2(w^Tx_i-y_i)x_i.
```

SGD update：

```math
w_{t+1}
=
w_t
-
2\gamma_t
(w_t^Tx_{i_t}-y_{i_t})
x_{i_t}.
```

等价写法：

```math
w_{t+1}
=
w_t
+
2\gamma_t
(y_{i_t}-w_t^Tx_{i_t})
x_{i_t}.
```

这就是 CS229 LMS update 的 finite-sum / stochastic-gradient 解释，常数差异取决于 loss 是否含有 $1/2$。

## 8. Logistic Regression Per-Sample Gradient

对 logistic loss：

```math
f_i(w)
=
\log(1+e^{-y_iw^Tx_i}).
```

Gradient：

```math
\nabla f_i(w)
=
-
\frac{y_ix_i}{1+e^{y_iw^Tx_i}}.
```

SGD update：

```math
w_{t+1}
=
w_t
+
\gamma_t
\frac{y_{i_t}x_{i_t}}{1+e^{y_{i_t}w_t^Tx_{i_t}}}.
```

如果加入 L2 penalty 且 penalty 不分摊到 $f_i$ 中，则 update 还需要加上 full penalty gradient：

```math
-2\gamma_t\lambda w_t.
```

## 9. Boundary

本文件只证明 stochastic gradient 对 empirical finite-sum objective 的 unbiasedness。它没有证明 convergence theorem、generalization theorem 或 stability result。
