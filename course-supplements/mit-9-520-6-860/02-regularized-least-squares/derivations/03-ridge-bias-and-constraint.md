# Ridge Bias and Constrained View

返回 [Module 02](../note.md)。

来源边界：参考 MIT 9.520 / 6.860 Class 03 中 regularization、bias、penalized / constrained views 的讨论。本文件不做完整 bias-variance decomposition。

## 1. Fixed-Design Linear Model

设：

```math
y
=
Xw^*
+
\varepsilon.
```

其中：

| Symbol | 含义 |
| --- | --- |
| $X$ | fixed design matrix |
| $w^*$ | true linear parameter |
| $\varepsilon$ | noise vector |

假设：

```math
\mathbb E[\varepsilon]
=
0.
```

这里 expectation 是对 noise 取的，$X$ 固定。

## 2. OLS Unbiasedness Under Full Rank

OLS estimator：

```math
\hat w_{\mathrm{OLS}}
=
(X^TX)^{-1}X^Ty.
```

代入 $y=Xw^*+\varepsilon$：

```math
\hat w_{\mathrm{OLS}}
=
(X^TX)^{-1}X^T(Xw^*+\varepsilon).
```

展开：

```math
\hat w_{\mathrm{OLS}}
=
(X^TX)^{-1}X^TXw^*
+
(X^TX)^{-1}X^T\varepsilon.
```

化简：

```math
\hat w_{\mathrm{OLS}}
=
w^*
+
(X^TX)^{-1}X^T\varepsilon.
```

取 expectation：

```math
\mathbb E[\hat w_{\mathrm{OLS}}]
=
w^*
+
(X^TX)^{-1}X^T\mathbb E[\varepsilon].
```

因为 $\mathbb E[\varepsilon]=0$：

```math
\mathbb E[\hat w_{\mathrm{OLS}}]
=
w^*.
```

在这些条件下 OLS unbiased。但它可能 variance 很大，尤其当 $X^TX$ ill-conditioned 时。

## 3. Ridge Bias

Ridge estimator：

```math
\hat w_{\lambda}
=
(X^TX+n\lambda I)^{-1}X^Ty.
```

代入 $y=Xw^*+\varepsilon$：

```math
\hat w_{\lambda}
=
(X^TX+n\lambda I)^{-1}X^T(Xw^*+\varepsilon).
```

展开：

```math
\hat w_{\lambda}
=
(X^TX+n\lambda I)^{-1}X^TXw^*
+
(X^TX+n\lambda I)^{-1}X^T\varepsilon.
```

取 expectation：

```math
\mathbb E[\hat w_{\lambda}]
=
(X^TX+n\lambda I)^{-1}X^TXw^*
+
(X^TX+n\lambda I)^{-1}X^T\mathbb E[\varepsilon].
```

因为 $\mathbb E[\varepsilon]=0$：

```math
\mathbb E[\hat w_{\lambda}]
=
(X^TX+n\lambda I)^{-1}X^TXw^*.
```

一般：

```math
(X^TX+n\lambda I)^{-1}X^TXw^*
\neq
w^*.
```

所以：

```math
\mathbb E[\hat w_{\lambda}]
\neq
w^*.
```

Ridge estimator biased。

## 4. Bias in Spectral Coordinates

若：

```math
X^TX
=
V
\mathrm{diag}(\sigma_j^2)
V^T,
```

则：

```math
(X^TX+n\lambda I)^{-1}X^TX
=
V
\mathrm{diag}
\left(
\frac{\sigma_j^2}{\sigma_j^2+n\lambda}
\right)
V^T.
```

因此：

```math
\mathbb E[\hat w_{\lambda}]
=
V
\mathrm{diag}
\left(
\frac{\sigma_j^2}{\sigma_j^2+n\lambda}
\right)
V^T
w^*.
```

每个 spectral coordinate 都被乘以：

```math
\frac{\sigma_j^2}{\sigma_j^2+n\lambda}.
```

该因子小于或等于 $1$。小 singular value directions shrinkage 更强，所以 bias 也更强。这是 intentional bias：用系统性收缩换取稳定性和较低 sensitivity。

## 5. Penalized View

Penalized ridge 写成：

```math
\underset{w}{\mathrm{minimize}}
\quad
\widehat R(w)
+
\lambda
\left\|w\right\|_2^2.
```

这里 $\lambda$ 控制 parameter norm 的代价。$\lambda$ 越大，large-norm solutions 越不受偏好。

## 6. Constrained View

Constrained view 写成：

```math
\underset{w}{\mathrm{minimize}}
\quad
\widehat R(w)
\quad
\mathrm{s.t.}
\quad
\left\|w\right\|_2^2
\leq
c.
```

它把 allowed parameter set 限制在 L2 ball 内。若 unconstrained OLS solution 已在 ball 内，则 constraint inactive，constrained solution 等于 OLS。若 OLS 在 ball 外，solution 会落在 boundary 附近。

## 7. Lagrange Multiplier Relation

Constrained problem 的 Lagrangian 可写为：

```math
\mathcal L(w,\mu)
=
\widehat R(w)
+
\mu
\left(
\left\|w\right\|_2^2-c
\right),
\quad
\mu\geq0.
```

若 constraint active，并满足适当 convexity / regularity conditions，则某个 $\mu$ 会使 constrained optimum 同时满足 penalized problem 的 first-order condition。

这说明 penalized 和 constrained views 有对应关系。但一般不能写：

```text
lambda = 1 / c
```

原因是 $c$ 决定 feasible set size，$\lambda$ 决定 penalty strength。二者之间的映射依赖 $\widehat R(w)$ 的 geometry、数据矩阵 $X$ 和 optimum 是否触碰 constraint boundary。

## 8. Geometric Meaning

对应图：

![Ridge geometry](../figures/mit9520-ridge-geometry.png)

Elliptical contours 来自 empirical square risk；L2 ball 来自 constrained regularization。Constrained solution 是 feasible set 内接触最低 contour 的点。Penalized solution 则通过在 objective 中持续惩罚 norm，把 optimum 从 OLS 方向拉向更稳定的小 norm 区域。

## 9. Boundary

本文件只建立 ridge bias 和 constrained / penalized equivalence 的基础。完整 bias-variance decomposition、model selection 和 generalization theory 留到 CS229 L8 附近。
