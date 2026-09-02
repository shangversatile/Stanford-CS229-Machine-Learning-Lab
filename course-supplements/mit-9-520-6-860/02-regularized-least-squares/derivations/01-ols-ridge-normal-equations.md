# OLS and Ridge Normal Equations

返回 [Module 02](../README.md)。

来源边界：参考 MIT 9.520 / 6.860 Class 03 对 Ordinary Least Squares、normal equation 和 ridge regression 的处理。本文件只讨论 CS229 L1-L5 已需理解的 linear least-squares regularization。

## 1. Setup

设 design matrix 为：

```math
X\in\mathbb R^{n\times d},
```

target vector 为：

```math
y\in\mathbb R^n,
```

parameter vector 为：

```math
w\in\mathbb R^d.
```

第 $i$ 个样本的预测是：

```math
f_w(x_i)
=
w^Tx_i.
```

把全部预测堆叠后：

```math
\hat y
=
Xw.
```

Residual 采用：

```math
r(w)
=
Xw-y.
```

## 2. OLS Objective as Empirical Risk

Ordinary Least Squares objective：

```math
J_{\mathrm{OLS}}(w)
=
\frac{1}{n}
\left\|y-Xw\right\|_2^2.
```

因为：

```math
\left\|y-Xw\right\|_2^2
=
(Xw-y)^T(Xw-y),
```

所以：

```math
J_{\mathrm{OLS}}(w)
=
\frac{1}{n}
(Xw-y)^T(Xw-y).
```

这正是 square loss 的 empirical risk：

```math
\widehat R_n(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-w^Tx_i)^2.
```

## 3. OLS Gradient

展开 quadratic form：

```math
(Xw-y)^T(Xw-y)
=
w^TX^TXw
-
2y^TXw
+
y^Ty.
```

因此：

```math
J_{\mathrm{OLS}}(w)
=
\frac{1}{n}
\left[
w^TX^TXw
-
2y^TXw
+
y^Ty
\right].
```

使用：

```math
\nabla_w(w^TAw)
=
(A+A^T)w,
```

并且 $X^TX$ symmetric：

```math
\nabla_w(w^TX^TXw)
=
2X^TXw.
```

又因为 $y^TXw=(X^Ty)^Tw$：

```math
\nabla_w(y^TXw)
=
X^Ty.
```

所以：

```math
\nabla_wJ_{\mathrm{OLS}}(w)
=
\frac{2}{n}
X^TXw
-
\frac{2}{n}
X^Ty.
```

写成 residual form：

```math
\nabla_wJ_{\mathrm{OLS}}(w)
=
\frac{2}{n}
X^T(Xw-y).
```

## 4. OLS Normal Equation

Stationary condition：

```math
\nabla_wJ_{\mathrm{OLS}}(\hat w)
=
0.
```

代入 gradient：

```math
\frac{2}{n}
X^T(X\hat w-y)
=
0.
```

去掉正比例常数：

```math
X^TX\hat w
-
X^Ty
=
0.
```

得到 normal equation：

```math
X^TX\hat w
=
X^Ty.
```

若 $X^TX$ invertible：

```math
\hat w_{\mathrm{OLS}}
=
(X^TX)^{-1}X^Ty.
```

这个推导说明 OLS closed form 是 empirical square risk 的 first-order optimality condition。

## 5. Ridge Objective

Ridge / Tikhonov objective：

```math
J_{\lambda}(w)
=
\frac{1}{n}
\left\|y-Xw\right\|_2^2
+
\lambda
\left\|w\right\|_2^2.
```

其中 $\lambda\geq0$。第一项是 empirical data fit；第二项是 parameter-norm preference。

## 6. Ridge Gradient

已有：

```math
\nabla_w
\frac{1}{n}
\left\|y-Xw\right\|_2^2
=
\frac{2}{n}
X^T(Xw-y).
```

同时：

```math
\nabla_w
\lambda
\left\|w\right\|_2^2
=
2\lambda w.
```

合并：

```math
\nabla_wJ_{\lambda}(w)
=
\frac{2}{n}
X^T(Xw-y)
+
2\lambda w.
```

## 7. Ridge Normal Equation

令 gradient 为 $0$：

```math
\frac{2}{n}
X^T(Xw-y)
+
2\lambda w
=
0.
```

除以 $2$：

```math
\frac{1}{n}
X^TXw
-
\frac{1}{n}
X^Ty
+
\lambda w
=
0.
```

乘以 $n$：

```math
X^TXw
-
X^Ty
+
n\lambda w
=
0.
```

合并 $w$：

```math
(X^TX+n\lambda I)w
=
X^Ty.
```

若 $\lambda>0$：

```math
\hat w_{\lambda}
=
(X^TX+n\lambda I)^{-1}X^Ty.
```

## 8. Why the Matrix Becomes Stable

对任意 $v\in\mathbb R^d$：

```math
v^TX^TXv
=
\left\|Xv\right\|_2^2
\geq
0.
```

所以 $X^TX$ positive semidefinite。加入 $n\lambda I$ 后：

```math
v^T(X^TX+n\lambda I)v
=
\left\|Xv\right\|_2^2
+
n\lambda\left\|v\right\|_2^2.
```

若 $\lambda>0$ 且 $v\neq0$：

```math
v^T(X^TX+n\lambda I)v
>
0.
```

因此 $X^TX+n\lambda I$ positive definite，可逆。Ridge 不只是“加一项惩罚”，它直接改变了 linear system 的谱结构。

## 9. Convention Check

如果 objective 写成：

```math
\left\|y-Xw\right\|_2^2
+
\lambda\left\|w\right\|_2^2,
```

则 ridge system 是：

```math
(X^TX+\lambda I)w
=
X^Ty.
```

如果 objective 写成 average loss：

```math
\frac{1}{n}
\left\|y-Xw\right\|_2^2
+
\lambda\left\|w\right\|_2^2,
```

则 system 是：

```math
(X^TX+n\lambda I)w
=
X^Ty.
```

两种 convention 都可以，但不能混写。本 supplement 采用 average-loss convention。
