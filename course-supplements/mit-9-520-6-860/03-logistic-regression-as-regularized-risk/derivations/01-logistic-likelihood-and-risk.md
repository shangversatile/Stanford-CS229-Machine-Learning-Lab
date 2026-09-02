# Logistic Likelihood and Empirical Risk

返回 [Module 03](../README.md)。

来源边界：本文件把 CS229 Lecture 3-4 的 Bernoulli logistic likelihood 与 MIT 9.520 / 6.860 Class 05 的 logistic empirical risk 连接起来。它只处理 logistic regression；Class 05 中的 SVM 内容不在当前范围。

## 1. Two Label Conventions

CS229 常用：

```math
y\in\{0,1\}.
```

MIT Class 05 的 margin notation 常用：

```math
y\in\{-1,+1\}.
```

二者转换为：

```math
y_{\pm}
=
2y_{01}-1.
```

反向转换为：

```math
y_{01}
=
\frac{y_{\pm}+1}{2}.
```

这只是 notation change，不是模型改变。

## 2. CS229 Bernoulli Logistic Model

令 linear score 为：

```math
z_i
=
w^Tx_i.
```

Sigmoid function：

```math
\sigma(z)
=
\frac{1}{1+e^{-z}}.
```

CS229 的 conditional probability model：

```math
P(Y=1\mid x_i;w)
=
\sigma(w^Tx_i).
```

于是：

```math
P(Y=0\mid x_i;w)
=
1-\sigma(w^Tx_i).
```

对 $y_i\in\{0,1\}$，Bernoulli conditional likelihood 的单样本项为：

```math
p(y_i\mid x_i;w)
=
\sigma(w^Tx_i)^{y_i}
\left(
1-\sigma(w^Tx_i)
\right)^{1-y_i}.
```

全样本 likelihood：

```math
L(w)
=
\prod_{i=1}^{n}
p(y_i\mid x_i;w).
```

## 3. Negative Log-Likelihood

平均 negative log-likelihood：

```math
-\frac{1}{n}\log L(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\left[
-y_i\log\sigma(w^Tx_i)
-
(1-y_i)
\log(1-\sigma(w^Tx_i))
\right].
```

这就是 empirical cross-entropy risk。CS229 从 maximum likelihood 得到的 optimization problem 等价于最小化上式。

因此：

```math
\underset{w}{\mathrm{argmax}}
\,
L(w)
=
\underset{w}{\mathrm{argmin}}
\left[
-\frac{1}{n}\log L(w)
\right].
```

Maximum conditional likelihood 和 empirical risk minimization 在这里给出同一个 estimator。

## 4. MIT Margin Form

使用 $y_i\in\{-1,+1\}$ 时，logistic model 可以写成：

```math
p(y_i\mid x_i;w)
=
\frac{1}{1+e^{-y_iw^Tx_i}}.
```

验证：

若 $y_i=+1$：

```math
p(y_i\mid x_i;w)
=
\frac{1}{1+e^{-w^Tx_i}}
=
\sigma(w^Tx_i).
```

若 $y_i=-1$：

```math
p(y_i\mid x_i;w)
=
\frac{1}{1+e^{w^Tx_i}}
=
1-\sigma(w^Tx_i).
```

单样本 negative log-likelihood：

```math
-\log p(y_i\mid x_i;w)
=
\log(1+e^{-y_iw^Tx_i}).
```

因此 empirical risk 为：

```math
\widehat R_n(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\log(1+e^{-y_iw^Tx_i}).
```

得到：

```math
-\frac{1}{n}\log L(w)
=
\widehat R_n(w).
```

这里没有隐藏常数；差别只来自 label convention 和是否把 loss 写成 cross-entropy form。

## 5. Why MIT Reads This as Risk

CS229 的解释顺序是：

```text
assume Bernoulli conditional model
-> write likelihood
-> maximize likelihood
```

MIT Class 05 的解释顺序是：

```text
choose linear score class
-> choose logistic loss
-> minimize empirical logistic risk
```

两者数学上落在同一个 objective 上，但含义不同。CS229 强调概率模型与参数估计；MIT 强调 prediction criterion、hypothesis space 和 finite-sample proxy。

## 6. Regularized Logistic Objective

Regularized ERM 写成：

```math
\hat w_{\lambda}
=
\underset{w}{\mathrm{argmin}}
\left[
\widehat R_n(w)
+
\lambda
\left\|w\right\|_2^2
\right],
\quad
\lambda\geq0.
```

展开：

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

三种读法：

| View | 同一数学对象的含义 |
| --- | --- |
| Likelihood | penalized negative conditional log-likelihood |
| Statistical learning | regularized empirical logistic risk |
| Bayesian intuition | Gaussian prior induces an L2 penalty under MAP |

Bayesian MAP 的细节不在本文件展开，避免和 CMU MLE / MAP supplement 重复。当前重点是：regularization 在 MIT 视角下是对 finite-sample risk minimization 的结构控制。

## 7. Population Logistic Target

Module 01 已推导：在 $Y\in\{-1,+1\}$ 下，若：

```math
\eta(x)
=
P(Y=+1\mid X=x),
```

则 logistic loss 的 population-optimal score 是：

```math
f^*(x)
=
\log
\frac{\eta(x)}{1-\eta(x)}.
```

因此 logistic regression 的 linear score assumption：

```math
f_w(x)
=
w^Tx
```

等价于把 conditional log-odds 限制在线性函数类中。CS229 Lecture 5 中 GDA 推出 logistic-shaped posterior；MIT 在这里强调：logistic regression 自身是在学习一个受限 log-odds function。

## 8. Boundary

Class 05 后续会把 logistic loss 与 hinge loss、SVM、margin 和 kernels 放在一起比较。当前不进入这些内容，因为 CS229 尚未到 SVM / kernels 节点。
