# Statistical Risk Framework

返回 [Module 01](../note.md)。

来源边界：本文件参考 MIT 9.520 / 6.860 Class 02 和 Rosasco / Poggio notes Chapter 1 的 statistical learning setup，并把它局部连接到 CS229 Lecture 1-5 已学模型。这里不证明 uniform convergence 或 generalization bounds。

## 1. Population Distribution

Statistical learning 的起点不是参数 $\theta$，而是一个未知的总体分布。令 $X$ 是 input random variable，$Y$ 是 output random variable，二者联合取值于：

```math
\mathcal X\times\mathcal Y.
```

未知数据生成分布写成：

```math
(X,Y)\sim\rho.
```

这里 $\rho$ 是 population-level object。它不是训练集，也不是模型参数。CS229 中经常写 $p(x,y)$、$p(y\mid x)$ 或 $p(x\mid y)p(y)$；这些是具体模型对数据分布的建模方式。MIT notation 中的 $\rho$ 更靠近“真实但未知的数据规律”。

训练样本是从 $\rho$ 抽出的 finite random sample：

```math
S
=
\{(x_i,y_i)\}_{i=1}^{n}.
```

标准假设为 iid：

```math
(x_i,y_i)
\overset{\mathrm{iid}}{\sim}
\rho.
```

在抽样之前，$S$ 是 random object；抽样之后，仓库中的训练数据文件或矩阵是它的一次 realization。这个区分很重要，因为 empirical quantity 会随样本改变，而 population quantity 由 $\rho$ 决定。

## 2. Predictors and Loss

学习算法输出一个 predictor：

```math
f:\mathcal X\to\mathcal A.
```

其中 $\mathcal A$ 是 prediction / score space。Regression 中常取 $\mathcal A=\mathbb R$；binary classification 中，$f(x)$ 可以是 score、logit 或 probability 的前置量。

Loss function：

```math
\ell:\mathcal A\times\mathcal Y\to\mathbb R_+.
```

给定 prediction $f(x)$ 和 label $y$，loss 为：

```math
\ell(f(x),y).
```

Loss 不是 probability。Probability 描述随机事件如何发生；loss 描述预测错到什么程度。一个 probabilistic model 可以诱导 NLL loss，但 loss 本身是评价准则。

CS229 L1-L5 已出现的直接例子：

| CS229 object | MIT loss view |
| --- | --- |
| Linear regression | squared loss $\ell(f(x),y)=(f(x)-y)^2$ |
| Logistic regression | logistic / cross-entropy loss |
| GDA | generative likelihood induces posterior prediction, but prediction quality still可由 loss 评价 |
| Naive Bayes | joint modeling gives classifier；classification loss / log loss 可用于评价 |

## 3. Expected Risk

Expected risk 是总体上真正想小的量：

```math
R(f)
=
\mathbb E_{(X,Y)\sim\rho}
\left[
\ell(f(X),Y)
\right].
```

如果 $\rho$ 有密度或 probability measure 表示，也可写成：

```math
R(f)
=
\int_{\mathcal X\times\mathcal Y}
\ell(f(x),y)
d\rho(x,y).
```

这里 $X,Y$ 是 random variables，$x,y$ 是积分变量。$R(f)$ 是固定 $f$ 后对未来随机样本的平均损失。

这个定义回答的是：

```text
如果未来数据仍来自 rho，predictor f 的长期平均损失是多少？
```

但 $\rho$ unknown，所以 $R(f)$ 不可直接计算。

## 4. Empirical Risk

对观测样本 $S$，empirical risk 定义为：

```math
\widehat R_n(f)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell(f(x_i),y_i).
```

它是样本平均，不是总体平均。抽样之前，$\widehat R_n(f)$ 是 random variable；抽样之后，它是一个可以由数据和代码算出来的数。

对 fixed predictor $f$，若 $\ell(f(X),Y)$ 有有限期望，iid 假设下可以由 law of large numbers 得到：

```math
\widehat R_n(f)
\to
R(f)
```

当 $n\to\infty$。这只是 fixed $f$ 的 statement。ERM 会在许多候选函数中根据同一份数据选择 $\hat f$，因此不能把 fixed-function convergence 直接当作所有函数同时成立的结论。后续 uniform convergence 等理论正是处理这个难点；当前不展开。

## 5. ERM as a Learning Rule

给定 hypothesis space $\mathcal H$，Empirical Risk Minimization 定义为：

```math
\hat f
\in
\underset{f\in\mathcal H}{\mathrm{argmin}}
\,
\widehat R_n(f).
```

这个式子有三个层次：

| Layer | 含义 |
| --- | --- |
| $\mathcal H$ | algorithm 被允许选择的函数集合 |
| $\widehat R_n(f)$ | 可观测训练样本上的 proxy objective |
| $\hat f$ | 根据样本选出的 random estimator |

ERM 的目标不是崇拜 training error，而是在 unknown $R(f)$ 不可计算时，用 $\widehat R_n(f)$ 作为可优化替代。

## 6. Linear Regression as ERM

选择 linear hypothesis space：

```math
\mathcal H_{\mathrm{lin}}
=
\{x\mapsto w^Tx:w\in\mathbb R^d\}.
```

选择 squared loss：

```math
\ell(f_w(x),y)
=
(w^Tx-y)^2.
```

则 empirical risk 为：

```math
\widehat R_n(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
(w^Tx_i-y_i)^2.
```

ERM estimator：

```math
\hat w
\in
\underset{w\in\mathbb R^d}{\mathrm{argmin}}
\,
\frac{1}{n}
\sum_{i=1}^{n}
(w^Tx_i-y_i)^2.
```

写成矩阵形式就是：

```math
\hat w
\in
\underset{w}{\mathrm{argmin}}
\,
\frac{1}{n}
\left\|Xw-y\right\|_2^2.
```

CS229 normal equation 是这个 ERM 在 linear-square-loss 组合下的 analytic solution，而不是一个脱离 statistical learning 的孤立代数技巧。

## 7. Logistic Regression as ERM

使用 MIT 的 label convention：

```math
y_i\in\{-1,+1\}.
```

Linear score：

```math
f_w(x)
=
w^Tx.
```

Logistic loss：

```math
\ell_{\log}(y_i,f_w(x_i))
=
\log(1+e^{-y_iw^Tx_i}).
```

Empirical risk：

```math
\widehat R_n(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\log(1+e^{-y_iw^Tx_i}).
```

CS229 用 Bernoulli likelihood / GLM 推导出同一 objective；MIT 把它看成 empirical logistic risk。这不是两个算法，而是同一 optimization problem 的两种解释。

## 8. Generative Models in the Same Framework

GDA 和 Naive Bayes 看起来不像直接选择 $f:\mathcal X\to\mathcal Y$，因为它们先建立 joint 或 class-conditional probability model。

GDA 选择：

```math
p(y)p(x\mid y)
```

并限制 $p(x\mid y)$ 为 Gaussian family。Naive Bayes 进一步限制：

```math
p(x\mid y)
=
\prod_{j=1}^{d}
p(x_j\mid y).
```

从 statistical-learning 视角看，它们仍然是在限制 admissible solution class，只是限制对象从 prediction function 扩展到了 data law / conditional distribution family。最终 classifier 仍通过 posterior 或 score 函数产生 prediction：

```math
\hat y(x)
=
\underset{y}{\mathrm{argmax}}
\,
p(y\mid x).
```

因此 CS229 L5 的 generative / discriminative split 可以被 MIT 框架统一为：

```text
different ways of imposing inductive structure
before learning from finite samples.
```

## 9. Generalization Gap as the Central Tension

对同一个 $f$，generalization gap 为：

```math
R(f)-\widehat R_n(f).
```

这个 gap 不是优化误差。即使 $\widehat R_n(f)$ 被优化到很低，$R(f)$ 仍可能高。原因是 $\widehat R_n$ 只看训练样本，而 $R$ 看未知 population。

CS229 L1-L5 已经隐含这个问题：

* linear regression 可能在训练数据上 residual 小，但遇到 distribution shift 会失败；
* logistic regression 可能在 separable training data 上把参数推到很大；
* GDA 在 Gaussian assumption 错误时 posterior 会 misspecified；
* Naive Bayes 在 feature dependence 强时可能 probability calibration 很差。

MIT supplement 的作用就是把这些现象统一解释为：

```text
finite-sample fit does not by itself guarantee population-level prediction.
```

## 10. Regularized ERM

Regularized ERM 写成：

```math
\hat f_{\lambda}
=
\underset{f\in\mathcal H}{\mathrm{argmin}}
\left[
\widehat R_n(f)
+
\lambda\Omega(f)
\right].
```

其中 $\Omega(f)$ 是 complexity / preference term。它的作用不是机械地“防止过拟合”，而是在 ill-posed finite-sample problem 中，对候选解施加结构排序。

对 linear models，常见形式是：

```math
\Omega(f_w)
=
\left\|w\right\|_2^2.
```

这会偏好小 norm 的 score function。偏好小 norm 不是因为“小”天然正确，而是因为在许多有限样本设置中，小 norm 解对扰动更稳定，并且不依赖数据中被弱约束的方向。

## 11. Boundary

本文件只完成 Class 02 的当前基础层。它没有证明：

```text
uniform convergence
Rademacher complexity
stability
generalization bounds
```

这些是后续 CS229 节点的内容。
