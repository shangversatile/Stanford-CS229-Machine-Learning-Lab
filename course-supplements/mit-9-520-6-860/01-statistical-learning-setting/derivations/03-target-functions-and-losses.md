# Target Functions and Losses

返回 [Module 01](../note.md)。

来源边界：本文件覆盖 MIT 9.520 / 6.860 Class 02 中 target function、conditional risk、loss choice 与 population optimum 的部分，并只保留 CS229 Lecture 5 之前已经需要的 square loss、0-1 classification loss 和 logistic loss。Hinge loss 在 Class 02 / Class 05 中出现，但当前推迟到 CS229 SVM 节点。

## 1. Distribution Factorization

Class 02 把 supervised learning 写在 probability space 上。令联合分布为：

```math
(X,Y)\sim\rho.
```

也可以按 Class 02 slides 的写法记为 $P$。联合分布可以分解为 input marginal 和 conditional label law：

```math
d\rho(x,y)
=
d\rho_{\mathcal X}(x)
d\rho(y\mid x).
```

其中：

| Symbol | 含义 |
| --- | --- |
| $\rho$ | joint population distribution |
| $\rho_{\mathcal X}$ | input marginal distribution |
| $\rho(y\mid x)$ | conditional distribution of output given input |

$\rho_{\mathcal X}$ 说明未来 input 更可能出现在 $\mathcal X$ 的哪些区域；$\rho(y\mid x)$ 说明同一个 input 附近的 output 如何随机变化。CS229 早期的 likelihood 写法通常直接假设某个 parametric form，而 MIT 在这里先把真实但未知的 population object 摆出来。

## 2. Conditional Risk

Expected risk 为：

```math
R(f)
=
\mathbb E_{(X,Y)\sim\rho}
\left[
\ell(f(X),Y)
\right].
```

用条件分布展开：

```math
R(f)
=
\int_{\mathcal X}
\left[
\int_{\mathcal Y}
\ell(f(x),y)
d\rho(y\mid x)
\right]
d\rho_{\mathcal X}(x).
```

对固定 input $x$，定义 conditional risk：

```math
L_x(a)
=
\mathbb E
\left[
\ell(a,Y)
\mid
X=x
\right],
```

其中 $a$ 是在 $x$ 处给出的 prediction / score。于是：

```math
R(f)
=
\int_{\mathcal X}
L_x(f(x))
d\rho_{\mathcal X}(x).
```

这个公式说明：如果没有 hypothesis-space restriction，population optimum 可以逐点选择每个 $x$ 上使 $L_x(a)$ 最小的 prediction。有限样本学习的困难在于 $\rho(y\mid x)$ unknown，而且我们不能真的在每个 $x$ 上单独观察无限次。

## 3. Square Loss Target

对 regression，使用 square loss：

```math
\ell(a,y)
=
(a-y)^2.
```

固定 $x$，conditional risk 为：

```math
L_x(a)
=
\mathbb E
\left[
(a-Y)^2
\mid
X=x
\right].
```

展开：

```math
L_x(a)
=
a^2
-
2a
\mathbb E[Y\mid X=x]
+
\mathbb E[Y^2\mid X=x].
```

对 $a$ 求导：

```math
\frac{d}{da}L_x(a)
=
2a
-
2
\mathbb E[Y\mid X=x].
```

令导数为 $0$：

```math
a
=
\mathbb E[Y\mid X=x].
```

因此 square loss 下的 population target function 是 conditional mean：

```math
f^*(x)
=
\mathbb E[Y\mid X=x].
```

这解释了 CS229 linear regression 的限制：OLS 不直接恢复任意 conditional mean，而是在线性函数类中近似它。如果真实 conditional mean 非线性，固定 linear hypothesis space 会产生 approximation limitation。

## 4. 0-1 Classification Target

对 binary classification，令：

```math
Y\in\{-1,+1\}.
```

定义：

```math
\eta(x)
=
P(Y=+1\mid X=x).
```

则：

```math
P(Y=-1\mid X=x)
=
1-\eta(x).
```

0-1 loss 衡量分类是否错误：

```math
\ell(a,y)
=
\mathbf 1\{a\neq y\},
\quad
a\in\{-1,+1\}.
```

若在 $x$ 处预测 $+1$，conditional risk 为：

```math
L_x(+1)
=
P(Y=-1\mid X=x)
=
1-\eta(x).
```

若预测 $-1$：

```math
L_x(-1)
=
P(Y=+1\mid X=x)
=
\eta(x).
```

因此 optimal classification rule 是：

```math
f^*(x)
=
\begin{cases}
+1,
&
\eta(x)\geq 1/2,
\\
-1,
&
\eta(x)<1/2.
\end{cases}
```

等价地：

```math
f^*(x)
=
\mathrm{sign}
\left(
P(Y=+1\mid X=x)
-
P(Y=-1\mid X=x)
\right).
```

这和 CS229 Lecture 5 的 GDA / logistic discussion 连接很紧：两者都试图给出或近似 posterior $P(Y\mid X=x)$，再由 posterior 决定分类。

## 5. Logistic Loss Target

Logistic regression 通常不直接输出 class label，而是输出 score：

```math
a
=
f(x)
\in
\mathbb R.
```

对 $Y\in\{-1,+1\}$，logistic loss 为：

```math
\ell_{\log}(y,a)
=
\log(1+e^{-ya}).
```

固定 $x$，令 $\eta(x)=P(Y=+1\mid X=x)$。Conditional logistic risk：

```math
L_x(a)
=
\eta(x)
\log(1+e^{-a})
+
(1-\eta(x))
\log(1+e^a).
```

求导：

```math
\frac{d}{da}L_x(a)
=
-
\frac{\eta(x)}{1+e^a}
+
\frac{(1-\eta(x))e^a}{1+e^a}.
```

令导数为 $0$：

```math
-
\eta(x)
+
(1-\eta(x))e^a
=
0.
```

所以：

```math
e^a
=
\frac{\eta(x)}{1-\eta(x)}.
```

当 $0<\eta(x)<1$ 时：

```math
a^*(x)
=
\log
\frac{\eta(x)}{1-\eta(x)}
=
\log
\frac{P(Y=+1\mid X=x)}{P(Y=-1\mid X=x)}.
```

因此 logistic loss 的 population-optimal score 是 conditional log-odds。CS229 用 Bernoulli GLM 说明 sigmoid / logit 的 probability-model 来源；MIT 的 risk view 说明同一 score 也是 logistic loss 下的 conditional-risk minimizer。

若 $\eta(x)=1$，则 $a^*(x)$ 形式上趋向 $+\infty$；若 $\eta(x)=0$，则趋向 $-\infty$。这也预示了 finite separable data 下 unregularized logistic regression 可能把 parameter norm 推向无穷大，详见 Module 03。

## 6. Loss Choice Changes the Target

Class 02 的一个关键点是：target function 不是只由 $\rho$ 决定，也由 loss 决定。Square loss 得到 conditional mean；0-1 loss 得到 Bayes classifier；logistic loss 得到 conditional log-odds score。

因此学习问题不是一句“拟合真实函数”就能说清楚。更准确地说：

```text
distribution rho
+ loss ell
+ hypothesis space H
+ finite sample S
+ algorithm
-> learned predictor
```

CS229 早期模型常从模型族和 likelihood 出发；MIT 这里强调先问清楚 prediction criterion，再看 empirical proxy 和 hypothesis restriction。

## 7. Current Boundary

本文件没有展开 hinge loss、margin、support vector 或 kernelized classifier。虽然 Class 02 的 loss overview 会列出这些对象，但它们要等 CS229 进入 SVM / kernels 后再学习。
