# MLE、MAP 与 Beta-Bernoulli

返回 [Module 01](../README.md)。

CS229 连接：[Lecture 4 sufficient statistics](../../../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-a-what-information-about-a-parameter-is-actually-in-the-data) 解释为什么某些统计量能保留和参数有关的 likelihood 信息。本推导把这个思想接到 CMU 的 closed-form MLE/MAP recipe。

来源边界：这是本仓库的独立推导，参考 CMU 10-601 Spring 2023 Lecture 16 的 MLE/MAP segment、Lecture 17 的 MAP opening、Tom Mitchell 的 MLE/MAP reading，以及模块索引中列出的 CMU historical notes。

## 1. Likelihood 是参数的函数

观测数据固定为：

```math
\mathcal D
=
\{z^{(i)}\}_{i=1}^{m}.
```

参数模型为 $p(z\mid\theta)$。likelihood 定义为：

```math
L(\theta;\mathcal D)
=
p(\mathcal D\mid\theta).
```

这里 $\theta$ 是 candidate parameter，$\mathcal D$ 是已经观察到的数据。优化时变化的是 $\theta$，不是数据。likelihood 不是 $p(\theta\mid\mathcal D)$。

iid assumption 下：

```math
L(\theta;\mathcal D)
=
\prod_{i=1}^{m}
p(z^{(i)}\mid\theta).
```

log-likelihood：

```math
\ell(\theta)
=
\log L(\theta;\mathcal D)
=
\sum_{i=1}^{m}
\log p(z^{(i)}\mid\theta).
```

MLE：

```math
\hat\theta_{\mathrm{MLE}}
=
\underset{\theta}{\mathrm{argmax}}
\,
\ell(\theta).
```

## 2. Sufficient-Statistic Bridge

若：

```math
L(\theta;\mathcal D)
=
h(\mathcal D)
g_{\theta}(S(\mathcal D)),
```

则 $h(\mathcal D)$ 和 $\theta$ 无关：

```math
\underset{\theta}{\mathrm{argmax}}
\,
L(\theta;\mathcal D)
=
\underset{\theta}{\mathrm{argmax}}
\,
g_{\theta}(S(\mathcal D)).
```

因此参数估计所需的数据证据被压缩进 $S(\mathcal D)$。这就是：

```text
sufficient statistics
-> likelihood compression
-> parameter estimation
```

Bernoulli 的 $N_1,N_0$、Naive Bayes 的 class-feature counts、Gaussian NB 的 class-wise sums 都是这个思想的实现形态。

## 3. MAP Objective

MAP 从 posterior mode 开始：

```math
\hat\theta_{\mathrm{MAP}}
=
\underset{\theta}{\mathrm{argmax}}
\,
p(\theta\mid\mathcal D).
```

Bayes rule：

```math
p(\theta\mid\mathcal D)
=
\frac{
p(\mathcal D\mid\theta)p(\theta)
}{
p(\mathcal D)
}.
```

因为 $p(\mathcal D)$ 不随 candidate $\theta$ 变化：

```math
\hat\theta_{\mathrm{MAP}}
=
\underset{\theta}{\mathrm{argmax}}
\,
p(\mathcal D\mid\theta)p(\theta).
```

log objective：

```math
\ell_{\mathrm{MAP}}(\theta)
=
\log p(\mathcal D\mid\theta)
+
\log p(\theta).
```

MLE 只看 data likelihood；MAP 看 data likelihood plus prior preference。

## 4. Bernoulli MLE

假设：

```math
Y_i
\sim
\mathrm{Bernoulli}(\phi),
```

且 $y_i\in\{0,1\}$。定义：

```math
N_1
=
\sum_{i=1}^{m}
y_i,
```

以及：

```math
N_0
=
m-N_1.
```

likelihood：

```math
p(\mathcal D\mid\phi)
=
\prod_{i=1}^{m}
\phi^{y_i}(1-\phi)^{1-y_i}
=
\phi^{N_1}(1-\phi)^{N_0}.
```

log-likelihood：

```math
\ell(\phi)
=
N_1\log\phi
+
N_0\log(1-\phi).
```

求导：

```math
\frac{d\ell}{d\phi}
=
\frac{N_1}{\phi}
-
\frac{N_0}{1-\phi}.
```

设为 $0$：

```math
\frac{N_1}{\phi}
=
\frac{N_0}{1-\phi}.
```

于是：

```math
N_1(1-\phi)
=
N_0\phi.
```

整理：

```math
N_1
=
(N_1+N_0)\phi
=
m\phi.
```

所以：

```math
\hat\phi_{\mathrm{MLE}}
=
\frac{N_1}{m}.
```

## 5. Beta Prior 和 Posterior

加入 prior：

```math
\phi
\sim
\mathrm{Beta}(\alpha,\beta).
```

Beta density 的 proportional form：

```math
p(\phi)
\propto
\phi^{\alpha-1}(1-\phi)^{\beta-1}.
```

posterior：

```math
p(\phi\mid\mathcal D)
\propto
\phi^{N_1}(1-\phi)^{N_0}
\phi^{\alpha-1}(1-\phi)^{\beta-1}.
```

合并幂次：

```math
p(\phi\mid\mathcal D)
\propto
\phi^{N_1+\alpha-1}
(1-\phi)^{N_0+\beta-1}.
```

因此：

```math
\phi\mid\mathcal D
\sim
\mathrm{Beta}
(
N_1+\alpha,
N_0+\beta
).
```

## 6. Beta-Bernoulli MAP

posterior log-density：

```math
\ell(\phi)
=
(
N_1+\alpha-1
)
\log\phi
+
(
N_0+\beta-1
)
\log(1-\phi)
+
C.
```

求导：

```math
\frac{d\ell}{d\phi}
=
\frac{
N_1+\alpha-1
}{
\phi
}
-
\frac{
N_0+\beta-1
}{
1-\phi
}.
```

设为 $0$：

```math
\frac{
N_1+\alpha-1
}{
\phi
}
=
\frac{
N_0+\beta-1
}{
1-\phi
}.
```

交叉相乘：

```math
(N_1+\alpha-1)(1-\phi)
=
(N_0+\beta-1)\phi.
```

整理：

```math
N_1+\alpha-1
=
(N_1+N_0+\alpha+\beta-2)\phi.
```

因为 $m=N_1+N_0$：

```math
\hat\phi_{\mathrm{MAP}}
=
\frac{
N_1+\alpha-1
}{
m+\alpha+\beta-2
}.
```

这个 derivation 只适用于 interior maximum。posterior mode 在内部的条件是：

```math
N_1+\alpha
>
1
```

以及：

```math
N_0+\beta
>
1.
```

否则 mode 可能出现在 boundary。特别是 $\alpha \leq 1$ 或 $\beta \leq 1$ 时，不能忽略 boundary case。

## 7. Posterior Mean versus MAP

posterior：

```math
\phi\mid\mathcal D
\sim
\mathrm{Beta}
(
N_1+\alpha,
N_0+\beta
).
```

posterior mean：

```math
E[\phi\mid\mathcal D]
=
\frac{
N_1+\alpha
}{
m+\alpha+\beta
}.
```

posterior mode / MAP：

```math
\hat\phi_{\mathrm{MAP}}
=
\frac{
N_1+\alpha-1
}{
m+\alpha+\beta-2
}.
```

这两个估计一般不同。posterior mean 是对整个 posterior 的一阶矩；MAP 是 posterior density 最大的点。

## 8. Pseudo-Count Interpretation

MAP 可以写成：

```math
\hat\phi_{\mathrm{MAP}}
=
\frac{
N_1+(\alpha-1)
}{
N_1+N_0+(\alpha-1)+(\beta-1)
}.
```

所以 $\alpha-1$ 和 $\beta-1$ 看起来像 prior pseudo-counts。严格说，这只是 prior 对 posterior mode 的作用解释，不是说这些样本真的存在于数据集中。

实现上，这个解释帮助理解 smoothing 为什么能避免 $0/1$ 极端估计；数学上，仍需区分 MAP mode、posterior mean 和具体 smoothing 公式。
