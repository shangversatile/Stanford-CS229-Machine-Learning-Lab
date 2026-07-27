# Exponential Family Anatomy

## 1. Normalized Form

Exponential family 的 normalized canonical form 是：

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

其中 $a(\eta)$ 不是任意项，而是 normalization 所要求的 log-partition function。

## 2. Normalization

定义：

```math
Z(\eta)=\int b(y)\exp\left(\eta^TT(y)\right)dy
```

令：

```math
a(\eta)=\log Z(\eta)
```

则：

```math
\int p(y;\eta)dy
=
\int b(y)\exp\left(\eta^TT(y)-a(\eta)\right)dy
```

```math
=
e^{-a(\eta)}Z(\eta)
```

```math
=
1
```

离散分布中把 integral 换成 sum，逻辑完全相同。

## 3. Natural Parameter

Natural parameter $\eta$ 是让 distribution 进入 canonical exponential form 的 parameter。它不一定等于常见参数。例如 Bernoulli 的常见参数是 success probability $\phi$，但 natural parameter 是 log odds：

```math
\eta=\log\frac{\phi}{1-\phi}
```

Poisson 的常见参数是 rate $\lambda$，natural parameter 是：

```math
\eta=\log\lambda
```

GLM 的 canonical link 通常让 natural parameter 等于 linear predictor：

```math
\eta=\theta^Tx
```

## 4. Sufficient Statistic

Sufficient statistic $T(y)$ 是 observation 进入 parameter-dependent likelihood 的方式。它不总是 $y$。

Examples：

| Model | Sufficient statistic | Comment |
| ----- | -------------------- | ------- |
| Bernoulli | $T(y)=y$ | binary event 的 count |
| Fixed-variance Gaussian | $T(y)=y$ | variance fixed 时只需 mean statistic |
| Unknown-variance Gaussian | $T(y)=(y,y^2)$ | mean 和 second moment 都参与 |
| Poisson | $T(y)=y$ | count sum is sufficient for rate |
| Categorical with reference class | indicator vector | 每类 count 是 sufficient evidence |

Unknown-variance Gaussian 是最常见的反例。若 mean 和 variance 都未知，density 展开会同时出现 $y$ 和 $y^2$，因此 sufficient statistic 必须包含二者。

## 5. Log-Partition Function

Log-partition function：

```math
a(\eta)=\log Z(\eta)
```

它的角色包括：

* 确保 probability distribution normalized；
* 通过 gradient 给出 sufficient statistic 的 mean；
* 通过 Hessian 给出 sufficient statistic 的 covariance；
* 决定 negative log likelihood 的 convex geometry。

因此 $a(\eta)$ 是 exponential family 的数学引擎。

## 6. Base Measure

Base measure $b(y)$ 收集与 $\eta$ 无关的部分。它不影响对 $\eta$ 的 gradient，但会影响完整 probability mass/density、support 和 likelihood value。

Examples：

```math
b_{\mathrm{Gaussian}}(y)
=
\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{y^2}{2}\right)
```

```math
b_{\mathrm{Bernoulli}}(y)=1
```

```math
b_{\mathrm{Poisson}}(y)=\frac{1}{y!}
```

## 7. Iid Factorization and Sufficiency

对 iid data：

```math
p(y^{(1)},\ldots,y^{(m)};\eta)
=
\prod_{i=1}^{m}
b(y^{(i)})
\exp\left(\eta^TT(y^{(i)})-a(\eta)\right)
```

整理：

```math
=
\left(\prod_{i=1}^{m}b(y^{(i)})\right)
\exp\left(
\eta^T\sum_{i=1}^{m}T(y^{(i)})
-ma(\eta)
\right)
```

所有依赖 $\eta$ 的 data information 都通过：

```math
\sum_{i=1}^{m}T(y^{(i)})
```

进入 likelihood。这就是 factorization/sufficiency interpretation：给定这个 statistic 后，样本中关于 $\eta$ 的信息已经被压缩。

## 8. Modeling Lesson

Exponential family 不是“所有分布的集合”。它是一类具有特殊 normalization、moment 和 convexity 结构的分布族。GLM 借用这类结构，把 response distribution 和 linear predictor 系统地连接起来。

