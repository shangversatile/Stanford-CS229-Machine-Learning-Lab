# Excess Risk and Consistency

返回 [Module 01](../README.md)。

来源边界：本文件覆盖 MIT 9.520 / 6.860 Class 02 中与 excess risk、consistency 和 learning target 有关的基础内容。它只建立定义和分解，不进入后续 learning-theory bound。

## 1. Population Benchmark

Expected risk 为：

```math
R(f)
=
\mathbb E_{(X,Y)\sim\rho}
\left[
\ell(f(X),Y)
\right].
```

理想 population-level predictor 写成：

```math
f^*
\in
\underset{f}{\mathrm{argmin}}
\,
R(f).
```

这里的 minimization 是在足够大的函数集合中进行，通常不是学习算法实际能搜索的集合。$f^*$ 表示由 loss 和 distribution $\rho$ 决定的理论最佳目标。

## 2. Hypothesis-Space Benchmark

学习算法通常只能在 hypothesis space $\mathcal H$ 中选函数。因此还需要定义：

```math
f_{\mathcal H}^*
\in
\underset{f\in\mathcal H}{\mathrm{argmin}}
\,
R(f).
```

$f_{\mathcal H}^*$ 是 $\mathcal H$ 内的 population-best predictor。它和 empirical minimizer 不同，因为它仍然依赖 unknown $\rho$，不能直接由训练集计算。

ERM 输出：

```math
\hat f_n
\in
\underset{f\in\mathcal H}{\mathrm{argmin}}
\,
\widehat R_n(f).
```

因此有三类函数：

| Symbol | 由什么决定 | 是否可由训练算法直接得到 |
| --- | --- | --- |
| $f^*$ | $\rho$、loss、unrestricted function class | 否 |
| $f_{\mathcal H}^*$ | $\rho$、loss、chosen $\mathcal H$ | 否 |
| $\hat f_n$ | sample $S$、loss、chosen $\mathcal H$、algorithm | 是 |

## 3. Excess Risk

Excess risk 衡量 learned predictor 相对理想 population benchmark 的损失差：

```math
R(\hat f_n)-R(f^*).
```

这个量是 statistical learning 的核心评价对象，因为它直接比较 population risk，而不是 training risk。

若只关心 $\mathcal H$ 内可达到的目标，也可以看：

```math
R(\hat f_n)-R(f_{\mathcal H}^*).
```

它衡量算法和有限样本造成的损失，而不惩罚 $\mathcal H$ 本身表达能力不足。

## 4. Exact Decomposition

从 total excess risk 开始：

```math
R(\hat f_n)-R(f^*).
```

加减同一个项 $R(f_{\mathcal H}^*)$：

```math
R(\hat f_n)-R(f^*)
=
R(\hat f_n)-R(f_{\mathcal H}^*)
+
R(f_{\mathcal H}^*)-R(f^*).
```

写成分组形式：

```math
R(\hat f_n)-R(f^*)
=
\left[
R(\hat f_n)-R(f_{\mathcal H}^*)
\right]
+
\left[
R(f_{\mathcal H}^*)-R(f^*)
\right].
```

第一项是 estimation / finite-sample limitation。第二项是 approximation limitation。

这是一条恒等式，不是 bound。它只把误差来源分开。

## 5. Approximation Limitation

Approximation limitation：

```math
R(f_{\mathcal H}^*)-R(f^*).
```

它来自 hypothesis space 的表达限制。若 $\mathcal H$ 太窄，即使无限数据也无法达到 $f^*$ 的 risk。

CS229 例子：

| Model | Approximation limitation |
| --- | --- |
| Linear regression | true conditional mean 非线性，而 $\mathcal H$ 只含 linear maps |
| Logistic regression | true log-odds 非线性，而 score restricted to $w^Tx$ |
| GDA | true class-conditional distribution 非 Gaussian |
| Naive Bayes | features 在 given class 后仍强相关 |

这类误差不能靠把 empirical optimization 做得更精确来解决。需要改变 feature representation、model family 或 hypothesis space。

## 6. Estimation Limitation

Estimation limitation：

```math
R(\hat f_n)-R(f_{\mathcal H}^*).
```

它来自有限样本。即使 $\mathcal H$ 中存在好的 predictor，ERM 也只能通过 $\widehat R_n$ 间接选择。样本不同，$\hat f_n$ 也可能不同。

这个项和 generalization gap 有关。因为 $\hat f_n$ 是用同一份样本选出来的，不能只用 fixed-function law of large numbers 草率替代。

当前只建立直觉：

```text
larger H can reduce approximation limitation
but may increase estimation difficulty.
```

完整 trade-off 的 quantitative theory 留到后续 CS229 regularization / model selection 节点。

## 7. Consistency Relative to H

令算法在样本量 $n$ 时输出 $\hat f_n$。如果：

```math
R(\hat f_n)
\to
R(f_{\mathcal H}^*)
```

当 $n\to\infty$，则可称它相对于 $\mathcal H$ 是 consistent 的一种形式。

收敛方式需要具体说明，常见有：

```text
in probability
in expectation
almost surely
```

本层不固定某个 theorem 的收敛模式，只强调 convergence target 必须是 population risk benchmark，而不是 training loss。

## 8. Consistency Toward the Bayes / Unrestricted Target

如果进一步希望：

```math
R(\hat f_n)
\to
R(f^*),
```

则要求不仅 estimation limitation 消失，approximation limitation 也要消失。固定的 linear hypothesis space 通常不能保证这一点，因为它可能永远表达不了 $f^*$。

若随 $n$ 增长扩大 hypothesis space：

```math
\mathcal H_1
\subseteq
\mathcal H_2
\subseteq
\cdots,
```

则有可能让 approximation limitation 降低。但 $\mathcal H_n$ 增长过快又可能让 estimation 变难。这是后续 theory 的核心 tension，当前不展开。

## 9. Consistency Is Not Training Error Convergence

Training error 收敛到很小不等于 consistency。一个高复杂度模型可以让：

```math
\widehat R_n(\hat f_n)
\approx
0
```

但如果它只记住样本噪声，$R(\hat f_n)$ 仍可能很高。

Consistency 的对象是：

```math
R(\hat f_n),
```

不是：

```math
\widehat R_n(\hat f_n).
```

这就是 MIT Class 02 对 CS229 早期模型的关键补充：优化 training objective 只是中间步骤，学习成功要回到 population risk。

## 10. Connection to Regularization

Regularization 可以影响两个误差项：

* 它可能增加 approximation limitation，因为更强限制会排除一些 low-risk functions；
* 它可能降低 estimation limitation，因为较小或更稳定的 effective hypothesis space 更容易从有限样本中学习。

因此 regularization 的理论角色不是“给 loss 加惩罚项”这么浅。它是在 finite-sample learning 中控制 approximation 与 estimation 张力的一种结构选择。

## 11. Current Scope

本文件完成：

```text
f*
f_H*
excess risk
approximation limitation
estimation limitation
consistency target
training risk vs population risk distinction
```

不包含后续 generalization bound。
