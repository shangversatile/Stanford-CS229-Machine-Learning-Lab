# Bernoulli Naive Bayes 参数估计与 Logistic Posterior

返回 [Module 01](../README.md)。

CS229 连接：[Lecture 5 Naive Bayes](../../../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#15-naive-bayes-for-discrete-features) 给出 CS229 主线的 factorization。本文件补全 CMU 10-601 更强调的 count-based estimation、MAP smoothing intuition 和 Naive Bayes linearity。

来源边界：这是本仓库的独立推导，参考 CMU 10-601 Spring 2023 Lecture 17、Cohen 10-601 Naive Bayes materials，以及 Tom Mitchell 的 Naive Bayes / Logistic Regression reading。

## 1. Model

令 $Y\in\{0,1,\ldots,K-1\}$，$X\in\{0,1\}^{d}$。class prior：

```math
P(Y=k)
=
\pi_k.
```

对 class $k$ 下的 binary feature：

```math
\phi_{jk}
=
P(X_j=1\mid Y=k).
```

Naive Bayes assumption：

```math
P(X=x\mid Y=k)
=
\prod_{j=1}^{d}
P(X_j=x_j\mid Y=k).
```

因此：

```math
P(X=x\mid Y=k)
=
\prod_{j=1}^{d}
\phi_{jk}^{x_j}
(1-\phi_{jk})^{1-x_j}.
```

joint model：

```math
P(X=x,Y=k)
=
\pi_k
\prod_{j=1}^{d}
\phi_{jk}^{x_j}
(1-\phi_{jk})^{1-x_j}.
```

## 2. Parameter Counting

没有 conditional independence 时，$P(X\mid Y=k)$ 是定义在 $2^d$ 个 binary configurations 上的 categorical distribution。概率和为 $1$，因此每个 class 需要：

```math
2^d-1
```

个自由参数。

Bernoulli NB 只需要每个 class 的 $d$ 个 feature probabilities：

```math
\phi_{1k},\ldots,\phi_{dk}.
```

class prior 另有 $K-1$ 个独立自由参数。参数规模从 exponential in $d$ 变为 linear in $d$，这正是 CMU 强调的可实现性来源。

## 3. Class Prior MLE

定义：

```math
N_k
=
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}.
```

class-prior likelihood factor：

```math
\prod_{i=1}^{m}
\pi_{y^{(i)}}.
```

在约束 $\sum_k\pi_k=1$ 下最大化 $\sum_kN_k\log\pi_k$，得到：

```math
\hat\pi_k
=
\frac{N_k}{m}.
```

## 4. Feature Parameter MLE

固定 feature $j$ 和 class $k$。只看 $y^{(i)}=k$ 的样本：

```math
N_{jk,1}
=
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}
x_j^{(i)}.
```

并定义：

```math
N_{jk,0}
=
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}
(1-x_j^{(i)}).
```

对于该 feature/class pair：

```math
N_k
=
N_{jk,1}+N_{jk,0}.
```

相关 log-likelihood：

```math
\ell(\phi_{jk})
=
N_{jk,1}\log\phi_{jk}
+
N_{jk,0}\log(1-\phi_{jk}).
```

由 Bernoulli MLE：

```math
\hat\phi_{jk}
=
\frac{
N_{jk,1}
}{
N_k
}.
```

展开为样本求和：

```math
\hat\phi_{jk}
=
\frac{
\sum_{i=1}^{m}
\mathbf{1}
\{
y^{(i)}=k
\}
x_j^{(i)}
}{
\sum_{i=1}^{m}
\mathbf{1}
\{
y^{(i)}=k
\}
}.
```

编程对应：

```text
class_count[k] = number of examples with y == k
feature_count[k, j] = number of class-k examples with feature j present
phi[k, j] = feature_count[k, j] / class_count[k]
```

## 5. Feature Parameter MAP

假设 independent Beta priors：

```math
\phi_{jk}
\sim
\mathrm{Beta}(\alpha,\beta).
```

posterior shape parameters：

```math
N_{jk,1}+\alpha,
\quad
N_{jk,0}+\beta.
```

interior MAP：

```math
\hat\phi_{jk,\mathrm{MAP}}
=
\frac{
N_{jk,1}+\alpha-1
}{
N_k+\alpha+\beta-2
}.
```

这个 prior 会把估计从极端 $0$ 或 $1$ 拉回来，但 boundary condition 仍然要检查。它给出 smoothing intuition，却不等同于任意 additive smoothing 公式。

## 6. Prediction in Log Space

Bayes rule：

```math
P(Y=k\mid X=x)
=
\frac{
\pi_kp(x\mid Y=k)
}{
\sum_{\ell}
\pi_{\ell}p(x\mid Y=\ell)
}.
```

分母对所有 candidate class 相同，所以：

```math
\hat y
=
\underset{k}{\mathrm{argmax}}
\,
\pi_kp(x\mid Y=k).
```

log-space：

```math
\hat y
=
\underset{k}{\mathrm{argmax}}
\,
\left[
\log\pi_k
+
\sum_{j=1}^{d}
x_j\log\phi_{jk}
+
(1-x_j)\log(1-\phi_{jk})
\right].
```

这是实现中的默认形式，避免很多小概率相乘导致 underflow。

## 7. Logistic Posterior

二分类时 $Y\in\{0,1\}$：

```math
\log
\frac{
P(Y=1\mid x)
}{
P(Y=0\mid x)
}
=
\log
\frac{
\pi_1p(x\mid Y=1)
}{
\pi_0p(x\mid Y=0)
}.
```

代入 Bernoulli NB：

```math
=
\log\frac{\pi_1}{\pi_0}
+
\sum_{j=1}^{d}
\left[
x_j\log\frac{\phi_{j1}}{\phi_{j0}}
+
(1-x_j)\log\frac{1-\phi_{j1}}{1-\phi_{j0}}
\right].
```

展开：

```math
=
\log\frac{\pi_1}{\pi_0}
+
\sum_{j=1}^{d}
\log\frac{1-\phi_{j1}}{1-\phi_{j0}}
+
\sum_{j=1}^{d}
x_j
\left[
\log\frac{\phi_{j1}}{\phi_{j0}}
-
\log\frac{1-\phi_{j1}}{1-\phi_{j0}}
\right].
```

定义 bias：

```math
b
=
\log\frac{\pi_1}{\pi_0}
+
\sum_{j=1}^{d}
\log\frac{1-\phi_{j1}}{1-\phi_{j0}}.
```

定义 weight：

```math
w_j
=
\log
\frac{
\phi_{j1}(1-\phi_{j0})
}{
\phi_{j0}(1-\phi_{j1})
}.
```

得到：

```math
\log
\frac{
P(Y=1\mid x)
}{
P(Y=0\mid x)
}
=
w^Tx+b.
```

所以：

```math
P(Y=1\mid x)
=
\sigma(w^Tx+b).
```

Bernoulli NB 可以诱导 logistic-form posterior；但参数 $w,b$ 是由 generative counts 派生出来的，不是 logistic regression 通过 conditional likelihood 直接训练出来的。

## 8. Implementation Consequence

如果只需要分类边界，训练后可以预计算 $w$ 和 $b$。对 batch $X$：

```text
score = X @ w + b
prediction = score >= 0
```

这解释了 Cohen 讲义中 “Naive Bayes is linear” 的含义：线性形式来自 generative model 的 posterior algebra，而不是来自直接拟合线性分类器。
