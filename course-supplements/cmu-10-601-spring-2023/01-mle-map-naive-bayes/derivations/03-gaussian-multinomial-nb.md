# Gaussian 与 Multinomial Naive Bayes

返回 [Module 01](../README.md)。

CS229 连接：[Lecture 5 GDA/QDA](../../../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#13-qda-unequal-covariance-and-quadratic-boundary) 已经说明 Gaussian covariance assumptions 如何影响边界形状。本文件补充 Gaussian NB 的 diagonal covariance case，以及 Multinomial NB 的 text count event model。

来源边界：这是本仓库的独立推导，参考 CMU 10-601 Spring 2023 Lecture 17、Cohen 10-601 Naive Bayes materials、Tom Mitchell 的 Naive Bayes / Logistic Regression reading，以及公开实现参考。

## 1. Gaussian Naive Bayes Model

令 $X\in\mathbb R^d$，$Y\in\{0,\ldots,K-1\}$。Gaussian NB 假设：

```math
X_j\mid Y=k
\sim
\mathcal N(\mu_{jk},\sigma_{jk}^2).
```

并且给定 class 后 features 条件独立：

```math
P(X=x\mid Y=k)
=
\prod_{j=1}^{d}
P(X_j=x_j\mid Y=k).
```

因此：

```math
p(x\mid Y=k)
=
\prod_{j=1}^{d}
\mathcal N(x_j;\mu_{jk},\sigma_{jk}^2).
```

这等价于 diagonal covariance 的 multivariate Gaussian：

```math
X\mid Y=k
\sim
\mathcal N(\mu_k,\Sigma_k),
```

其中：

```math
\Sigma_k
=
\mathrm{diag}
(
\sigma_{1k}^2,\ldots,\sigma_{dk}^2
).
```

## 2. Gaussian NB MLE

class count：

```math
N_k
=
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}.
```

mean estimator：

```math
\hat\mu_{jk}
=
\frac{1}{N_k}
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}
x_j^{(i)}.
```

variance MLE：

```math
\hat\sigma_{jk}^2
=
\frac{1}{N_k}
\sum_{i=1}^{m}
\mathbf{1}\{y^{(i)}=k\}
(
x_j^{(i)}-\hat\mu_{jk}
)^2.
```

class prior：

```math
\hat\pi_k
=
\frac{N_k}{m}.
```

实现上可维护：

```text
count[k]
sum_x[k, j]
sum_x2[k, j]
```

再由：

```math
\hat\sigma_{jk}^2
=
\frac{\mathrm{sum\_x2}_{kj}}{N_k}
-
\hat\mu_{jk}^2
```

得到 variance。实际代码需要 variance floor，避免数值上出现 $0$ 方差。

## 3. Covariance Constraints

GDA / LDA-style shared-covariance model：

```math
X\mid Y=k
\sim
\mathcal N(\mu_k,\Sigma).
```

QDA class-specific full-covariance model：

```math
X\mid Y=k
\sim
\mathcal N(\mu_k,\Sigma_k).
```

Gaussian NB class-specific diagonal-covariance model：

```math
X\mid Y=k
\sim
\mathcal N
(
\mu_k,
\mathrm{diag}
(
\sigma_{1k}^2,\ldots,\sigma_{dk}^2
)
).
```

三者都属于 Gaussian class-conditional generative classifiers，但 covariance constraints 不同。

| Model | Covariance | $K$ classes 的 covariance 参数规模 |
| --- | --- | --- |
| GDA / LDA-style | shared full $\Sigma$ | $d(d+1)/2$ |
| QDA | class-specific full $\Sigma_k$ | $Kd(d+1)/2$ |
| Gaussian NB | class-specific diagonal $\Sigma_k$ | $Kd$ |

不要不加限定地说 Gaussian NB 是 CS229 GDA 的严格子模型。classical CS229 GDA 共享 full covariance；Gaussian NB 常见形式使用 class-specific diagonal covariance。更准确的比较对象是 Gaussian class-conditional family 中的 parameter constraints。

## 4. Gaussian NB Prediction Score

class score：

```math
s_k(x)
=
\log\pi_k
+
\sum_{j=1}^{d}
\log
\mathcal N(x_j;\mu_{jk},\sigma_{jk}^2).
```

展开 univariate Gaussian log-density：

```math
s_k(x)
=
\log\pi_k
-
\frac{1}{2}
\sum_{j=1}^{d}
\left[
\log(2\pi\sigma_{jk}^2)
+
\frac{(x_j-\mu_{jk})^2}{\sigma_{jk}^2}
\right].
```

若 variances class-specific，boundary 可以含 quadratic terms。若所有 class 共享同一个 diagonal covariance，class-to-class 比较时二次项会抵消，边界变成 linear in $x$。

## 5. Multinomial Naive Bayes Model

文本计数模型中，$X_j$ 是 vocabulary item $j$ 在文档中的出现次数：

```math
X_j
=
\text{count of vocabulary item }j.
```

document length：

```math
N
=
\sum_{j=1}^{d}
X_j.
```

对 class $k$：

```math
X\mid Y=k
\sim
\mathrm{Multinomial}(N,\theta_k),
```

并且：

```math
\sum_{j=1}^{d}
\theta_{jk}
=
1.
```

probability mass：

```math
p(x\mid Y=k)
=
\frac{N!}{\prod_{j=1}^{d}x_j!}
\prod_{j=1}^{d}
\theta_{jk}^{x_j}.
```

预测同一个 document $x$ 时，组合系数 $N!/\prod_jx_j!$ 与 class 无关，不影响 $\mathrm{argmax}$。

## 6. Multinomial NB MLE

定义 class-word count：

```math
C_{jk}
=
\sum_{i:y^{(i)}=k}
x_j^{(i)}.
```

class total count：

```math
C_k
=
\sum_{j=1}^{d}
C_{jk}.
```

class-specific log-likelihood：

```math
\ell(\theta_k)
=
\sum_{j=1}^{d}
C_{jk}\log\theta_{jk}
+
C,
```

约束：

```math
\sum_{j=1}^{d}
\theta_{jk}
=
1.
```

Lagrangian：

```math
\mathcal L(\theta_k,\lambda)
=
\sum_{j=1}^{d}
C_{jk}\log\theta_{jk}
+
\lambda
\left(
\sum_{j=1}^{d}
\theta_{jk}
-
1
\right).
```

stationarity：

```math
\frac{\partial \mathcal L}{\partial \theta_{jk}}
=
\frac{C_{jk}}{\theta_{jk}}
+
\lambda
=
0.
```

于是：

```math
\theta_{jk}
=
-
\frac{C_{jk}}{\lambda}.
```

对 $j$ 求和：

```math
1
=
\sum_{j=1}^{d}
\theta_{jk}
=
-
\frac{1}{\lambda}
\sum_{j=1}^{d}
C_{jk}
=
-
\frac{C_k}{\lambda}.
```

所以 $\lambda=-C_k$，得到：

```math
\hat\theta_{jk}
=
\frac{C_{jk}}{C_k}.
```

实现上就是 class-wise word counts 的归一化。

## 7. Dirichlet MAP 和 Posterior Mean

加入 Dirichlet prior：

```math
\theta_k
\sim
\mathrm{Dirichlet}(\alpha_1,\ldots,\alpha_d).
```

posterior：

```math
\theta_k\mid\mathcal D
\sim
\mathrm{Dirichlet}
(
C_{1k}+\alpha_1,
\ldots,
C_{dk}+\alpha_d
).
```

interior MAP：

```math
\hat\theta_{jk,\mathrm{MAP}}
=
\frac{
C_{jk}+\alpha_j-1
}{
C_k+\sum_{\ell=1}^{d}\alpha_{\ell}-d
}.
```

posterior mean：

```math
E[\theta_{jk}\mid\mathcal D]
=
\frac{
C_{jk}+\alpha_j
}{
C_k+\sum_{\ell=1}^{d}\alpha_{\ell}
}.
```

和 Beta-Bernoulli 一样，MAP 与 posterior mean 不是同一个估计器。

## 8. Implementation Consequence

Multinomial NB 的训练核心可以写成：

```text
for each document i:
    k = y[i]
    class_count[k] += 1
    word_count[k, :] += x_counts[i, :]
```

如果使用 sparse matrix，`word_count` 可以由 class mask 加矩阵聚合得到。预测时：

```text
score = X_counts @ log_theta.T + log_pi
```

这就是 CMU 视角中的重点：模型假设直接决定存储结构、训练复杂度和预测的向量化形式。
