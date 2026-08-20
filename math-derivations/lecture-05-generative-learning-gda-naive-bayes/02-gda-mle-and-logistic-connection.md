# GDA MLE and Logistic Connection

Cross-link: see [Lecture 5 Section 9: GDA Model and Generative Story](../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#9-gda-model-and-generative-story), [Lecture 5 Section 10: GDA Joint Likelihood and MLE](../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#10-gda-joint-likelihood-and-mle), and [Lecture 5 Section 11: GDA Posterior Has Logistic Form](../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#11-gda-posterior-has-logistic-form).

## 1. Model and Data

Training set:

```math
\mathcal D=\{(x^{(i)},y^{(i)})\}_{i=1}^m.
```

其中：

```math
x^{(i)}\in\mathbb R^d,\quad y^{(i)}\in\{0,1\}.
```

GDA model:

```math
Y\sim\mathrm{Bernoulli}(\phi).
```

```math
X\mid Y=0\sim\mathcal N(\mu_0,\Sigma).
```

```math
X\mid Y=1\sim\mathcal N(\mu_1,\Sigma).
```

用 indicator 记号：

```math
\mathbf{1}\{A\}=\begin{cases}1,&\text{if }A\text{ is true}\\0,&\text{otherwise}\end{cases}.
```

定义 class counts：

```math
m_1=\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}.
```

```math
m_0=\sum_{i=1}^m\mathbf{1}\{y^{(i)}=0\}.
```

显然：

```math
m_0+m_1=m.
```

下面的 MLE 推导默认两个 classes 都至少出现一次：

```math
m_0>0,\quad m_1>0.
```

如果某个 class 在 training set 中没有样本，对应的 class mean 不能从数据中估计；这不是 GDA 公式的细节问题，而是参数没有被 likelihood 识别出来。

## 2. GDA MLE without Skipped Steps

Joint likelihood:

```math
L(\phi,\mu_0,\mu_1,\Sigma)=\prod_{i=1}^m p(x^{(i)},y^{(i)}).
```

Generative factorization:

```math
p(x^{(i)},y^{(i)})=p(x^{(i)}\mid y^{(i)})p(y^{(i)}).
```

Log likelihood:

```math
\ell(\phi,\mu_0,\mu_1,\Sigma)=\sum_{i=1}^m\log p(y^{(i)};\phi)+\sum_{i=1}^m\log p(x^{(i)}\mid y^{(i)};\mu_0,\mu_1,\Sigma).
```

把 Bernoulli prior 和 Gaussian class-conditional density 全部代入。由于 $y^{(i)}\in\{0,1\}$，prior term 是：

```math
\sum_{i=1}^m\left[y^{(i)}\log\phi+(1-y^{(i)})\log(1-\phi)\right].
```

Gaussian term 可以用 indicators 统一写成：

```math
\sum_{i=1}^m\mathbf{1}\{y^{(i)}=0\}\log p(x^{(i)};\mu_0,\Sigma)
+\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}\log p(x^{(i)};\mu_1,\Sigma).
```

展开 density 后得到：

```math
\ell
=
\sum_{i=1}^m\left[y^{(i)}\log\phi+(1-y^{(i)})\log(1-\phi)\right]
-\frac{md}{2}\log(2\pi)
-\frac m2\log|\Sigma|
```

```math
-\frac12\sum_{i=1}^m
\left[
\mathbf{1}\{y^{(i)}=0\}
\left(x^{(i)}-\mu_0\right)^\top\Sigma^{-1}\left(x^{(i)}-\mu_0\right)
+\mathbf{1}\{y^{(i)}=1\}
\left(x^{(i)}-\mu_1\right)^\top\Sigma^{-1}\left(x^{(i)}-\mu_1\right)
\right].
```

这个展开式也说明了参数如何分离：$\phi$ 只出现在 Bernoulli prior term 中；$\mu_0,\mu_1$ 只出现在各自 class 的 quadratic residual term 中；$\Sigma$ 同时接收两个 classes 的 residual information。

### 2.1 MLE for the Class Prior

Bernoulli probability 可以写成：

```math
p(y;\phi)=\phi^y(1-\phi)^{1-y}.
```

因此 prior 部分的 log likelihood 是：

```math
\ell_\phi=\sum_{i=1}^m\left[y^{(i)}\log\phi+(1-y^{(i)})\log(1-\phi)\right].
```

用 $m_1=\sum_i y^{(i)}$ 和 $m_0=m-m_1$：

```math
\ell_\phi=m_1\log\phi+m_0\log(1-\phi).
```

求导：

```math
\frac{\partial\ell_\phi}{\partial\phi}=\frac{m_1}{\phi}-\frac{m_0}{1-\phi}.
```

令 derivative 为 $0$：

```math
\frac{m_1}{\phi}=\frac{m_0}{1-\phi}.
```

交叉相乘：

```math
m_1(1-\phi)=m_0\phi.
```

展开：

```math
m_1-m_1\phi=m_0\phi.
```

合并：

```math
m_1=(m_0+m_1)\phi.
```

由于 $m_0+m_1=m$：

```math
\hat\phi=\frac{m_1}{m}.
```

也就是：

```math
\hat\phi=\frac1m\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}.
```

### 2.2 MLE for Class Means

令：

```math
A=\Sigma^{-1}.
```

对 class $k\in\{0,1\}$，只看依赖 $\mu_k$ 的部分：

```math
\ell_{\mu_k}=-\frac12\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}\left(x^{(i)}-\mu_k\right)^\top A\left(x^{(i)}-\mu_k\right).
```

对单个 residual 定义：

```math
r_i=x^{(i)}-\mu_k.
```

因为 $A$ symmetric：

```math
\frac{\partial}{\partial\mu_k}\left(r_i^\top A r_i\right)=-2A r_i.
```

所以：

```math
\nabla_{\mu_k}\ell_{\mu_k}=\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}A\left(x^{(i)}-\mu_k\right).
```

令 gradient 为 $0$：

```math
\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}A\left(x^{(i)}-\mu_k\right)=0.
```

由于 $A$ invertible，可以左乘 $A^{-1}$：

```math
\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}\left(x^{(i)}-\mu_k\right)=0.
```

展开：

```math
\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}x^{(i)}-\mu_k\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}=0.
```

因此：

```math
\hat\mu_k=\frac{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}x^{(i)}}{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}}.
```

写成两个 classes：

```math
\hat\mu_0=\frac{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=0\}x^{(i)}}{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=0\}}.
```

```math
\hat\mu_1=\frac{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}x^{(i)}}{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}}.
```

这说明 GDA 的 class mean MLE 就是每个 class 内的 empirical mean。

### 2.3 MLE for Shared Pooled Covariance

在 $\mu_0,\mu_1$ 固定时，定义：

```math
r_i=x^{(i)}-\mu_{y^{(i)}}.
```

并定义 residual scatter matrix：

```math
S=\sum_{i=1}^m r_i r_i^\top.
```

只看依赖 $\Sigma$ 的 Gaussian log likelihood 部分：

```math
\ell_\Sigma=-\frac m2\log|\Sigma|-\frac12\sum_{i=1}^m r_i^\top\Sigma^{-1}r_i+\mathrm{constant}.
```

用 trace identity：

```math
r_i^\top\Sigma^{-1}r_i=\mathrm{tr}\left(\Sigma^{-1}r_i r_i^\top\right).
```

所以：

```math
\sum_{i=1}^m r_i^\top\Sigma^{-1}r_i=\mathrm{tr}\left(\Sigma^{-1}\sum_{i=1}^m r_i r_i^\top\right)=\mathrm{tr}\left(\Sigma^{-1}S\right).
```

直接对 $\Sigma$ 求导也可以，但对 precision matrix 更清楚。令：

```math
A=\Sigma^{-1}.
```

则：

```math
\log|\Sigma|=-\log|A|.
```

因此：

```math
\ell_A=\frac m2\log|A|-\frac12\mathrm{tr}(AS)+\mathrm{constant}.
```

使用 differentials：

```math
d\log|A|=\mathrm{tr}(A^{-1}dA).
```

以及：

```math
d\,\mathrm{tr}(AS)=\mathrm{tr}(S\,dA).
```

于是：

```math
d\ell_A=\frac m2\mathrm{tr}(A^{-1}dA)-\frac12\mathrm{tr}(S\,dA).
```

把两项合并成同一个 trace inner product：

```math
d\ell_A=\mathrm{tr}\left(\left(\frac m2A^{-1}-\frac12S\right)dA\right).
```

令 first-order condition 为 $0$，对任意 symmetric perturbation $dA$ 都必须成立，所以：

```math
\frac m2A^{-1}-\frac12S=0.
```

乘以 $2$：

```math
mA^{-1}=S.
```

因为 $A^{-1}=\Sigma$：

```math
\hat\Sigma=\frac1m S.
```

代回 $S$：

```math
\hat\Sigma=\frac1m\sum_{i=1}^m\left(x^{(i)}-\hat\mu_{y^{(i)}}\right)\left(x^{(i)}-\hat\mu_{y^{(i)}}\right)^\top.
```

这里 denominator 是 $m$，不是 $m_0$ 或 $m_1$，因为模型只有一个 shared covariance parameter。所有 classes 的 residuals 都被 pooled 到同一个 $\Sigma$ 中。这里也不是 unbiased covariance estimation，而是 maximum likelihood estimation。

更具体地说，GDA 的 class-conditional sampling story 中，每个 training example 的 $X^{(i)}\mid Y^{(i)}$ 都共享同一个 covariance $\Sigma$。因此 likelihood 中有 $m$ 个 Gaussian conditional draws 对同一个 $\Sigma$ 提供信息，log determinant term 也是 $-\frac m2\log|\Sigma|$。这就是 first-order condition 给出 $S/m$ 的原因。

如果模型改成 separate covariance：

```math
X\mid Y=k\sim\mathcal N(\mu_k,\Sigma_k),
```

那么每个 $\Sigma_k$ 只由 class $k$ 内的 residuals 决定，MLE 会变成：

```math
\hat\Sigma_k=\frac1{m_k}\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}
\left(x^{(i)}-\hat\mu_k\right)\left(x^{(i)}-\hat\mu_k\right)^\top.
```

GDA 使用 pooled covariance 不是因为两个 empirical class covariances 必然相等，而是因为模型假设它们由同一个 population covariance 生成。

## 3. GDA Prediction Rule

GDA 的 posterior 是：

```math
P(Y=y\mid X=x)=\frac{p(x\mid Y=y)P(Y=y)}{p(x)}.
```

对 classification，只比较不同 $y$ 的 numerator：

```math
\hat y=\underset{y\in\{0,1\}}{\mathrm{argmax}}\ p(x\mid Y=y)P(Y=y).
```

这条 rule 不是 heuristic，而是 Bayes classifier 的直接结果。

## 4. Posterior Odds Expansion

从 odds 开始：

```math
\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=\frac{p(x\mid Y=1)P(Y=1)}{p(x\mid Y=0)P(Y=0)}.
```

取 log：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=\log p(x\mid Y=1)-\log p(x\mid Y=0)+\log\frac{\phi}{1-\phi}.
```

Shared covariance GDA 给出：

```math
\log p(x\mid Y=k)=-\frac d2\log(2\pi)-\frac12\log|\Sigma|-\frac12(x-\mu_k)^\top\Sigma^{-1}(x-\mu_k).
```

因此 common normalizing terms 抵消：

```math
\log p(x\mid Y=1)-\log p(x\mid Y=0)=-\frac12(x-\mu_1)^\top\Sigma^{-1}(x-\mu_1)+\frac12(x-\mu_0)^\top\Sigma^{-1}(x-\mu_0).
```

展开：

```math
(x-\mu_k)^\top\Sigma^{-1}(x-\mu_k)=x^\top\Sigma^{-1}x-2\mu_k^\top\Sigma^{-1}x+\mu_k^\top\Sigma^{-1}\mu_k.
```

代入 $k=1$ 和 $k=0$：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=-\frac12x^\top\Sigma^{-1}x+\mu_1^\top\Sigma^{-1}x-\frac12\mu_1^\top\Sigma^{-1}\mu_1
```

```math
+\frac12x^\top\Sigma^{-1}x-\mu_0^\top\Sigma^{-1}x+\frac12\mu_0^\top\Sigma^{-1}\mu_0+\log\frac{\phi}{1-\phi}.
```

Quadratic terms 抵消：

```math
-\frac12x^\top\Sigma^{-1}x+\frac12x^\top\Sigma^{-1}x=0.
```

剩下：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=(\mu_1-\mu_0)^\top\Sigma^{-1}x-\frac12\mu_1^\top\Sigma^{-1}\mu_1+\frac12\mu_0^\top\Sigma^{-1}\mu_0+\log\frac{\phi}{1-\phi}.
```

定义：

```math
w=\Sigma^{-1}(\mu_1-\mu_0).
```

和：

```math
b=-\frac12\mu_1^\top\Sigma^{-1}\mu_1+\frac12\mu_0^\top\Sigma^{-1}\mu_0+\log\frac{\phi}{1-\phi}.
```

得到：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=w^\top x+b.
```

## 5. Logistic Form

令：

```math
s=w^\top x+b.
```

则：

```math
\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=\exp(s).
```

记：

```math
\pi(x)=P(Y=1\mid X=x).
```

那么：

```math
P(Y=0\mid X=x)=1-\pi(x).
```

于是：

```math
\frac{\pi(x)}{1-\pi(x)}=\exp(s).
```

解这个方程：

```math
\pi(x)=\exp(s)(1-\pi(x)).
```

```math
\pi(x)=\exp(s)-\exp(s)\pi(x).
```

```math
\pi(x)(1+\exp(s))=\exp(s).
```

```math
\pi(x)=\frac{\exp(s)}{1+\exp(s)}.
```

等价地：

```math
\pi(x)=\frac{1}{1+\exp(-s)}.
```

代回 $s=w^\top x+b$：

```math
P(Y=1\mid X=x)=\frac{1}{1+\exp[-(w^\top x+b)]}.
```

这就是 GDA posterior 的 logistic form。

## 6. Why the Converse Does Not Hold

上面的推导证明的是：

```math
\text{shared-covariance Gaussian class-conditionals}\quad\Longrightarrow\quad\text{logistic-form posterior}.
```

它没有证明反方向。也就是说，看到

```math
P(Y=1\mid X=x)=\frac1{1+\exp[-(w^\top x+b)]}
```

并不能推出 $X\mid Y=0$ 和 $X\mid Y=1$ 一定是 Gaussian，甚至不能推出它们有 shared covariance。

一个简单例子是 binary features 的 Naive Bayes。它的 posterior log-odds 也可以写成 linear form：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}
=b_{\mathrm{NB}}+\sum_{j=1}^d w_j^{\mathrm{NB}}x_j.
```

但这里的 $X_j\mid Y$ 是 Bernoulli distribution，不是 Gaussian。更一般地，很多不同的 generative assumptions 都可能诱导出 logistic-looking posterior。Logistic regression 直接把这个 posterior form 当作 conditional model；GDA 则从更强的 Gaussian class-conditional model 推导出它。

## 7. Boundary Shape and QDA Contrast

Decision boundary 满足：

```math
P(Y=1\mid X=x)=P(Y=0\mid X=x).
```

等价于：

```math
w^\top x+b=0.
```

所以 shared-covariance GDA 的 boundary 是 linear hyperplane。

如果改成：

```math
X\mid Y=0\sim\mathcal N(\mu_0,\Sigma_0)
```

和：

```math
X\mid Y=1\sim\mathcal N(\mu_1,\Sigma_1),
```

log density ratio 中的二次项变为：

```math
-\frac12x^\top\Sigma_1^{-1}x+\frac12x^\top\Sigma_0^{-1}x.
```

当 $\Sigma_0\neq\Sigma_1$ 时：

```math
-\frac12\Sigma_1^{-1}+\frac12\Sigma_0^{-1}\neq0
```

一般成立，所以 quadratic part 留下，boundary 一般是 quadratic。GDA 的 linearity 来自 shared covariance 的 cancellation，不来自 Gaussian 这个词本身。
