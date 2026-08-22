# Naive Bayes Factorization and MLE

Cross-link: see [Lecture 5 Section 15: Naive Bayes for Discrete Features](../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#15-naive-bayes-for-discrete-features), [Lecture 5 Section 16: Conditional Independence](../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#16-conditional-independence), [Lecture 5 Section 17: Naive Bayes Parameters and MLE](../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#17-naive-bayes-parameters-and-mle), and [Lecture 5 Section 18: Naive Bayes Prediction](../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#18-naive-bayes-prediction).

## 1. Discrete Feature Setup

Lecture 5 的 Naive Bayes 从 text classification 的 binary feature model 开始。Label 是：

```math
Y\in\{0,1\}.
```

Feature vector 是：

```math
X=(X_1,\ldots,X_d).
```

设 vocabulary：

```math
V=\{v_1,\ldots,v_d\}.
```

每个 coordinate 是 word-presence indicator：

```math
X_j
=
\mathbf1
\{
v_j
\text{ appears in the email}
\}.
```

Realized value $x_j=1$ 表示第 $j$ 个 word 出现；$x_j=0$ 表示没有出现。

如果不加结构地估计 $p(X\mid Y=k)$，一般要处理：

```math
2^d
```

种 binary feature configurations。Naive Bayes 的 conditional-independence assumption 正是为了解决这个 high-dimensional discrete distribution 不可直接估计的问题。

## 2. Chain Rule before the Naive Assumption

不要直接从 joint conditional 跳到 product。先写 chain rule：

```math
p(x_1,\ldots,x_d\mid y)=p(x_1\mid y)p(x_2\mid x_1,y)p(x_3\mid x_1,x_2,y)\cdots p(x_d\mid x_1,\ldots,x_{d-1},y).
```

更紧凑地：

```math
p(x_1,\ldots,x_d\mid y)=\prod_{j=1}^d p(x_j\mid x_1,\ldots,x_{j-1},y).
```

这个等式只是 probability chain rule，没有任何 independence assumption。

## 3. Conditional Independence Assumption

Naive Bayes assumption 是：

```math
X_1,\ldots,X_d\text{ are conditionally independent given }Y.
```

也就是对每个 $j$：

```math
p(x_j\mid x_1,\ldots,x_{j-1},y)=p(x_j\mid y).
```

把这个 assumption 代入 chain rule：

```math
p(x_1,\ldots,x_d\mid y)=\prod_{j=1}^d p(x_j\mid y).
```

这不是 unconditional independence。Marginally，features 仍可以相关：

```math
X_j\not\!\perp X_k.
```

但模型假设 conditioned on $Y$ 后：

```math
X_j\perp X_k\mid Y.
```

共同 class label 可以解释一部分 marginal correlation。Naive Bayes 的强假设是：知道 class 后，features 之间剩余 dependence 被忽略。

可以把这个区别写成一个具体等式。由 total probability：

```math
p(x_j,x_k)=\sum_{y\in\{0,1\}}p(y)p(x_j,x_k\mid y).
```

在 Naive Bayes assumption 下：

```math
p(x_j,x_k)=\sum_{y\in\{0,1\}}p(y)p(x_j\mid y)p(x_k\mid y).
```

而两个 marginal probabilities 是：

```math
p(x_j)=\sum_{y\in\{0,1\}}p(y)p(x_j\mid y),
```

```math
p(x_k)=\sum_{y\in\{0,1\}}p(y)p(x_k\mid y).
```

一般来说：

```math
\sum_{y\in\{0,1\}}p(y)p(x_j\mid y)p(x_k\mid y)
\neq
\left[\sum_{y\in\{0,1\}}p(y)p(x_j\mid y)\right]
\left[\sum_{y\in\{0,1\}}p(y)p(x_k\mid y)\right].
```

所以 $X_j\perp X_k\mid Y$ 并不推出 $X_j\perp X_k$。这也是 text classification 中 Naive Bayes 仍可能合理的原因之一：word features 可以因为共享 topic 或 class 而 marginally correlated，但模型只要求 class 已知以后继续忽略剩余相关。

## 4. Bernoulli Feature Likelihood

定义 class prior：

```math
\phi_y=P(Y=1).
```

因此：

```math
P(Y=0)=1-\phi_y.
```

对每个 feature 和 class，定义：

```math
\phi_{j\mid 1}=P(X_j=1\mid Y=1).
```

```math
\phi_{j\mid 0}=P(X_j=1\mid Y=0).
```

对 $k\in\{0,1\}$：

```math
p(x_j\mid y=k)=\phi_{j\mid k}^{x_j}(1-\phi_{j\mid k})^{1-x_j}.
```

所以一个 example 的 joint probability 是：

```math
p(x,y)=p(y)\prod_{j=1}^d\phi_{j\mid y}^{x_j}(1-\phi_{j\mid y})^{1-x_j}.
```

Dataset likelihood 是：

```math
L=\prod_{i=1}^m p(y^{(i)})\prod_{j=1}^d p(x_j^{(i)}\mid y^{(i)}).
```

Log likelihood 是：

```math
\ell=\sum_{i=1}^m\log p(y^{(i)})+\sum_{i=1}^m\sum_{j=1}^d\log p(x_j^{(i)}\mid y^{(i)}).
```

## 5. MLE for the Class Prior

Prior 部分是 Bernoulli likelihood：

```math
\ell_y=\sum_{i=1}^m\left[y^{(i)}\log\phi_y+(1-y^{(i)})\log(1-\phi_y)\right].
```

令：

```math
m_1=\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}.
```

以及：

```math
m_0=\sum_{i=1}^m\mathbf{1}\{y^{(i)}=0\}.
```

下面的 closed-form MLE 默认：

```math
m_0>0,\quad m_1>0.
```

如果某个 class 没有 training examples，对应的 class-conditional Bernoulli parameters 不能由 likelihood 唯一估计。

则：

```math
\ell_y=m_1\log\phi_y+m_0\log(1-\phi_y).
```

求导：

```math
\frac{\partial\ell_y}{\partial\phi_y}=\frac{m_1}{\phi_y}-\frac{m_0}{1-\phi_y}.
```

令 derivative 为 $0$：

```math
\frac{m_1}{\phi_y}=\frac{m_0}{1-\phi_y}.
```

得到：

```math
\hat\phi_y=\frac{m_1}{m}.
```

也就是：

```math
\hat\phi_y=\frac1m\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}.
```

## 6. MLE for Bernoulli Word-Feature Parameters

固定 feature $j$ 和 class $k$。只取依赖 $\phi_{j\mid k}$ 的 log likelihood 部分：

```math
\ell_{j\mid k}=\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}\left[x_j^{(i)}\log\phi_{j\mid k}+(1-x_j^{(i)})\log(1-\phi_{j\mid k})\right].
```

定义 class $k$ 中 word $j$ 出现的次数：

```math
c_{jk}=\sum_{i=1}^m\mathbf{1}\{x_j^{(i)}=1,y^{(i)}=k\}.
```

Class $k$ 的样本数是：

```math
m_k=\sum_{i=1}^m\mathbf{1}\{y^{(i)}=k\}.
```

因为 $x_j^{(i)}\in\{0,1\}$，class $k$ 中 word $j$ 不出现的次数是：

```math
m_k-c_{jk}.
```

所以：

```math
\ell_{j\mid k}=c_{jk}\log\phi_{j\mid k}+(m_k-c_{jk})\log(1-\phi_{j\mid k}).
```

求导：

```math
\frac{\partial\ell_{j\mid k}}{\partial\phi_{j\mid k}}=\frac{c_{jk}}{\phi_{j\mid k}}-\frac{m_k-c_{jk}}{1-\phi_{j\mid k}}.
```

令 derivative 为 $0$：

```math
\frac{c_{jk}}{\phi_{j\mid k}}=\frac{m_k-c_{jk}}{1-\phi_{j\mid k}}.
```

交叉相乘：

```math
c_{jk}(1-\phi_{j\mid k})=(m_k-c_{jk})\phi_{j\mid k}.
```

展开：

```math
c_{jk}-c_{jk}\phi_{j\mid k}=m_k\phi_{j\mid k}-c_{jk}\phi_{j\mid k}.
```

两边同时消去 $-c_{jk}\phi_{j\mid k}$：

```math
c_{jk}=m_k\phi_{j\mid k}.
```

因此：

```math
\hat\phi_{j\mid k}=\frac{c_{jk}}{m_k}.
```

写成 Lecture 5 的两个 binary classes：

```math
\hat\phi_{j\mid1}=\frac{\sum_{i=1}^m\mathbf{1}\{x_j^{(i)}=1,y^{(i)}=1\}}{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}}.
```

```math
\hat\phi_{j\mid0}=\frac{\sum_{i=1}^m\mathbf{1}\{x_j^{(i)}=1,y^{(i)}=0\}}{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=0\}}.
```

这就是 class-conditional empirical frequency。

## 7. Prediction

Bayes rule 给出：

```math
P(Y=y\mid X=x)=\frac{P(Y=y)P(X=x\mid Y=y)}{P(X=x)}.
```

对 classification，denominator 不依赖 $y$：

```math
\hat y=\underset{y\in\{0,1\}}{\mathrm{argmax}}\ P(Y=y)P(X=x\mid Y=y).
```

代入 Naive Bayes factorization：

```math
\hat y=\underset{y\in\{0,1\}}{\mathrm{argmax}}\ P(Y=y)\prod_{j=1}^dP(X_j=x_j\mid Y=y).
```

对 Bernoulli features：

```math
P(X_j=x_j\mid Y=y)=\phi_{j\mid y}^{x_j}(1-\phi_{j\mid y})^{1-x_j}.
```

所以 score 是：

```math
P(Y=y)\prod_{j=1}^d\phi_{j\mid y}^{x_j}(1-\phi_{j\mid y})^{1-x_j}.
```

## 8. Log-Space Form

实际实现中乘积可能 underflow。由于 log strictly increasing，argmax 不变：

```math
\underset{y}{\mathrm{argmax}}\ a_y=\underset{y}{\mathrm{argmax}}\ \log a_y.
```

因此使用 log score：

```math
s(y)=\log P(Y=y)+\sum_{j=1}^d\log P(X_j=x_j\mid Y=y).
```

对 Bernoulli feature model，进一步写成：

```math
s(y)=\log P(Y=y)+\sum_{j=1}^d\left[x_j\log\phi_{j\mid y}+(1-x_j)\log(1-\phi_{j\mid y})\right].
```

Prediction:

```math
\hat y=\underset{y\in\{0,1\}}{\mathrm{argmax}}\ s(y).
```

## 9. Posterior Log-Odds for Bernoulli Naive Bayes

二分类时，也可以直接比较 posterior odds：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}
=
\log\frac{P(Y=1)}{P(Y=0)}
+
\sum_{j=1}^d
\log\frac{P(X_j=x_j\mid Y=1)}{P(X_j=x_j\mid Y=0)}.
```

代入 Bernoulli feature likelihood：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}
=
\log\frac{\phi_y}{1-\phi_y}
+
\sum_{j=1}^d
\log
\frac{
\phi_{j\mid1}^{x_j}(1-\phi_{j\mid1})^{1-x_j}
}{
\phi_{j\mid0}^{x_j}(1-\phi_{j\mid0})^{1-x_j}
}.
```

把每个 feature 的 contribution 展开：

```math
=
\log\frac{\phi_y}{1-\phi_y}
+
\sum_{j=1}^d
\left[
x_j\log\frac{\phi_{j\mid1}}{\phi_{j\mid0}}
+
(1-x_j)\log\frac{1-\phi_{j\mid1}}{1-\phi_{j\mid0}}
\right].
```

继续把和 $x_j$ 无关的部分放进 intercept：

```math
=
\log\frac{\phi_y}{1-\phi_y}
+
\sum_{j=1}^d
\log\frac{1-\phi_{j\mid1}}{1-\phi_{j\mid0}}
+
\sum_{j=1}^d
x_j
\left[
\log\frac{\phi_{j\mid1}}{\phi_{j\mid0}}
-
\log\frac{1-\phi_{j\mid1}}{1-\phi_{j\mid0}}
\right].
```

因此 Bernoulli Naive Bayes 也诱导一个 linear log-odds：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}
=
b_{\mathrm{NB}}+\sum_{j=1}^d w_j^{\mathrm{NB}}x_j,
```

其中：

```math
b_{\mathrm{NB}}
=
\log\frac{\phi_y}{1-\phi_y}
+
\sum_{j=1}^d
\log\frac{1-\phi_{j\mid1}}{1-\phi_{j\mid0}},
```

```math
w_j^{\mathrm{NB}}
=
\log\frac{\phi_{j\mid1}}{\phi_{j\mid0}}
-
\log\frac{1-\phi_{j\mid1}}{1-\phi_{j\mid0}}.
```

这个结果和 GDA 的 lesson 平行：不同 generative assumptions 可以产生 linear/logistic-looking posterior，但它们对 $p(x\mid y)$ 的假设完全不同。

## 10. Boundary of Lecture 5

如果某个 word 在某个 class 中从未出现，MLE 可能给出：

```math
\hat\phi_{j\mid k}=0.
```

这会让包含该 word 的 test example 在 class $k$ 下得到 zero likelihood。Laplace smoothing 正是为这个问题引入 pseudo-counts。按照 Autumn 2018 syllabus，Laplace smoothing 属于 Lecture 6，因此本文件只指出问题来源，不展开 smoothing derivation。

Multinomial event model 也是 text classification 的重要 extension，但它在 official `cs229-notes2.pdf` 中位于 Laplace smoothing 之后。本 Lecture 5 文件保持 Bernoulli event model 主线，不展开 multinomial event model。
