# Lecture 5: Generative Learning Algorithms, GDA, and Naive Bayes

Canonical references: [Stanford CS229 Autumn 2018 syllabus](https://cs229.stanford.edu/syllabus-autumn2018.html), [Stanford Online Lecture 5 video](https://www.youtube.com/watch?v=nt63k3bfXS0), [official CS229 Generative Algorithms notes](https://cs229.stanford.edu/notes_archive/cs229-notes2.pdf), [official multivariate Gaussian section notes](https://cs229.stanford.edu/section/gaussians.pdf), and [More on Multivariate Gaussians](https://cs229.stanford.edu/section/more_on_gaussians.pdf).

Scope note: Autumn 2018 syllabus 把 Lecture 5 标为 Gaussian Discriminant Analysis 和 Naive Bayes；Lecture 6 才开始 Laplace Smoothing 和 Support Vector Machines。官方 `cs229-notes2.pdf` 的 Generative Algorithms notes 会继续讲 Laplace smoothing 和 text event models，所以本笔记把这些内容作为 notes2 continuation material 处理，不把它们塞进 Lecture 5 主线。

## Navigation

| Module | Purpose |
|---|---|
| [1. Course Position](#1-course-position) | 从 Lecture 4 的 conditional GLM 接到 Lecture 5 的 joint modeling |
| [2. Learning Objectives](#2-learning-objectives) | 明确本笔记已经完成的理解目标 |
| [3. Source Coverage and Boundary](#3-source-coverage-and-boundary) | 区分 official Lecture 5 内容和 notes2 continuation |
| [4. Notation Convention](#4-notation-convention) | 固定 random variable、realization、sample 和 dimension 符号 |
| [5. Generative versus Discriminative Learning](#5-generative-versus-discriminative-learning) | 解释两类学习器到底在学习什么 |
| [6. Multivariate Gaussian Distribution](#6-multivariate-gaussian-distribution) | 定义 density、mean、covariance、PSD 和 PD |
| [7. Mahalanobis Geometry](#7-mahalanobis-geometry) | 推导 covariance-aware quadratic form |
| [8. Gaussian Isocontours and Determinant](#8-gaussian-isocontours-and-determinant) | 连接 covariance、ellipse、density height 和 volume |
| [9. GDA Model and Generative Story](#9-gda-model-and-generative-story) | 定义 GDA assumptions 和 sampling direction |
| [10. GDA Joint Likelihood and MLE](#10-gda-joint-likelihood-and-mle) | 推导参数估计和 pooled covariance 逻辑 |
| [11. GDA Posterior Has Logistic Form](#11-gda-posterior-has-logistic-form) | 展开 posterior odds 并展示 quadratic cancellation |
| [12. Why the GDA Boundary Is Linear](#12-why-the-gda-boundary-is-linear) | 精确说明 linear boundary 从哪里来 |
| [13. GDA versus Logistic Regression](#13-gda-versus-logistic-regression) | 比较 assumptions、efficiency、misspecification 和 robustness |
| [14. Short QDA Contrast](#14-short-qda-contrast) | 说明 unequal covariance 为什么产生 quadratic boundary |
| [15. Naive Bayes for Discrete Features](#15-naive-bayes-for-discrete-features) | 从 continuous GDA features 转到 binary word features |
| [16. Conditional Independence](#16-conditional-independence) | 从 chain rule 推出 Naive Bayes factorization |
| [17. Naive Bayes Parameters and MLE](#17-naive-bayes-parameters-and-mle) | 推导 Bernoulli feature parameters 和 class prior |
| [18. Naive Bayes Prediction](#18-naive-bayes-prediction) | 得到 posterior score 和 log-space prediction |
| [19. Why Naive Bayes Can Work](#19-why-naive-bayes-can-work) | 解释 naive assumption 虽强但分类仍可能有效 |
| [20. Research-Level Synthesis](#20-research-level-synthesis) | 连接 assumptions、shift、reliability 和 failure modes |
| [21. Official Coverage Audit](#21-official-coverage-audit) | 核对 Lecture 5 官方主题是否覆盖 |
| [22. Fast Review Answers and Checklist](#22-fast-review-answers-and-checklist) | 快速复习答案，不放未完成问题 |
| [23. Learning Status](#23-learning-status) | 记录 first-pass note 和 continuing study 状态 |
| [24. Concept Map Summary](#24-concept-map-summary) | 用一页 map 总结本讲逻辑 |

**Related math derivations**

| Topic in this note | Deep-dive |
|---|---|
| Multivariate Gaussian density, covariance, Mahalanobis geometry, isocontours, determinant | [Multivariate Gaussian Geometry](../../math-derivations/lecture-05-generative-learning-gda-naive-bayes/01-multivariate-gaussian-geometry.md) |
| GDA MLE, pooled covariance, posterior odds, and logistic connection | [GDA MLE and Logistic Connection](../../math-derivations/lecture-05-generative-learning-gda-naive-bayes/02-gda-mle-and-logistic-connection.md) |
| Naive Bayes factorization, Bernoulli feature likelihood, MLE, and log-space prediction | [Naive Bayes Factorization and MLE](../../math-derivations/lecture-05-generative-learning-gda-naive-bayes/03-naive-bayes-factorization-and-mle.md) |
| Lecture 4 bridge: conditional exponential-family modeling | [GLM Construction Recipe](../../math-derivations/lecture-04-perceptron-exponential-family-glm/07-glm-construction-recipe.md) |

**Figures**

| Figure | File |
|---|---|
| Generative versus discriminative modeling schematic | [lecture05-generative-vs-discriminative.png](../../assets/figures/lecture05-generative-vs-discriminative.png) |
| Bivariate Gaussian 3D density | [lecture05-bivariate-gaussian-density-3d.png](../../assets/figures/lecture05-bivariate-gaussian-density-3d.png) |
| Same Gaussian 2D contours | [lecture05-bivariate-gaussian-contours.png](../../assets/figures/lecture05-bivariate-gaussian-contours.png) |
| Covariance geometry variants | [lecture05-covariance-geometry-variants.png](../../assets/figures/lecture05-covariance-geometry-variants.png) |
| GDA shared-covariance geometry | [lecture05-gda-shared-covariance-boundary.png](../../assets/figures/lecture05-gda-shared-covariance-boundary.png) |
| Shared covariance versus unequal covariance boundary | [lecture05-gda-qda-boundary-comparison.png](../../assets/figures/lecture05-gda-qda-boundary-comparison.png) |
| Naive Bayes conditional-independence schematic | [lecture05-naive-bayes-conditional-independence.png](../../assets/figures/lecture05-naive-bayes-conditional-independence.png) |

## 1. Course Position

Lecture 4 的主线是 conditional probabilistic modeling。GLM 的建模入口是：

```math
p(y\mid x;\theta)
```

也就是先指定给定 input 后 response 的 conditional distribution，再用 conditional likelihood 学习参数。Logistic regression 是最重要的 binary example：

```math
P(Y=1\mid X=x;\theta)=\frac{1}{1+\exp(-\theta^Tx)}.
```

Lecture 5 换了一条建模路线。它不先直接选择 $p(y\mid x)$，而是通过 class prior 和 class-conditional distribution 建模 joint distribution：

```math
p(x,y)=p(x\mid y)p(y).
```

同一个 joint distribution 也可以写成：

```math
p(x,y)=p(x\mid y)p(y)=p(y\mid x)p(x).
```

所以 generative classifier 最终仍然预测 $p(y\mid x)$。区别在于：它先建模 $p(y)$ 和 $p(x\mid y)$，再通过 Bayes rule 反推出 posterior。

Lecture 4 到 Lecture 5 的核心桥梁是：

```text
Lecture 4: choose p(y | x) through exponential-family / GLM structure.
Lecture 5: choose p(x | y) and p(y), then derive p(y | x).
```

这也解释了为什么两条完全不同的建模路线最后都可能得到 logistic-looking posterior：logistic regression 直接假设 posterior 的形式；GDA 假设 shared-covariance Gaussian class-conditionals，然后推出 posterior 具有 logistic form。

## 2. Learning Objectives

本笔记完成以下理解目标：

* 说清 discriminative learner 和 generative learner 分别建模什么；
* 用 Bayes rule 把 generative model 变成 classifier；
* 定义 non-degenerate multivariate Gaussian density，并解释每个符号；
* 区分 covariance matrix 一般只保证 PSD 和普通 density formula 要求 PD；
* 把 Gaussian quadratic form 理解成 squared Mahalanobis distance；
* 从 density level set 推导 Gaussian ellipse / ellipsoid；
* 把 determinant 解释为 uncertainty-volume scaling，而不是 point probability；
* 写出 GDA model、generative story、joint likelihood 和 MLE；
* 推导 GDA posterior log-odds，并说明为什么 shared covariance 会产生 linear boundary；
* 比较 GDA 和 logistic regression 的 assumptions、sample efficiency、misspecification 和 robustness；
* 从 chain rule 加 conditional independence 推出 Naive Bayes factorization；
* 推导 Bernoulli Naive Bayes 的 MLE 和 log-space prediction；
* 把 GDA / NB 的 assumptions 连接到 distribution shift、reliability 和 failure modes。

## 3. Source Coverage and Boundary

本讲的 source hierarchy 是：

```text
Stanford CS229 Autumn 2018 official lecture/video
-> official CS229 written notes
-> official CS229 supplementary notes
-> rigorous derivations and geometric explanations in this repository
-> future study-derived revisions integrated into relevant sections
-> research connections and critical extensions
```

Autumn 2018 official syllabus 给出的边界是：

| Date | Lecture | Official topic |
|---|---|---|
| 2018-10-08 | Lecture 5 | Gaussian Discriminant Analysis. Naive Bayes. |
| 2018-10-10 | Lecture 6 | Laplace Smoothing. Support Vector Machines. |

Stanford Online 视频 metadata 标题为 Lecture 5 - GDA and Naive Bayes。可抽取的 YouTube chapter metadata 包含 GDA 相关章节，没有暴露单独的 Laplace smoothing 或 multinomial event model 章节。审计时页面 metadata 中存在 caption track，但 caption endpoint 返回空文本，所以本笔记不声称已经获得 transcript-level evidence。

官方 `cs229-notes2.pdf` 覆盖范围比 Autumn 2018 Lecture 5 更长：它先讲 generative-learning introduction、GDA、GDA versus logistic regression、Naive Bayes，然后继续讲 Laplace smoothing 和 text event models。本笔记按照 Autumn 2018 Lecture 5 边界处理：Laplace smoothing 和 multinomial event model 不纳入 Lecture 5 主线。

## 4. Notation Convention

本笔记使用：

| Symbol | Meaning |
|---|---|
| $X$ | random feature vector |
| $x$ | realized feature vector |
| $X_j$ | 第 $j$ 个 feature random variable |
| $x_j$ | realized feature coordinate |
| $Y$ | label random variable |
| $y$ | realized label |
| $x^{(i)}$ | 第 $i$ 个 training example 的 feature vector |
| $y^{(i)}$ | 第 $i$ 个 training example 的 observed label |
| $m$ | number of training examples |
| $d$ | feature dimension |
| $\mathcal D$ | training set |

Training set 写成：

```math
\mathcal D=\{(x^{(i)},y^{(i)})\}_{i=1}^m.
```

部分 official CS229 notes 使用 $n$ 表示 feature 数量。本仓库用 $d$ 表示 feature dimension，用 $m$ 表示 sample size，避免把样本数和维度混在一起。

## 5. Generative versus Discriminative Learning

Discriminative learner 直接学习 conditional distribution 或 direct decision rule：

```math
p(y\mid x;\theta)
```

或者：

```math
h_\theta:\mathcal X\to\mathcal Y.
```

Lecture 3 和 Lecture 4 中的 logistic regression 是 probabilistic discriminative model：

```math
P(Y=1\mid X=x;\theta)=g(\theta^Tx).
```

Generative learner 建模：

```math
p(y)
```

和：

```math
p(x\mid y).
```

因此它建模 joint distribution：

```math
p(x,y)=p(x\mid y)p(y).
```

这里的 generate 不是说模型一定要输出逼真的样本，而是说模型描述了 complete labeled example 的 sampling process：先根据 class prior 抽 $Y$，再根据该 class 的 class-conditional distribution 抽 $X$。预测时方向反过来，用 Bayes rule：

```math
p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}.
```

分类时可以不用显式计算 $p(x)$，因为对所有候选 $y$ 来说它相同：

```math
\underset{y}{\mathrm{argmax}}\ p(y\mid x)=\underset{y}{\mathrm{argmax}}\ \frac{p(x\mid y)p(y)}{p(x)}
```

```math
=\underset{y}{\mathrm{argmax}}\ p(x\mid y)p(y).
```

这只是 argmax decision rule 中的 cancellation，不是说 $p(x)$ 不存在。若要 calibrated posterior probabilities，仍然需要 normalization。

![Generative versus discriminative modeling schematic](../../assets/figures/lecture05-generative-vs-discriminative.png)

Figure 1. Discriminative learning 直接建模 $p(y\mid x)$ 或 decision rule；generative learning 建模 $p(y)$ 和 $p(x\mid y)$，再用 Bayes rule 推断 label。

Generative modeling 增加了对 world 的结构假设：不仅要能分 label，还要描述每个 class 内 feature 怎么分布。这个额外结构在 assumption 近似正确时可能带来 sample-efficiency advantage；在 assumption 错误时也会带来 misspecification risk。

## 6. Multivariate Gaussian Distribution

Detailed companion: [Multivariate Gaussian Geometry](../../math-derivations/lecture-05-generative-learning-gda-naive-bayes/01-multivariate-gaussian-geometry.md).

令：

```math
X\in\mathbb R^d.
```

若：

```math
X\sim\mathcal N(\mu,\Sigma),
```

其中：

```math
\mu\in\mathbb R^d
```

并且：

```math
\Sigma\in\mathbb R^{d\times d}.
```

对 non-degenerate multivariate Gaussian，它的 density 是：

```math
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right).
```

符号含义：

| Symbol | Meaning |
|---|---|
| $x$ | $\mathbb R^d$ 中的 realized point |
| $\mu$ | mean vector，也就是 density 的中心 |
| $\Sigma$ | covariance matrix，控制 spread 和 correlation |
| $|\Sigma|$ | determinant of $\Sigma$ |
| $\Sigma^{-1}$ | inverse covariance matrix，也叫 precision matrix |
| $(x-\mu)^\top\Sigma^{-1}(x-\mu)$ | $x$ 到 $\mu$ 的 squared Mahalanobis distance |
| $(2\pi)^{d/2}|\Sigma|^{1/2}$ | normalization scale，使总 probability mass 等于 $1$ |

Mean 是：

```math
\mathbb E[X]=\mu.
```

Covariance matrix 是：

```math
\Sigma=\mathbb E\left[(X-\mu)(X-\mu)^\top\right].
```

展开 matrix entry：

```math
\Sigma_{jk}=\mathrm{Cov}(X_j,X_k).
```

Diagonal entries 是 variances：

```math
\Sigma_{jj}=\mathrm{Var}(X_j).
```

Off-diagonal entries 是 covariances。Positive covariance 表示两个 coordinates 倾向于一起高于或低于各自 mean；negative covariance 表示一个高于 mean 时另一个倾向于低于 mean；zero covariance 表示没有 linear covariance signal，但一般不等于 independence。对 multivariate Gaussian，zero covariance 会推出 independence，这是 Gaussian 的特殊性质，不是 covariance 的一般性质。

### 6.1 PSD versus PD

任意具有有限二阶矩的 random vector，其 covariance matrix 总是 positive semidefinite：

```math
\Sigma\succeq 0.
```

证明：

```math
v^\top\Sigma v=v^\top\mathbb E\left[(X-\mu)(X-\mu)^\top\right]v
```

```math
=\mathbb E\left[v^\top(X-\mu)(X-\mu)^\top v\right]
```

```math
=\mathbb E\left[\left(v^\top(X-\mu)\right)^2\right]\geq 0.
```

这只证明 PSD，不证明 positive definite。如果存在非零 $v$ 使得：

```math
v^\top(X-\mu)=0
```

almost surely，则：

```math
v^\top\Sigma v=0.
```

此时 covariance singular，distribution 退化在一个 lower-dimensional affine subspace 上。

普通 Gaussian density formula 使用：

```math
\Sigma^{-1}
```

和：

```math
|\Sigma|^{-1/2}.
```

所以具有 Lebesgue density 的 non-degenerate multivariate Gaussian 通常要求：

```math
\Sigma\succ 0.
```

正确区分是：

```text
general covariance matrix -> PSD
ordinary full-dimensional Gaussian density -> PD covariance
degenerate Gaussian -> PSD but singular, no ordinary full-dimensional density
```

## 7. Mahalanobis Geometry

Gaussian density 中最核心的几何对象是：

```math
D_M^2(x,\mu)=(x-\mu)^\top\Sigma^{-1}(x-\mu).
```

它不是需要背诵的装饰项，而是 covariance-aware squared distance。使用 eigendecomposition：

```math
\Sigma=Q\Lambda Q^\top
```

其中 $Q$ 是 orthogonal matrix，并且：

```math
\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_d).
```

因为 $\Sigma\succ0$，所以：

```math
\lambda_j>0.
```

于是：

```math
\Sigma^{-1}=Q\Lambda^{-1}Q^\top.
```

令：

```math
z=Q^\top(x-\mu).
```

完整展开：

```math
(x-\mu)^\top\Sigma^{-1}(x-\mu)=(x-\mu)^\top Q\Lambda^{-1}Q^\top(x-\mu)
```

```math
=z^\top\Lambda^{-1}z
```

```math
=\sum_{j=1}^d\frac{z_j^2}{\lambda_j}.
```

含义是：

* eigenvectors 决定 Gaussian 的 principal directions；
* eigenvalues 决定对应方向的 spread；
* 大 variance direction 中相同 Euclidean displacement 被惩罚得更小；
* 小 variance direction 中 displacement 被惩罚得更大；
* Euclidean distance 把所有方向等权处理，Mahalanobis distance 按 covariance geometry 重新缩放距离。

核心区分是：多元高斯的等密度线不是“把 covariance matrix 直接画成椭圆”。真正进入 density exponent 的对象是 precision quadratic form：

```math
(x-\mu)^\top\Sigma^{-1}(x-\mu).
```

$\Sigma$ 仍然决定 geometry，但它是通过 $\Sigma^{-1}$ 出现在 exponent 中，并通过 eigendecomposition 给出主方向和尺度。换句话说，$\Sigma$ 控制 spread 和 correlation；$\Sigma^{-1}$ 直接控制离开 mean 后 density 下降得多快。

二维时令 $\mu=0$ 且：

```math
\Sigma=
\begin{bmatrix}
\sigma_x^2 & \sigma_{xy}\\
\sigma_{xy} & \sigma_y^2
\end{bmatrix}.
```

则：

```math
\Sigma^{-1}
=
\frac{1}{\sigma_x^2\sigma_y^2-\sigma_{xy}^2}
\begin{bmatrix}
\sigma_y^2 & -\sigma_{xy}\\
-\sigma_{xy} & \sigma_x^2
\end{bmatrix},
```

所以：

```math
\begin{bmatrix}x\\y\end{bmatrix}^\top
\Sigma^{-1}
\begin{bmatrix}x\\y\end{bmatrix}
=
\frac{
\sigma_y^2x^2-2\sigma_{xy}xy+\sigma_x^2y^2
}{
\sigma_x^2\sigma_y^2-\sigma_{xy}^2
}.
```

这说明非对角 covariance 在 exponent 里体现为 $xy$ 交叉项。若 $\sigma_{xy}=0$，等密度椭圆不旋转，主轴沿原坐标轴；若 $\sigma_{xy}\neq0$，交叉项改变主方向，椭圆会相对原坐标轴旋转。

在二维等方差特例：

```math
\Sigma=\sigma^2
\begin{bmatrix}
1 & \rho\\
\rho & 1
\end{bmatrix},
```

主方向是：

```math
\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}
\qquad
\frac1{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix},
```

也就是 $x=y$ 和 $x=-y$ 两个 $45^\circ$ 方向。若 $\rho>0$，沿 $x=y$ 拉长、沿 $x=-y$ 压缩；若 $\rho<0$，方向反过来。严格说，$45^\circ$ 是 $\sigma_x^2=\sigma_y^2$ 的对称特例；一般二维 covariance matrix

```math
\Sigma=
\begin{bmatrix}
a & c\\
c & b
\end{bmatrix}
```

的主轴旋转角满足：

```math
\tan(2\theta)=\frac{2c}{a-b}.
```

![Bivariate Gaussian 3D density](../../assets/figures/lecture05-bivariate-gaussian-density-3d.png)

Figure 2. Bivariate Gaussian density，参数为 $\mu=[0.6,-0.4]^\top$ 和 $\Sigma=\begin{bmatrix}1.70&0.90\\0.90&0.85\end{bmatrix}$。非零 covariance 让 bell shape 相对坐标轴发生倾斜。

![Same Gaussian 2D contours](../../assets/figures/lecture05-bivariate-gaussian-contours.png)

Figure 3. 与 Figure 2 完全相同的 Gaussian，用 2D density contours 展示 level-set geometry。3D density surface 和 2D contours 使用同一个 $\mu$ 和 $\Sigma$。

## 8. Gaussian Isocontours and Determinant

Isocontour 固定 density height：

```math
p(x;\mu,\Sigma)=c.
```

代入 density：

```math
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right)=c.
```

移项：

```math
\exp\left(-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right)=c(2\pi)^{d/2}|\Sigma|^{1/2}.
```

取 log：

```math
-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)=\log c+\frac d2\log(2\pi)+\frac12\log|\Sigma|.
```

乘以 $-2$：

```math
(x-\mu)^\top\Sigma^{-1}(x-\mu)=C.
```

其中：

```math
C=-2\log c-d\log(2\pi)-\log|\Sigma|.
```

对低于 peak density 的合法 level set，有 $C\geq0$。使用同一个 eigen-coordinate $z=Q^\top(x-\mu)$：

```math
\sum_{j=1}^d\frac{z_j^2}{\lambda_j}=C.
```

当 $C>0$ 时：

```math
\sum_{j=1}^d\frac{z_j^2}{C\lambda_j}=1.
```

这在 $d=2$ 中是 ellipse，在更高维是 ellipsoid。第 $j$ 个 principal semi-axis length 为：

```math
\sqrt{C\lambda_j}.
```

所以 semi-axis length 与：

```math
\sqrt{\lambda_j}
```

成比例。

完整理解链条是：

```text
covariance matrix -> quadratic form -> Mahalanobis geometry -> density contour
```

Determinant 控制 volume scale：

```math
|\Sigma|=\prod_{j=1}^d\lambda_j.
```

因此：

```math
|\Sigma|^{1/2}=\prod_{j=1}^d\sqrt{\lambda_j}.
```

这和 Gaussian uncertainty volume / ellipsoid volume scaling 相关。Covariance volume 越大，probability mass 越分散，peak density 必须降低。但这说的是 density height，不是某一点的 probability。连续随机变量满足：

```math
P(X=x)=0
```

对任意单点 $x$ 成立；probability mass 来自对 region 的积分。

![Covariance geometry variants](../../assets/figures/lecture05-covariance-geometry-variants.png)

Figure 4. 四种 covariance geometry：isotropic、unequal diagonal variances、positive covariance、negative covariance。图像展示 $\Sigma$ 如何决定 ellipse scale 和 orientation。

## 9. GDA Model and Generative Story

Detailed companion: [GDA MLE and Logistic Connection](../../math-derivations/lecture-05-generative-learning-gda-naive-bayes/02-gda-mle-and-logistic-connection.md).

Gaussian Discriminant Analysis 处理 binary labels：

```math
Y\in\{0,1\}.
```

模型定义为：

```math
Y\sim\mathrm{Bernoulli}(\phi),
```

```math
X\mid Y=0\sim\mathcal N(\mu_0,\Sigma),
```

```math
X\mid Y=1\sim\mathcal N(\mu_1,\Sigma).
```

参数含义：

| Parameter | Meaning |
|---|---|
| $\phi$ | class prior $P(Y=1)$ |
| $\mu_0$ | class $0$ 的 mean feature vector |
| $\mu_1$ | class $1$ 的 mean feature vector |
| $\Sigma$ | 两个 classes 共享的 covariance matrix |

这里最重要的 modeling assumption 是：

```math
\Sigma_0=\Sigma_1=\Sigma.
```

这不是 Gaussian distribution 的必然性质。两个 Gaussian classes 完全可以有不同 covariance；Lecture 5 的 GDA 选择 shared covariance，是为了得到特定的 parameter sharing 和 linear posterior boundary。

GDA 的 generative story 是：

```text
1. draw Y from Bernoulli(phi)
2. condition on the realized class Y
3. draw X from the corresponding Gaussian class-conditional distribution
```

形式化写成：

```math
p(x,y)=p(x\mid y)p(y).
```

预测时用 Bayes rule：

```math
p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}.
```

所以生成方向是：

```text
Y -> X
```

而 inference direction 是：

```text
observed X -> infer Y
```

## 10. GDA Joint Likelihood and MLE

GDA 最大化 observed labeled examples 的 joint likelihood：

```math
L(\phi,\mu_0,\mu_1,\Sigma)=\prod_{i=1}^m p(x^{(i)},y^{(i)}).
```

利用 generative factorization：

```math
p(x^{(i)},y^{(i)})=p(x^{(i)}\mid y^{(i)})p(y^{(i)}).
```

得到：

```math
L(\phi,\mu_0,\mu_1,\Sigma)=\prod_{i=1}^m p(x^{(i)}\mid y^{(i)};\mu_0,\mu_1,\Sigma)p(y^{(i)};\phi).
```

这和 logistic regression 的 conditional likelihood 不同：

```math
L_{\mathrm{logistic}}(\theta)=\prod_{i=1}^m p(y^{(i)}\mid x^{(i)};\theta).
```

核心比较是：

```text
GDA: joint likelihood over (x, y)
logistic regression: conditional likelihood over y given x
```

定义 indicator：

```math
\mathbf{1}\{A\}=\begin{cases}1,&\text{if }A\text{ is true}\\0,&\text{otherwise}\end{cases}.
```

GDA 的 MLE 是：

```math
\hat\phi=\frac1m\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}.
```

```math
\hat\mu_0=\frac{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=0\}x^{(i)}}{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=0\}}.
```

```math
\hat\mu_1=\frac{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}x^{(i)}}{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}}.
```

令 $\hat\mu_{y^{(i)}}$ 表示：当 $y^{(i)}=0$ 时取 $\hat\mu_0$，当 $y^{(i)}=1$ 时取 $\hat\mu_1$。Shared covariance 的 MLE 是：

```math
\hat\Sigma=\frac1m\sum_{i=1}^m\left(x^{(i)}-\hat\mu_{y^{(i)}}\right)\left(x^{(i)}-\hat\mu_{y^{(i)}}\right)^\top.
```

Denominator 是 $m$，因为这是 Gaussian joint likelihood 下 shared covariance 的 maximum likelihood estimator。每个 example 都贡献一个 residual outer product，并且所有 residual 都估计同一个 covariance parameter。它是 pooled covariance，因为模型假设两个 classes 共享一个 $\Sigma$。这不是带 degrees-of-freedom correction 的 unbiased covariance estimator；CS229 这里推导的是 MLE。

完整 derivative calculation 见 [GDA MLE and Logistic Connection](../../math-derivations/lecture-05-generative-learning-gda-naive-bayes/02-gda-mle-and-logistic-connection.md#2-gda-mle-without-skipped-steps)。

## 11. GDA Posterior Has Logistic Form

GDA 最关键的数学连接是：它能推出 logistic-form posterior。从 posterior odds 开始：

```math
\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}.
```

由 Bayes rule：

```math
\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=\frac{p(x\mid Y=1)P(Y=1)}{p(x\mid Y=0)P(Y=0)}.
```

取 log：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=\log p(x\mid Y=1)-\log p(x\mid Y=0)+\log\frac{\phi}{1-\phi}.
```

两个 class-conditional log density 是：

```math
\log p(x\mid Y=1)=-\frac d2\log(2\pi)-\frac12\log|\Sigma|-\frac12(x-\mu_1)^\top\Sigma^{-1}(x-\mu_1).
```

```math
\log p(x\mid Y=0)=-\frac d2\log(2\pi)-\frac12\log|\Sigma|-\frac12(x-\mu_0)^\top\Sigma^{-1}(x-\mu_0).
```

相减后 common constants 抵消：

```math
\log p(x\mid Y=1)-\log p(x\mid Y=0)=-\frac12(x-\mu_1)^\top\Sigma^{-1}(x-\mu_1)+\frac12(x-\mu_0)^\top\Sigma^{-1}(x-\mu_0).
```

展开第一个 quadratic form：

```math
(x-\mu_1)^\top\Sigma^{-1}(x-\mu_1)=x^\top\Sigma^{-1}x-2\mu_1^\top\Sigma^{-1}x+\mu_1^\top\Sigma^{-1}\mu_1.
```

展开第二个 quadratic form：

```math
(x-\mu_0)^\top\Sigma^{-1}(x-\mu_0)=x^\top\Sigma^{-1}x-2\mu_0^\top\Sigma^{-1}x+\mu_0^\top\Sigma^{-1}\mu_0.
```

代回 posterior log-odds：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=-\frac12x^\top\Sigma^{-1}x+\mu_1^\top\Sigma^{-1}x-\frac12\mu_1^\top\Sigma^{-1}\mu_1
```

```math
+\frac12x^\top\Sigma^{-1}x-\mu_0^\top\Sigma^{-1}x+\frac12\mu_0^\top\Sigma^{-1}\mu_0+\log\frac{\phi}{1-\phi}.
```

因为 covariance shared，两个 $x^\top\Sigma^{-1}x$ 项完全抵消。剩下：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=(\mu_1-\mu_0)^\top\Sigma^{-1}x-\frac12\mu_1^\top\Sigma^{-1}\mu_1+\frac12\mu_0^\top\Sigma^{-1}\mu_0+\log\frac{\phi}{1-\phi}.
```

利用 $\Sigma^{-1}$ symmetric：

```math
(\mu_1-\mu_0)^\top\Sigma^{-1}x=\left(\Sigma^{-1}(\mu_1-\mu_0)\right)^\top x.
```

定义：

```math
w=\Sigma^{-1}(\mu_1-\mu_0).
```

以及：

```math
b=-\frac12\mu_1^\top\Sigma^{-1}\mu_1+\frac12\mu_0^\top\Sigma^{-1}\mu_0+\log\frac{\phi}{1-\phi}.
```

于是：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=w^\top x+b.
```

如果 log-odds 为 $s=w^\top x+b$，则：

```math
\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=\exp(s).
```

又因为 $P(Y=0\mid X=x)=1-P(Y=1\mid X=x)$：

```math
\frac{P(Y=1\mid X=x)}{1-P(Y=1\mid X=x)}=\exp(s).
```

解得：

```math
P(Y=1\mid X=x)=\frac{1}{1+\exp[-(w^\top x+b)]}.
```

所以 GDA 产生了和 logistic regression 相同形状的 posterior，但它不是直接假设 sigmoid；它是从 $p(x\mid y)$ 和 $p(y)$ 推出来的。

## 12. Why the GDA Boundary Is Linear

Decision boundary 是两个 posterior probabilities 相等的位置：

```math
P(Y=1\mid X=x)=P(Y=0\mid X=x).
```

等价于 posterior odds 为 $1$：

```math
\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=1.
```

取 log：

```math
\log\frac{P(Y=1\mid X=x)}{P(Y=0\mid X=x)}=0.
```

在 shared-covariance GDA 中：

```math
w^\top x+b=0.
```

因此 boundary 是 hyperplane。原因不是 Gaussian 自然产生直线，而是：

```text
shared covariance -> equal quadratic terms -> cancellation -> linear log-likelihood ratio -> linear decision boundary
```

![GDA shared-covariance geometry](../../assets/figures/lecture05-gda-shared-covariance-boundary.png)

Figure 5. GDA class-conditionals，shared covariance 为 $\Sigma=\begin{bmatrix}1.25&0.55\\0.55&0.80\end{bmatrix}$，means 为 $\mu_0=[-1.3,-0.7]^\top$ 和 $\mu_1=[1.15,0.9]^\top$，class prior 为 $\phi=0.55$，红线是 Bayes/GDA decision boundary。

## 13. GDA versus Logistic Regression

GDA 假设 feature distribution inside each class 具有 Gaussian structure：

```math
X\mid Y=k\sim\mathcal N(\mu_k,\Sigma).
```

由这些 assumptions 推出：

```math
P(Y=1\mid X=x)=\frac{1}{1+\exp[-(w^\top x+b)]}.
```

所以：

```text
GDA assumptions imply a logistic-form posterior.
```

但反过来一般不成立：

```text
logistic-form posterior does not imply Gaussian class-conditionals.
```

很多不同的 joint distributions 可以产生同样的 conditional posterior shape；很多 non-Gaussian class-conditionals 也可能对应有用的 logistic-style decision rule。

| Aspect | GDA | Logistic regression |
|---|---|---|
| Modeling target | $p(x\mid y)$ and $p(y)$ | $p(y\mid x)$ |
| Likelihood | joint likelihood of $(x,y)$ | conditional likelihood of $y$ given $x$ |
| Assumptions | 对 $X\mid Y$ 有更强 distributional assumptions | 对 $X$ 的 assumptions 更弱 |
| Data efficiency | assumptions 近似正确时可以更有效利用数据 | assumptions 更少，可能需要更多数据来学同样结构 |
| Misspecification | class-conditionals 非 Gaussian 或 covariance 不 shared 时可能失败严重 | 通常对 non-Gaussian $X\mid Y$ 更 robust |
| Output | posterior 由 Bayes rule 推出 | posterior 直接指定 |
| Diagnostics | 检查 class means、covariance、shape、priors | 检查 conditional calibration、residual patterns、boundary fit |

这里比较的是 inductive bias。Stronger assumptions 不自动更好或更坏，而是在 flexibility 和 structure 之间做 tradeoff。真实机制接近 GDA 时，GDA 可能更 data efficient；真实机制偏离 Gaussian/shared covariance 时，logistic regression 通常更稳健。

## 14. Short QDA Contrast

如果 class covariances 不同：

```math
X\mid Y=0\sim\mathcal N(\mu_0,\Sigma_0)
```

并且：

```math
X\mid Y=1\sim\mathcal N(\mu_1,\Sigma_1),
```

log posterior odds 中会出现：

```math
-\frac12x^\top\Sigma_1^{-1}x+\frac12x^\top\Sigma_0^{-1}x.
```

当：

```math
\Sigma_0\neq\Sigma_1
```

时，这些 quadratic terms 一般不会抵消，所以 boundary 一般是 quadratic。

简短对照是：

```text
shared covariance -> LDA/GDA linear geometry
different covariance -> QDA-style quadratic geometry
```

![Shared covariance versus unequal covariance boundary](../../assets/figures/lecture05-gda-qda-boundary-comparison.png)

Figure 6. Shared covariance 会抵消 quadratic terms，产生 linear boundary；unequal covariance 会保留 quadratic terms，从而产生 curved boundary。

## 15. Naive Bayes for Discrete Features

GDA 适合 continuous real-valued features。Lecture 5 中的 Naive Bayes 进入 high-dimensional discrete features，典型例子是 text classification。

Spam classification 中：

```math
Y\in\{0,1\}.
```

设 vocabulary size 为 $d$。定义 binary word-presence features：

```math
X_j\in\{0,1\}.
```

Realized coordinate：

```math
x_j=1
```

表示第 $j$ 个 word 出现在 email 中，而：

```math
x_j=0
```

表示没有出现。完整 feature vector 是：

```math
x=(x_1,\ldots,x_d)\in\{0,1\}^d.
```

如果直接建模 $\{0,1\}^d$ 上任意 joint distribution，需要为每个 binary vector 指定 probability；当 $d$ 很大时不可行。Naive Bayes 通过强 conditional-independence assumption 大幅减少参数数量。

## 16. Conditional Independence

先从 chain rule 开始：

```math
p(x_1,\ldots,x_d\mid y)=p(x_1\mid y)p(x_2\mid x_1,y)p(x_3\mid x_1,x_2,y)\cdots p(x_d\mid x_1,\ldots,x_{d-1},y).
```

Naive Bayes assumption 是 conditional independence：

```math
X_1,\ldots,X_d\text{ are conditionally independent given }Y.
```

也就是一旦已知 $Y$，每一项都删去 previous feature conditioning：

```math
p(x_j\mid x_1,\ldots,x_{j-1},y)=p(x_j\mid y).
```

因此：

```math
p(x_1,\ldots,x_d\mid y)=\prod_{j=1}^d p(x_j\mid y).
```

Conditional independence given $Y$ 不等于 unconditional independence。可以同时成立：

```math
X_j\not\!\perp X_k
```

以及模型假设：

```math
X_j\perp X_k\mid Y.
```

在 text 中，两个 words 可能 marginally correlated，因为它们都和 spam 相关；但模型假设在已知 spam / non-spam 后，剩余 dependence 被忽略。

![Naive Bayes conditional-independence schematic](../../assets/figures/lecture05-naive-bayes-conditional-independence.png)

Figure 7. Naive Bayes graphical intuition。$Y$ 是 factorization 中的 shared parent，features 在给定 $Y$ 后被建模为 conditionally independent。这个图只表示 probabilistic factorization assumption，不是完整 causal claim。

## 17. Naive Bayes Parameters and MLE

Detailed companion: [Naive Bayes Factorization and MLE](../../math-derivations/lecture-05-generative-learning-gda-naive-bayes/03-naive-bayes-factorization-and-mle.md).

对 Bernoulli word-presence features，定义：

```math
\phi_{j\mid 1}=P(X_j=1\mid Y=1).
```

```math
\phi_{j\mid 0}=P(X_j=1\mid Y=0).
```

以及 class prior：

```math
\phi_y=P(Y=1).
```

对一个 example：

```math
p(x,y)=p(y)\prod_{j=1}^d p(x_j\mid y).
```

对 $k\in\{0,1\}$：

```math
p(x_j\mid y=k)=\phi_{j\mid k}^{x_j}(1-\phi_{j\mid k})^{1-x_j}.
```

Class prior 的 MLE 是：

```math
\hat\phi_y=\frac1m\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}.
```

Word-presence probabilities 的 MLE 是：

```math
\hat\phi_{j\mid 1}=\frac{\sum_{i=1}^m\mathbf{1}\{x_j^{(i)}=1,y^{(i)}=1\}}{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=1\}}.
```

```math
\hat\phi_{j\mid 0}=\frac{\sum_{i=1}^m\mathbf{1}\{x_j^{(i)}=1,y^{(i)}=0\}}{\sum_{i=1}^m\mathbf{1}\{y^{(i)}=0\}}.
```

这些就是 class-conditional empirical frequencies：在 spam examples 中 word $j$ 出现的频率，以及在 non-spam examples 中 word $j$ 出现的频率。

Laplace smoothing 会通过 pseudo-counts 改变这些 estimates。它实践上非常重要，但 Autumn 2018 syllabus 把 Laplace smoothing 放在 Lecture 6，因此本 Lecture 5 主线不展开 smoothing formula。

## 18. Naive Bayes Prediction

使用 Bayes rule：

```math
P(Y=y\mid X=x)\propto P(Y=y)P(X=x\mid Y=y).
```

在 Naive Bayes factorization 下：

```math
P(Y=y\mid X=x)\propto P(Y=y)\prod_{j=1}^d P(X_j=x_j\mid Y=y).
```

Classifier 是：

```math
\hat y=\underset{y\in\{0,1\}}{\mathrm{argmax}}\ P(Y=y)\prod_{j=1}^d P(X_j=x_j\mid Y=y).
```

Practical / numerical note: 实现时通常在 log space 中计算：

```math
s(y)=\log P(Y=y)+\sum_{j=1}^d\log P(X_j=x_j\mid Y=y).
```

然后：

```math
\hat y=\underset{y\in\{0,1\}}{\mathrm{argmax}}\ s(y).
```

Log transform 保持 argmax 不变，因为 log strictly increasing；同时它避免大量小概率相乘造成 numerical underflow。

## 19. Why Naive Bayes Can Work

Naive Bayes 的 independence assumption 非常强。真实 text features 通常相关：words co-occur，phrases 有意义，topic 会让一组 vocabulary 同时出现，document length 也会一起影响很多 features。模型被称为 naive，正是因为它在 conditioning on $Y$ 后忽略这些 dependence。

但 classifier 仍可能工作，因为 classification 不要求 full joint density 的每个细节都正确。很多时候只需要 posterior ranking 正确：

```math
P(Y=1\mid X=x)>P(Y=0\mid X=x).
```

即使模型 overcounts correlated evidence，两个 classes 的 ranking 仍可能有效。真正危险的是 probability overconfidence：高度相关的 words 可能被当作 independent repeated evidence，使 posterior probabilities 过度接近 $0$ 或 $1$。

所以 classification accuracy 和 probability calibration 是不同性质：

```text
a classifier may classify correctly while producing poorly calibrated posterior probabilities
```

这和长期 reliability / calibration 方向有直接联系，但本讲只保留这条连接，不提前展开 calibration 专题。

## 20. Research-Level Synthesis

Generative models 的一个研究价值是 assumptions 更暴露。GDA 暴露的是 class prior、class means、Gaussian class-conditional shape 和 shared covariance。Naive Bayes 暴露的是 class prior、feature likelihood family 和 conditional independence。

Misspecification 指真实 data-generating process 不在所选 model family 中。对 GDA，真实 $p(x\mid y)$ 可能 non-Gaussian、multimodal、heavy-tailed、heteroscedastic，或者 class covariance 不同。对 Naive Bayes，features 可能在 conditioning on class 后仍有强 dependence。

Distribution shift 可以沿 generative factorization 拆开分析。如果：

```math
p_{\mathrm{train}}(x\mid y)\neq p_{\mathrm{test}}(x\mid y),
```

说明 class-conditional shape、covariance 或 word-frequency mechanism 发生变化。如果：

```math
p_{\mathrm{train}}(y)\neq p_{\mathrm{test}}(y),
```

说明 class prior 发生变化。前者改变 evidence model，后者改变 observing evidence 前的 base rate。

对 reliable ML 来说，generative modeling 的价值不只是 prediction，而是把 failure modes 分解成可检查的假设：

* class-conditional shape change；
* covariance change；
* class-prior shift；
* violated conditional independence；
* density model mismatch；
* acceptable accuracy 下的 calibration failure。

这自然连接到 distribution shift、reliability 和 mechanism change，但仍然服务于 Lecture 5 的 official models，而不是扩展成独立 research survey。

## 21. Official Coverage Audit

| Official Lecture 5 topic | Status in this note |
|---|---|
| Discriminative versus generative algorithms | Covered in Sections 1 and 5 |
| Bayes classification rule | Covered in Section 5 |
| Multivariate Gaussian density | Covered in Section 6 |
| Mean and covariance matrix | Covered in Section 6 |
| PSD covariance versus PD density condition | Covered in Section 6.1 |
| Mahalanobis quadratic form | Covered in Section 7 |
| Gaussian contours and determinant intuition | Covered in Section 8 |
| GDA model | Covered in Section 9 |
| GDA generative story | Covered in Section 9 |
| GDA joint likelihood | Covered in Section 10 |
| GDA MLE formulas | Covered in Section 10 and detailed derivation file |
| GDA prediction | Covered in Sections 11 and 12 |
| GDA posterior logistic form | Covered in Section 11 and detailed derivation file |
| GDA versus logistic regression | Covered in Section 13 |
| Naive Bayes discrete-feature setup | Covered in Section 15 |
| Conditional-independence assumption | Covered in Section 16 |
| Naive Bayes likelihood | Covered in Section 17 and detailed derivation file |
| Naive Bayes MLE | Covered in Section 17 and detailed derivation file |
| Naive Bayes prediction | Covered in Section 18 |

Boundary audit:

| Topic | Lecture 5 status |
|---|---|
| Laplace smoothing | Not part of Lecture 5 mainline: Autumn 2018 syllabus assigns it to Lecture 6; it appears later in `cs229-notes2.pdf` as continuation material |
| Multinomial event model for text classification | Not part of Lecture 5 mainline: it appears after Laplace smoothing in `cs229-notes2.pdf`, beyond the audited Lecture 5 scope |

## 22. Fast Review Answers and Checklist

* Generative learning 建模 $p(y)$ 和 $p(x\mid y)$，再用 Bayes rule 得到 $p(y\mid x)$。
* Discriminative learning 直接建模 $p(y\mid x)$ 或从 $x$ 到 $y$ 的 decision rule。
* 分类时可以用 $\underset{y}{\mathrm{argmax}}\ p(x\mid y)p(y)$，因为 $p(x)$ 不依赖于 $y$。
* Non-degenerate multivariate Gaussian density 要求 $\Sigma\succ0$；一般 covariance matrix 只保证 PSD。
* Quadratic form $(x-\mu)^\top\Sigma^{-1}(x-\mu)$ 是 squared Mahalanobis distance。
* 多元高斯 contour 由 $\Sigma^{-1}$ 中的 quadratic form 决定；$\Sigma$ 的非对角 entry 在二维展开中产生 $xy$ 交叉项。
* $\Sigma$ 的 eigenvectors 给 principal directions；eigenvalues 给 spread；semi-axis lengths 按 $\sqrt{\lambda_j}$ 缩放。
* 二维等方差时，相关方向自然分解为 $x+y$ 和 $x-y$，所以主轴是 $x=y$ 与 $x=-y$；一般不等方差时旋转角满足 $\tan(2\theta)=2\sigma_{xy}/(\sigma_x^2-\sigma_y^2)$。
* $|\Sigma|^{1/2}$ 是 volume-scaling term；density height 不是 point probability。
* GDA 假设 $Y\sim\mathrm{Bernoulli}(\phi)$ 且 $X\mid Y=k\sim\mathcal N(\mu_k,\Sigma)$，两个 classes 共享 covariance。
* GDA maximizes joint likelihood；logistic regression maximizes conditional likelihood。
* GDA covariance MLE 的 denominator 是 $m$，因为所有 examples 的 residual vectors 都估计同一个 shared covariance。
* GDA 产生 logistic posterior，是因为 shared covariance 抵消了 quadratic $x^\top\Sigma^{-1}x$ terms。
* GDA decision boundary 是 linear，因为 posterior log-odds 是 $w^\top x+b$。
* 如果 class covariances 不同，quadratic terms 留下，boundary 一般变成 quadratic。
* Naive Bayes 假设 $X_1,\ldots,X_d$ are conditionally independent given $Y$，不是 unconditionally independent。
* Bernoulli Naive Bayes 用 class-conditional empirical frequencies 估计 word-presence probabilities。
* Naive Bayes prediction 使用 $P(Y=y)\prod_jP(X_j=x_j\mid Y=y)$，实际实现通常在 log space 中做。
* Naive Bayes 即使 density misspecified 也可能分类正确，但 correlated evidence 会造成 overconfident probabilities。

## 23. Learning Status

Lecture 5 note: complete first-pass research-level note.

Lecture 5 interactive study: in progress.

下一讲的 official continuation 是 Lecture 6：Laplace smoothing 和 Support Vector Machines。通过学习过程产生的新理解，应直接整合回本笔记中最自然的正文位置。

## 24. Concept Map Summary

Course development map:

```text
Lecture 4 conditional GLMs
-> discriminative p(y | x)
-> Lecture 5 generative p(y), p(x | y)
-> Bayes rule
-> multivariate Gaussian geometry
-> GDA class-conditionals
-> joint likelihood MLE
-> posterior odds
-> logistic-form posterior
-> linear boundary from shared covariance
-> GDA vs logistic regression
-> discrete feature modeling
-> Naive Bayes conditional independence
-> NB likelihood, MLE, prediction
-> reliability and shift analysis
```

Generative classification map:

```text
choose prior p(y)
-> choose class-conditional model p(x | y)
-> learn parameters by joint likelihood
-> observe x_new
-> compute class scores p(x_new | y)p(y)
-> normalize if probabilities are needed
-> choose class by posterior comparison
```

Geometry map:

```text
Sigma
-> inverse covariance in quadratic form
-> eigendirections, eigenvalues, and possible cross terms
-> Mahalanobis distance
-> Gaussian density contours
-> GDA class-conditional comparisons
-> boundary shape
```

Reliability map:

```text
assumption
-> fitted parameter
-> posterior score
-> decision
-> calibration and shift diagnostics
```
