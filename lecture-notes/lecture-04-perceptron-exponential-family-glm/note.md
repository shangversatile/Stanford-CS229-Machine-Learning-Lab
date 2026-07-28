# Lecture 4: Perceptron, Exponential Family, and Generalized Linear Models

Canonical reference: [Stanford CS229 supervised learning notes](https://cs229.stanford.edu/notes_archive/cs229-notes1.pdf), especially the sections on Perceptron, exponential family, generalized linear models, and softmax regression.

## Navigation

| Module | Purpose |
|---|---|
| [1. Core Question](#1-core-question) | What Lecture 4 is really trying to unify |
| [2. Perceptron](#2-perceptron-as-a-non-probabilistic-linear-classifier) | Linear classification without probabilistic modeling |
| [3. Perceptron vs GLM](#3-why-perceptron-is-discussed-before-glm) | Why linear score alone does not define a model |
| [4. Newton Bridge](#4-newton-method-as-an-optimization-bridge) | Why nonlinear likelihood models need iterative optimization |
| [5. Exponential Family Motivation](#5-why-exponential-family-is-introduced) | Why this family is introduced |
| [6. Exponential Family Anatomy](#6-anatomy-of-the-exponential-family) | What $\eta$, $T(y)$, $a(\eta)$, and $b(y)$ mean |
| [Conceptual Interlude A](#conceptual-interlude-a-from-response-space-to-probability-distribution) | How output space guides distribution choice |
| [7. Log-Partition Function](#7-log-partition-function-as-the-mathematical-engine) | Why $a(\eta)$ controls mean, variance, and convexity |
| [Mathematical Interlude B](#mathematical-interlude-b-why-exponential-family-mle-is-convex-friendly) | Why MLE/NLL has favorable geometry |
| [8. GLM Components](#8-glm-three-components-and-official-assumptions) | Random component, systematic component, link, response |
| [9. GLM Workflow](#9-the-complete-glm-modeling-workflow) | How to build a GLM from data semantics |
| [10. Hypothesis Function](#10-deep-meaning-of-the-hypothesis-function) | Why $h_\theta(x)$ is a conditional mean |
| [11. Gaussian GLM](#11-gaussian-glm) | Real-valued regression |
| [12. Bernoulli GLM](#12-bernoulli-glm) | Binary classification |
| [13. Poisson GLM](#13-poisson-glm) | Count regression |
| [14. Multinomial Form](#14-multinomial-exponential-family-form) | Multiclass sufficient statistic |
| [15. Softmax Response](#15-softmax-response-function) | Multiclass probability model |
| [16. Softmax Cross-Entropy](#16-softmax-likelihood-and-cross-entropy) | Multiclass NLL and gradient |
| [17. Reliability View](#17-reliability-view) | Failure modes and diagnostics |
| [18. PS1 Connection](#18-connection-to-ps1) | Assignment-gate connection |
| [19. Takeaways](#19-takeaways) | Final synthesis |

## 1. Core Question

Lecture 4 的核心问题不是“再学几个模型”，而是：给定一个 supervised learning 任务，怎样从输出变量的语义出发，构造一个合法、可解释、可优化、可诊断的 conditional model？

完整 pipeline 是：

```text
response semantics
-> conditional distribution
-> exponential-family representation
-> natural parameter
-> linear predictor
-> response function
-> likelihood
-> parameter learning
```

也就是说，先问 $y$ 是什么：real-valued measurement、binary event、multiclass label、count、positive duration、probability，还是 probability vector。然后选择一个支持集、mean-variance behavior、tail behavior 和数据生成机制都合理的 conditional distribution。若这个分布属于 exponential family，就可以通过 natural parameter $\eta$ 和 log-partition function $a(\eta)$ 得到统一的 response function 和 likelihood geometry。

两个容易混淆的点必须先分清：

* Multiclass classification 对应 categorical/multinomial distribution，不是 Poisson。
* Poisson 用于 count data，例如单位时间内事件次数、缺陷数、访问次数或事故数。

Newton method 是 Lecture 3 到 Lecture 4 的桥：它解释 logistic regression 和 GLM likelihood 如何用 curvature 加速优化。但 Lecture 4 的主线不是 Newton；主线是 Perceptron、exponential family、GLM，以及 multinomial/softmax model 的构造逻辑。

## 2. Perceptron as a Non-probabilistic Linear Classifier

Perceptron 使用 signed labels：

```math
y\in\{-1,+1\}
```

模型用一个 linear score 判断类别：

```math
\hat y=\mathrm{sign}(\theta^Tx)
```

这里 $\theta$ 是 decision boundary 的 normal vector。边界是所有满足 $\theta^Tx=0$ 的点；$\theta$ 的方向指向 score 为正的一侧，$\theta$ 的长度影响 score scale，但不改变边界方向。

对单个样本 $(x,y)$，如果它被正确分类，就应有 $y\theta^Tx>0$。Misclassification condition 是：

```math
y\theta^Tx\leq0
```

Perceptron 在犯错时更新：

```math
\theta_{\mathrm{new}}=\theta+\alpha yx
```

这个 update 的核心性质是：它必然提高当前样本的 signed score。证明如下：

```math
y\theta_{\mathrm{new}}^Tx
=
y(\theta+\alpha yx)^Tx
```

```math
=
y\theta^Tx+\alpha y^2x^Tx
```

```math
=
y\theta^Tx+\alpha\|x\|_2^2
```

因为 $y^2=1$ 且 $\alpha>0$，当前样本的 signed score 增加了 $\alpha\|x\|_2^2$。

这个公式给出几何解释：

* 若 $y=+1$ 且当前被错分，update 把 $\theta$ 往 $x$ 的方向拉，使正类样本更可能落在 positive side。
* 若 $y=-1$ 且当前被错分，update 把 $\theta$ 往 $-x$ 的方向拉，使负类样本更可能落在 negative side。
* 在二维中，$\theta$ 的旋转会带动 boundary $\theta^Tx=0$ 旋转，因为 boundary 总是垂直于 $\theta$。
* Perceptron 是 mistake-driven，不是 likelihood-driven；它没有默认的 calibrated probability interpretation。

![Perceptron vector update](../../assets/figures/lecture04-perceptron-vector-update.png)

## 3. Why Perceptron Is Discussed Before GLM

Perceptron 和 logistic regression 可以共享同一个 linear score $\theta^Tx$，也可以产生同一个 linear decision boundary $\theta^Tx=0$。但它们代表两种完全不同的建模哲学：Perceptron 直接修正 classification mistakes；logistic regression 先建模 $P(y=1\mid x;\theta)$，再通过 Bernoulli likelihood 学习参数。

| Aspect | Perceptron | Logistic regression |
| ------ | ---------- | ------------------- |
| Label convention | 常用 $y\in\{-1,+1\}$ | 常用 $y\in\{0,1\}$ |
| Linear score | $\theta^Tx$ | $\theta^Tx$ |
| Response function | hard sign or step | sigmoid probability |
| Prediction | $\hat y=\mathrm{sign}(\theta^Tx)$ | $P(y=1\mid x;\theta)=g(\theta^Tx)$ |
| Loss or criterion | mistake-driven update | Bernoulli NLL / cross-entropy |
| Update behavior | 只在错分时更新 | 每个样本按 probability residual 贡献 gradient |
| Probability interpretation | 默认没有 | 有 conditional probability interpretation |
| Calibration | 不保证 calibrated | 可做 calibration diagnostics |
| Convergence assumption | separable data 下有限犯错界 | convex objective；separation 时 MLE 可能发散 |
| Noise behavior | label noise 下可能持续震荡 | 可加 regularization 并分析 likelihood |

因此，Perceptron 放在 GLM 之前的价值是：它先展示 linear score 与 decision boundary geometry，再让 logistic regression 和 Bernoulli GLM 说明“同一个 score 怎样被概率化”。

![Perceptron versus logistic response](../../assets/figures/lecture04-perceptron-vs-logistic-response.png)

## 4. Newton Method as an Optimization Bridge

Newton method 的 multivariate optimization update 是：

```math
\theta_{t+1}
=
\theta_t-H(\theta_t)^{-1}\nabla J(\theta_t)
```

把优化问题写成 root finding，会更清楚。最优点满足 first-order condition：

```math
\nabla J(\theta)=0
```

令 $F(\theta)=\nabla J(\theta)$，Newton method 就是在当前点用一阶线性近似解 $F(\theta)=0$。因为 $F$ 的 Jacobian 是 Hessian，所以得到上面的 update。

对 quadratic objective：

```math
J(\theta)=\frac12\theta^TA\theta-b^T\theta+c
```

若 $A=A^T$ 且 $A$ nonsingular，则：

```math
\nabla J(\theta)=A\theta-b
```

```math
H(\theta)=A
```

Newton update 为：

```math
\theta_{t+1}
=
\theta_t-A^{-1}(A\theta_t-b)
```

```math
=
\theta_t-\theta_t+A^{-1}b
```

```math
=
A^{-1}b
```

最优解满足 $A\theta^\star=b$，所以 $\theta_{t+1}=\theta^\star$。因此，对 exact quadratic 且 Hessian 可逆的问题，Newton method one-step convergence。

Least squares 是这个结论的直接例子。令：

```math
J(\theta)=\frac12\|X\theta-y\|_2^2
```

则：

```math
\nabla J(\theta)=X^T(X\theta-y)
```

```math
H=X^TX
```

如果 $X^TX$ invertible，Newton update 一步到达：

```math
\theta^\star=(X^TX)^{-1}X^Ty
```

这解释了 normal equation 与 Newton method 的关系：least squares 的 curvature 是 constant，所以 Newton 不需要迭代来更新 curvature。

实践中要谨慎：

* Singular Hessian：例如 feature rank deficient，$H^{-1}$ 不存在，需要 regularization、pseudoinverse 或重新设计 features。
* Ill-conditioned Hessian：数值误差会放大，直接求逆不稳定，应解线性系统并考虑 damping。
* Indefinite Hessian：对 minimization，Newton step 可能朝非下降方向走，需要 modified Newton 或 trust-region。
* Initialization：Newton 是局部二阶方法，离 optimum 太远时 quadratic approximation 可能误导。
* Computation：每步构造和分解 Hessian 成本高，large-scale GLM 常用 gradient、stochastic 或 quasi-Newton。
* Separation：logistic/softmax 在 separable data 上可能没有 finite MLE，Newton 可能追逐越来越大的参数。
* Damping and line search：实际 update 常写成 $\theta_{t+1}=\theta_t-\gamma_tH^{-1}\nabla J$，其中 $0<\gamma_t\leq1$。

![Newton curvature bridge](../../assets/figures/lecture04-newton-curvature-bridge.png)

## 5. Why Exponential Family Is Introduced

Exponential family 的 canonical form 是：

```math
p(y;\eta)=b(y)\exp\left(\eta^TT(y)-a(\eta)\right)
```

它不是单纯为了“形式漂亮”，而是同时统一了五件事：

* probability representation：Gaussian、Bernoulli、Poisson、multinomial 等可以用同一种代数结构表示；
* sufficient statistics：数据通过 $T(y)$ 进入 likelihood；
* moment identities：$a(\eta)$ 的导数直接给出 mean 和 covariance；
* convex likelihood geometry：log-likelihood 对 natural parameter 通常是 concave；
* GLM construction：把 $\eta$ 设成 linear predictor，就得到 distribution-derived response function。

因此，exponential family 是从“单个算法公式”到“系统建模 recipe”的关键桥梁。

## 6. Anatomy of the Exponential Family

| Component | Symbol | Meaning | Modeling role |
| --------- | ------ | ------- | ------------- |
| Natural parameter | $\eta$ | 控制分布形状的 canonical parameter | GLM 中通常由 linear predictor 给出 |
| Sufficient statistic | $T(y)$ | 数据进入 likelihood 的统计量 | 决定被模型直接匹配的 empirical summary |
| Log-partition function | $a(\eta)$ | normalization 的对数 | 生成 moments 并决定 convex geometry |
| Base measure | $b(y)$ | 与 $\eta$ 无关的部分 | 保留 support 和 reference weighting |
| Dispersion parameter | optional | 控制 scale 或 variance | 有些 GLM 推广会显式加入 |

$T(y)$ 不总是 $y$。它是分布族需要从 observation 中提取的 sufficient statistic，而不是“标签本身”的同义词。

| Distribution setting | Common sufficient statistic | Why |
| -------------------- | --------------------------- | --- |
| Bernoulli | $T(y)=y$ | 单个 binary outcome 的成功次数就是信息 |
| Fixed-variance Gaussian | $T(y)=y$ | 已知方差时只需估计 mean |
| Unknown-variance Gaussian | $T(y)=(y,y^2)$ | mean 和 second moment 都携带参数信息 |
| Multinomial / categorical | indicator vector | 每一类的 one-hot count 是信息 |

对 iid observations，factorization 使全部样本通过 sufficient statistic 的和进入 natural-parameter-dependent likelihood：

```math
\sum_{i=1}^{m}T(y^{(i)})
```

这就是为什么 exponential family 会自然连接到 MLE：重复观测把 sample evidence 聚合成一个 statistic，而不是保留每个 $y^{(i)}$ 的全部细节。

![Exponential-family anatomy](../../assets/figures/lecture04-exponential-family-anatomy.png)

---

# Conceptual Interlude A: From Response Space to Probability Distribution

> This interlude is a modeling map. It explains how the semantic type of $Y$ constrains the probability distribution and therefore the GLM response function.

---

## A. The modeling question comes before the algorithm

Before choosing a loss, optimizer, or activation-looking function, define what the response variable means. The first modeling questions are:

* What kind of object is $Y$?
* What values can $Y$ legally take?
* Is $Y$ continuous, binary, categorical, count-valued, positive, or probability-valued?
* What uncertainty structure is plausible?
* Is variance constant, mean-dependent, heavy-tailed, bounded, or compositional?

The output space does not uniquely determine the distribution, but it rules out many invalid distributions. For example, a count target should not be modeled with unconstrained Gaussian regression when negative predictions are impossible or harmful. A Gaussian model can still be a useful approximation in some large-count regimes, but then the approximation and its failure modes must be explicit.

## B. Response support is necessary but not sufficient

A support set is the set of values a random variable can legally take.

```math
\mathbb R=(-\infty,\infty)
```

```math
\mathbb R_{>0}=(0,\infty)
```

```math
\mathbb R_{\geq0}=[0,\infty)
```

```math
\Delta^{K-1}=\left\{p\in\mathbb R^K:p_k\geq0,\ \sum_{k=1}^{K}p_k=1\right\}
```

Support tells us where predictions or random outcomes are allowed to live. But choosing a distribution also requires thinking about variance, skewness, tails, discreteness, zero-inflation, dependence, and mechanism. Two distributions can share the same support while encoding very different assumptions; Gamma and Exponential are both positive continuous families, but Gamma can express a wider range of positive skew and variance behavior.

## C. Distribution map: support, meaning, and model role

| Response type | Support | Task type | Candidate distribution | Mean | Variance pattern | GLM response | Official CS229 status |
| ------------- | ------- | --------- | ---------------------- | ---- | ---------------- | ------------ | --------------------- |
| Real-valued continuous | $\mathbb R$ | regression | Gaussian | $\mu$ | constant if variance is fixed | identity | fully derived |
| Binary | $\{0,1\}$ | binary classification | Bernoulli | $\phi$ | $\phi(1-\phi)$ | sigmoid | fully derived |
| Multiclass categorical | $\{1,\dots,K\}$ | multiclass classification | categorical / multinomial one-trial | $\phi_k$ | coupled categorical covariance | softmax | fully derived |
| Count-valued | $\mathbb N_0=\{0,1,2,\dots\}$ | count regression | Poisson | $\lambda$ | variance equals mean | exponential | mentioned / problem-set related / useful GLM example |
| Positive continuous | $\mathbb R_{>0}$ | waiting time, duration, survival-like positive response | Exponential / Gamma | positive mean | right-skewed, often mean-dependent | usually inverse or log-linked depending parameterization | mentioned as exponential-family members |
| Scalar probability | $(0,1)$ | rate/proportion as target | Beta | $\alpha/(\alpha+\beta)$ | bounded, shape-dependent | mean in $(0,1)$, link chosen separately | mentioned as distribution over probabilities |
| Probability vector | $\Delta^{K-1}$ | composition/proportion vector | Dirichlet | simplex-valued mean | negative covariance across components | simplex-valued mean | mentioned as distribution over probabilities |

The important distinction is that multiclass is not Poisson. Poisson is for count data generated by a rate mechanism. A category ID such as $3$ is not three events; it is a label for one mutually exclusive class.

## D. Distribution-by-distribution explanations

#### Gaussian: real-valued continuous response

**When to use.** Use Gaussian modeling when $Y$ is a real-valued measurement and additive, roughly symmetric noise is plausible. Examples include temperature residuals, height after controlling for features, or a continuous sensor measurement.

**Random variable.** $Y\in\mathbb R$ with location parameter $\mu$ and variance parameter $\sigma^2>0$.

**Density.**

```math
p(y;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
```

**Mean and variance.**

```math
\mathbb E[Y]=\mu,\qquad \mathrm{Var}(Y)=\sigma^2
```

**GLM meaning.** In the fixed-variance canonical Gaussian GLM, the response mean is identity: $h_\theta(x)=\theta^Tx$. Squared loss appears because it is the Gaussian negative log likelihood up to constants.

**Failure modes.** Heavy tails, asymmetric residuals, bounded outcomes, nonconstant variance, or harmful negative predictions can make a Gaussian conditional model misleading.

#### Bernoulli: binary response

**When to use.** Use Bernoulli modeling when each observation is a yes/no event: default or no default, click or no click, disease present or absent.

**Random variable.** $Y\in\{0,1\}$ with success probability $\phi\in(0,1)$.

**PMF.**

```math
p(y;\phi)=\phi^y(1-\phi)^{1-y},\qquad y\in\{0,1\}
```

**Mean and variance.**

```math
\mathbb E[Y]=\phi,\qquad \mathrm{Var}(Y)=\phi(1-\phi)
```

**GLM meaning.** The natural parameter is log-odds. Setting log-odds equal to $\theta^Tx$ gives the sigmoid response and binary cross-entropy NLL.

**Failure modes.** Class imbalance, complete separation, label noise, uncalibrated probabilities, and time-varying base rates can break the practical interpretation even when the support is correct.

#### Categorical / Multinomial one-trial: multiclass response

**When to use.** Use categorical modeling when each sample belongs to exactly one of $K$ mutually exclusive classes, such as digit identity, species class, or topic label.

**Random variable.** $Y\in\{1,\dots,K\}$ with class probabilities $\phi_1,\dots,\phi_K$ satisfying $\phi_k\geq0$ and $\sum_k\phi_k=1$.

**PMF.**

```math
p(y=k;\phi)=\phi_k,\qquad \sum_{k=1}^{K}\phi_k=1
```

For one-hot vector $T(Y)$:

```math
\mathbb E[T_k(Y)]=\phi_k
```

**Covariance.**

```math
\mathrm{Cov}(T_i(Y),T_j(Y))=
\begin{cases}
\phi_i(1-\phi_i), & i=j\\
-\phi_i\phi_j, & i\neq j
\end{cases}
```

**GLM meaning.** Softmax is the response function because all class probabilities must be jointly normalized. Raising one class probability lowers available probability mass for others.

**Failure modes.** Treating class IDs as ordered numbers, using Poisson on labels, or fitting independent one-vs-rest probabilities without normalization can produce invalid multiclass probability semantics.

#### Poisson: count-valued response

**When to use.** Use Poisson modeling for nonnegative integer counts under a rate/exposure mechanism, such as arrivals per hour, defects per batch, or calls per minute.

**Random variable.** $Y\in\mathbb N_0=\{0,1,2,\dots\}$ with rate $\lambda>0$.

**PMF.**

```math
p(y;\lambda)=\frac{\lambda^y e^{-\lambda}}{y!},\qquad y\in\mathbb N_0
```

**Mean and variance.**

```math
\mathbb E[Y]=\lambda,\qquad \mathrm{Var}(Y)=\lambda
```

**GLM meaning.** The natural parameter is $\eta=\log\lambda$. Setting $\eta=\theta^Tx$ gives $h_\theta(x)=e^{\theta^Tx}$, a nonnegative conditional mean.

**Failure modes.** Overdispersion, underdispersion, excess zeros, varying exposure, dependence between events, and bursty arrival processes often require offsets, quasi-Poisson, negative binomial, or richer count models.

#### Exponential: positive waiting-time response

**When to use.** Use Exponential modeling for positive waiting times when a memoryless mechanism is plausible, such as the time until the next event in a simple constant-rate process.

**Random variable.** $Y\in\mathbb R_{>0}$ with rate $\lambda>0$.

**Density.**

```math
p(y;\lambda)=\lambda e^{-\lambda y},\qquad y>0
```

**Mean and variance.**

```math
\mathbb E[Y]=\frac{1}{\lambda},\qquad \mathrm{Var}(Y)=\frac{1}{\lambda^2}
```

**GLM meaning.** It shows how a positive continuous response can be tied to a rate or mean parameter. It is useful conceptually in Lecture 4 as an exponential-family member, even though the note does not develop a full Exponential GLM workflow.

**Failure modes.** Non-memoryless hazards, delayed effects, censoring, heavy tails, or a point mass near zero make a plain Exponential model too restrictive.

#### Gamma: positive continuous response

**When to use.** Use Gamma modeling for positive continuous outcomes with right skew, such as costs, durations, rainfall amounts, or biological concentrations.

**Random variable.** $Y\in\mathbb R_{>0}$ with shape $\alpha>0$ and rate $\beta>0$.

**Density.**

```math
p(y;\alpha,\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}y^{\alpha-1}e^{-\beta y},\qquad y>0
```

**Mean and variance.**

```math
\mathbb E[Y]=\frac{\alpha}{\beta},\qquad \mathrm{Var}(Y)=\frac{\alpha}{\beta^2}
```

**GLM meaning.** Gamma GLMs are common for positive responses where variance grows with the mean. Link choice is separate from support; log and inverse links are both used depending on parameterization and modeling goals.

**Failure modes.** Exact zeros, extreme heavy tails, multimodality, censoring, or mixtures of mechanisms can violate a single Gamma conditional model.

#### Beta: scalar probability response

**When to use.** Use Beta modeling when the response itself is a scalar probability or proportion in $(0,1)$, such as a conversion rate measured over a unit interval or a fractional occupancy target.

**Random variable.** $Y\in(0,1)$ with shape parameters $\alpha>0$ and $\beta>0$.

**Density.**

```math
p(y;\alpha,\beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}y^{\alpha-1}(1-y)^{\beta-1},\qquad 0<y<1
```

**Mean.**

```math
\mathbb E[Y]=\frac{\alpha}{\alpha+\beta}
```

**Variance.**

```math
\mathrm{Var}(Y)=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
```

**GLM meaning.** Beta is not the usual distribution for a binary label; Bernoulli is. Beta is for random variables whose observed values are probabilities or proportions.

**Failure modes.** Exact $0$ or $1$ values, denominators with different reliability, and aggregated binomial counts can require boundary handling or a binomial model with exposure instead.

#### Dirichlet: probability-vector response

**When to use.** Use Dirichlet modeling when the response itself is a composition or probability vector, such as topic proportions, mixture weights, or normalized budget shares.

**Random variable.** $p\in\Delta^{K-1}$ with concentration vector $\alpha_k>0$.

**Density.**

```math
p(p;\alpha)=\frac{\Gamma(\sum_{k=1}^{K}\alpha_k)}{\prod_{k=1}^{K}\Gamma(\alpha_k)}\prod_{k=1}^{K}p_k^{\alpha_k-1}
```

with:

```math
p\in\Delta^{K-1}
```

**Mean.**

```math
\mathbb E[p_k]=\frac{\alpha_k}{\sum_{j=1}^{K}\alpha_j}
```

**Covariance.**

```math
\mathrm{Cov}(p_i,p_j)=
\begin{cases}
\frac{\alpha_i(\alpha_0-\alpha_i)}{\alpha_0^2(\alpha_0+1)}, & i=j\\
-\frac{\alpha_i\alpha_j}{\alpha_0^2(\alpha_0+1)}, & i\neq j
\end{cases}
```

where:

```math
\alpha_0=\sum_{k=1}^{K}\alpha_k
```

**GLM meaning.** Dirichlet models a probability vector as the response or as a distribution over probability parameters. Basic softmax classification instead models a class label whose conditional mean is a probability vector.

**Failure modes.** Structural zeros, stronger correlations than Dirichlet allows, multimodal compositions, or subcomposition effects can require logistic-normal or other compositional models.

## E. What this distribution controls inside a GLM

The chosen distribution controls:

1. legal output support;
2. conditional mean form;
3. mean-variance relationship;
4. likelihood;
5. loss function;
6. Hessian/curvature;
7. calibration interpretation;
8. failure diagnostics.

For Gaussian regression, choosing a fixed-variance Gaussian makes $\theta^Tx$ the conditional mean and turns MLE into squared-loss minimization. For Bernoulli classification, choosing Bernoulli makes the prediction a calibrated event probability when the model is correct and turns NLL into binary cross-entropy. For Poisson regression, choosing the distribution enforces a nonnegative mean prediction through $e^{\theta^Tx}$ and ties variance to the mean. For softmax classification, choosing categorical/multinomial modeling makes probabilities coupled through one normalization denominator and turns NLL into multiclass cross-entropy.

## F. Official CS229 core vs extension layer

| Distribution | Lecture 4 role |
| ------------ | -------------- |
| Gaussian | Official core derivation |
| Bernoulli | Official core derivation |
| Multinomial / Softmax | Official core derivation |
| Poisson | Mentioned and natural GLM extension / problem-set-level |
| Gamma / Exponential | Mentioned as exponential-family examples |
| Beta / Dirichlet | Mentioned as distributions over probabilities |
| Others | Outside this lecture |

The official Lecture 4 core is the derivation pattern: write a distribution in exponential-family form, identify the natural parameter, set that parameter to a linear predictor, and derive the response mean. The extension layer broadens modeling intuition without claiming that every listed family is fully developed in the lecture.

## G. Common misunderstandings

1. Multiclass is not Poisson. Multiclass labels are mutually exclusive categories; Poisson variables are nonnegative event counts.
2. Softmax is not independent one-vs-rest logistic regression. Softmax probabilities are coupled and sum to one by construction.
3. Support does not uniquely determine a distribution. Positive continuous data could be Exponential, Gamma, log-normal, Weibull, or another family depending on mechanism and tails.
4. $h_\theta(x)$ is not the likelihood-maximizing parameter; it is the conditional mean under the learned model.
5. The response function is not merely an arbitrary activation function. In a GLM it comes from a distributional assumption plus a link function.
6. Beta and Dirichlet are not the usual starting point for basic classification; they model probability-valued random variables or priors over probability parameters.

## H. Applying the map to real modeling problems

Use the same reasoning pattern every time: define $Y$, write its legal support, name the data-generating mechanism, choose a first candidate distribution, then state what would falsify that choice.

| Problem | First modeling read | Candidate start | What to check next |
| ------- | ------------------- | --------------- | ------------------ |
| Loan default | one binary event per applicant | Bernoulli / logistic GLM | calibration, separation, class imbalance, subgroup shift |
| Arrivals per hour | nonnegative event count under exposure | Poisson GLM with possible exposure offset | overdispersion, excess zeros, time dependence |
| Medical cost | positive continuous, right-skewed amount | Gamma GLM or log-normal model | exact zeros, heavy tails, mixture of patient groups |
| Time until repair | positive duration | Exponential as simple baseline, Gamma or survival model if richer | censoring, nonconstant hazard, delayed effects |
| Image class label | one mutually exclusive class | categorical / softmax GLM | rare classes, label ambiguity, calibration |
| Click-through rate as an observed proportion | probability-valued or aggregated binomial outcome | Beta if direct proportion, binomial if successes/trials known | boundary values, denominator size, heterogeneity |
| Topic mixture vector | probability vector response | Dirichlet or compositional model | structural zeros, component dependence, multimodality |

A strong answer does not say only “the label is numeric, so use regression.” It says what the number means. If the number is a class ID, use categorical/softmax; if it is a count, use a count model; if it is a positive amount, use a positive continuous model; if it is a probability, use a model whose random variable lives in $(0,1)$ or on the simplex.
---

Return to main Lecture 4 flow: [7. Log-Partition Function](#7-log-partition-function-as-the-mathematical-engine).

## 7. Log-Partition Function as the Mathematical Engine

先定义 unnormalized normalizer：

```math
Z(\eta)=\int b(y)e^{\eta^TT(y)}dy
```

离散情形把 integral 换成 sum。Log-partition function 定义为：

```math
a(\eta)=\log Z(\eta)
```

因为：

```math
\int b(y)\exp\left(\eta^TT(y)-a(\eta)\right)dy
=
e^{-a(\eta)}Z(\eta)
=
1
```

$a(\eta)$ 正是让 distribution normalized 的那一项。

在可交换 differentiation and integration 的 regularity conditions 下：

```math
\nabla Z(\eta)
=
\int b(y)e^{\eta^TT(y)}T(y)dy
```

因此：

```math
\nabla a(\eta)
=
\frac{1}{Z(\eta)}\nabla Z(\eta)
```

```math
=
\int T(y)
\frac{b(y)e^{\eta^TT(y)}}{Z(\eta)}
dy
```

```math
=
\mathbb E_{\eta}[T(Y)]
```

再对 mean 求导。令 $\mu_T(\eta)=\mathbb E_\eta[T(Y)]$，则：

```math
\nabla^2a(\eta)
=
\nabla \mu_T(\eta)
```

更直接地，用二阶导数展开：

```math
\nabla^2a(\eta)
=
\frac{1}{Z(\eta)}
\nabla^2Z(\eta)
-
\frac{1}{Z(\eta)^2}
\nabla Z(\eta)\nabla Z(\eta)^T
```

其中：

```math
\nabla^2Z(\eta)
=
\int b(y)e^{\eta^TT(y)}T(y)T(y)^Tdy
```

代回去得到：

```math
\nabla^2a(\eta)
=
\mathbb E_\eta[T(Y)T(Y)^T]
-
\mathbb E_\eta[T(Y)]\mathbb E_\eta[T(Y)]^T
```

也就是：

```math
\nabla^2a(\eta)=\mathrm{Cov}_{\eta}(T(Y))
```

Scalar case 中：

```math
a''(\eta)=\mathrm{Var}_{\eta}(T(Y))
```

任意向量 $v$ 满足：

```math
v^T\nabla^2a(\eta)v
=
v^T\mathrm{Cov}_{\eta}(T(Y))v
=
\mathrm{Var}_{\eta}(v^TT(Y))
\geq0
```

所以：

```math
\nabla^2a(\eta)\succeq0
```

因此 $a(\eta)$ 是 convex function。这个 convexity 不是额外假设，而是 normalizing a probability distribution 的代数后果。

![Log-partition moments](../../assets/figures/lecture04-log-partition-moments.png)

---

# Mathematical Interlude B: Why Exponential-Family MLE Is Convex-Friendly

> This interlude explains why exponential-family likelihoods have favorable optimization geometry through the log-partition function.

---

For iid data $y^{(1)},\ldots,y^{(m)}$, the log-likelihood is:

```math
\ell(\eta)
=
\sum_{i=1}^{m}
\log b(y^{(i)})
+
\eta^T\sum_{i=1}^{m}T(y^{(i)})
-
ma(\eta)
```

Equivalently:

```math
\ell(\eta)
=
\eta^T\sum_{i=1}^{m}T(y^{(i)})
-ma(\eta)
+\sum_{i=1}^{m}\log b(y^{(i)})
```

The sample statistic term is linear in $\eta$, and $\log b(y^{(i)})$ is independent of $\eta$. Therefore the second derivative comes only from $-ma(\eta)$:

```math
\nabla^2\ell(\eta)
=
-m\nabla^2a(\eta)
```

Using $\nabla^2a(\eta)=\mathrm{Cov}(T(Y))$:

```math
\nabla^2\ell(\eta)
=
-m\,\mathrm{Cov}(T(Y))
\preceq0
```

The conclusions are:

* log-likelihood is concave in the natural parameter;
* negative log likelihood is convex in the natural parameter;
* the MLE estimate itself is not a concave object, because concavity describes the objective;
* strict convexity, finite MLE, and unique optimum also need rank, identifiability, support, and existence conditions.

In a scalar natural-parameter GLM, set:

```math
\eta_i=\theta^Tx^{(i)}
```

Ignoring constants, the single-sample NLL is:

```math
J_i(\theta)=a(\eta_i)-\eta_iT(y^{(i)})
```

Its gradient is:

```math
\nabla_\theta J_i
=
\left(a'(\eta_i)-T(y^{(i)})\right)x^{(i)}
```

Its Hessian is:

```math
\nabla_\theta^2J_i
=
a''(\eta_i)x^{(i)}x^{(i)T}
```

Using $a''(\eta_i)=\mathrm{Var}(T(Y^{(i)})\mid x^{(i)})$:

```math
\nabla_{\theta}^{2}J_{\mathrm{NLL}}
=
\sum_{i=1}^{m}
\mathrm{Var}\left(T(Y^{(i)})\mid x^{(i)}\right)
x^{(i)}x^{(i)T}
```

So a GLM Hessian can be read as variance-weighted feature geometry: features determine directions, and conditional variance determines how much curvature each sample contributes.

---

Return to main Lecture 4 flow: [8. GLM Components](#8-glm-three-components-and-official-assumptions).

## 8. GLM Three Components and Official Assumptions

| Modern GLM term | CS229 assumption/design | Role |
| --------------- | ----------------------- | ---- |
| Random component | $Y$ given $x;\theta$ belongs to exponential family | conditional response distribution |
| Prediction target | $h_\theta(x)=\mathbb E[T(Y)\mid x;\theta]$ | response mean |
| Systematic component | $\eta=\theta^Tx$ | linear predictor |
| Link function | $g(\mu)=\eta$ | maps mean to linear predictor |
| Response function | $\mu=g^{-1}(\eta)$ | maps predictor to mean |

Terminology convention 很重要。许多 statistics texts 使用 $g$ 表示 link function，即 $g(\mu)=\eta$；response function 是 $g^{-1}$。CS229 notes 在一些位置把 $g$ 用作 response function，例如 logistic regression 中 $g(z)=1/(1+e^{-z})$。因此读 Lecture 4 时要看上下文：如果 $g$ 输入 linear score 并输出 mean/probability，它是 response function；如果 $g$ 输入 mean 并输出 natural parameter，它是 link function。

## 9. The Complete GLM Modeling Workflow

A GLM is built by moving from response semantics to distribution, then from distribution to likelihood and prediction.

1. Define the response variable $Y$ precisely. This prevents a numeric code from being mistaken for a measurement, count, or probability.
2. Identify support and measurement mechanism. Support rules out illegal distributions, while mechanism explains whether the data are measurements, events, arrivals, durations, or compositions.
3. Choose a candidate conditional distribution for $Y\mid x$. The distribution encodes uncertainty, support, and a mean-variance relationship.
4. Write its PMF/PDF. This makes the likelihood concrete instead of choosing a loss by habit.
5. Rewrite it in exponential-family form. The canonical form reveals the natural parameter and the log-partition function.
6. Identify $T(y)$, $\eta$, $a(\eta)$, and $b(y)$. These components tell us what statistic is modeled, what is linearized, what normalizes the distribution, and what support/base weighting remains.
7. Decide whether to use canonical link. The canonical link often gives simpler gradients and convex-friendly likelihood geometry, but noncanonical links may be useful for domain reasons.
8. Set the linear predictor. In the scalar canonical case this means $\eta=\theta^Tx$; in multiclass models it means class-specific scores.
9. Derive the response mean. The prediction is $\mathbb E[T(Y)\mid x;\theta]$, not an arbitrary nonlinear transformation.
10. Write likelihood over all samples. Multiplying conditional probabilities states the iid or conditional-independence assumption being used for training.
11. Convert likelihood into NLL. The NLL is the loss induced by the chosen distribution.
12. Optimize parameters. Training estimates the shared parameter values that make the observed data most plausible under the model family.
13. Predict using conditional mean or class probability. Regression reports a mean, binary classification reports an event probability or thresholded label, and multiclass classification reports coupled class probabilities.
14. Diagnose model assumptions. After fitting, check support, calibration, residual structure, overdispersion, separation, identifiability, and train/deployment shift.

Frequentist MLE should be read carefully:

* $\theta$ is a fixed but unknown parameter;
* data are random because they come from a sampling or data-generating process;
* $\hat\theta(D)$ is an estimator that changes with the dataset $D$;
* training chooses the parameter that makes the observed sample plausible, not a posterior distribution over $\theta$.

Bernoulli mini-example:

```text
Binary outcome: loan default yes/no.
Y in {0,1}.
Choose Bernoulli.
Natural parameter is log-odds.
Set log-odds = theta^T x.
Response mean becomes sigmoid.
Likelihood becomes Bernoulli likelihood.
NLL becomes binary cross-entropy.
```

Poisson mini-example:

```text
Count outcome: number of arrivals per hour.
Y in N0.
Choose Poisson.
Natural parameter is log lambda.
Set log rate = theta^T x.
Response mean becomes exp(theta^T x).
NLL becomes Poisson deviance-like objective.
```

![GLM construction pipeline](../../assets/figures/lecture04-glm-construction-pipeline.png)

## 10. Deep Meaning of the Hypothesis Function

In a GLM, the hypothesis function is not the parameter that maximizes probability. It is the model prediction after parameters have been learned.

### A. Separate parameter learning from prediction

Training estimates parameters from the dataset:

```math
\hat\theta=\underset{\theta}{\mathrm{argmax}}\ p(D\mid\theta)
```

Prediction uses the learned parameter inside the conditional mean:

```math
h_{\hat\theta}(x)=\mathbb E[T(Y)\mid x;\hat\theta]
```

The hypothesis function is therefore not “the parameter that maximizes the probability.” The learned parameter is $\hat\theta$; the prediction is $h_{\hat\theta}(x)$, the conditional mean implied by the fitted model.

### B. Response function as inverse link

Statistics convention usually defines the link function as the map from mean to linear predictor:

```math
g(\mu)=\eta
```

The response function is the inverse map:

```math
\mu=g^{-1}(\eta)
```

In a canonical GLM:

```math
\eta=\theta^Tx
```

Therefore:

```math
h_\theta(x)=\mu=g^{-1}(\theta^Tx)
```

The exponential-family moment identity gives the same object through the log-partition function:

```math
\mu=\mathbb E[T(Y)\mid \eta]=\nabla a(\eta)
```

So in the canonical case:

```math
h_\theta(x)=\nabla a(\theta^Tx)
```

Examples:

| Distribution | Natural linear predictor | Response mean |
| ------------ | ------------------------ | ------------- |
| Gaussian | $\eta=\theta^Tx$ | identity: $h_\theta(x)=\theta^Tx$ |
| Bernoulli | log-odds $\eta=\theta^Tx$ | sigmoid: $h_\theta(x)=1/(1+e^{-\theta^Tx})$ |
| Poisson | log-rate $\eta=\theta^Tx$ | exponential: $h_\theta(x)=e^{\theta^Tx}$ |
| Multinomial | class scores $\eta_k=\theta_k^Tx$ | softmax probabilities |

### C. Activation-function comparison

It is legitimate to notice that sigmoid, exponential, and softmax are nonlinear maps from scores to outputs. But GLM response functions are not chosen only for computational convenience. They are derived from distributional assumptions and link functions, so they carry a likelihood, a mean-variance relationship, a calibration interpretation, and model-specific diagnostics.

## 11. Gaussian GLM

采用 official fixed-variance derivation，令 variance 为 $1$。Gaussian density：

```math
p(y;\mu)
=
\frac{1}{\sqrt{2\pi}}
\exp\left(-\frac12(y-\mu)^2\right)
```

展开：

```math
p(y;\mu)
=
\frac{1}{\sqrt{2\pi}}
\exp\left(-\frac12y^2+\mu y-\frac12\mu^2\right)
```

写成 exponential-family form：

```math
p(y;\eta)=b(y)\exp\left(\eta T(y)-a(\eta)\right)
```

其中：

```math
\eta=\mu
```

```math
T(y)=y
```

```math
a(\eta)=\frac{\eta^2}{2}
```

```math
b(y)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{y^2}{2}\right)
```

Moment identity 给出：

```math
h_\theta(x)
=
\mathbb E[Y\mid x;\theta]
=
a'(\eta)
=
\eta
```

canonical link 设 $\eta=\theta^Tx$，所以：

```math
h_\theta(x)=\theta^Tx
```

Gaussian log likelihood 忽略与 $\theta$ 无关的常数后等价于：

```math
-\frac12\sum_{i=1}^{m}
\left(y^{(i)}-\theta^Tx^{(i)}\right)^2
```

因此 maximizing Gaussian likelihood 等价于 minimizing squared loss。Squared loss 不是任意选择；它是 fixed-variance Gaussian conditional model 的 NLL。

## 12. Bernoulli GLM

Bernoulli distribution：

```math
p(y;\phi)=\phi^y(1-\phi)^{1-y},
\quad y\in\{0,1\}
```

先取 log 并整理 exponent：

```math
p(y;\phi)
=
\exp\left(y\log\phi+(1-y)\log(1-\phi)\right)
```

```math
=
\exp\left(
y\log\frac{\phi}{1-\phi}
+
\log(1-\phi)
\right)
```

与 exponential-family form 对比：

```math
p(y;\eta)=b(y)\exp\left(\eta T(y)-a(\eta)\right)
```

得到：

```math
T(y)=y
```

```math
\eta=\log\frac{\phi}{1-\phi}
```

```math
b(y)=1
```

为了得到 $a(\eta)$，先解出 $\phi$。从：

```math
\eta=\log\frac{\phi}{1-\phi}
```

指数化：

```math
e^\eta=\frac{\phi}{1-\phi}
```

整理：

```math
e^\eta(1-\phi)=\phi
```

```math
e^\eta=\phi(1+e^\eta)
```

所以：

```math
\phi=\frac{e^\eta}{1+e^\eta}
=
\frac{1}{1+e^{-\eta}}
```

同时：

```math
1-\phi=\frac{1}{1+e^\eta}
```

因为 exponential-family form 中 $\log(1-\phi)=-a(\eta)$，所以：

```math
a(\eta)=\log(1+e^\eta)
```

Moment identity 给出：

```math
h_\theta(x)
=
\mathbb E[Y\mid x;\theta]
=
a'(\eta)
=
\frac{e^\eta}{1+e^\eta}
```

canonical link 设 $\eta=\theta^Tx$，得到：

```math
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
```

所以 sigmoid 不是随意画出的 S-curve，而是 Bernoulli exponential-family representation 加 canonical linear predictor 的必然结果。

## 13. Poisson GLM

Poisson distribution：

```math
p(y;\lambda)=\frac{\lambda^ye^{-\lambda}}{y!}
```

其中 $y\in\{0,1,2,\ldots\}$，$\lambda>0$ 是 rate 或 mean count。整理为：

```math
p(y;\lambda)
=
\frac{1}{y!}
\exp\left(y\log\lambda-\lambda\right)
```

设：

```math
\eta=\log\lambda
```

则 $\lambda=e^\eta$，所以：

```math
p(y;\eta)
=
\frac{1}{y!}
\exp\left(y\eta-e^\eta\right)
```

因此：

```math
T(y)=y
```

```math
a(\eta)=e^\eta
```

```math
b(y)=\frac{1}{y!}
```

Moment identity：

```math
h_\theta(x)
=
\mathbb E[Y\mid x;\theta]
=
a'(\eta)
=
e^\eta
```

canonical link 设 $\eta=\theta^Tx$，于是：

```math
h_\theta(x)=\mathbb E[Y\mid x;\theta]=e^{\theta^Tx}
```

这保证 prediction 是 positive count-rate。注意 prediction 是 expected count/rate，不是必须输出整数；实际 observed $Y$ 仍是非负整数随机变量。

![Gaussian, Bernoulli, and Poisson response functions](../../assets/figures/lecture04-gaussian-bernoulli-poisson-response.png)

## 14. Multinomial Exponential-Family Form

对 $K$ 类 categorical outcome，使用 $K-1$ 个 independent probability parameters，并把第 $K$ 类作为 reference class。令：

```math
\phi_K=1-\sum_{k=1}^{K-1}\phi_k
```

定义 sufficient statistic：

```math
(T(y))_k=\mathbf1\{y=k\},
\quad k=1,\ldots,K-1
```

它的 expectation 是 class probability：

```math
\mathbb E[(T(Y))_k]=P(Y=k)=\phi_k
```

Categorical PMF 可以写成：

```math
p(y;\phi)
=
\prod_{k=1}^{K}\phi_k^{\mathbf1\{y=k\}}
```

把 reference class 拆出来：

```math
p(y;\phi)
=
\phi_K
\prod_{k=1}^{K-1}
\left(\frac{\phi_k}{\phi_K}\right)^{\mathbf1\{y=k\}}
```

取 exponential form：

```math
p(y;\phi)
=
\exp\left(
\sum_{k=1}^{K-1}\mathbf1\{y=k\}
\log\frac{\phi_k}{\phi_K}
+
\log\phi_K
\right)
```

因此 natural parameters 是 log odds against the reference class：

```math
\eta_k=\log\frac{\phi_k}{\phi_K},
\quad k=1,\ldots,K-1
```

指数化并归一化：

```math
e^{\eta_k}=\frac{\phi_k}{\phi_K}
```

```math
\phi_k=e^{\eta_k}\phi_K
```

利用 $\sum_{k=1}^{K}\phi_k=1$：

```math
\phi_K\left(1+\sum_{k=1}^{K-1}e^{\eta_k}\right)
=
1
```

```math
\phi_K
=
\frac{1}{1+\sum_{j=1}^{K-1}e^{\eta_j}}
```

所以：

```math
\phi_k
=
\frac{e^{\eta_k}}
{1+\sum_{j=1}^{K-1}e^{\eta_j}},
\quad k=1,\ldots,K-1
```

这就是 reference-class parameterization 下 softmax 的来源。

## 15. Softmax Response Function

若为每个 class 定义 linear score：

```math
s_k(x)=\theta_k^Tx
```

则 symmetric softmax response 是：

```math
p(y=k\mid x;\Theta)
=
\frac{\exp(\theta_k^Tx)}
{\sum_{j=1}^{K}\exp(\theta_j^Tx)}
```

解释：

* 每个 $\theta_k$ 是 class-specific linear score direction。
* 所有 scores 被 jointly normalized；一个 class 的 probability 上升会挤压其他 class。
* 参数是 relative，不是 absolute；只有 score differences 影响概率。
* 给所有 class parameter 加同一个 vector $v$，即 $\theta_k'=\theta_k+v$，所有 score 同时增加 $v^Tx$，分子分母同乘同一因子，probability 不变。
* Reference-class parameterization 通过固定一个 class 的 parameter，例如 $\theta_K=0$，解决 identifiability。

Binary reduction 也可以直接证明。对 $K=2$：

```math
p(y=1\mid x)
=
\frac{\exp(\theta_1^Tx)}
{\exp(\theta_1^Tx)+\exp(\theta_2^Tx)}
```

分子分母同时除以 $\exp(\theta_1^Tx)$：

```math
p(y=1\mid x)
=
\frac{1}
{1+\exp(\theta_2^Tx-\theta_1^Tx)}
```

整理：

```math
p(y=1\mid x)
=
\frac{1}
{1+\exp\left(-(\theta_1-\theta_2)^Tx\right)}
```

因此 binary softmax 等价于 logistic regression with parameter difference $\theta_1-\theta_2$。

Softmax 不是 independent one-vs-rest logistic regression。One-vs-rest 会训练 $K$ 个 binary probability models，它们的 outputs 一般不保证 sum to one；softmax 是一个 joint multinomial conditional model，概率从定义上互相耦合并归一化。

![Softmax coupled probabilities](../../assets/figures/lecture04-softmax-coupled-probabilities.png)

![Softmax simplex](../../assets/figures/lecture04-softmax-simplex.png)

## 16. Softmax Likelihood and Cross-Entropy

定义 one-hot target：

```math
t_{ik}=\mathbf1\{y^{(i)}=k\}
```

记：

```math
p_{ik}=p(y=k\mid x^{(i)};\Theta)
```

Likelihood 是：

```math
L(\Theta)
=
\prod_{i=1}^{m}
\prod_{k=1}^{K}
p_{ik}^{t_{ik}}
```

Log-likelihood：

```math
\ell(\Theta)
=
\sum_{i=1}^{m}
\sum_{k=1}^{K}
t_{ik}\log p_{ik}
```

Negative log likelihood，也就是 multiclass cross-entropy：

```math
J(\Theta)
=
-\sum_{i=1}^{m}
\sum_{k=1}^{K}
t_{ik}\log p_{ik}
```

为了推导 gradient，先写单样本 loss：

```math
J_i
=
-\sum_{k=1}^{K}t_{ik}\log p_{ik}
```

由于：

```math
\log p_{ik}
=
\theta_k^Tx^{(i)}
-
\log\sum_{j=1}^{K}\exp(\theta_j^Tx^{(i)})
```

所以：

```math
J_i
=
-\sum_{k=1}^{K}t_{ik}\theta_k^Tx^{(i)}
+
\sum_{k=1}^{K}t_{ik}
\log\sum_{j=1}^{K}\exp(\theta_j^Tx^{(i)})
```

one-hot target 满足 $\sum_k t_{ik}=1$，故：

```math
J_i
=
-\sum_{k=1}^{K}t_{ik}\theta_k^Tx^{(i)}
+
\log\sum_{j=1}^{K}\exp(\theta_j^Tx^{(i)})
```

对 $\theta_r$ 求 gradient：

```math
\nabla_{\theta_r}J_i
=
-t_{ir}x^{(i)}
+
\frac{\exp(\theta_r^Tx^{(i)})}
{\sum_{j=1}^{K}\exp(\theta_j^Tx^{(i)})}
x^{(i)}
```

即：

```math
\nabla_{\theta_r}J_i
=
(p_{ir}-t_{ir})x^{(i)}
```

汇总所有 samples：

```math
\nabla_{\theta_k}J
=
\sum_{i=1}^{m}
(p_{ik}-t_{ik})x^{(i)}
```

所有 class parameters 是 jointly trained 的，因为每个 $p_{ik}$ 的 denominator 包含全部 class scores。

## 17. Reliability View

GLM 的可靠性来自两个层面：优化 objective 是否被正确求解，以及 statistical assumptions 是否适合真实数据。后一层往往更难，因为 likelihood 可以被优化得很好，但 model family 仍然 misspecified。

| Assumption | Diagnostic | Likely symptom | Mitigation |
| ---------- | ---------- | -------------- | ---------- |
| Support matches response | 检查 prediction range 和 observed $y$ | negative count、probability outside range、invalid class encoding | 换分布或 link；重定义 response |
| Correct distribution family | residual plots、PIT、posterior predictive style checks | tail errors、skew errors、systematic residual pattern | richer family、robust loss、mixture or nonparametric model |
| Correct mean-variance relation | plot residual variance against fitted mean | Poisson data overdispersed；Gaussian heteroscedastic | quasi-Poisson、negative binomial、variance modeling |
| Correct link | residual pattern vs linear predictor | probability saturation、count underfit at high rates | alternative link、feature transform |
| Sufficient linear predictor | validation error by subgroup、partial residuals | stable bias in regions of feature space | interactions、basis expansion、nonlinear model |
| iid conditional samples | time/group correlation checks | overconfident standard errors、leakage-like validation | clustered SE、mixed model、time split |
| No complete separation | inspect margins and coefficient growth | logistic/softmax coefficients diverge | regularization、Bayesian prior、feature review |
| Balanced enough classes | class-frequency and per-class metrics | rare class ignored、poor recall | class weighting、resampling、threshold policy |
| Calibration | reliability diagram, Brier score, ECE | confident wrong probabilities | calibration set、temperature scaling、regularization |
| Stable distribution | train/test shift diagnostics | likelihood good in train but poor deployment | shift monitoring、domain adaptation、robust evaluation |
| Identifiability | rank checks and invariance checks | multiple parameters yield same probabilities | constraints, reference class, regularization |
| Parameter uncertainty acceptable | standard errors, bootstrap, profile likelihood | unstable signs and wide intervals | more data, simpler model, shrinkage |

## 18. Connection to PS1

Lecture 4 completes the conceptual prerequisites for the first supervised-learning assignment gate. At this point，linear regression、locally weighted regression、logistic regression、Newton method、Perceptron、exponential family、GLM 和 softmax 的核心关系已经具备。

这里不复制 official problem statements，也不提供 official solutions。PS1 gate 接下来应进入 independent derivation attempts、implementation planning、tests 和 mistake log。

## 19. Takeaways

Lecture 4 把 supervised learning 从一组孤立算法转化为 principled model-construction system：

* Perceptron 展示 non-probabilistic linear decision geometry。
* Exponential family 提供 unified probability representation。
* Log-partition function 生成 mean、variance 和 convexity。
* GLM 把 response semantics、distribution、link、linear predictor 和 likelihood 接成一条链。
* Softmax 是 multinomial conditional model，不是多个独立 sigmoid 的拼接。
* Reliability analysis 要同时检查 support、distribution、link、feature geometry、optimization 和 deployment shift。

## Fast Review Checklist

- [ ] I can explain why multiclass uses categorical/multinomial rather than Poisson.
- [ ] I can distinguish support, distribution, link function, response function, and loss.
- [ ] I can derive sigmoid from Bernoulli.
- [ ] I can derive exponential response from Poisson.
- [ ] I can explain why softmax probabilities are coupled.
- [ ] I can explain what $T(y)$ does when it is not equal to $y$.
- [ ] I can explain why $a'(\eta)$ gives the mean and $a''(\eta)$ gives variance.
- [ ] I can choose a candidate distribution for a new supervised-learning problem and justify it.

## Concept Map Summary

| Modeling question | Mathematical object | Example |
|---|---|---|
| What can $Y$ be? | support | $\mathbb R$, $\{0,1\}$, $\mathbb N_0$, simplex |
| What uncertainty model? | conditional distribution | Gaussian, Bernoulli, Poisson |
| What statistic matters? | $T(y)$ | scalar, one-hot vector |
| What is linear? | natural parameter $\eta$ | $\eta=\theta^Tx$ |
| What is predicted? | conditional mean | fitted mean or class probabilities |
| What is optimized? | NLL | squared loss, cross-entropy |

The long prediction formula is kept outside the table so Markdown renders it reliably:

```math
h_\theta(x)=\mathbb E[T(Y)\mid x;\theta]
```
