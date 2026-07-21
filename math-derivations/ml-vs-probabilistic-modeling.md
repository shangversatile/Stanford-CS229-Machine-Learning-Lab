# Machine Learning View vs Probabilistic Modeling View

## 1. Core Distinction

同一个 supervised learning objective 经常可以被两种语言解释。Machine learning / ERM view 把训练看成 empirical objective optimization：选 hypothesis class、选 loss、在 training data 上最小化 empirical risk，并关心 generalization。Probabilistic modeling / MLE view 把训练看成 likelihood-based parameter estimation：先指定 conditional distribution，再选择让 observed data 最 plausible 的 parameter。

这两种 view 不是互斥模型。它们常常导向同一个优化目标，但回答的问题不同。ML view 从 prediction error 出发；probabilistic view 从 data-generating story 出发。

## 2. Empirical Risk Minimization

给定 training dataset：

```math
D=\{(x^{(i)},y^{(i)})\}_{i=1}^{m}
```

选择 hypothesis class 中的一个 predictor：

```math
h_{\theta}:\mathcal{X}\to\mathcal{Y}
```

以及 pointwise loss：

```math
\ell(h_{\theta}(x),y)
```

真正想控制的是 population risk：

```math
R(\theta)=\mathbb{E}_{(x,y)\sim P_{\mathrm{data}}}\left[\ell(h_{\theta}(x),y)\right]
```

但真实分布 $P_{\mathrm{data}}$ 不可见，所以训练时用 empirical risk 近似：

```math
J_{\mathrm{ERM}}(\theta)=\frac{1}{m}\sum_{i=1}^{m}\ell(h_{\theta}(x^{(i)}),y^{(i)})
```

ERM estimator 是：

```math
\hat{\theta}_{\mathrm{ERM}}=\underset{\theta}{\mathrm{argmin}}\ J_{\mathrm{ERM}}(\theta)
```

Generalization interpretation 是：training minimizes $J_{\mathrm{ERM}}(\theta)$，但 deployment 关心的是 $R(\theta)$。因此可靠的 ML 分析不能只看 training loss，还要看 validation performance、generalization gap、robustness 和 distribution shift。

## 3. Probabilistic Modeling and Likelihood

Probabilistic view 先指定 conditional model：

```math
p(y|x;\theta)
```

如果 examples 条件独立，则 observed dataset 的概率为：

```math
p(D|\theta)=\prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta)
```

Likelihood 是把 observed data 固定后得到的关于 $\theta$ 的函数：

```math
L(\theta)=p(D|\theta)
```

MLE 选择最大化 likelihood 的 parameter：

```math
\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmax}}\ L(\theta)
```

通常使用 log likelihood：

```math
\ell_{\mathrm{log}}(\theta)=\log L(\theta)=\sum_{i=1}^{m}\log p(y^{(i)}|x^{(i)};\theta)
```

Negative log likelihood 是：

```math
J_{\mathrm{NLL}}(\theta)=-\ell_{\mathrm{log}}(\theta)
```

因此 MLE 也可以写成 minimization：

```math
\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmin}}\ J_{\mathrm{NLL}}(\theta)
```

## 4. Why Negative Log Likelihood Becomes a Loss

在 ERM view 中，loss 可以先被定义出来。在 probabilistic view 中，loss 是 conditional distribution 诱导出来的。对单个 example，NLL loss 是：

```math
\ell_{\mathrm{NLL}}(h_{\theta}(x),y)=-\log p(y|x;\theta)
```

整个 dataset 上的 objective 是：

```math
J_{\mathrm{NLL}}(\theta)=\sum_{i=1}^{m}\ell_{\mathrm{NLL}}(h_{\theta}(x^{(i)}),y^{(i)})
```

这说明 probabilistic model 一旦确定，loss 的形式就不再是纯粹的数值偏好。Gaussian conditional model 诱导 squared loss；Bernoulli conditional model 诱导 cross-entropy。

## 5. Linear Regression from Both Views

ML view 选择 linear hypothesis：

```math
h_{\theta}(x)=\theta^Tx
```

并选择 squared loss objective：

```math
J(\theta)=\frac{1}{2}\sum_{i=1}^{m}\left(y^{(i)}-\theta^Tx^{(i)}\right)^2
```

Probabilistic view 假设 additive Gaussian noise：

```math
y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}
```

```math
\epsilon^{(i)}\sim\mathcal{N}(0,\sigma^2)
```

因此 conditional density 是：

```math
p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\left(y^{(i)}-\theta^Tx^{(i)}\right)^2}{2\sigma^2}\right)
```

对 independent samples，likelihood 为：

```math
L(\theta)=\prod_{i=1}^{m}\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\left(y^{(i)}-\theta^Tx^{(i)}\right)^2}{2\sigma^2}\right)
```

Log likelihood 展开：

```math
\ell_{\mathrm{log}}(\theta)
=
\sum_{i=1}^{m}\left[-\log(\sqrt{2\pi}\sigma)-\frac{\left(y^{(i)}-\theta^Tx^{(i)}\right)^2}{2\sigma^2}\right]
```

也就是：

```math
\ell_{\mathrm{log}}(\theta)
=
-m\log(\sqrt{2\pi}\sigma)-\frac{1}{2\sigma^2}\sum_{i=1}^{m}\left(y^{(i)}-\theta^Tx^{(i)}\right)^2
```

第一项与 $\theta$ 无关，$1/(2\sigma^2)$ 是 positive scaling。因此 maximizing log likelihood 等价于 minimizing residual sum of squares：

```math
\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmin}}\sum_{i=1}^{m}\left(y^{(i)}-\theta^Tx^{(i)}\right)^2
```

所以 least squares 同时有两个解释：在 ML view 中，它是 squared prediction error minimization；在 probabilistic view 中，它是 independent homoscedastic Gaussian noise 假设下的 MLE。

## 6. Logistic Regression from Both Views

二分类中 $y\in\{0,1\}$。ML view 选择 sigmoid output：

```math
h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}
```

并使用 binary cross-entropy：

```math
J(\theta)=-\sum_{i=1}^{m}\left[y^{(i)}\log h_{\theta}(x^{(i)})+\left(1-y^{(i)}\right)\log\left(1-h_{\theta}(x^{(i)})\right)\right]
```

Probabilistic view 假设 Bernoulli conditional model：

```math
p(y|x;\theta)=h_{\theta}(x)^y\left(1-h_{\theta}(x)\right)^{1-y}
```

Dataset likelihood 为：

```math
L(\theta)=\prod_{i=1}^{m}h_{\theta}(x^{(i)})^{y^{(i)}}\left(1-h_{\theta}(x^{(i)})\right)^{1-y^{(i)}}
```

Log likelihood 是：

```math
\ell_{\mathrm{log}}(\theta)=\sum_{i=1}^{m}\left[y^{(i)}\log h_{\theta}(x^{(i)})+\left(1-y^{(i)}\right)\log\left(1-h_{\theta}(x^{(i)})\right)\right]
```

Negative log likelihood 是：

```math
J_{\mathrm{NLL}}(\theta)=-\ell_{\mathrm{log}}(\theta)
```

这正是 binary cross-entropy。因此 logistic regression 不是 thresholded linear regression，而是 Bernoulli conditional likelihood model。sigmoid 输出的是 $p(y=1|x;\theta)$，cross-entropy 是该 Bernoulli model 的 NLL。

## 7. Probability vs Likelihood

Probability 和 likelihood 使用同一个数学表达式，但固定和变化的对象不同。

Probability: 固定 $\theta$，把 data 当成 random variable：

```math
p(y|x;\theta)\quad\text{as a function of }y
```

Likelihood: 固定 observed data，把 $\theta$ 当成函数 argument：

```math
L(\theta)=p(D|\theta)\quad\text{as a function of }\theta
```

所以 likelihood 本身不是 $\theta$ 的 probability distribution。它只是一个用于比较不同 parameter values 的函数。只有在 Bayesian modeling 中引入 prior $p(\theta)$ 后，posterior $p(\theta|D)$ 才是关于 $\theta$ 的 probability distribution。

## 8. Frequentist Interpretation in Early CS229

Early CS229 的 MLE 推导主要是 frequentist。它的解释是：

```math
\theta\ \text{is fixed but unknown}
```

```math
D\ \text{is random because it is sampled from the data-generating process}
```

```math
\hat{\theta}(D)\ \text{is random because it depends on the sampled dataset}
```

true parameter 不被赋予 probability distribution。estimator 是 random variable，因为它是 random dataset 的函数。反复从同一个 data-generating process 抽样，会得到不同的 dataset，也就可能得到不同的 $\hat{\theta}(D)$。

## 9. Bayesian Contrast

Bayesian view 把 $\theta$ 建模为 random variable，并指定 prior：

```math
p(\theta)
```

观察到 dataset 后，posterior 为：

```math
p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}
```

Evidence 是 normalization constant：

```math
p(D)=\int p(D|\theta)p(\theta)d\theta
```

MAP estimator 是 posterior mode：

```math
\hat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\mathrm{argmax}}\ p(D|\theta)p(\theta)
```

Full Bayesian prediction 不只使用一个 point estimate，而是对 posterior uncertainty 做积分：

```math
p(y_*|x_*,D)=\int p(y_*|x_*;\theta)p(\theta|D)d\theta
```

Bayesian view 的核心对象是 posterior distribution，而不是单个参数估计。

## 10. MLE vs MAP

MLE 只最大化 likelihood：

```math
\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmax}}\ p(D|\theta)
```

MAP 最大化 likelihood times prior：

```math
\hat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\mathrm{argmax}}\ p(D|\theta)p(\theta)
```

Negative log version 是：

```math
\hat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\mathrm{argmin}}\left[-\log p(D|\theta)-\log p(\theta)\right]
```

这解释了 regularization connection：many regularization terms can be interpreted as negative log priors。比如 Gaussian prior 会导出 L2-style penalty，Laplace prior 会导出 L1-style penalty。这里的重点不是把所有 regularization 都等同于 Bayesian modeling，而是说明 prior 会改变 optimization objective。

## 11. Reliability Implications

ML / ERM view 的 reliability checks 包括 optimization stability、validation performance、generalization gap、robustness、class imbalance 和 distribution shift。它回答的问题是：模型在 observed and held-out data 上是否真的预测得好？

Probabilistic / MLE view 的 reliability checks 包括 noise model、conditional independence、calibration、likelihood misspecification、uncertainty interpretation 和 estimator behavior。它回答的问题是：这个 loss 背后的 data-generating assumption 是否 plausible？

Bayesian view 进一步要求检查 prior sensitivity、posterior concentration、posterior predictive calibration 和 uncertainty propagation。一个 model 可以 empirical loss 很低但 calibration 很差；也可以 likelihood 很优雅但严重 misspecified。Reliable ML 需要 empirical validation plus probabilistic assumption checking。

## 12. Summary Table

| Question | ML / ERM View | Probabilistic / MLE View | Bayesian View |
| -------- | ------------- | ------------------------ | ------------- |
| What is learned? | Predictor or parameter $`h_{\theta}`$, $`\theta`$ | Parameter $`\theta`$ of a conditional model | Posterior distribution $`p(\theta|D)`$ |
| What is optimized? | Empirical risk $`J_{\mathrm{ERM}}(\theta)`$ | Likelihood $`L(\theta)`$ or NLL $`J_{\mathrm{NLL}}(\theta)`$ | Posterior, MAP objective, or posterior predictive quantity |
| What is random? | Training data are samples; analysis focuses on empirical vs population performance | Data are random before observation; $`\theta`$ is fixed but unknown in frequentist MLE | $`\theta`$ is random and updated after observing $`D`$ |
| How is uncertainty represented? | Usually through validation error, confidence intervals, or robustness tests | Sampling distribution of estimators and likelihood curvature | Posterior uncertainty and posterior predictive distribution |
| How does linear regression arise? | Linear hypothesis plus squared loss | Gaussian conditional model plus MLE | Gaussian likelihood plus prior, giving posterior over $`\theta`$ |
| How does logistic regression arise? | Sigmoid output plus cross-entropy | Bernoulli conditional model plus MLE | Bernoulli likelihood plus prior, giving posterior over $`\theta`$ |
| What can go wrong? | Overfitting, underfitting, optimization failure, poor generalization | Misspecified noise model, dependence violations, poor calibration | Bad prior, posterior miscalibration, computational approximation error |