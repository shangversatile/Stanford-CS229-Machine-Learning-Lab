# Machine Learning View vs Probabilistic Modeling View

## 1. Core Distinction

同一个 supervised learning model 常常有两种互补解释。Machine learning view 把训练看成 empirical objective optimization：选择 hypothesis class、选择 loss、在 training data 上最小化 empirical risk。Probabilistic modeling view 把训练看成 parameter estimation：指定 conditional distribution、写出 likelihood、寻找让 observed data 最 plausible 的 parameter。

这两种视角不互相排斥。它们经常导向同一个公式，但关注点不同。ML view 更适合讨论 prediction、optimization 和 generalization；probabilistic view 更适合讨论 loss 的统计假设、noise model、calibration 和 uncertainty。

## 2. Empirical Risk Minimization

设 training set 为：

$$D=\{(x^{(i)},y^{(i)})\}_{i=1}^{m}.$$

选择 hypothesis：

$$h_{\theta}:\mathcal{X}\to\mathcal{Y}.$$

选择 loss：

$$\ell(h_{\theta}(x),y).$$

Empirical risk 是：

$$J_{\mathrm{ERM}}(\theta)=\frac{1}{m}\sum_{i=1}^{m}\ell(h_{\theta}(x^{(i)}),y^{(i)}).$$

ERM estimate 是：

$$\hat{\theta}_{\mathrm{ERM}}=\underset{\theta}{\mathrm{argmin}}\ J_{\mathrm{ERM}}(\theta).$$

在这个视角下，squared loss、cross-entropy、hinge loss 等都可以作为 optimization objective 来讨论。核心问题是 objective 是否容易优化、是否能 generalize、是否对 outliers 和 distribution shift 稳定。

## 3. Probabilistic Modeling and Likelihood

Probabilistic view 从 conditional model 开始：

$$p(y|x;\theta).$$

若 training examples 条件独立，则：

$$p(y^{(1)},\dots,y^{(m)}|X;\theta)=\prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta).$$

Likelihood 是 observed data 固定后关于 $\theta$ 的函数：

$$L(\theta)=\prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta).$$

MLE 选择最大化 likelihood 的 parameter：

$$\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmax}}\ L(\theta).$$

通常使用 negative log likelihood：

$$J_{\mathrm{NLL}}(\theta)=-\sum_{i=1}^{m}\log p(y^{(i)}|x^{(i)};\theta).$$

于是：

$$\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmin}}\ J_{\mathrm{NLL}}(\theta).$$

在这个视角下，loss function 是由 probability model 推出的，而不是单纯的数值偏好。

## 4. Linear Regression from Both Views

ML view 选择 linear hypothesis：

$$h_{\theta}(x)=\theta^Tx.$$

并选择 squared loss：

$$J(\theta)=\frac{1}{2}\sum_{i=1}^{m}\left(y^{(i)}-\theta^Tx^{(i)}\right)^2.$$

Probabilistic view 假设 additive Gaussian noise：

$$y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}.$$

$$\epsilon^{(i)}\sim\mathcal{N}(0,\sigma^2).$$

因此：

$$p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\left(y^{(i)}-\theta^Tx^{(i)}\right)^2}{2\sigma^2}\right).$$

对 independent samples 取 negative log likelihood，去掉与 $\theta$ 无关的常数和 positive scaling 后，得到同一个 least-squares objective：

$$\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmin}}\sum_{i=1}^{m}\left(y^{(i)}-\theta^Tx^{(i)}\right)^2.$$

所以 least squares 可以同时解释为 empirical prediction error minimization 和 Gaussian MLE。

## 5. Logistic Regression from Both Views

ML view 选择 sigmoid probability output：

$$h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}.$$

并使用 binary cross-entropy：

$$J(\theta)=-\sum_{i=1}^{m}\left[y^{(i)}\log h_{\theta}(x^{(i)})+\left(1-y^{(i)}\right)\log\left(1-h_{\theta}(x^{(i)})\right)\right].$$

Probabilistic view 假设 Bernoulli conditional model：

$$p(y|x;\theta)=h_{\theta}(x)^y\left(1-h_{\theta}(x)\right)^{1-y}.$$

对 observed labels 的 likelihood 取 negative log，正好得到 cross-entropy。这个推导说明 logistic regression 不是 thresholded linear regression，而是 Bernoulli likelihood model。

## 6. Probability vs Likelihood

Probability 和 likelihood 使用同一个函数形式，但固定和变化的对象不同。Probability view 固定 $\theta$，把 data 当作 random：

$$p(y|x;\theta).$$

Likelihood view 固定 observed data，把 $\theta$ 当作要比较的变量：

$$L(\theta)=p(D|\theta).$$

因此 likelihood 本身不是 $\theta$ 的 probability distribution。只有引入 prior $p(\theta)$ 后，Bayesian posterior 才是关于 $\theta$ 的 probability distribution。

## 7. Frequentist Interpretation in Early CS229

早期 CS229 的 MLE 推导主要是 frequentist 的。它的解释是：

$$\theta\ \text{is fixed but unknown.}$$

$$D\ \text{is random because it is sampled from the data-generating process.}$$

$$\hat{\theta}(D)\ \text{is random because it depends on the sampled dataset.}$$

因此 true parameter 不是 random variable；estimator 是 random variable，因为它是 random dataset 的函数。MLE 的目标是构造一个从 data 到 parameter estimate 的 rule。

## 8. Bayesian Contrast

Bayesian view 把 $\theta$ 本身建模为 random variable。先指定 prior：

$$p(\theta).$$

观察 data $D$ 后得到 posterior：

$$p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}.$$

Evidence 是 normalization constant：

$$p(D)=\int p(D|\theta)p(\theta)d\theta.$$

Bayesian view 的核心对象不是单个 estimate，而是 posterior distribution $p(\theta|D)$。

## 9. MLE vs MAP

MLE 只最大化 likelihood：

$$\hat{\theta}_{\mathrm{MLE}}=\underset{\theta}{\mathrm{argmax}}\ p(D|\theta).$$

MAP 最大化 posterior，等价于最大化 likelihood times prior：

$$\hat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\mathrm{argmax}}\ p(D|\theta)p(\theta).$$

取 log 后：

$$\hat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\mathrm{argmax}}\left[\log p(D|\theta)+\log p(\theta)\right].$$

这解释了为什么许多 regularization terms 可以被解释为 negative log prior。Full Bayesian prediction 进一步对 posterior 做积分：

$$p(y_*|x_*,D)=\int p(y_*|x_*;\theta)p(\theta|D)d\theta.$$

## 10. Reliability Implications

ML view 的 reliability diagnostics 包括 empirical performance、generalization gap、optimization stability、robustness to perturbations 和 distribution-shift stress tests。

Probabilistic view 的 reliability diagnostics 包括 noise model plausibility、conditional independence assumption、calibration、likelihood misspecification、posterior or estimator uncertainty。

一个 model 可以有很低 empirical loss，但 probability estimates 不 calibrated。另一个 model 可以有优雅的 likelihood，但 data-generating process 与假设不符。Reliable ML 需要同时问 optimization objective 是否工作，以及 probability model 是否可信。

## 11. Summary Table

| Aspect | Machine Learning / ERM View | Probabilistic Modeling View |
| ------ | --------------------------- | --------------------------- |
| Starting point | Hypothesis and loss | Conditional distribution |
| Objective | Minimize empirical risk | Maximize likelihood or minimize NLL |
| Data role | Training examples for optimization | Samples from assumed data-generating process |
| Linear regression | Linear hypothesis plus squared loss | Gaussian conditional model plus MLE |
| Logistic regression | Sigmoid output plus cross-entropy | Bernoulli conditional model plus MLE |
| Main risk | Poor generalization or optimization failure | Model misspecification or miscalibration |
| Reliable ML check | Test performance and robustness | Test assumptions and uncertainty |