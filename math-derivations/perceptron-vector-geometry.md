# Perceptron Vector Geometry

## 1. Setup with Signed Labels

Perceptron 使用 signed labels：

```math
y\in\{-1,+1\}
```

给定 feature vector $x\in\mathbb R^d$ 和 parameter vector $\theta\in\mathbb R^d$，linear score 是：

```math
s_\theta(x)=\theta^Tx
```

Prediction rule：

```math
\hat y=\mathrm{sign}(\theta^Tx)
```

若 $y\theta^Tx>0$，样本在正确的一侧；若 $y\theta^Tx\leq0$，样本被错分或落在 boundary 上。

## 2. Boundary Normal

Decision boundary 是 hyperplane：

```math
\theta^Tx=0
```

对 boundary 上任意两个点 $u,v$，都有：

```math
\theta^Tu=0
```

```math
\theta^Tv=0
```

相减：

```math
\theta^T(u-v)=0
```

所以 boundary 内的任意 tangent direction $u-v$ 都与 $\theta$ 正交。于是 $\theta$ 是 boundary normal vector。

## 3. Mistake-Driven Update

Perceptron 在当前样本被错分时更新：

```math
\theta_{\mathrm{new}}=\theta+\alpha yx
```

其中 $\alpha>0$ 是 learning rate。这个 update 只依赖当前 mistake，不来自 likelihood，也不需要 probability model。

## 4. Signed-Score Improvement

当前样本更新后的 signed score 是：

```math
y\theta_{\mathrm{new}}^Tx
=
y(\theta+\alpha yx)^Tx
```

展开：

```math
y\theta_{\mathrm{new}}^Tx
=
y\theta^Tx+\alpha y^2x^Tx
```

因为 $y^2=1$：

```math
y\theta_{\mathrm{new}}^Tx
=
y\theta^Tx+\alpha\|x\|_2^2
```

因此，只要 $\alpha>0$ 且 $x\neq0$，update 必然提高当前样本的 signed score。

## 5. Angular Interpretation

若 $y=+1$ 且样本被错分，说明 $\theta^Tx\leq0$。更新 $\theta+\alpha x$ 会让 $\theta$ 朝 $x$ 的方向移动，减小 $\theta$ 与 $x$ 的夹角，使 $x$ 更可能被放在 positive side。

若 $y=-1$ 且样本被错分，说明 $-\theta^Tx\leq0$，等价于 $\theta^Tx\geq0$。更新 $\theta-\alpha x$ 会让 $\theta$ 远离 $x$ 的方向，使 $x$ 更可能落到 negative side。

在二维中，因为 boundary 垂直于 $\theta$，更新 $\theta$ 会旋转 boundary。positive mistake 和 negative mistake 会把 normal vector 推向不同方向，因此 boundary 的旋转方向也不同。

## 6. Relation to Margin

Signed margin 常写成：

```math
\gamma_i(\theta)=\frac{y^{(i)}\theta^Tx^{(i)}}{\|\theta\|_2}
```

Perceptron update 增加当前样本的 numerator，但也可能改变 denominator $\|\theta\|_2$，并可能降低其他样本的 margin。因此它是 local correction，不是一次全局 margin optimization。

## 7. Comparison with Logistic Gradient

Logistic regression 使用 $y\in\{0,1\}$ 时，NLL gradient 是：

```math
\nabla_\theta J(\theta)
=
\sum_{i=1}^{m}
\left(h_\theta(x^{(i)})-y^{(i)}\right)x^{(i)}
```

每个样本都有 gradient contribution，且 contribution 随 probability error 连续变化。

Perceptron update 可以写成单样本形式：

```math
\theta\leftarrow\theta+\alpha y^{(i)}x^{(i)}
```

但只有在 $y^{(i)}\theta^Tx^{(i)}\leq0$ 时触发。它不关心模型对正确样本的 confidence，也不输出 calibrated probabilities。

| Aspect | Perceptron | Logistic regression |
| ------ | ---------- | ------------------- |
| Trigger | only mistakes | every sample |
| Signal | signed label times feature | probability residual times feature |
| Objective | implicit mistake correction | explicit Bernoulli NLL |
| Smoothness | discontinuous decision rule | smooth sigmoid response |
| Probability | no default probability | conditional probability |

## 8. Separability and Noise Limitations

若数据 linearly separable，经典 Perceptron convergence theorem 给出 finite mistake bound，其大小依赖 maximum norm 和 margin。直观上，存在某个 separating direction 时，mistake updates 会持续把 $\theta$ 推向能分开数据的方向。

若数据不可分或有 label noise，Perceptron 可能持续震荡。因为任何固定 boundary 都会错分一些点，而算法会不断被这些点拉动。实践中常见缓解方式包括 averaged perceptron、margin perceptron、regularization、early stopping，或改用概率模型如 logistic regression。

