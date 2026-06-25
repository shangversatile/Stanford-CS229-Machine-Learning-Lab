# Linear Regression: Least Squares, Normal Equation, and MLE

## 1. Setup and Dimensions

设有 $m$ 个 training examples，每个 raw input 有 $n$ 个 features。加入 intercept coordinate $x_0=1$ 后：

| Symbol | Meaning | Shape |
| ------ | ------- | ----- |
| $m$ | Number of samples | Scalar |
| $n$ | Number of original features | Scalar |
| $x^{(i)}$ | Augmented feature vector for sample $i$ | $(n+1)\times1$ |
| $y^{(i)}$ | Target for sample $i$ | Scalar |
| $X$ | Design matrix; row $i$ is $(x^{(i)})^T$ | $m\times(n+1)$ |
| $\theta$ | Parameter vector including intercept | $(n+1)\times1$ |
| $y$ | Target vector | $m\times1$ |
| $X\theta$ | Vector of fitted values | $m\times1$ |
| $r=X\theta-y$ | Residual vector under this note's convention | $m\times1$ |
| $X^TX$ | Gram matrix | $(n+1)\times(n+1)$ |
| $X^Ty$ | Feature-target cross-product | $(n+1)\times1$ |
| $J(\theta)$ | Least-squares objective | Scalar |

Hypothesis：

\[
h_\theta(x^{(i)})=\theta^Tx^{(i)}.
\]

Design matrix：

\[
X=
\begin{bmatrix}
(x^{(1)})^T\\
\vdots\\
(x^{(m)})^T
\end{bmatrix},
\qquad
y=
\begin{bmatrix}
y^{(1)}\\
\vdots\\
y^{(m)}
\end{bmatrix}.
\]

于是第 $i$ 个分量满足

\[
(X\theta)_i=(x^{(i)})^T\theta=h_\theta(x^{(i)}).
\]

## 2. Scalar Objective

对样本 $i$ 定义 residual

\[
r^{(i)}=\theta^Tx^{(i)}-y^{(i)}.
\]

Squared loss 为

\[
L_i(\theta)=\frac12(r^{(i)})^2
=\frac12(\theta^Tx^{(i)}-y^{(i)})^2.
\]

对全部样本求和：

\[
\boxed{
J(\theta)=
\frac12\sum_{i=1}^{m}
(\theta^Tx^{(i)}-y^{(i)})^2
}.
\]

由于

\[
\lVert X\theta-y\rVert_2^2
=\sum_{i=1}^{m}
(\theta^Tx^{(i)}-y^{(i)})^2,
\]

scalar objective 与 matrix objective 完全相同：

\[
J(\theta)
=\frac12\lVert X\theta-y\rVert_2^2
=\frac12(X\theta-y)^T(X\theta-y).
\]

## 3. Gradient of One Example

固定一个样本，令

\[
e_i(\theta)=\theta^Tx^{(i)}-y^{(i)}
=\sum_{k=0}^{n}\theta_kx_k^{(i)}-y^{(i)}.
\]

则

\[
J_i(\theta)=\frac12e_i(\theta)^2.
\]

对 $\theta_j$ 求偏导。Chain rule 给出

\[
\frac{\partial J_i}{\partial\theta_j}
=
\frac{\partial J_i}{\partial e_i}
\cdot
\frac{\partial e_i}{\partial\theta_j}.
\]

分别计算：

\[
\frac{\partial J_i}{\partial e_i}
=\frac{\partial}{\partial e_i}\frac12e_i^2
=e_i,
\]

以及

\[
\begin{aligned}
\frac{\partial e_i}{\partial\theta_j}
&=
\frac{\partial}{\partial\theta_j}
\left(
\sum_{k=0}^{n}\theta_kx_k^{(i)}-y^{(i)}
\right)\\
&=x_j^{(i)}.
\end{aligned}
\]

因此

\[
\boxed{
\frac{\partial J_i}{\partial\theta_j}
=
(\theta^Tx^{(i)}-y^{(i)})x_j^{(i)}
}.
\]

Vector form：

\[
\boxed{
\nabla_\theta J_i(\theta)
=
(\theta^Tx^{(i)}-y^{(i)})x^{(i)}
}.
\]

Gradient descent update：

\[
\theta_j
\leftarrow
\theta_j-\alpha
(\theta^Tx^{(i)}-y^{(i)})x_j^{(i)}.
\]

把负号吸收到 residual 后：

\[
\boxed{
\theta_j
\leftarrow
\theta_j+\alpha
(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
}.
\]

这就是 one-example LMS / SGD update。

## 4. Batch Gradient

### 4.1 From scalar summation

从

\[
J(\theta)
=\frac12\sum_{i=1}^{m}
(\theta^Tx^{(i)}-y^{(i)})^2
\]

开始。对 $\theta_j$ 求偏导：

\[
\begin{aligned}
\frac{\partial J}{\partial\theta_j}
&=
\sum_{i=1}^{m}
\frac{\partial}{\partial\theta_j}
\frac12(\theta^Tx^{(i)}-y^{(i)})^2\\
&=
\sum_{i=1}^{m}
(\theta^Tx^{(i)}-y^{(i)})x_j^{(i)}.
\end{aligned}
\]

把所有 coordinates 组合成 vector：

\[
\nabla_\theta J(\theta)
=
\sum_{i=1}^{m}
x^{(i)}\left((x^{(i)})^T\theta-y^{(i)}\right).
\]

而 $X\theta-y$ 收集了所有 scalar residual，左乘 $X^T$ 正好对 samples 求上述加权和：

\[
\boxed{
\nabla_\theta J(\theta)=X^T(X\theta-y)
}.
\]

Dimension check：

\[
X^T(X\theta-y):
\quad
((n+1)\times m)(m\times1)
=(n+1)\times1.
\]

这与 $\theta$ 的 shape 一致。

### 4.2 From matrix form

令

\[
r(\theta)=X\theta-y.
\]

则

\[
J(\theta)=\frac12r(\theta)^Tr(\theta).
\]

Differential method：

\[
dJ
=\frac12\left((dr)^Tr+r^Tdr\right).
\]

两项都是 scalar，并且互为 transpose，因此相等：

\[
dJ=r^Tdr.
\]

因为

\[
dr=X\,d\theta,
\]

所以

\[
dJ=r^TX\,d\theta
=(X^Tr)^T d\theta.
\]

由 gradient 的定义

\[
dJ=(\nabla_\theta J)^T d\theta,
\]

得到

\[
\boxed{
\nabla_\theta J=X^Tr=X^T(X\theta-y)
}.
\]

## 5. Normal Equation

从

\[
J(\theta)
=\frac12(X\theta-y)^T(X\theta-y)
\]

开始。

### Step 1: transpose

\[
(X\theta-y)^T
=\theta^TX^T-y^T.
\]

### Step 2: distribute

\[
\begin{aligned}
J(\theta)
&=\frac12(\theta^TX^T-y^T)(X\theta-y)\\
&=\frac12\left(
\theta^TX^TX\theta
-\theta^TX^Ty
-y^TX\theta
+y^Ty
\right).
\end{aligned}
\]

### Step 3: identify equal scalar cross terms

由于 $\theta^TX^Ty$ 是 scalar，

\[
\theta^TX^Ty
=(\theta^TX^Ty)^T
=y^TX\theta.
\]

因此

\[
J(\theta)
=\frac12\left(
\theta^TX^TX\theta
-2y^TX\theta
+y^Ty
\right).
\]

### Step 4: differentiate each term

$X^TX$ symmetric，因为

\[
(X^TX)^T=X^T(X^T)^T=X^TX.
\]

因此

\[
\nabla_\theta(\theta^TX^TX\theta)
=2X^TX\theta.
\]

又因为

\[
y^TX\theta=(X^Ty)^T\theta,
\]

所以

\[
\nabla_\theta(y^TX\theta)=X^Ty.
\]

$y^Ty$ 与 $\theta$ 无关，所以

\[
\nabla_\theta(y^Ty)=0.
\]

合并：

\[
\begin{aligned}
\nabla_\theta J(\theta)
&=\frac12
\left(
2X^TX\theta-2X^Ty
\right)\\
&=X^TX\theta-X^Ty\\
&=X^T(X\theta-y).
\end{aligned}
\]

### Step 5: stationary condition

令

\[
\nabla_\theta J(\hat{\theta})=0,
\]

得到

\[
X^TX\hat{\theta}-X^Ty=0,
\]

即

\[
\boxed{
X^TX\hat{\theta}=X^Ty
}.
\]

### Step 6: solve when invertible

若 $X^TX$ invertible，在等式左侧乘 $(X^TX)^{-1}$：

\[
(X^TX)^{-1}X^TX\hat{\theta}
=(X^TX)^{-1}X^Ty.
\]

因为

\[
(X^TX)^{-1}X^TX=I_{n+1},
\]

所以

\[
\boxed{
\hat{\theta}=(X^TX)^{-1}X^Ty
}.
\]

Dimension check：

\[
((n+1)\times(n+1))
((n+1)\times m)
(m\times1)
=(n+1)\times1.
\]

实际计算中不应默认显式构造 inverse。`solve`、QR、SVD 或 pseudo-inverse 通常更稳定。

## 6. Matrix Derivative Facts

### 6.1 Gradient of a linear form

令

\[
f(x)=a^Tx=\sum_{i=1}^{d}a_ix_i.
\]

则第 $j$ 个偏导为

\[
\frac{\partial f}{\partial x_j}=a_j.
\]

所以

\[
\boxed{\nabla_x(a^Tx)=a}.
\]

### 6.2 Gradient of a quadratic form

令

\[
f(x)=x^TAx
=\sum_{i=1}^{d}\sum_{j=1}^{d}x_iA_{ij}x_j.
\]

对 $x_k$ 求偏导。$x_k$ 可能出现在左侧因子 $x_i$，也可能出现在右侧因子 $x_j$：

\[
\begin{aligned}
\frac{\partial f}{\partial x_k}
&=
\sum_{j=1}^{d}A_{kj}x_j
+
\sum_{i=1}^{d}x_iA_{ik}\\
&=(Ax)_k+(A^Tx)_k.
\end{aligned}
\]

因此

\[
\boxed{
\nabla_x(x^TAx)=(A+A^T)x
}.
\]

若 $A=A^T$，

\[
\boxed{
\nabla_x(x^TAx)=2Ax
}.
\]

### 6.3 Gradient of squared residual norm

令

\[
f(\theta)=\frac12\lVert X\theta-y\rVert_2^2.
\]

展开为

\[
f(\theta)
=\frac12
\left(
\theta^TX^TX\theta
-2y^TX\theta
+y^Ty
\right).
\]

使用前两条规则：

\[
\begin{aligned}
\nabla_\theta f
&=\frac12
\left(
2X^TX\theta-2X^Ty
\right)\\
&=X^T(X\theta-y).
\end{aligned}
\]

所以

\[
\boxed{
\nabla_\theta
\frac12\lVert X\theta-y\rVert_2^2
=X^T(X\theta-y)
}.
\]

## 7. Convexity and Global Optimum

Gradient 为

\[
\nabla_\theta J(\theta)
=X^TX\theta-X^Ty.
\]

再对 $\theta$ 求导得到 Hessian：

\[
\boxed{
\nabla_\theta^2J(\theta)=X^TX
}.
\]

对任意 $v\in\mathbb{R}^{n+1}$，

\[
v^TX^TXv
=(Xv)^T(Xv)
=\lVert Xv\rVert_2^2
\geq0.
\]

所以

\[
X^TX\succeq0,
\]

从而 $J(\theta)$ convex。Convex differentiable function 的任何 stationary point 都是 global minimizer。

若 $X$ full column rank，则对任意 $v\neq0$，

\[
Xv\neq0,
\]

因此

\[
v^TX^TXv=\lVert Xv\rVert_2^2>0.
\]

于是

\[
X^TX\succ0,
\]

$J$ strictly convex，minimizer 唯一。

若 $X$ rank deficient，objective 仍 convex，但可能存在多个 parameter vectors 产生相同 fitted values。Pseudo-inverse 可选择 minimum-norm solution。

## 8. Projection Geometry

Here, \(\mathrm{Col}(X)\) denotes the column space of \(X\).

The fitted vector is

\[
\hat{y}=X\hat{\theta}.
\]

Because \(\hat{y}\) is a linear combination of the columns of \(X\), it lies in the column space:

\[
\hat{y}\in\mathrm{Col}(X).
\]

The normal equation can be rewritten as

\[
X^T(y-X\hat{\theta})=0.
\]

This means the residual vector is orthogonal to the column space:

\[
y-X\hat{\theta}\perp\mathrm{Col}(X).
\]

Therefore, \(\hat{y}\) is the orthogonal projection of \(y\) onto \(\mathrm{Col}(X)\).

Equivalently, with \(e=y-\hat{y}\),

\[
y=\hat{y}+e,
\qquad
\hat{y}\in\mathrm{Col}(X),
\qquad
e\in\mathrm{Col}(X)^\perp.
\]

Thus,

\[
X\hat{\theta}
=
\mathrm{Proj}_{\mathrm{Col}(X)}y.
\]

“Normal equation”中的 normal 正对应 residual 对 column space 的 normal / orthogonal 关系。

## 9. Probabilistic Interpretation

### 9.1 Data-generating model

假设

\[
y^{(i)}
=\theta^Tx^{(i)}+\epsilon^{(i)},
\]

其中

\[
\epsilon^{(i)}
\overset{\mathrm{iid}}{\sim}
\mathcal{N}(0,\sigma^2).
\]

因此

\[
y^{(i)}\mid x^{(i)};\theta
\sim
\mathcal{N}(\theta^Tx^{(i)},\sigma^2).
\]

### 9.2 Conditional density

\[
p(y^{(i)}\mid x^{(i)};\theta)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[
-\frac{
(y^{(i)}-\theta^Tx^{(i)})^2
}{2\sigma^2}
\right].
\]

### 9.3 Likelihood

Conditional independence 给出

\[
\begin{aligned}
L(\theta)
&=
p(y\mid X;\theta)\\
&=
\prod_{i=1}^{m}
p(y^{(i)}\mid x^{(i)};\theta).
\end{aligned}
\]

这里 observed $X,y$ 固定，likelihood 是 $\theta$ 的函数。

### 9.4 Log likelihood

\[
\begin{aligned}
\ell(\theta)
&=\log L(\theta)\\
&=
\sum_{i=1}^{m}
\left[
-\frac12\log(2\pi\sigma^2)
-\frac{
(y^{(i)}-\theta^Tx^{(i)})^2
}{2\sigma^2}
\right]\\
&=
-\frac{m}{2}\log(2\pi\sigma^2)
-\frac{1}{2\sigma^2}
\sum_{i=1}^{m}
(y^{(i)}-\theta^Tx^{(i)})^2.
\end{aligned}
\]

### 9.5 MLE to least squares

第一项不依赖 $\theta$。因为 $\sigma^2>0$，

\[
-\frac{1}{2\sigma^2}<0.
\]

Therefore, maximizing the Gaussian log likelihood is equivalent to minimizing the squared-error objective:

\[
\underset{\theta}{\mathrm{argmax}}\ \ell(\theta)
=
\underset{\theta}{\mathrm{argmin}}\;
\sum_{i=1}^{m}
\left(
y^{(i)}-\theta^T x^{(i)}
\right)^2.
\]

So Gaussian noise MLE and ordinary least squares produce the same parameter estimate.

这个推导依赖 zero-mean、constant-variance、independent Gaussian noise 的统计模型。Squared loss 也可以脱离严格 Gaussian assumption 作为 convex objective 使用，但此时不能自动继承所有基于该 likelihood 的统计结论。

## 10. Reliability Notes

Least squares 在以下条件下可能不可靠：

* **Outliers**：quadratic penalty 产生高 influence；
* **Heavy-tailed noise**：极端 residual 比 Gaussian model 预期更频繁；
* **Heteroscedasticity**：constant variance assumption 失效；
* **Correlated errors**：iid assumption 失效，常见于 temporal / spatial data；
* **Model misspecification**：$\mathbb{E}[y\mid x]$ 不能由所选 features 的线性组合表达；
* **Multicollinearity**：$X^TX$ ill-conditioned，parameter estimates 对扰动敏感；
* **Rank deficiency**：parameter solution 不唯一；
* **Label noise**：错误 observations 被平方项放大；
* **Distribution shift**：训练条件下的 relation 不再适用于测试或部署；
* **Objective mismatch**：squared error 与真实 operational cost 不一致。

可靠使用 linear regression 需要同时检查 residual plots、feature scaling、rank / conditioning、outlier sensitivity、train-test shift 和 deployment metric。Closed-form solution 只解决 optimization，不自动解决 specification、generalization 或 reliability。
