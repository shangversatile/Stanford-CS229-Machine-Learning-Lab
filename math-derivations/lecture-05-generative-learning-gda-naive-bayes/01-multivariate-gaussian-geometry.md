# Multivariate Gaussian Geometry

Cross-link: see [Lecture 5 Section 6: Multivariate Gaussian Distribution](../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#6-multivariate-gaussian-distribution), [Lecture 5 Section 7: Mahalanobis Geometry](../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#7-mahalanobis-geometry), and [Lecture 5 Section 8: Gaussian Isocontours and Determinant](../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#8-gaussian-isocontours-and-determinant).

## 1. Definition and Dimensions

令 random vector：

```math
X\in\mathbb R^d.
```

Non-degenerate multivariate Gaussian 写成：

```math
X\sim\mathcal N(\mu,\Sigma).
```

其中 mean vector 的 shape 是：

```math
\mu\in\mathbb R^d.
```

Covariance matrix 的 shape 是：

```math
\Sigma\in\mathbb R^{d\times d}.
```

Density 为：

```math
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right).
```

Dimension check:

```math
x-\mu\in\mathbb R^d.
```

```math
\Sigma^{-1}(x-\mu)\in\mathbb R^d.
```

```math
(x-\mu)^\top\Sigma^{-1}(x-\mu)\in\mathbb R.
```

因此 exponent 是 scalar，determinant $|\Sigma|$ 也是 scalar。这些 shape check 可以防止把 row/column convention 写乱。

## 2. Mean and Covariance

Mean 定义为：

```math
\mathbb E[X]=\mu.
```

Covariance matrix 定义为：

```math
\Sigma=\mathbb E\left[(X-\mu)(X-\mu)^\top\right].
```

第 $j,k$ 个 entry 是：

```math
\Sigma_{jk}=\mathbb E\left[(X_j-\mu_j)(X_k-\mu_k)\right].
```

也就是：

```math
\Sigma_{jk}=\mathrm{Cov}(X_j,X_k).
```

Diagonal entry 是 variance：

```math
\Sigma_{jj}=\mathbb E\left[(X_j-\mu_j)^2\right]=\mathrm{Var}(X_j).
```

Off-diagonal entry 是 covariance。它只捕捉 linear co-movement；两个变量 covariance 为 $0$ 并不自动意味着 independent。对 jointly Gaussian variables，zero covariance 会推出 independence，但这是 Gaussian family 的特殊结构。

## 3. Covariance Is Always PSD

对任意 $v\in\mathbb R^d$：

```math
v^\top\Sigma v=v^\top\mathbb E\left[(X-\mu)(X-\mu)^\top\right]v.
```

利用 expectation linearity：

```math
v^\top\Sigma v=\mathbb E\left[v^\top(X-\mu)(X-\mu)^\top v\right].
```

因为 $v^\top(X-\mu)$ 是 scalar：

```math
v^\top(X-\mu)(X-\mu)^\top v=\left(v^\top(X-\mu)\right)^2.
```

所以：

```math
v^\top\Sigma v=\mathbb E\left[\left(v^\top(X-\mu)\right)^2\right]\geq0.
```

这证明：

```math
\Sigma\succeq0.
```

这个结论对任何 covariance matrix 成立，不依赖 Gaussian assumption。

## 4. Why the Density Uses PD Covariance

普通 multivariate Gaussian density 里有两处要求 covariance 可逆：

```math
\Sigma^{-1}
```

和：

```math
|\Sigma|^{-1/2}.
```

如果 $\Sigma$ singular，则 inverse 不存在，determinant 为 $0$，上面的 full-dimensional density formula 失效。因此对有 ordinary Lebesgue density 的 non-degenerate Gaussian，要求：

```math
\Sigma\succ0.
```

如果 $\Sigma$ 只是 PSD 且 singular，仍可能对应 degenerate Gaussian。此时 probability mass 集中在 lower-dimensional affine subspace 上，不再有相对于整个 $\mathbb R^d$ volume 的普通 density。

需要保留的精确区分是：

```text
covariance matrix in general -> PSD
ordinary full-dimensional Gaussian density -> PD covariance
degenerate Gaussian -> PSD but singular
```

## 5. Mahalanobis Distance from Eigendecomposition

由于 $\Sigma$ symmetric positive definite，可以做 orthonormal eigendecomposition：

```math
\Sigma=Q\Lambda Q^\top.
```

其中：

```math
Q^\top Q=QQ^\top=I.
```

并且：

```math
\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_d).
```

因为 $\Sigma\succ0$：

```math
\lambda_j>0.
```

所以 inverse 是：

```math
\Sigma^{-1}=Q\Lambda^{-1}Q^\top.
```

其中：

```math
\Lambda^{-1}=\mathrm{diag}\left(\frac1{\lambda_1},\ldots,\frac1{\lambda_d}\right).
```

令 centered vector：

```math
u=x-\mu.
```

把它旋转到 eigen-coordinate：

```math
z=Q^\top u.
```

现在展开 quadratic form：

```math
u^\top\Sigma^{-1}u=u^\top Q\Lambda^{-1}Q^\top u.
```

因为 $z=Q^\top u$ 且 $z^\top=u^\top Q$：

```math
u^\top Q\Lambda^{-1}Q^\top u=z^\top\Lambda^{-1}z.
```

由于 $\Lambda^{-1}$ 是 diagonal：

```math
z^\top\Lambda^{-1}z=\sum_{j=1}^d\frac{z_j^2}{\lambda_j}.
```

因此：

```math
(x-\mu)^\top\Sigma^{-1}(x-\mu)=\sum_{j=1}^d\frac{z_j^2}{\lambda_j}.
```

这说明 Gaussian exponent 测量的是 principal covariance directions 中按 variance 缩放后的 displacement。大 eigenvalue 方向允许更大偏移，小 eigenvalue 方向对偏移更敏感。

## 6. Diagonal Covariance as Independent Coordinates

先看 diagonal covariance，这能帮助理解 full covariance 的几何含义。设：

```math
\Sigma=\mathrm{diag}(\sigma_1^2,\ldots,\sigma_d^2).
```

则：

```math
|\Sigma|=\prod_{j=1}^d\sigma_j^2.
```

所以：

```math
|\Sigma|^{1/2}=\prod_{j=1}^d\sigma_j.
```

并且：

```math
\Sigma^{-1}=\mathrm{diag}\left(\frac1{\sigma_1^2},\ldots,\frac1{\sigma_d^2}\right).
```

Quadratic form 变成：

```math
(x-\mu)^\top\Sigma^{-1}(x-\mu)=\sum_{j=1}^d\frac{(x_j-\mu_j)^2}{\sigma_j^2}.
```

代入 density：

```math
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{d/2}\prod_{j=1}^d\sigma_j}\exp\left(-\frac12\sum_{j=1}^d\frac{(x_j-\mu_j)^2}{\sigma_j^2}\right).
```

把 normalizer 和 exponent 拆成 product：

```math
p(x;\mu,\Sigma)=\prod_{j=1}^d\frac{1}{\sqrt{2\pi}\sigma_j}\exp\left(-\frac12\frac{(x_j-\mu_j)^2}{\sigma_j^2}\right).
```

每一项都是 univariate Gaussian density：

```math
p(x;\mu,\Sigma)=\prod_{j=1}^d p(x_j;\mu_j,\sigma_j^2).
```

所以 diagonal covariance Gaussian 可以看成 independent univariate Gaussians 的 product。Full covariance 的 off-diagonal entries 则通过 rotation / correlation 打破 coordinate-axis factorization，但仍可在 principal directions 中理解。

## 7. Whitening Interpretation

定义 matrix square root：

```math
\Sigma^{1/2}=Q\Lambda^{1/2}Q^\top.
```

相应地：

```math
\Sigma^{-1/2}=Q\Lambda^{-1/2}Q^\top.
```

令 whitened coordinate：

```math
r=\Sigma^{-1/2}(x-\mu).
```

则：

```math
\|r\|_2^2=r^\top r.
```

代入：

```math
r^\top r=(x-\mu)^\top\Sigma^{-1/2}\Sigma^{-1/2}(x-\mu).
```

由于 $\Sigma^{-1/2}$ symmetric，且：

```math
\Sigma^{-1/2}\Sigma^{-1/2}=\Sigma^{-1},
```

得到：

```math
\|r\|_2^2=(x-\mu)^\top\Sigma^{-1}(x-\mu).
```

所以 Mahalanobis distance 可以理解为：先 center，再 rotate，再按每个 principal standard deviation 缩放，最后计算 Euclidean distance。

## 8. Normalization by Change of Variables

Multivariate Gaussian density 的 normalizer 不是任意设置的。它可以从 standard Gaussian 通过 linear transform 得到。

令：

```math
Z\sim\mathcal N(0,I).
```

其 density 是：

```math
p_Z(z)=\frac{1}{(2\pi)^{d/2}}\exp\left(-\frac12z^\top z\right).
```

取一个 matrix $B$ 满足：

```math
BB^\top=\Sigma.
```

例如可以取：

```math
B=\Sigma^{1/2}.
```

定义：

```math
X=\mu+BZ.
```

则：

```math
Z=B^{-1}(X-\mu).
```

Change-of-variables formula 给出：

```math
p_X(x)=p_Z\left(B^{-1}(x-\mu)\right)\left|B^{-1}\right|.
```

代入 standard Gaussian density：

```math
p_X(x)=\frac{1}{(2\pi)^{d/2}}\exp\left(-\frac12(B^{-1}(x-\mu))^\top B^{-1}(x-\mu)\right)\left|B^{-1}\right|.
```

Quadratic exponent 化简为：

```math
(B^{-1}(x-\mu))^\top B^{-1}(x-\mu)=(x-\mu)^\top B^{-\top}B^{-1}(x-\mu).
```

因为：

```math
B^{-\top}B^{-1}=(BB^\top)^{-1}=\Sigma^{-1},
```

所以 exponent 变成：

```math
(x-\mu)^\top\Sigma^{-1}(x-\mu).
```

Jacobian factor 是：

```math
\left|B^{-1}\right|=\frac1{|B|}.
```

又因为：

```math
|\Sigma|=|BB^\top|=|B|^2,
```

所以：

```math
|B|=|\Sigma|^{1/2}.
```

最终得到：

```math
p_X(x)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right).
```

这说明 density formula 同时来自两个操作：standard Gaussian 的 spherical density，以及 linear transform $B$ 对 volume 的伸缩。

## 9. Isocontours

固定 density level：

```math
p(x;\mu,\Sigma)=c.
```

代入 density：

```math
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right)=c.
```

乘回 normalizer：

```math
\exp\left(-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right)=c(2\pi)^{d/2}|\Sigma|^{1/2}.
```

取 log：

```math
-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)=\log c+\frac d2\log(2\pi)+\frac12\log|\Sigma|.
```

定义：

```math
C=-2\log c-d\log(2\pi)-\log|\Sigma|.
```

于是 contour 满足：

```math
(x-\mu)^\top\Sigma^{-1}(x-\mu)=C.
```

使用 eigen-coordinate：

```math
\sum_{j=1}^d\frac{z_j^2}{\lambda_j}=C.
```

当 $C>0$：

```math
\sum_{j=1}^d\frac{z_j^2}{C\lambda_j}=1.
```

这就是 ellipsoid。沿 eigenvector $q_j$ 的 semi-axis length 是：

```math
\sqrt{C\lambda_j}.
```

所以 covariance eigenvectors 决定 orientation，eigenvalues 决定 squared scale。

## 10. Determinant and Density Height

由：

```math
\Sigma=Q\Lambda Q^\top
```

得到：

```math
|\Sigma|=|Q||\Lambda||Q^\top|.
```

Orthogonal matrix 的 determinant magnitude 是 $1$：

```math
|Q||Q^\top|=1.
```

因此：

```math
|\Sigma|=|\Lambda|=\prod_{j=1}^d\lambda_j.
```

Density normalizer 中出现：

```math
|\Sigma|^{1/2}=\prod_{j=1}^d\sqrt{\lambda_j}.
```

这正是 principal directions 上 standard-deviation scales 的乘积。Covariance volume 越大，同样的总 probability mass 被摊到更大 region 上，density peak 就越低。

注意 density height 不是 point probability。连续分布对 region 分配 probability：

```math
P(X\in A)=\int_Ap(x;\mu,\Sigma)\,dx.
```

单点 probability 为：

```math
P(X=x)=0.
```

## 11. Connection Back to GDA

GDA 比较：

```math
p(x\mid Y=1)P(Y=1)
```

和：

```math
p(x\mid Y=0)P(Y=0).
```

Gaussian class-conditional term 的几何由三部分控制：

* $x$ 到 class mean $\mu_k$ 的 Mahalanobis distance；
* covariance volume $|\Sigma|^{1/2}$；
* class prior $P(Y=k)$。

当 covariance shared 时，两个 classes 使用同一套 distance geometry。Log density difference 中 common quadratic term 抵消，这就是 GDA 产生 linear boundary 的几何来源。
