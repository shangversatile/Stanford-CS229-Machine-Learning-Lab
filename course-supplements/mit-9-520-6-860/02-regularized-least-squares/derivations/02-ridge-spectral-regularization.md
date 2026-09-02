# Ridge Spectral Regularization

返回 [Module 02](../README.md)。

来源边界：参考 MIT 9.520 / 6.860 Class 03 的 SVD / pseudoinverse / ridge spectral view。本文件只处理 linear least squares 的谱解释。

## 1. SVD Setup

令 design matrix 的 thin SVD 为：

```math
X
=
U\Sigma V^T.
```

其中：

| Symbol | 含义 |
| --- | --- |
| $U$ | left singular vectors |
| $V$ | right singular vectors |
| $\Sigma$ | diagonal singular-value matrix |
| $\sigma_j$ | 第 $j$ 个 singular value |

若 $X$ full column rank，则所有 $\sigma_j>0$。

## 2. OLS Through SVD

OLS solution：

```math
\hat w_{\mathrm{OLS}}
=
(X^TX)^{-1}X^Ty.
```

代入 SVD：

```math
X^TX
=
V\Sigma^TU^TU\Sigma V^T.
```

因为 $U^TU=I$：

```math
X^TX
=
V\Sigma^T\Sigma V^T.
```

因此：

```math
(X^TX)^{-1}
=
V(\Sigma^T\Sigma)^{-1}V^T.
```

又：

```math
X^T
=
V\Sigma^TU^T.
```

所以：

```math
\hat w_{\mathrm{OLS}}
=
V(\Sigma^T\Sigma)^{-1}V^TV\Sigma^TU^Ty.
```

化简：

```math
\hat w_{\mathrm{OLS}}
=
V\Sigma^{-1}U^Ty.
```

展开到 singular directions：

```math
\hat w_{\mathrm{OLS}}
=
\sum_{j=1}^{d}
\frac{u_j^Ty}{\sigma_j}
v_j.
```

## 3. Noise Amplification

若 target 扰动为：

```math
y
\to
y+\delta y,
```

则 OLS solution perturbation：

```math
\delta\hat w_{\mathrm{OLS}}
=
V\Sigma^{-1}U^T\delta y.
```

展开：

```math
\delta\hat w_{\mathrm{OLS}}
=
\sum_{j=1}^{d}
\frac{u_j^T\delta y}{\sigma_j}
v_j.
```

若某个 $\sigma_j$ 很小，即使 $u_j^T\delta y$ 很小，除以 $\sigma_j$ 后也可能变大。小 singular value direction 表示数据矩阵在该方向上提供的约束弱；OLS inverse 会放大这些弱约束方向上的 noise。

## 4. Ridge Through SVD

Ridge solution under average-loss convention：

```math
\hat w_{\lambda}
=
(X^TX+n\lambda I)^{-1}X^Ty.
```

代入 SVD：

```math
X^TX+n\lambda I
=
V\Sigma^T\Sigma V^T
+
n\lambda VV^T.
```

因为 $VV^T=I$：

```math
X^TX+n\lambda I
=
V(\Sigma^T\Sigma+n\lambda I)V^T.
```

因此：

```math
(X^TX+n\lambda I)^{-1}
=
V(\Sigma^T\Sigma+n\lambda I)^{-1}V^T.
```

继续：

```math
\hat w_{\lambda}
=
V(\Sigma^T\Sigma+n\lambda I)^{-1}V^TV\Sigma^TU^Ty.
```

化简：

```math
\hat w_{\lambda}
=
V(\Sigma^T\Sigma+n\lambda I)^{-1}\Sigma^TU^Ty.
```

由于 diagonal entries 为 $\sigma_j^2+n\lambda$，得到：

```math
\hat w_{\lambda}
=
V
\mathrm{diag}
\left(
\frac{\sigma_j}{\sigma_j^2+n\lambda}
\right)
U^Ty.
```

展开：

```math
\hat w_{\lambda}
=
\sum_{j=1}^{d}
\frac{\sigma_j}{\sigma_j^2+n\lambda}
(u_j^Ty)
v_j.
```

## 5. Shrinkage Relative to OLS

OLS multiplier：

```math
\frac{1}{\sigma_j}.
```

Ridge multiplier：

```math
\frac{\sigma_j}{\sigma_j^2+n\lambda}.
```

Ridge relative to OLS：

```math
\frac{
\sigma_j
/
(\sigma_j^2+n\lambda)
}{
1/\sigma_j
}
=
\frac{\sigma_j^2}{\sigma_j^2+n\lambda}.
```

如果 $\sigma_j^2\gg n\lambda$：

```math
\frac{\sigma_j^2}{\sigma_j^2+n\lambda}
\approx
1.
```

如果 $\sigma_j^2\ll n\lambda$：

```math
\frac{\sigma_j^2}{\sigma_j^2+n\lambda}
\approx
0.
```

所以 ridge 对 small-singular-value directions 施加强 shrinkage，对 large-singular-value directions 保留较多信息。

## 6. Why This Is Regularization

从 inverse problem 看，OLS 试图用 $1/\sigma_j$ 恢复所有方向，包括数据几乎没有稳定观测到的方向。Ridge 把恢复因子改成：

```math
\frac{\sigma_j}{\sigma_j^2+n\lambda}.
```

当 $\sigma_j$ 很小时，这个因子不会像 $1/\sigma_j$ 那样爆炸，而是接近 $0$。因此 ridge 的本质是 spectral filtering：

```text
do not trust directions that the finite sample barely constrains.
```

这正是 MIT regularization perspective：regularization 是对 ill-posed inverse problem 的稳定化，不只是工程上的 overfitting trick。

## 7. Figure Link

对应图：

![Ridge spectral shrinkage](../figures/mit9520-ridge-spectral-shrinkage.png)

左图比较 OLS inverse factor 与 ridge filter；右图显示 ridge 相对 OLS 的 shrinkage factor。
