# Pseudoinverse and Minimal-Norm Least Squares

返回 [Module 02](../note.md)。

来源边界：本文件覆盖 MIT 9.520 / 6.860 Class 03 中 overdetermined systems、underdetermined systems、Moore-Penrose pseudoinverse、minimal-norm solution 与 ridge limit 的部分。它服务于 regularized least squares 的理论理解，不进入 kernels 或后续 representation theorem。

## 1. Why Normal Equations Are Not the Whole Story

CS229 Lecture 2 中常见的 normal equation 写法是：

```math
\hat w_{\mathrm{OLS}}
=
(X^TX)^{-1}X^Ty.
```

这个公式需要 $X^TX$ invertible。MIT Class 03 会立刻追问：如果 $X$ 不是 full column rank，或者样本数与维度关系导致系统没有唯一解，该怎么办？

这不是技术细节，而是 regularization theory 的入口。因为当 least-squares inverse 不稳定或不唯一时，算法必须引入额外选择原则。

## 2. Overdetermined Systems

若：

```math
n>d,
```

则样本数多于参数数，线性方程：

```math
Xw=y
```

通常没有 exact solution。OLS 不再要求 residual 为 $0$，而是最小化：

```math
\left\|Xw-y\right\|_2^2.
```

如果 $X$ full column rank，则 $X^TX$ invertible，解为：

```math
\hat w
=
(X^TX)^{-1}X^Ty.
```

对应 pseudoinverse：

```math
X^{\dagger}
=
(X^TX)^{-1}X^T.
```

于是：

```math
\hat w
=
X^{\dagger}y.
```

## 3. Underdetermined Systems

若：

```math
n<d,
```

则参数数多于样本数。只要 $y$ 落在 $X$ 的 row constraints 可实现范围内，方程：

```math
Xw=y
```

可能有 infinitely many solutions。此时仅要求 empirical square loss 为 $0$ 不足以选出唯一参数。

MIT Class 03 的关键补充是 minimal-norm principle：在所有 interpolating solutions 中，选 Euclidean norm 最小的一个：

```math
\underset{w}{\mathrm{minimize}}
\quad
\left\|w\right\|_2^2
\quad
\mathrm{s.t.}
\quad
Xw=y.
```

这个选择已经是一种 inductive bias：它不是由 training residual 决定的，而是由 solution norm preference 决定的。

## 4. Minimal-Norm Derivation

为了简化常数，写 Lagrangian：

```math
\mathcal L(w,\alpha)
=
\frac{1}{2}
\left\|w\right\|_2^2
+
\alpha^T(Xw-y),
```

其中 $\alpha\in\mathbb R^n$ 是 Lagrange multiplier。

对 $w$ 求导：

```math
\nabla_w\mathcal L(w,\alpha)
=
w
+
X^T\alpha.
```

Stationarity 给出：

```math
w
=
-X^T\alpha.
```

代入 constraint：

```math
Xw
=
-XX^T\alpha
=
y.
```

若 $XX^T$ invertible，则：

```math
\alpha
=
-(XX^T)^{-1}y.
```

因此：

```math
w
=
X^T(XX^T)^{-1}y.
```

所以 underdetermined full-row-rank case 的 pseudoinverse 为：

```math
X^{\dagger}
=
X^T(XX^T)^{-1}.
```

Minimal-norm solution：

```math
w^{\dagger}
=
X^{\dagger}y.
```

## 5. SVD Definition of Pseudoinverse

令 thin SVD 为：

```math
X
=
U\Sigma V^T.
```

若 rank 为 $r$，则：

```math
\Sigma
=
\mathrm{diag}(\sigma_1,\ldots,\sigma_r),
\quad
\sigma_j>0.
```

Pseudoinverse 定义为：

```math
X^{\dagger}
=
V\Sigma^{\dagger}U^T,
```

其中：

```math
\Sigma^{\dagger}
=
\mathrm{diag}
\left(
\frac{1}{\sigma_1},
\ldots,
\frac{1}{\sigma_r}
\right).
```

对 null-space directions，pseudoinverse 不做任意恢复，而是给出 minimal-norm 选择。这一点和 regularization 的精神一致：在数据没有约束的方向上，不任意放大参数。

## 6. Ridge Limit

Ridge solution 在本 supplement 的 average-loss convention 下为：

```math
\hat w_{\lambda}
=
(X^TX+n\lambda I)^{-1}X^Ty.
```

用 SVD 写成：

```math
\hat w_{\lambda}
=
\sum_{j=1}^{r}
\frac{\sigma_j}{\sigma_j^2+n\lambda}
(u_j^Ty)
v_j.
```

令 $\lambda\to0^+$：

```math
\frac{\sigma_j}{\sigma_j^2+n\lambda}
\to
\frac{1}{\sigma_j}.
```

因此在非零 singular directions 上：

```math
\hat w_{\lambda}
\to
\sum_{j=1}^{r}
\frac{u_j^Ty}{\sigma_j}
v_j
=
X^{\dagger}y.
```

如果 $X$ rank deficient，ridge limit 仍不会在 null-space 中加入任意分量。它收敛到 pseudoinverse minimal-norm solution。

## 7. Why This Matters for Regularization

Pseudoinverse 和 ridge 都在回答同一个问题：

```text
when finite samples do not determine a stable unique inverse,
which solution should the algorithm prefer?
```

Pseudoinverse 在 exact interpolation 或 least-squares solution 中选择 minimal norm。Ridge 在 objective 中显式加入 norm penalty，并对 small singular values 做 spectral filtering。

二者的区别是：pseudoinverse 是 $\lambda\to0^+$ 的极限选择；ridge 使用 positive $\lambda$ 保留稳定化强度。MIT Class 03 借此说明，least squares 从一开始就是 inverse problem，而 regularization 是让这个 inverse problem 可控的结构原则。
