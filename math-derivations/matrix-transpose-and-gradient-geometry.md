# Matrix Transpose Identity and Gradient-Contour Geometry

# Part I: Why $(AB)^T=B^TA^T$

## 1. Dimension Setup

设矩阵 dimensions 为

$$A\in\mathbb{R}^{m\times n},\qquad B\in\mathbb{R}^{n\times p}$$

令

$$C=AB\in\mathbb{R}^{m\times p}.$$

这里的 dimension condition 是 essential: the number of columns of $A$ must equal the number of rows of $B$; this common inner dimension is $n$. 后面的求和都沿着这个 shared index 展开。

本文使用以下 notation：

* $A_{i:}$ 表示 $A$ 的第 $i$ 行；
* $A_{:k}$ 表示 $A$ 的第 $k$ 列；
* $B_{k:}$ 表示 $B$ 的第 $k$ 行；
* $B_{:j}$ 表示 $B$ 的第 $j$ 列；
* $C_{ij}$ 表示 $C$ 的第 $i$ 行、第 $j$ 列 entry。

## 2. Entry-wise Row-Column View

Matrix product 的最局部定义是单个 entry：

$$C_{ij}=(AB)_{ij}=A_{i:}B_{:j}=\sum_{k=1}^{n}A_{ik}B_{kj}.$$

也就是说，product $AB$ 的 $(i,j)$ entry 是 $A$ 的第 $i$ 行与 $B$ 的第 $j$ 列的 inner product。这个视角关注一个 scalar output：每个输出 entry 都由一次 row-column interaction 产生。

## 3. Column View: Columns of $AB$ as Linear Combinations of Columns of $A$

对第 $j$ 列，

$$C_{:j}=(AB)_{:j}=AB_{:j}.$$

由于 $B_{:j}$ 是一个 $n$-dimensional coefficient vector，

$$AB_{:j}=B_{1j}A_{:1}+B_{2j}A_{:2}+\cdots+B_{nj}A_{:n}.$$

因此

$$C_{:j}=\sum_{k=1}^{n}B_{kj}A_{:k}.$$

这说明 $AB$ 的第 $j$ 列是 $A$ 的 columns 的 linear combination，coefficients 来自 $B$ 的第 $j$ 列。更精确地说，column vector $B_{:j}$ 提供 coefficients；真正被组合的 vectors 是 $A$ 的 columns。由此也可以看出

$$\mathrm{Col}(AB)\subseteq\mathrm{Col}(A).$$

## 4. Row View: Rows of $AB$ as Linear Combinations of Rows of $B$

对第 $i$ 行，

$$C_{i:}=(AB)_{i:}=A_{i:}B.$$

由于 $A_{i:}$ 是一个 $n$-dimensional row vector of coefficients，

$$A_{i:}B=A_{i1}B_{1:}+A_{i2}B_{2:}+\cdots+A_{in}B_{n:}.$$

因此

$$C_{i:}=\sum_{k=1}^{n}A_{ik}B_{k:}.$$

这说明 $AB$ 的第 $i$ 行是 $B$ 的 rows 的 linear combination，coefficients 来自 $A$ 的第 $i$ 行。也就是说，每一行 $A_{i:}$ 提供 coefficients，用来组合 $B$ 的 rows。相应地，

$$\mathrm{Row}(AB)\subseteq\mathrm{Row}(B).$$

## 5. Why the Row and Column Views Are Consistent

Entry-wise view、column view 和 row view 是同一个 matrix multiplication 的三种观察尺度：

* Entry-wise view 关注一个 scalar output；
* Column view 关注每个 output column 如何形成；
* Row view 关注每个 output row 如何形成。

它们最终都回到同一个表达式：

$$C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}.$$

因此，matrix multiplication 不只是“row times column”的局部规则；它也是一种 structured linear-combination operation：从 column 角度组合 $A$ 的 columns，从 row 角度组合 $B$ 的 rows。

## 6. Entry-wise Proof of $(AB)^T=B^TA^T$

令

$$D=(AB)^T$$

则

$$D_{ij}=((AB)^T)_{ij}=(AB)_{ji}.$$

由 entry-wise multiplication rule，

$$(AB)_{ji}=\sum_{k=1}^{n}A_{jk}B_{ki}.$$

现在考虑

$$E=B^TA^T$$

它的 $(i,j)$ entry 为

$$E_{ij}=(B^TA^T)_{ij}=\sum_{k=1}^{n}(B^T)_{ik}(A^T)_{kj}.$$

根据 transpose 的定义，

$$(B^T)_{ik}=B_{ki}.$$

$$(A^T)_{kj}=A_{jk}.$$

因此

$$E_{ij}=\sum_{k=1}^{n}B_{ki}A_{jk}.$$

因为 $A_{jk}$ 与 $B_{ki}$ 都是 scalars，multiplication commutes：

$$\sum_{k=1}^{n}B_{ki}A_{jk}=\sum_{k=1}^{n}A_{jk}B_{ki}.$$

所以

$$E_{ij}=D_{ij}.$$

由于这对每个 $i$ 和 $j$ 都成立，

$$(AB)^T=B^TA^T.$$

## 7. Intuitive Explanation of Why the Order Reverses

Transpose operation swaps rows and columns。因为 matrix multiplication 既可以从 row-combination 角度读，也可以从 column-combination 角度读，转置 product 会迫使我们把 final rows 重新解释为 columns，把 final columns 重新解释为 rows。

形成 product $AB$ 时，matrix $A$ acts on the columns of matrix $B$; equivalently, rows of $B$ 按照 rows of $A$ 提供的 coefficients 被组合。转置之后，rows 与 columns 的角色交换，所以原来最后作用的 factor 必须先出现。

Linear map interpretation 给出更直接的解释。若 vectors 都看作 column vectors，则 $ABx$ 表示

$$ABx=A(Bx).$$

所以 $B$ 先作用，而 $A$ 后作用。对 transpose，corresponding composition reverses：

$$(AB)^T=B^TA^T.$$

这不是 symbolic trick，而是 transposition 在 dual / inner-product representation 中反转 composition direction 的结果。

## 8. Connection to CS229 Linear Regression

在 linear regression 中，normal equation derivation 会使用

$$J(\theta)=\frac{1}{2}(X\theta-y)^T(X\theta-y).$$

为了正确展开这个 expression，需要

$$(X\theta-y)^T=\theta^TX^T-y^T.$$

其中

$$(X\theta)^T=\theta^TX^T$$

正是 transpose-of-product rule。因此，identity $(AB)^T=B^TA^T$ 不是孤立的 linear algebra fact；它直接支撑 least squares、gradient 和 normal equation 的推导。

# Part II: Why Gradients Are Perpendicular to Contours

## 1. Contours of a Two-Variable Function

设

$$f:\mathbb{R}^2\to\mathbb{R}$$

可微。给定常数 $c$ 后，level set / contour 定义为

$$\mathcal{C}_c = \{(x,y):f(x,y)=c\}.$$

Contour 上所有点具有相同 function value。对 topographic map，它表示相同高度；对 loss landscape，它表示相同 objective value。

Gradient 为

$$\nabla f(x,y) = \begin{bmatrix} f_x(x,y)\\ f_y(x,y) \end{bmatrix}.$$

## 2. Curve-Based Proof

令一条 differentiable curve

$$r(t) = \begin{bmatrix} x(t) \\ y(t) \end{bmatrix}$$

完全位于 contour $\mathcal{C}_c$ 上。Since $r(t)$ stays on the contour, the function value is constant:

$$f(x(t),y(t))=c.$$

Differentiating both sides with respect to $t$:

$$\frac{d}{dt}f(x(t),y(t))=f_x(x(t),y(t))x'(t)+f_y(x(t),y(t))y'(t).$$

Using vector notation:

$$\frac{d}{dt}f(x(t),y(t))=\nabla f(x(t),y(t))^T\begin{bmatrix}x'(t)\\y'(t)\end{bmatrix}=0.$$

Therefore,

$$\nabla f(x(t),y(t))^Tr'(t)=0.$$

So,

$$\nabla f(x(t),y(t))\perp r'(t).$$

这个证明揭示了关键逻辑：沿 contour 的 directional derivative 为零，而 directional derivative 正是 gradient 与移动方向的 inner product。

在 $\nabla f=0$ 的情形下，等式仍成立，但 zero vector 不能定义唯一 normal direction；此时 level set 的局部形状可能需要 higher-order information。

## 3. Surface-Normal Projection Proof

同一个结论也可以从几何角度理解：把 $f(x,y)$ 看成三维 surface 的高度，surface normal 投影到 $xy$-plane 后应当给出二维 gradient。下面严格化这个想法。

考虑 graph surface：

$$z=f(x,y).$$

写成 implicit form：

$$F(x,y,z)=f(x,y)-z=0.$$

由于 $F$ 在 surface 上恒为零，the vector $\nabla F$ 是 surface 的 normal vector：

$$\boxed{ \nabla F = \begin{bmatrix} f_x\\ f_y\\ -1 \end{bmatrix} }.$$

把该 vector orthogonally projected onto the $xy$-plane，删除 $z$ component：

$$\mathrm{Proj}_{xy}(\nabla F) = \begin{bmatrix} f_x\\ f_y\\ 0 \end{bmatrix}.$$

其前两个 coordinates 正是

$$\nabla f= \begin{bmatrix} f_x\\ f_y \end{bmatrix}.$$

现在取 horizontal plane

$$z=c.$$

它与 surface 的 intersection 满足

$$f(x,y)=c,$$

所以该 intersection 投影到 $xy$-plane 后就是 contour。

设 intersection curve 的三维 tangent 为

$$T= \begin{bmatrix} x'(t)\\ y'(t)\\ 0 \end{bmatrix}.$$

$z$ component 为零，因为 curve 位于 horizontal plane。由于 $T$ 也位于 surface tangent plane，

$$\nabla F^TT=0.$$

展开：

$$\begin{bmatrix} f_x&f_y&-1 \end{bmatrix} \begin{bmatrix} x'(t)\\ y'(t)\\ 0 \end{bmatrix} =f_x\,x'(t)+f_y\,y'(t)=0.$$

这等价于

$$\nabla f^Tr'(t)=0.$$

因此，surface normal 的 $xy$ projection 与 projected contour tangent orthogonal。这个三维证明与 curve-based proof 不是两个互不相关的事实，而是同一个 chain-rule geometry 的不同表示。

## 4. Connection to Gradient Descent

在 point $x$ 附近，对小 displacement $\Delta$ 使用 first-order approximation：

$$f(x+\Delta) \approx f(x)+\nabla f(x)^T\Delta.$$

By Cauchy--Schwarz,

$$\nabla f(x)^T\Delta\geq-\left\|\nabla f(x)\right\|_2\left\|\Delta\right\|_2.$$

If $\|\Delta\|_2=\varepsilon$, then

$$\nabla f(x)^T\Delta\geq-\varepsilon\left\|\nabla f(x)\right\|_2.$$

The bound is attained when

$$\Delta=-\varepsilon\frac{\nabla f(x)}{\left\|\nabla f(x)\right\|_2}.$$

Therefore, the negative gradient direction is the steepest local descent direction under the Euclidean norm. In particular:

* $\nabla f$ points in the direction of greatest local first-order increase;
* $-\nabla f$ points in the direction of greatest local first-order decrease;
* because the gradient is orthogonal to contours, gradient descent crosses loss contours rather than moving along one level set.

这一定义依赖 Euclidean norm。若使用不同 norm、preconditioner 或 Riemannian metric，“steepest” direction 会改变；这也是 feature scaling、natural gradient 和 second-order methods 会改变 optimization path 的几何原因。
