# Matrix Transpose Identity and Gradient-Contour Geometry

# Part I: Why \((AB)^T=B^TA^T\)

## 1. Matrix Multiplication as Row-Column Interaction

设

$$A\in\mathbb{R}^{p\times q}, \qquad B\in\mathbb{R}^{q\times r}.$$

内维 \(q\) 相同，因此

$$AB\in\mathbb{R}^{p\times r}.$$

Product 的 \((i,j)\) entry 定义为

$$\boxed{ (AB)_{ij} =\sum_{k=1}^{q}A_{ik}B_{kj} }.$$

也就是说，\((AB)_{ij}\) 是 \(A\) 的第 \(i\) 行与 \(B\) 的第 \(j\) 列的 inner product。Matrix multiplication 不是逐元素相乘；它通过共享索引 \(k\) 聚合中间维度。

## 2. Column-Space View

\(AB\) 的第 \(j\) 列为

$$\boxed{ (AB)_{:j}=AB_{:j} }.$$

若把 \(A\) 写成 columns：

$$A= \begin{bmatrix} A_{:1}&A_{:2}&\cdots&A_{:q} \end{bmatrix},$$

并把 \(B\) 的第 \(j\) 列写成

$$B_{:j} = \begin{bmatrix} B_{1j}\\ B_{2j}\\ \vdots\\ B_{qj} \end{bmatrix},$$

则

$$(AB)_{:j} = \sum_{k=1}^{q}B_{kj}A_{:k}.$$

因此，\(AB\) 的第 \(j\) 列是 \(A\) columns 的 linear combination，coefficients 来自 \(B\) 的第 \(j\) 列。由此立即得到

$$\mathrm{Col}(AB) \subseteq \mathrm{Col}(A).$$

## 3. Row-Space View

\(AB\) 的第 \(i\) 行为

$$\boxed{ (AB)_{i:}=A_{i:}B }.$$

若

$$A_{i:} = \begin{bmatrix} A_{i1}&A_{i2}&\cdots&A_{iq} \end{bmatrix},$$

则

$$(AB)_{i:} = \sum_{k=1}^{q}A_{ik}B_{k:}.$$

所以 \(AB\) 的第 \(i\) 行是 \(B\) rows 的 linear combination，coefficients 来自 \(A\) 的第 \(i\) 行。相应地，

$$\mathrm{Row}(AB) \subseteq \mathrm{Row}(B).$$

Column view 与 row view 描述的是同一个 operation：中间 index \(k\) 决定如何组合 \(A\) 的 columns 或 \(B\) 的 rows。

## 4. Entry-wise Proof of Transpose Identity

目标是证明

$$(AB)^T=B^TA^T.$$

先检查 dimensions：

$$AB\in\mathbb{R}^{p\times r} \quad\Longrightarrow\quad (AB)^T\in\mathbb{R}^{r\times p}.$$

另一方面，

$$B^T\in\mathbb{R}^{r\times q}, \qquad A^T\in\mathbb{R}^{q\times p},$$

所以

$$B^TA^T\in\mathbb{R}^{r\times p}.$$

两边 shape 一致。现在比较任意 \((i,j)\) entry。

由 transpose 定义：

$$\left((AB)^T\right)_{ij}=(AB)_{ji}$$

$$\left((AB)^T\right)_{ij}=\sum_{k=1}^{q}A_{jk}B_{ki}.$$

另一方面，由 matrix multiplication 定义：

$$(B^TA^T)_{ij}=\sum_{k=1}^{q}(B^T)_{ik}(A^T)_{kj}$$

$$(B^TA^T)_{ij}=\sum_{k=1}^{q}B_{ki}A_{jk}.$$

对每个 \(k\)，\(A_{jk}\) 与 \(B_{ki}\) 是 scalars，scalar multiplication commutes：

$$A_{jk}B_{ki}=B_{ki}A_{jk}.$$

因此

$$\sum_{k=1}^{q}A_{jk}B_{ki} = \sum_{k=1}^{q}B_{ki}A_{jk}.$$

所以对所有 \(i,j\)，

$$\left((AB)^T\right)_{ij} =(B^TA^T)_{ij}.$$

由矩阵相等的 entry-wise definition，

$$\boxed{ (AB)^T=B^TA^T }.$$

## 5. Intuitive Explanation of Order Reversal

用户的直觉是：matrix multiplication 通过 linear combinations 生成 rows 或 columns；transpose 交换 row / column roles，因此 composition 的顺序也必须反转。

从 linear maps 看，这个结论更直接。对 column vector \(x\)，

$$(AB)x=A(Bx).$$

所以 \(AB\) 表示：

1. \(B\) 先作用于 \(x\)；
2. \(A\) 再作用于结果。

Transpose 描述 inner product 中把 linear map 从一侧移到另一侧的 adjoint operation。对任意兼容向量 \(u,v\)，

$$\langle u,ABv\rangle =u^TABv.$$

先把 \(A\) 移到左侧：

$$u^TABv =(A^Tu)^TBv.$$

再把 \(B\) 移到左侧：

$$(A^Tu)^TBv =(B^TA^Tu)^Tv.$$

因此 composite map 的 transpose 必须先反转最后作用的 \(A\)，再反转先作用的 \(B\)：

$$(AB)^T=B^TA^T.$$

这与函数复合求逆时 order reversal 的逻辑相似：要撤回或对偶化一串 operations，必须从最后一步开始。

扩展到三个 matrices：

$$(ABC)^T=C^TB^TA^T.$$

# Part II: Why Gradients Are Perpendicular to Contours

## 1. Contours of a Two-Variable Function

设

$$f:\mathbb{R}^2\to\mathbb{R}$$

可微。给定常数 \(c\)，level set / contour 定义为

$$\mathcal{C}_c = \{(x,y):f(x,y)=c\}.$$

Contour 上所有点具有相同 function value。对 topographic map，它表示相同高度；对 loss landscape，它表示相同 objective value。

Gradient 为

$$\nabla f(x,y) = \begin{bmatrix} f_x(x,y)\\ f_y(x,y) \end{bmatrix}.$$

## 2. Curve-Based Proof

令一条 differentiable curve

$$r(t) = \begin{bmatrix} x(t) \\ y(t) \end{bmatrix}$$

完全位于 contour \(\mathcal{C}_c\) 上。Since \(r(t)\) stays on the contour, the function value is constant:

$$f(x(t),y(t))=c.$$

Differentiating both sides with respect to \(t\):

$$\frac{d}{dt}f(x(t),y(t))=f_x(x(t),y(t))x'(t)+f_y(x(t),y(t))y'(t).$$

Using vector notation:

$$\frac{d}{dt}f(x(t),y(t))=\nabla f(x(t),y(t))^T\begin{bmatrix}x'(t)\\y'(t)\end{bmatrix}=0.$$

Therefore,

$$\nabla f(x(t),y(t))^Tr'(t)=0.$$

So,

$$\nabla f(x(t),y(t))\perp r'(t).$$

这个证明揭示了关键逻辑：沿 contour 的 directional derivative 为零，而 directional derivative 正是 gradient 与移动方向的 inner product。

若 \(\nabla f=0\)，等式仍成立，但 zero vector 不能定义唯一 normal direction；此时 level set 的局部形状可能需要 higher-order information。

## 3. Surface-Normal Projection Proof

用户提出的几何思路是：把 \(f(x,y)\) 看成三维 surface 的高度，surface normal 投影到 \(xy\)-plane 后应当给出二维 gradient。下面严格化这个想法。

考虑 graph surface：

$$z=f(x,y).$$

写成 implicit form：

$$F(x,y,z)=f(x,y)-z=0.$$

由于 \(F\) 在 surface 上恒为零，\(\nabla F\) 是 surface 的 normal vector：

$$\boxed{ \nabla F = \begin{bmatrix} f_x\\ f_y\\ -1 \end{bmatrix} }.$$

把该 vector orthogonally projected onto the \(xy\)-plane，删除 \(z\) component：

$$\mathrm{Proj}_{xy}(\nabla F) = \begin{bmatrix} f_x\\ f_y\\ 0 \end{bmatrix}.$$

其前两个 coordinates 正是

$$\nabla f= \begin{bmatrix} f_x\\ f_y \end{bmatrix}.$$

现在取 horizontal plane

$$z=c.$$

它与 surface 的 intersection 满足

$$f(x,y)=c,$$

所以该 intersection 投影到 \(xy\)-plane 后就是 contour。

设 intersection curve 的三维 tangent 为

$$T= \begin{bmatrix} x'(t)\\ y'(t)\\ 0 \end{bmatrix}.$$

\(z\) component 为零，因为 curve 位于 horizontal plane。由于 \(T\) 也位于 surface tangent plane，

$$\nabla F^TT=0.$$

展开：

$$\begin{bmatrix} f_x&f_y&-1 \end{bmatrix} \begin{bmatrix} x'(t)\\ y'(t)\\ 0 \end{bmatrix} =f_x\,x'(t)+f_y\,y'(t)=0.$$

这等价于

$$\nabla f^Tr'(t)=0.$$

因此，surface normal 的 \(xy\) projection 与 projected contour tangent orthogonal。这个三维证明与 curve-based proof 不是两个互不相关的事实，而是同一个 chain-rule geometry 的不同表示。

## 4. Connection to Gradient Descent

在 point \(x\) 附近，对小 displacement \(\Delta\)，

$$f(x+\Delta) \approx f(x)+\nabla f(x)^T\Delta.$$

By Cauchy--Schwarz,

$$\nabla f(x)^T\Delta\geq-\left\|\nabla f(x)\right\|_2\left\|\Delta\right\|_2.$$

If \(\|\Delta\|_2=\varepsilon\), then

$$\nabla f(x)^T\Delta\geq-\varepsilon\left\|\nabla f(x)\right\|_2.$$

The bound is attained when

$$\Delta=-\varepsilon\frac{\nabla f(x)}{\left\|\nabla f(x)\right\|_2}.$$

Therefore, the negative gradient direction is the steepest local descent direction under the Euclidean norm. In particular:

* \(\nabla f\) points in the direction of greatest local first-order increase;
* \(-\nabla f\) points in the direction of greatest local first-order decrease;
* because the gradient is orthogonal to contours, gradient descent crosses loss contours rather than moving along one level set.

这一定义依赖 Euclidean norm。若使用不同 norm、preconditioner 或 Riemannian metric，“steepest” direction 会改变；这也是 feature scaling、natural gradient 和 second-order methods 会改变 optimization path 的几何原因。
