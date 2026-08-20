# Markdown Math Style Guide

Use these rules for notes intended to render in this repository.

1. Inline math uses `$...$`.
2. New lecture notes and derivation files use fenced `math` blocks for display math.
3. Older notes may still contain legacy single-line `$$...$$` display formulas.
4. Do not use `&#92;(...&#92;)`.
5. Do not use `&#92;[` or `&#92;]`.
6. Do not use standalone `$$` lines.
7. Do not put real line breaks inside legacy single-line display math.
8. Split long derivations into multiple display equations.
9. Keep matrices on one physical Markdown line using LaTeX `\\`.
10. Avoid unsupported macros such as `&#92;operatorname`.
11. Use `\mathrm{Col}(X)`, `\mathrm{rank}(X)`, `\mathrm{tr}(A)`, and `\mathrm{diag}(A)`.
12. Use `\underset{\theta}{\mathrm{argmax}}` and `\underset{\theta}{\mathrm{argmin}}`.

Valid inline formulas use single-dollar delimiters:

```markdown
对 $\theta_j$ 求偏导。

The fitted vector is $X\hat{\theta}$.
```

Preferred display formulas for new notes use fenced `math` blocks:

````markdown
```math
J(\theta)=\frac{1}{2}(X\theta-y)^T(X\theta-y).
```

```math
\begin{bmatrix}x_1\\x_2\end{bmatrix}
```
````

Legacy display formulas and matrices occupy one physical Markdown line:

```latex
$$J(\theta)=\frac{1}{2}(X\theta-y)^T(X\theta-y).$$

$$\begin{bmatrix}x_1\\x_2\end{bmatrix}$$
```

For a long derivation, use several display lines separated by blank Markdown
lines instead of inserting line breaks between one pair of delimiters.

Run the repository audit after editing mathematical Markdown:

```bash
python tools/audit_markdown_math.py
```
