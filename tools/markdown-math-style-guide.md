# Markdown Math Style Guide

Use these rules for notes intended to render in this repository.

1. Inline math uses `$...$`.
2. Display math uses single-line `$$...$$`.
3. Do not use `&#92;(...&#92;)`.
4. Do not use `&#92;[` or `&#92;]`.
5. Do not use standalone `$$` lines.
6. Do not put real line breaks inside display math.
7. Split long derivations into multiple single-line display equations.
8. Keep matrices on one physical Markdown line using LaTeX `\\`.
9. Avoid unsupported macros such as `&#92;operatorname`.
10. Use `\mathrm{Col}(X)`, `\mathrm{rank}(X)`, `\mathrm{tr}(A)`, and `\mathrm{diag}(A)`.
11. Use `\underset{\theta}{\mathrm{argmax}}` and `\underset{\theta}{\mathrm{argmin}}`.

Valid inline formulas use single-dollar delimiters:

```markdown
对 $\theta_j$ 求偏导。

The fitted vector is $X\hat{\theta}$.
```

Valid display formulas and matrices occupy one physical Markdown line:

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
