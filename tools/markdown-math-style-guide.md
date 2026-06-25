# Markdown Math Style Guide

Use these rules for notes intended to render in this repository.

1. Use inline math `\( ... \)` for short expressions.
2. Use paired double-dollar delimiters for display math, with the entire formula on one physical Markdown line.
3. Do not use the bracket-style delimiters `&#92;[` and `&#92;]`.
4. Do not use standalone double-dollar delimiter lines.
5. Do not put real line breaks inside display math.
6. Split long derivations into multiple single-line display equations.
7. Keep an entire matrix on one Markdown line and use LaTeX row separators inside the matrix.
8. Avoid the unsupported `operatorname` command; use renderer-safe `\mathrm{...}` notation.
9. Use `\mathrm{Col}(X)`, `\mathrm{rank}(X)`, `\mathrm{tr}(A)`, and `\mathrm{diag}(A)`.
10. Use `\underset{\theta}{\mathrm{argmax}}` and `\underset{\theta}{\mathrm{argmin}}`.

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
