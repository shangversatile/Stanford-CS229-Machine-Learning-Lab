# Markdown Math Style Guide

Use these rules for notes intended to render in this repository.

1. Use inline math `\( ... \)` for short expressions.
2. Use single-line display math with paired double-dollar delimiters.
3. Do not use the bracket-style delimiters `&#92;[` and `&#92;]`.
4. Do not use standalone double-dollar delimiter lines.
5. Do not put real line breaks inside display math.
6. Split long derivations into multiple single-line display equations.
7. Keep an entire matrix on one Markdown line and use LaTeX `\\` between matrix rows.
8. Avoid unsupported named-operator macros; use renderer-safe `\mathrm{...}` notation.
9. Use `\mathrm{Col}(X)`, `\mathrm{rank}(X)`, `\mathrm{tr}(A)`, and `\mathrm{diag}(A)`.
10. Use `\underset{\theta}{\mathrm{argmax}}` and `\underset{\theta}{\mathrm{argmin}}`.

A valid displayed matrix is written on one physical Markdown line:

```latex
$$\begin{bmatrix}x_1\\x_2\end{bmatrix}$$
```

Run the repository audit after editing mathematical Markdown:

```bash
python tools/audit_markdown_math.py
```