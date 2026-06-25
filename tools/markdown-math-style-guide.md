# Markdown Math Style Guide

Use these rules for notes intended to render on GitHub or a strict Markdown math renderer.

1. Use inline math only for short expressions, for example `\(x\)`.
2. Use display math for derivations and multi-line equations.
3. Put the opening `$$` alone on its own line.
4. Put the closing `$$` alone on its own line.
5. Do not write same-line display blocks such as `$$ formula $$`.
6. Do not mix prose inside display math. Put explanatory sentences before or after the block.
7. Avoid unsupported named-operator macros such as `&#92;operatorname`.
8. Prefer renderer-safe notation such as `\mathrm{Col}(X)`, `\mathrm{rank}(X)`, and `\mathrm{tr}(A)`.
9. Use `\underset{\theta}{\mathrm{argmax}}` instead of the fragile `&#92;arg&#92;max` form.
10. Write multi-row vectors and matrices with one row per source line:

```latex
$$
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$
```

Run the repository audit after editing mathematical Markdown:

```bash
python tools/audit_markdown_math.py
```
