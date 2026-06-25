# Markdown Math Style Guide

Use these rules for notes intended to render on GitHub or a strict Markdown math renderer.

1. Inline math: use `\( ... \)` only for short expressions.
2. Display math: use `\[` and `\]`.
3. Never use double-dollar display math delimiters.
4. The opening `\[` must be alone on its own line.
5. The closing `\]` must be alone on its own line.
6. Do not mix prose inside display math. Put explanatory sentences before or after the block.
7. Avoid unsupported named-operator macros; use renderer-safe `\mathrm{...}` notation.
8. Use `\mathrm{Col}(X)`, `\mathrm{rank}(X)`, `\mathrm{tr}(A)`, and `\mathrm{diag}(A)`.
9. Use `\underset{\theta}{\mathrm{argmax}}` and `\underset{\theta}{\mathrm{argmin}}`.
10. Write multi-row vectors and matrices with one row per source line:

```latex
\[
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
\]
```

Run the repository audit after editing mathematical Markdown:

```bash
python tools/audit_markdown_math.py
```