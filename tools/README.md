# Tools

## Lecture 1 Figures

Run the following command from the repository root to regenerate the Lecture 1 educational figures:

```bash
python tools/generate_lecture01_figures.py
```

The generated PNG files are saved in `assets/figures/` and are referenced by `lecture-notes/lecture-01-introduction/note.md`.

## Lecture 2 Figures

Run the following command from the repository root to regenerate the Lecture 2 educational figures:

```bash
python tools/generate_lecture02_figures.py
```

The deterministic script uses only NumPy and Matplotlib. It creates six PNG files in `assets/figures/` covering regression residuals, the quadratic loss surface, gradient geometry, surface-normal projection, batch GD versus SGD, and normal-equation projection.

## Markdown Math Audit

Run the read-only compatibility audit from the repository root:

```bash
python tools/audit_markdown_math.py
```

The script recursively scans Markdown files and reports renderer-sensitive operator, optimization, column-space, rank, span, trace, diagonal, and projection notation with file paths and line numbers. It does not modify files and exits with a nonzero status when suspicious patterns are found.
