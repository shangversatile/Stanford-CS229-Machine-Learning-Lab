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


## Lecture 3 Figures

Run the following command from the repository root to regenerate the Lecture 3 educational figures:

```bash
python tools/generate_lecture03_figures.py
```

The deterministic script uses only NumPy and Matplotlib. It creates nine PNG files in `assets/figures/` covering underfitting and overfitting, locally weighted regression weights, bandwidth effects, high-dimensional distance concentration, sigmoid output, linear versus logistic output, logistic decision boundaries, Newton tangent iteration, and quadratic convergence.

## Lecture 4 Figures

Run the following command from the repository root to regenerate the Lecture 4 educational figures:

```bash
python tools/generate_lecture04_figures.py
```

The deterministic script uses only NumPy and Matplotlib. It creates eleven PNG files in `assets/figures/` covering Perceptron vector updates, Perceptron versus logistic response behavior, exponential-family anatomy, why exponential-family GLMs emerge, log-partition moments, response-distribution mapping, the GLM construction pipeline, canonical Gaussian/Bernoulli/Poisson responses, softmax probability coupling, the softmax simplex, and the Newton curvature bridge.

## Lecture 5 Figures

Run the following command from the repository root to regenerate the Lecture 5 educational figures:

```bash
python tools/generate_lecture05_figures.py
```

The deterministic script uses only NumPy and Matplotlib. It creates seven PNG files in `assets/figures/` covering generative versus discriminative modeling, a correlated bivariate Gaussian 3D density, the matching 2D Gaussian contours, covariance geometry variants, GDA shared-covariance decision geometry, shared-covariance versus unequal-covariance boundary comparison, and the Naive Bayes conditional-independence schematic.

## Markdown Math Audit

Run the read-only compatibility audit from the repository root:

```bash
python tools/audit_markdown_math.py
```

The script recursively scans Markdown files, ignores ordinary fenced code blocks, and validates GitHub `math` fences. Older notes still permit single-line `$$...$$` display formulas, while Lecture 4+ newer files are checked for fenced `math` display blocks only. The audit reports legacy backslash-parenthesis inline delimiters, bracket-style display delimiters, standalone double-dollar lines, unsupported macros, malformed arg notation, raw LaTeX command lines, and malformed display math. It does not modify files and exits with a nonzero status when rendering issues are found.

See [Markdown Math Style Guide](markdown-math-style-guide.md) for the repository conventions enforced by the audit.
