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
