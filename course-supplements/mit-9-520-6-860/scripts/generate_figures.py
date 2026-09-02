"""Generate deterministic figures for MIT 9.520 / 6.860 selected supplements.

Run from the repository root or from this supplement directory. Output paths
are resolved relative to this script and written to the supplement-local
figures directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


SUPPLEMENT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = SUPPLEMENT_DIR / "figures"

COLORS = {
    "blue": "#2F6BBA",
    "orange": "#D55E00",
    "green": "#00876C",
    "red": "#B23A48",
    "purple": "#7B3F98",
    "gray": "#555555",
    "light_gray": "#D9D9D9",
    "black": "#222222",
}


def configure_matplotlib() -> None:
    """Set a compact academic plotting style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 12.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.8,
            "font.family": "DejaVu Sans",
            "grid.color": "#D9D9D9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
        }
    )


def save(fig: plt.Figure, filename: str) -> Path:
    """Save one figure and return its path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_expected_vs_empirical_risk() -> Path:
    """Show population risk as an unknown expectation and empirical risk as samples."""
    rng = np.random.default_rng(4)
    x_grid = np.linspace(-3.0, 3.0, 500)
    true_mean = 0.55 * x_grid - 0.65 * np.sin(1.25 * x_grid)
    predictor = 0.42 * x_grid
    train_x = np.linspace(-2.7, 2.7, 18)
    train_y = 0.55 * train_x - 0.65 * np.sin(1.25 * train_x) + rng.normal(0.0, 0.45, size=train_x.size)
    sample_pred = 0.42 * train_x

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.fill_between(
        x_grid,
        true_mean - 0.75,
        true_mean + 0.75,
        color=COLORS["light_gray"],
        alpha=0.7,
        label="population uncertainty",
    )
    ax.plot(x_grid, true_mean, color=COLORS["black"], linewidth=2.2, label="unknown data law")
    ax.plot(x_grid, predictor, color=COLORS["blue"], linewidth=2.5, label="candidate predictor")
    ax.scatter(train_x, train_y, color=COLORS["orange"], s=40, zorder=4, label="finite training sample")
    for x_value, y_value, pred_value in zip(train_x, train_y, sample_pred):
        ax.plot([x_value, x_value], [pred_value, y_value], color=COLORS["red"], linewidth=0.9, alpha=0.65)

    ax.text(-2.85, 2.15, "Expected risk: average over unknown population", color=COLORS["black"], fontsize=10.5)
    ax.text(-2.85, 1.75, "Empirical risk: average over observed residuals", color=COLORS["red"], fontsize=10.5)
    ax.set_title("Expected Risk vs Empirical Risk")
    ax.set_xlabel("input x")
    ax.set_ylabel("output y")
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-2.65, 2.55)
    ax.grid(True)
    ax.legend(frameon=True, loc="lower right")
    return save(fig, "mit9520-risk-empirical-schematic.png")


def figure_ill_posed_learning() -> Path:
    """Plot multiple functions that fit the same finite observations."""
    x_train = np.array([-2.6, -1.6, -0.65, 0.15, 1.05, 2.2])
    y_train = np.array([-0.5, -1.15, -0.1, 0.7, 0.35, 1.15])
    x_grid = np.linspace(-2.9, 2.55, 650)
    interpolating = np.poly1d(np.polyfit(x_train, y_train, deg=len(x_train) - 1))(x_grid)
    ridge_like = np.poly1d(np.polyfit(x_train, y_train, deg=3))(x_grid)
    linear = np.poly1d(np.polyfit(x_train, y_train, deg=1))(x_grid)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(x_grid, interpolating, color=COLORS["red"], linewidth=2.0, label="interpolating high-degree fit")
    ax.plot(x_grid, ridge_like, color=COLORS["green"], linewidth=2.4, label="smoother structured fit")
    ax.plot(x_grid, linear, color=COLORS["blue"], linewidth=2.2, label="linear hypothesis")
    ax.scatter(x_train, y_train, color=COLORS["black"], s=52, zorder=5, label="same finite sample")
    ax.set_title("Finite Samples Do Not Identify a Unique Function")
    ax.set_xlabel("input x")
    ax.set_ylabel("prediction f(x)")
    ax.set_xlim(-2.9, 2.55)
    ax.set_ylim(-2.4, 2.4)
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "mit9520-ill-posed-functions.png")


def figure_ridge_geometry() -> Path:
    """Draw least-squares contours, an L2 ball, OLS, and constrained ridge solution."""
    a_matrix = np.array([[2.7, 1.05], [1.05, 0.9]])
    w_ols = np.array([1.85, 1.15])
    radius = 1.38
    theta = np.linspace(0.0, 2.0 * np.pi, 1000)
    circle_points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    values = np.einsum("ij,jk,ik->i", circle_points - w_ols, a_matrix, circle_points - w_ols)
    w_ridge = circle_points[np.argmin(values)]

    x = np.linspace(-2.2, 2.5, 420)
    y = np.linspace(-2.0, 2.2, 420)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack((xx - w_ols[0], yy - w_ols[1]), axis=-1)
    objective = np.einsum("...i,ij,...j->...", grid, a_matrix, grid)

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    levels = [0.15, 0.45, 0.85, 1.35, 2.05, 3.05, 4.35]
    contours = ax.contour(xx, yy, objective, levels=levels, colors=COLORS["gray"], linewidths=1.0)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.2g")
    ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor=COLORS["blue"], linewidth=2.4, label="L2 constraint"))
    ax.scatter([w_ols[0]], [w_ols[1]], color=COLORS["red"], s=72, zorder=5, label="OLS")
    ax.scatter([w_ridge[0]], [w_ridge[1]], color=COLORS["green"], s=72, zorder=6, label="ridge / constrained solution")
    ax.plot([0.0, w_ols[0]], [0.0, w_ols[1]], color=COLORS["red"], linestyle=":", linewidth=1.4)
    ax.plot([0.0, w_ridge[0]], [0.0, w_ridge[1]], color=COLORS["green"], linestyle=":", linewidth=1.4)
    ax.set_title("Ridge Geometry in Parameter Space")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_xlim(-2.2, 2.5)
    ax.set_ylim(-2.0, 2.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(frameon=True, loc="lower left")
    return save(fig, "mit9520-ridge-geometry.png")


def figure_singular_value_shrinkage() -> Path:
    """Plot OLS inverse factors and ridge spectral filters."""
    sigma = np.linspace(0.04, 5.0, 900)
    n = 50.0
    lambdas = [0.005, 0.03, 0.12]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    axes[0].plot(sigma, 1.0 / sigma, color=COLORS["gray"], linewidth=2.0, label="OLS inverse 1/sigma")
    for lam, color in zip(lambdas, [COLORS["blue"], COLORS["green"], COLORS["orange"]]):
        axes[0].plot(sigma, sigma / (sigma**2 + n * lam), linewidth=2.35, color=color, label=f"ridge lambda={lam:g}")
    axes[0].set_title("Spectral Filter Factors")
    axes[0].set_xlabel("singular value sigma")
    axes[0].set_ylabel("coefficient multiplier")
    axes[0].set_ylim(0.0, 8.0)
    axes[0].grid(True)
    axes[0].legend(frameon=True, loc="upper right")

    for lam, color in zip(lambdas, [COLORS["blue"], COLORS["green"], COLORS["orange"]]):
        axes[1].plot(sigma, sigma**2 / (sigma**2 + n * lam), linewidth=2.35, color=color, label=f"lambda={lam:g}")
    axes[1].set_title("Shrinkage Relative to OLS")
    axes[1].set_xlabel("singular value sigma")
    axes[1].set_ylabel("ridge / OLS factor")
    axes[1].set_ylim(-0.02, 1.03)
    axes[1].grid(True)
    axes[1].legend(frameon=True, loc="lower right")

    fig.suptitle("Ridge Shrinks Small-Singular-Value Directions First", y=1.03, fontsize=14, fontweight="semibold")
    return save(fig, "mit9520-ridge-spectral-shrinkage.png")


def figure_batch_vs_sgd_path() -> Path:
    """Compare smooth full-gradient descent and noisy stochastic-gradient updates."""
    rng = np.random.default_rng(9)
    a_matrix = np.array([[2.6, 0.9], [0.9, 0.85]])
    optimum = np.array([0.0, 0.0])
    start = np.array([-2.4, 1.85])
    batch_path = [start.copy()]
    sgd_path = [start.copy()]

    w_batch = start.copy()
    for _ in range(22):
        grad = a_matrix @ (w_batch - optimum)
        w_batch = w_batch - 0.17 * grad
        batch_path.append(w_batch.copy())

    w_sgd = start.copy()
    for step in range(64):
        grad = a_matrix @ (w_sgd - optimum)
        noise_scale = 1.65 / ((step + 2.0) ** 0.35)
        noisy_grad = grad + rng.normal(0.0, noise_scale, size=2)
        w_sgd = w_sgd - 0.10 * noisy_grad
        sgd_path.append(w_sgd.copy())

    x = np.linspace(-2.8, 2.3, 420)
    y = np.linspace(-2.2, 2.2, 420)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack((xx, yy), axis=-1)
    objective = 0.5 * np.einsum("...i,ij,...j->...", grid, a_matrix, grid)
    batch_path_arr = np.array(batch_path)
    sgd_path_arr = np.array(sgd_path)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.contour(xx, yy, objective, levels=12, colors=COLORS["light_gray"], linewidths=0.95)
    ax.plot(batch_path_arr[:, 0], batch_path_arr[:, 1], "-o", color=COLORS["blue"], markersize=3.2, linewidth=2.0, label="batch GD")
    ax.plot(sgd_path_arr[:, 0], sgd_path_arr[:, 1], "-o", color=COLORS["orange"], markersize=2.6, linewidth=1.55, alpha=0.9, label="SGD")
    ax.scatter([0.0], [0.0], color=COLORS["green"], s=80, marker="*", zorder=6, label="optimum region")
    ax.set_title("Batch Gradient Descent vs SGD")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_xlim(-2.8, 2.3)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(frameon=True, loc="upper right")
    return save(fig, "mit9520-batch-vs-sgd-path.png")


def main() -> None:
    """Generate every MIT supplement figure and print relative paths."""
    configure_matplotlib()
    generators = [
        figure_expected_vs_empirical_risk,
        figure_ill_posed_learning,
        figure_ridge_geometry,
        figure_singular_value_shrinkage,
        figure_batch_vs_sgd_path,
    ]
    paths = [generator() for generator in generators]
    print("Generated MIT 9.520 / 6.860 selected supplement figures:")
    for path in paths:
        print(f"- {path.relative_to(SUPPLEMENT_DIR)}")


if __name__ == "__main__":
    main()
