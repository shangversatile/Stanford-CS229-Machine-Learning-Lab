"""Generate deterministic figures for CMU 10-601 Supplement 01.

Run from the repository root or from this module. Output paths are resolved
relative to this script and written to the module-local figures directory.
"""

from __future__ import annotations

from math import lgamma
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


MODULE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = MODULE_DIR / "figures"

COLORS = {
    "blue": "#2F6BBA",
    "orange": "#D55E00",
    "green": "#00876C",
    "red": "#B23A48",
    "purple": "#7B3F98",
    "gray": "#555555",
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


def beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Return beta density values for x in (0, 1)."""
    log_beta = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    return np.exp((alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log1p(-x) - log_beta)


def scale_to_unit(values: np.ndarray) -> np.ndarray:
    """Scale positive values to a maximum of one."""
    max_value = float(np.max(values))
    if max_value <= 0.0:
        return values
    return values / max_value


def figure_beta_bernoulli_mle_map() -> Path:
    """Show one Bernoulli likelihood combined with different Beta priors."""
    n_ones = 3
    n_zeros = 7
    sample_count = n_ones + n_zeros
    mle = n_ones / sample_count
    x = np.linspace(0.002, 0.998, 900)
    likelihood = scale_to_unit((x**n_ones) * ((1.0 - x) ** n_zeros))
    priors = [
        ("uniform prior\nBeta(1, 1)", 1.0, 1.0, COLORS["blue"]),
        ("balanced prior\nBeta(5, 5)", 5.0, 5.0, COLORS["green"]),
        ("low-rate prior\nBeta(2, 8)", 2.0, 8.0, COLORS["orange"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), sharey=True)
    for ax, (title, alpha, beta, color) in zip(axes, priors):
        posterior_alpha = n_ones + alpha
        posterior_beta = n_zeros + beta
        posterior = scale_to_unit(beta_pdf(x, posterior_alpha, posterior_beta))
        map_estimate = (n_ones + alpha - 1.0) / (sample_count + alpha + beta - 2.0)
        posterior_mean = (n_ones + alpha) / (sample_count + alpha + beta)

        ax.plot(x, likelihood, color=COLORS["gray"], linewidth=2.0, label="likelihood")
        ax.plot(x, posterior, color=color, linewidth=2.4, label="posterior")
        ax.axvline(mle, color=COLORS["red"], linestyle="-", linewidth=1.8, label="MLE")
        ax.axvline(map_estimate, color=COLORS["purple"], linestyle="--", linewidth=1.8, label="MAP")
        ax.axvline(posterior_mean, color=COLORS["black"], linestyle=":", linewidth=2.1, label="posterior mean")
        ax.set_title(title)
        ax.set_xlabel("phi")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.08)
        ax.grid(True)
    axes[0].set_ylabel("scaled density / likelihood")
    axes[-1].legend(frameon=True, loc="upper right")
    fig.suptitle("Same Bernoulli Likelihood, Different Beta Priors", y=1.05, fontsize=14, fontweight="semibold")
    return save(fig, "cmu10601-beta-bernoulli-mle-map.png")


def figure_nb_parameter_reduction() -> Path:
    """Compare full binary class-conditional parameters with NB parameters."""
    dims = np.arange(1, 13)
    full_params = (2**dims) - 1
    nb_params = dims

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    axes[0].plot(dims, full_params, marker="o", linewidth=2.3, color=COLORS["red"], label="full p(X | Y=k): 2^d - 1")
    axes[0].plot(dims, nb_params, marker="s", linewidth=2.3, color=COLORS["blue"], label="Bernoulli NB: d")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of binary features d")
    axes[0].set_ylabel("free parameters per class")
    axes[0].set_title("Parameter Growth")
    axes[0].grid(True, which="both")
    axes[0].legend(frameon=True, loc="upper left")

    categories = ["full\nclass conditional", "Naive Bayes\nfactorized"]
    values = [2**10 - 1, 10]
    colors = [COLORS["red"], COLORS["blue"]]
    axes[1].bar(categories, values, color=colors, alpha=0.88)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("free parameters per class, d=10")
    axes[1].set_title("One Class-Conditional Table")
    axes[1].grid(True, axis="y", which="both")
    for index, value in enumerate(values):
        color = "white" if value > 100 else COLORS["black"]
        y_position = value * 0.72 if value > 100 else value * 1.18
        axes[1].text(index, y_position, f"{value:,}", ha="center", va="center", fontsize=10.5, color=color)

    fig.suptitle("Naive Bayes Turns Exponential Growth into Linear Growth", y=1.03, fontsize=14, fontweight="semibold")
    return save(fig, "cmu10601-nb-parameter-reduction.png")


def covariance_ellipse(ax: plt.Axes, mu: np.ndarray, sigma: np.ndarray, color: str, label: str | None = None) -> None:
    """Draw a two-standard-deviation covariance ellipse."""
    eigvals, eigvecs = np.linalg.eigh(sigma)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 4.0 * np.sqrt(eigvals)
    ellipse = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=2.3,
        label=label,
    )
    ax.add_patch(ellipse)
    ax.scatter([mu[0]], [mu[1]], color=color, marker="x", s=70, linewidth=2.2)


def figure_covariance_assumptions() -> Path:
    """Compare GDA/LDA, QDA, and Gaussian NB covariance assumptions."""
    mu0 = np.array([-1.1, -0.55])
    mu1 = np.array([1.15, 0.65])
    shared_full = np.array([[1.35, 0.62], [0.62, 0.75]])
    qda0 = np.array([[1.15, 0.45], [0.45, 0.50]])
    qda1 = np.array([[0.72, -0.38], [-0.38, 1.20]])
    gnb0 = np.array([[1.05, 0.0], [0.0, 0.42]])
    gnb1 = np.array([[0.55, 0.0], [0.0, 1.05]])

    panels = [
        ("GDA / LDA-style\nshared full covariance", shared_full, shared_full),
        ("QDA\nclass-specific full covariance", qda0, qda1),
        ("Gaussian NB\nclass-specific diagonal covariance", gnb0, gnb1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharex=True, sharey=True)
    for ax, (title, sigma0, sigma1) in zip(axes, panels):
        covariance_ellipse(ax, mu0, sigma0, COLORS["blue"], "class 0")
        covariance_ellipse(ax, mu1, sigma1, COLORS["orange"], "class 1")
        ax.set_title(title)
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].legend(frameon=True, loc="upper left")
    fig.suptitle("Gaussian Class-Conditional Models Differ by Covariance Constraints", y=1.04, fontsize=14, fontweight="semibold")
    return save(fig, "cmu10601-gda-qda-gnb-covariance.png")


def main() -> None:
    """Generate every CMU supplement figure and print relative paths."""
    configure_matplotlib()
    generators = [
        figure_beta_bernoulli_mle_map,
        figure_nb_parameter_reduction,
        figure_covariance_assumptions,
    ]
    paths = [generator() for generator in generators]
    print("Generated CMU 10-601 Supplement 01 figures:")
    for path in paths:
        print(f"- {path.relative_to(MODULE_DIR)}")


if __name__ == "__main__":
    main()
