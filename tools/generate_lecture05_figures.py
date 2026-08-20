"""Generate deterministic educational figures for CS229 Lecture 5.

The script uses only NumPy and Matplotlib. Run it from the repository root;
output paths are resolved relative to the repository root.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "figures"
RNG = np.random.default_rng(22905)

COLORS = {
    "blue": "#2F6BBA",
    "orange": "#D55E00",
    "green": "#00876C",
    "purple": "#7B3F98",
    "red": "#B23A48",
    "gray": "#555555",
    "light_gray": "#D8D8D8",
    "yellow": "#E6AB02",
    "black": "#222222",
}

GAUSSIAN_MU = np.array([0.6, -0.4])
GAUSSIAN_SIGMA = np.array([[1.7, 0.9], [0.9, 0.85]])


def configure_matplotlib() -> None:
    """Set a compact academic plotting style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "font.family": "DejaVu Sans",
            "grid.color": "#D9D9D9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
        }
    )


def save(fig: plt.Figure, filename: str) -> Path:
    """Save one figure and return its absolute path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def gaussian_pdf_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Return bivariate Gaussian density values on a mesh grid."""
    points = np.stack([xx - mu[0], yy - mu[1]], axis=-1)
    inv = np.linalg.inv(sigma)
    exponent = np.einsum("...i,ij,...j->...", points, inv, points)
    normalizer = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(sigma)))
    return normalizer * np.exp(-0.5 * exponent)


def make_gaussian_grid(
    mu: np.ndarray = GAUSSIAN_MU,
    sigma: np.ndarray = GAUSSIAN_SIGMA,
    size: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a grid and density values for the shared Lecture 5 Gaussian."""
    eigvals, eigvecs = np.linalg.eigh(sigma)
    radius = 3.4 * np.sqrt(eigvals.max())
    x1 = np.linspace(mu[0] - radius, mu[0] + radius, size)
    x2 = np.linspace(mu[1] - radius, mu[1] + radius, size)
    xx, yy = np.meshgrid(x1, x2)
    zz = gaussian_pdf_grid(xx, yy, mu, sigma)
    return xx, yy, zz


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    """Draw a labeled conceptual box."""
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.42,rounding_size=0.08",
            "facecolor": color,
            "edgecolor": "#333333",
            "linewidth": 1.1,
            "alpha": 0.16,
        },
    )


def draw_connector(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Draw an arrow between conceptual boxes."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.4,
        color="#444444",
        shrinkA=13,
        shrinkB=13,
    )
    ax.add_patch(arrow)


def figure_generative_vs_discriminative() -> Path:
    """Show conditional and generative modeling routes to classification."""
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    ax.set_axis_off()

    positions = {
        "data_left": (0.09, 0.70),
        "cond": (0.28, 0.70),
        "score": (0.49, 0.70),
        "decision_left": (0.70, 0.70),
        "data_right": (0.09, 0.30),
        "prior": (0.30, 0.42),
        "classcond": (0.30, 0.18),
        "bayes": (0.55, 0.30),
        "decision_right": (0.80, 0.30),
    }

    boxes = {
        "data_left": ("observed pair\n(x, y)", COLORS["blue"]),
        "cond": ("discriminative\nmodel p(y | x)", COLORS["green"]),
        "score": ("conditional score\nor posterior", COLORS["purple"]),
        "decision_left": ("predict y\nfrom x", COLORS["red"]),
        "data_right": ("observed pair\n(x, y)", COLORS["blue"]),
        "prior": ("class prior\np(y)", COLORS["orange"]),
        "classcond": ("class-conditional\nmodel p(x | y)", COLORS["green"]),
        "bayes": ("Bayes rule\ncompare p(x | y)p(y)", COLORS["purple"]),
        "decision_right": ("predict y\nfrom x", COLORS["red"]),
    }

    for key, (label, color) in boxes.items():
        draw_box(ax, positions[key], label, color)

    connector_points = [
        ((0.16, 0.70), (0.22, 0.70)),
        ((0.35, 0.70), (0.41, 0.70)),
        ((0.57, 0.70), (0.64, 0.70)),
        ((0.16, 0.34), (0.25, 0.41)),
        ((0.16, 0.26), (0.25, 0.19)),
        ((0.36, 0.40), (0.46, 0.34)),
        ((0.36, 0.20), (0.46, 0.26)),
        ((0.64, 0.30), (0.73, 0.30)),
    ]
    for start, end in connector_points:
        draw_connector(ax, start, end)

    ax.text(0.48, 0.90, "Two modeling routes can both produce a classifier", ha="center", fontsize=14, fontweight="semibold")
    ax.text(0.87, 0.70, "learn conditional\nstructure directly", ha="left", va="center", fontsize=10.5, color=COLORS["gray"])
    ax.text(0.88, 0.30, "learn a joint model\nthen infer labels", ha="left", va="center", fontsize=10.5, color=COLORS["gray"])
    ax.text(0.28, 0.56, "Lecture 4 route", ha="center", fontsize=10.5, color=COLORS["gray"])
    ax.text(0.30, 0.04, "Lecture 5 route", ha="center", fontsize=10.5, color=COLORS["gray"])
    ax.set_title("Generative versus Discriminative Modeling")
    return save(fig, "lecture05-generative-vs-discriminative.png")


def figure_bivariate_gaussian_density_3d() -> Path:
    """Plot a correlated bivariate Gaussian density surface."""
    xx, yy, zz = make_gaussian_grid()
    mean_density = gaussian_pdf_grid(
        np.array([[GAUSSIAN_MU[0]]]),
        np.array([[GAUSSIAN_MU[1]]]),
        GAUSSIAN_MU,
        GAUSSIAN_SIGMA,
    )[0, 0]

    fig = plt.figure(figsize=(8.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
    ax.scatter(
        [GAUSSIAN_MU[0]],
        [GAUSSIAN_MU[1]],
        [mean_density],
        s=54,
        color=COLORS["red"],
        depthshade=False,
        label="mean",
    )
    ax.plot(
        [GAUSSIAN_MU[0], GAUSSIAN_MU[0]],
        [GAUSSIAN_MU[1], GAUSSIAN_MU[1]],
        [0.0, mean_density],
        color=COLORS["red"],
        linewidth=1.6,
    )
    ax.view_init(elev=31, azim=-55)
    ax.set_title("Bivariate Gaussian Density")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("density")
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture05-bivariate-gaussian-density-3d.png")


def figure_bivariate_gaussian_contours() -> Path:
    """Plot contours for the same Gaussian used in the 3D density figure."""
    xx, yy, zz = make_gaussian_grid()
    eigvals, eigvecs = np.linalg.eigh(GAUSSIAN_SIGMA)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    contours = ax.contour(xx, yy, zz, levels=8, cmap="viridis", linewidths=1.8)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.3f")
    ax.scatter([GAUSSIAN_MU[0]], [GAUSSIAN_MU[1]], s=76, color=COLORS["red"], marker="*", label=r"mean $\mu$")

    for value, vector, color in zip(eigvals, eigvecs.T, [COLORS["orange"], COLORS["blue"]]):
        start = GAUSSIAN_MU
        end = GAUSSIAN_MU + 1.4 * np.sqrt(value) * vector
        ax.arrow(
            start[0],
            start[1],
            end[0] - start[0],
            end[1] - start[1],
            width=0.015,
            head_width=0.12,
            length_includes_head=True,
            color=color,
            zorder=6,
        )

    ax.set_title("Same Gaussian as 2D Density Contours")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture05-bivariate-gaussian-contours.png")


def covariance_ellipse(
    ax: plt.Axes,
    mu: np.ndarray,
    sigma: np.ndarray,
    c_value: float,
    color: str,
    label: str | None = None,
) -> None:
    """Draw the ellipse given by (x-mu)^T Sigma^{-1} (x-mu) = c_value."""
    eigvals, eigvecs = np.linalg.eigh(sigma)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2.0 * np.sqrt(c_value * eigvals)
    ellipse = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=2.0,
        label=label,
    )
    ax.add_patch(ellipse)


def figure_covariance_geometry_variants() -> Path:
    """Compare four covariance matrices through their ellipses."""
    matrices = [
        ("isotropic", np.array([[1.0, 0.0], [0.0, 1.0]]), COLORS["blue"]),
        ("unequal diagonal\nvariances", np.array([[1.8, 0.0], [0.0, 0.45]]), COLORS["orange"]),
        ("positive\ncovariance", np.array([[1.5, 0.8], [0.8, 0.9]]), COLORS["green"]),
        ("negative\ncovariance", np.array([[1.5, -0.8], [-0.8, 0.9]]), COLORS["red"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2), sharex=True, sharey=True)
    for ax, (title, sigma, color) in zip(axes.flat, matrices):
        covariance_ellipse(ax, np.zeros(2), sigma, 1.0, color, "1 Mahalanobis unit")
        covariance_ellipse(ax, np.zeros(2), sigma, 4.0, color, "2 Mahalanobis units")
        eigvals, eigvecs = np.linalg.eigh(sigma)
        for value, vector in zip(eigvals, eigvecs.T):
            end = 1.3 * np.sqrt(value) * vector
            ax.arrow(
                0.0,
                0.0,
                end[0],
                end[1],
                width=0.012,
                head_width=0.11,
                length_includes_head=True,
                color=COLORS["gray"],
                zorder=6,
            )
        ax.scatter([0.0], [0.0], s=35, color=COLORS["black"], zorder=7)
        ax.set_title(title)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
    axes[0, 0].legend(frameon=True, loc="upper right")
    fig.suptitle("Covariance Matrix Controls Ellipse Scale and Orientation", y=0.98, fontsize=14, fontweight="semibold")
    return save(fig, "lecture05-covariance-geometry-variants.png")


def gda_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return deterministic two-class GDA parameters."""
    mu0 = np.array([-1.3, -0.7])
    mu1 = np.array([1.15, 0.9])
    sigma = np.array([[1.25, 0.55], [0.55, 0.8]])
    phi = 0.55
    return mu0, mu1, sigma, phi


def gda_line(mu0: np.ndarray, mu1: np.ndarray, sigma: np.ndarray, phi: float) -> tuple[np.ndarray, float]:
    """Return w and b for the GDA posterior log-odds boundary."""
    inv = np.linalg.inv(sigma)
    w = inv @ (mu1 - mu0)
    b = -0.5 * mu1.T @ inv @ mu1 + 0.5 * mu0.T @ inv @ mu0 + np.log(phi / (1.0 - phi))
    return w, float(b)


def draw_decision_line(ax: plt.Axes, w: np.ndarray, b: float, color: str, label: str) -> None:
    """Draw the line w^T x + b = 0."""
    x_values = np.linspace(-4.2, 4.2, 300)
    if abs(w[1]) < 1e-10:
        ax.axvline(-b / w[0], color=color, linewidth=2.3, label=label)
        return
    y_values = -(w[0] * x_values + b) / w[1]
    ax.plot(x_values, y_values, color=color, linewidth=2.4, label=label)


def figure_gda_shared_covariance_boundary() -> Path:
    """Show GDA class-conditional contours and the linear Bayes boundary."""
    mu0, mu1, sigma, phi = gda_parameters()
    grid = np.linspace(-4.2, 4.2, 220)
    xx, yy = np.meshgrid(grid, grid)
    z0 = gaussian_pdf_grid(xx, yy, mu0, sigma)
    z1 = gaussian_pdf_grid(xx, yy, mu1, sigma)
    w, b = gda_line(mu0, mu1, sigma, phi)

    samples0 = RNG.multivariate_normal(mu0, sigma, size=70)
    samples1 = RNG.multivariate_normal(mu1, sigma, size=70)

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.scatter(samples0[:, 0], samples0[:, 1], s=22, color=COLORS["blue"], alpha=0.40, label="class 0 samples")
    ax.scatter(samples1[:, 0], samples1[:, 1], s=22, color=COLORS["orange"], alpha=0.40, label="class 1 samples")
    ax.contour(xx, yy, z0, levels=5, colors=COLORS["blue"], linewidths=1.5)
    ax.contour(xx, yy, z1, levels=5, colors=COLORS["orange"], linewidths=1.5)
    ax.scatter([mu0[0]], [mu0[1]], s=78, color=COLORS["blue"], marker="X", edgecolor="white", linewidth=0.7, label=r"$\mu_0$")
    ax.scatter([mu1[0]], [mu1[1]], s=78, color=COLORS["orange"], marker="X", edgecolor="white", linewidth=0.7, label=r"$\mu_1$")
    draw_decision_line(ax, w, b, COLORS["red"], "GDA boundary")
    ax.set_title("GDA Geometry with Shared Covariance")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.0, 4.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture05-gda-shared-covariance-boundary.png")


def qda_log_ratio(
    xx: np.ndarray,
    yy: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    sigma0: np.ndarray,
    sigma1: np.ndarray,
    phi: float,
) -> np.ndarray:
    """Return log p(y=1|x) odds up to the Bayes-normalizing denominator."""
    points = np.stack([xx, yy], axis=-1)
    diff0 = points - mu0
    diff1 = points - mu1
    inv0 = np.linalg.inv(sigma0)
    inv1 = np.linalg.inv(sigma1)
    quad0 = np.einsum("...i,ij,...j->...", diff0, inv0, diff0)
    quad1 = np.einsum("...i,ij,...j->...", diff1, inv1, diff1)
    return (
        np.log(phi / (1.0 - phi))
        - 0.5 * np.log(np.linalg.det(sigma1) / np.linalg.det(sigma0))
        - 0.5 * quad1
        + 0.5 * quad0
    )


def figure_gda_qda_boundary_comparison() -> Path:
    """Compare shared-covariance linear boundary with unequal-covariance boundary."""
    mu0, mu1, shared_sigma, phi = gda_parameters()
    sigma0 = np.array([[1.15, 0.25], [0.25, 0.55]])
    sigma1 = np.array([[0.75, -0.38], [-0.38, 1.25]])
    grid = np.linspace(-4.2, 4.2, 260)
    xx, yy = np.meshgrid(grid, grid)
    z_shared0 = gaussian_pdf_grid(xx, yy, mu0, shared_sigma)
    z_shared1 = gaussian_pdf_grid(xx, yy, mu1, shared_sigma)
    z_qda0 = gaussian_pdf_grid(xx, yy, mu0, sigma0)
    z_qda1 = gaussian_pdf_grid(xx, yy, mu1, sigma1)
    w, b = gda_line(mu0, mu1, shared_sigma, phi)
    qda_ratio = qda_log_ratio(xx, yy, mu0, mu1, sigma0, sigma1, phi)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.7), sharex=True, sharey=True)

    axes[0].contour(xx, yy, z_shared0, levels=5, colors=COLORS["blue"], linewidths=1.4)
    axes[0].contour(xx, yy, z_shared1, levels=5, colors=COLORS["orange"], linewidths=1.4)
    draw_decision_line(axes[0], w, b, COLORS["red"], "linear boundary")
    axes[0].set_title("shared covariance")

    axes[1].contour(xx, yy, z_qda0, levels=5, colors=COLORS["blue"], linewidths=1.4)
    axes[1].contour(xx, yy, z_qda1, levels=5, colors=COLORS["orange"], linewidths=1.4)
    axes[1].contour(xx, yy, qda_ratio, levels=[0.0], colors=COLORS["red"], linewidths=2.4)
    axes[1].plot([], [], color=COLORS["red"], linewidth=2.4, label="quadratic boundary")
    axes[1].set_title("unequal covariance")

    for ax in axes:
        ax.scatter([mu0[0]], [mu0[1]], s=72, color=COLORS["blue"], marker="X", edgecolor="white", linewidth=0.7, label=r"$\mu_0$")
        ax.scatter([mu1[0]], [mu1[1]], s=72, color=COLORS["orange"], marker="X", edgecolor="white", linewidth=0.7, label=r"$\mu_1$")
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-4.0, 4.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x_1$")
        ax.grid(True)
        ax.legend(frameon=True, loc="upper left")
    axes[0].set_ylabel(r"$x_2$")
    fig.suptitle("Boundary Shape Comes from Quadratic-Term Cancellation", y=1.02, fontsize=14, fontweight="semibold")
    return save(fig, "lecture05-gda-qda-boundary-comparison.png")


def figure_naive_bayes_conditional_independence() -> Path:
    """Draw the Naive Bayes conditional-independence factorization."""
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.set_axis_off()
    y_pos = (0.50, 0.78)
    feature_positions = [(0.16, 0.34), (0.34, 0.34), (0.52, 0.34), (0.70, 0.34), (0.88, 0.34)]

    draw_box(ax, y_pos, "class label\nY", COLORS["orange"])
    for index, pos in enumerate(feature_positions, start=1):
        label = f"word feature\n$X_{index}$" if index < 5 else "word feature\n$X_d$"
        draw_box(ax, pos, label, COLORS["blue"])
        draw_connector(ax, y_pos, pos)

    ax.text(
        0.50,
        0.16,
        r"$p(x_1,\ldots,x_d\mid y)=\prod_{j=1}^d p(x_j\mid y)$",
        ha="center",
        va="center",
        fontsize=13,
        color=COLORS["black"],
    )
    ax.text(
        0.50,
        0.055,
        "The diagram represents a factorization assumption, not a causal claim.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["gray"],
    )
    ax.set_title("Naive Bayes Conditional-Independence Schematic")
    return save(fig, "lecture05-naive-bayes-conditional-independence.png")


def main() -> None:
    """Generate every Lecture 5 figure and print relative paths."""
    configure_matplotlib()
    generators = [
        figure_generative_vs_discriminative,
        figure_bivariate_gaussian_density_3d,
        figure_bivariate_gaussian_contours,
        figure_covariance_geometry_variants,
        figure_gda_shared_covariance_boundary,
        figure_gda_qda_boundary_comparison,
        figure_naive_bayes_conditional_independence,
    ]
    paths = [generator() for generator in generators]
    print("Generated Lecture 5 figures:")
    for path in paths:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
