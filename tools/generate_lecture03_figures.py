"""Generate deterministic educational figures for CS229 Lecture 3.

The script uses only NumPy and Matplotlib. Run it from the repository root;
output paths are resolved relative to the repository root.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "figures"
RNG = np.random.default_rng(22903)

COLORS = {
    "blue": "#2F6BBA",
    "orange": "#D55E00",
    "green": "#00876C",
    "purple": "#7B3F98",
    "red": "#B23A48",
    "gray": "#555555",
    "light_gray": "#D8D8D8",
    "yellow": "#E6AB02",
}


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


def true_curve(x: np.ndarray) -> np.ndarray:
    """Smooth nonlinear regression target used by several figures."""
    return 0.45 * np.sin(2.4 * x) + 0.18 * x + 0.08 * x**2


def make_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic one-dimensional nonlinear regression data."""
    x = np.linspace(-2.8, 2.8, 38)
    y = true_curve(x) + RNG.normal(0.0, 0.12, size=x.shape)
    return x, y


def local_linear_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Predict with one-dimensional locally weighted linear regression."""
    design = np.column_stack([np.ones_like(train_x), train_x])
    predictions = []
    ridge = 1e-8 * np.eye(2)
    for x_0 in query_x:
        weights = np.exp(-((train_x - x_0) ** 2) / (2.0 * tau**2))
        weighted_design = design * weights[:, None]
        lhs = design.T @ weighted_design + ridge
        rhs = design.T @ (weights * train_y)
        theta = np.linalg.solve(lhs, rhs)
        predictions.append(theta[0] + theta[1] * x_0)
    return np.asarray(predictions)


def figure_underfitting_overfitting() -> Path:
    """Show true curve, underfit line, and overfit polynomial curve."""
    x, y = make_regression_data()
    grid = np.linspace(-3.0, 3.0, 400)
    line_coef = np.polyfit(x, y, deg=1)
    overfit_coef = np.polyfit(x, y, deg=13)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(
        x,
        y,
        s=38,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.7,
        label="Noisy observations",
        zorder=4,
    )
    ax.plot(
        grid,
        true_curve(grid),
        color="#222222",
        linewidth=2.2,
        label="True smooth trend",
    )
    ax.plot(
        grid,
        np.polyval(line_coef, grid),
        color=COLORS["orange"],
        linewidth=2.0,
        label="Underfit global line",
    )
    ax.plot(
        grid,
        np.polyval(overfit_coef, grid),
        color=COLORS["red"],
        linewidth=1.8,
        label="Overfit high-degree curve",
    )
    ax.set_title("Underfitting and Overfitting")
    ax.set_xlabel("Input x")
    ax.set_ylabel("Target y")
    ax.set_ylim(-1.05, 1.25)
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture03-underfitting-overfitting.png")


def figure_lwr_weights_query_point() -> Path:
    """Show Gaussian-kernel-like LWR weights around a query point."""
    x = np.linspace(-3.0, 3.0, 70)
    query = 0.85
    tau = 0.65
    weights = np.exp(-((x - query) ** 2) / (2.0 * tau**2))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(
        x,
        weights,
        color=COLORS["blue"],
        linewidth=2.3,
        label=r"Weight $w^{(i)}(x)$",
    )
    ax.scatter(
        x,
        weights,
        s=22 + 95 * weights,
        c=weights,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
        label="Training examples",
    )
    ax.axvline(
        query,
        color=COLORS["red"],
        linestyle="--",
        linewidth=2.0,
        label="Query point",
    )
    ax.set_title("Locally Weighted Regression Weights Around a Query")
    ax.set_xlabel("Input coordinate")
    ax.set_ylabel("Weight")
    ax.set_ylim(-0.04, 1.08)
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture03-lwr-weights-query-point.png")


def figure_tau_small_medium_large() -> Path:
    """Compare LWR fits under small, medium, and large bandwidths."""
    x, y = make_regression_data()
    grid = np.linspace(-3.0, 3.0, 300)
    taus = [0.18, 0.55, 1.6]
    labels = [r"Small $\tau$", r"Medium $\tau$", r"Large $\tau$"]
    colors = [COLORS["red"], COLORS["green"], COLORS["purple"]]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.scatter(
        x,
        y,
        s=34,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.65,
        label="Training data",
        zorder=4,
    )
    ax.plot(
        grid,
        true_curve(grid),
        color="#222222",
        linewidth=2.0,
        alpha=0.85,
        label="True trend",
    )
    for tau, label, color in zip(taus, labels, colors):
        ax.plot(
            grid,
            local_linear_predict(x, y, grid, tau),
            color=color,
            linewidth=2.0,
            label=label,
        )
    ax.set_title("Effect of Bandwidth on LWR")
    ax.set_xlabel("Input x")
    ax.set_ylabel("Prediction")
    ax.set_ylim(-1.05, 1.25)
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left", ncol=2)
    return save(fig, "lecture03-tau-small-medium-large.png")


def figure_high_dimensional_distance_concentration() -> Path:
    """Plot nearest-to-farthest distance ratios as dimension increases."""
    dimensions = np.array([1, 2, 5, 10, 20, 50, 100, 200])
    trials = 70
    sample_count = 700
    means = []
    lows = []
    highs = []

    for dim in dimensions:
        ratios = []
        for _ in range(trials):
            points = RNG.normal(0.0, 1.0, size=(sample_count, dim))
            query = RNG.normal(0.0, 1.0, size=dim)
            distances = np.linalg.norm(points - query, axis=1)
            ratios.append(np.min(distances) / np.max(distances))
        ratios = np.asarray(ratios)
        means.append(float(np.mean(ratios)))
        lows.append(float(np.percentile(ratios, 10)))
        highs.append(float(np.percentile(ratios, 90)))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        dimensions,
        means,
        color=COLORS["blue"],
        linewidth=2.3,
        marker="o",
        label="Mean nearest / farthest distance",
    )
    ax.fill_between(
        dimensions,
        lows,
        highs,
        color=COLORS["blue"],
        alpha=0.18,
        label="10th to 90th percentile",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Distance Concentration in High Dimensions")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Nearest distance / farthest distance")
    ax.grid(True, which="both")
    ax.legend(frameon=True, loc="lower right")
    return save(fig, "lecture03-high-dimensional-distance-concentration.png")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable enough sigmoid for plotting."""
    return 1.0 / (1.0 + np.exp(-z))


def figure_sigmoid_curve() -> Path:
    """Plot sigmoid curve and mark g(0)=0.5."""
    z = np.linspace(-8.0, 8.0, 500)
    g = sigmoid(z)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(z, g, color=COLORS["blue"], linewidth=2.5, label=r"$g(z)$")
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.4)
    ax.axvline(0.0, color=COLORS["gray"], linestyle="--", linewidth=1.4)
    ax.scatter([0.0], [0.5], s=58, color=COLORS["red"], zorder=5)
    ax.annotate(
        r"$g(0)=0.5$",
        xy=(0.0, 0.5),
        xytext=(1.0, 0.38),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=11,
    )
    ax.set_title("Sigmoid Maps Logits to Probabilities")
    ax.set_xlabel("Logit z")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture03-sigmoid-curve.png")


def figure_linear_vs_logistic_output() -> Path:
    """Compare unbounded linear output with bounded logistic output."""
    x = np.linspace(-7.0, 7.0, 500)
    linear = 0.2 + 0.65 * x
    logistic = sigmoid(linear)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        x,
        linear,
        color=COLORS["orange"],
        linewidth=2.2,
        label=r"Linear score $\theta^Tx$",
    )
    ax.plot(
        x,
        logistic,
        color=COLORS["blue"],
        linewidth=2.4,
        label=r"Logistic output $g(\theta^Tx)$",
    )
    ax.axhspan(0.0, 1.0, color=COLORS["green"], alpha=0.08, label="Probability range")
    ax.axhline(0.0, color=COLORS["gray"], linewidth=1.1, linestyle=":")
    ax.axhline(1.0, color=COLORS["gray"], linewidth=1.1, linestyle=":")
    ax.set_title("Linear Output is Unbounded; Logistic Output is Bounded")
    ax.set_xlabel("One-dimensional input")
    ax.set_ylabel("Output")
    ax.set_ylim(-3.0, 4.2)
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture03-linear-vs-logistic-output.png")


def figure_logistic_decision_boundary() -> Path:
    """Plot two-dimensional binary data and a linear logistic boundary."""
    class_0 = RNG.normal(loc=[-1.4, -0.85], scale=[0.55, 0.45], size=(55, 2))
    class_1 = RNG.normal(loc=[1.25, 1.05], scale=[0.62, 0.50], size=(55, 2))
    theta_0, theta_1, theta_2 = -0.05, 1.0, -0.82
    x_line = np.linspace(-2.8, 2.9, 200)
    y_line = -(theta_0 + theta_1 * x_line) / theta_2

    xx, yy = np.meshgrid(np.linspace(-3.0, 3.0, 180), np.linspace(-2.5, 2.7, 180))
    logits = theta_0 + theta_1 * xx + theta_2 * yy
    probs = sigmoid(logits)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    contour = ax.contourf(
        xx,
        yy,
        probs,
        levels=np.linspace(0.0, 1.0, 11),
        cmap="RdBu_r",
        alpha=0.28,
    )
    fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, label=r"$P(y=1|x;\theta)$")
    ax.scatter(
        class_0[:, 0],
        class_0[:, 1],
        s=44,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.7,
        label="Class 0",
        zorder=4,
    )
    ax.scatter(
        class_1[:, 0],
        class_1[:, 1],
        s=44,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.7,
        label="Class 1",
        zorder=4,
    )
    ax.plot(
        x_line,
        y_line,
        color="#222222",
        linewidth=2.2,
        label=r"Boundary $\theta^Tx=0$",
        zorder=5,
    )
    ax.set_title("Logistic Regression Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.5, 2.7)
    ax.grid(True, alpha=0.38)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture03-logistic-decision-boundary.png")


def figure_newton_tangent_iteration() -> Path:
    """Show one Newton tangent step for root finding."""
    def f(x: np.ndarray) -> np.ndarray:
        return x**2 - 2.0

    def fp(x: float) -> float:
        return 2.0 * x

    x0 = 2.2
    y0 = float(f(np.array([x0]))[0])
    slope = fp(x0)
    x1 = x0 - y0 / slope
    root = np.sqrt(2.0)
    grid = np.linspace(0.5, 2.6, 400)
    tangent = y0 + slope * (grid - x0)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(grid, f(grid), color=COLORS["blue"], linewidth=2.4, label=r"$F(\theta)=\theta^2-2$")
    ax.plot(grid, tangent, color=COLORS["orange"], linewidth=2.0, linestyle="--", label="Tangent at current iterate")
    ax.axhline(0.0, color=COLORS["gray"], linewidth=1.2)
    ax.scatter([x0], [y0], s=60, color=COLORS["red"], zorder=5, label=r"Current $\theta_t$")
    ax.scatter([x1], [0.0], s=62, color=COLORS["green"], zorder=5, label=r"Next $\theta_{t+1}$")
    ax.scatter([root], [0.0], s=70, marker="*", color="#222222", zorder=6, label="True root")
    ax.annotate(
        "tangent root",
        xy=(x1, 0.0),
        xytext=(x1 + 0.18, -0.65),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10.5,
    )
    ax.set_title("Newton Step from a Tangent Approximation")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$F(\theta)$")
    ax.set_ylim(-1.2, 4.0)
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture03-newton-tangent-iteration.png")


def figure_newton_quadratic_convergence() -> Path:
    """Plot a synthetic quadratic-convergence error sequence."""
    errors = [0.42]
    for _ in range(6):
        errors.append(0.82 * errors[-1] ** 2)
    errors = np.asarray(errors)
    linear_errors = errors[0] * (0.32 ** np.arange(len(errors)))
    steps = np.arange(len(errors))

    fig, ax = plt.subplots(figsize=(7.7, 5.0))
    ax.semilogy(
        steps,
        errors,
        marker="o",
        linewidth=2.3,
        color=COLORS["blue"],
        label=r"Quadratic pattern $e_{t+1}\approx C e_t^2$",
    )
    ax.semilogy(
        steps,
        linear_errors,
        marker="s",
        linewidth=1.8,
        color=COLORS["orange"],
        linestyle="--",
        label="Reference linear convergence",
    )
    ax.set_title("Quadratic Convergence Shrinks Error Rapidly")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Absolute error")
    ax.grid(True, which="both")
    ax.legend(frameon=True, loc="upper right")
    return save(fig, "lecture03-newton-quadratic-convergence.png")


def main() -> None:
    """Generate every Lecture 3 figure and print relative paths."""
    configure_matplotlib()
    generators = [
        figure_underfitting_overfitting,
        figure_lwr_weights_query_point,
        figure_tau_small_medium_large,
        figure_high_dimensional_distance_concentration,
        figure_sigmoid_curve,
        figure_linear_vs_logistic_output,
        figure_logistic_decision_boundary,
        figure_newton_tangent_iteration,
        figure_newton_quadratic_convergence,
    ]
    paths = [generator() for generator in generators]
    print("Generated Lecture 3 figures:")
    for path in paths:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
