"""Generate deterministic educational figures for CS229 Lecture 4.

The script uses only NumPy and Matplotlib. Run it from the repository root;
output paths are resolved relative to the repository root.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "figures"
RNG = np.random.default_rng(22904)

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


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Return sigmoid values for plotting."""
    return 1.0 / (1.0 + np.exp(-z))


def softmax(scores: np.ndarray) -> np.ndarray:
    """Return row-wise softmax probabilities."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def boundary_y(theta: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """Return y coordinates for theta[0] * x + theta[1] * y = 0."""
    return -(theta[0] / theta[1]) * x_values


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    label: str | None = None,
    mutation_scale: float = 13.0,
) -> None:
    """Draw an annotated arrow in axes coordinates."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=2.2,
        color=color,
        label=label,
        zorder=6,
    )
    ax.add_patch(arrow)


def figure_perceptron_vector_update() -> Path:
    """Show a Perceptron update rotating the boundary normal."""
    positives = RNG.normal(loc=[0.9, 1.0], scale=[0.26, 0.25], size=(18, 2))
    negatives = RNG.normal(loc=[0.9, -1.0], scale=[0.30, 0.28], size=(18, 2))
    mistaken = np.array([-0.75, -1.0])
    theta = np.array([0.75, -0.45])
    alpha = 0.75
    theta_new = theta + alpha * mistaken
    x_line = np.linspace(-1.8, 1.8, 200)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.scatter(
        positives[:, 0],
        positives[:, 1],
        s=48,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.7,
        label="Positive samples",
        zorder=4,
    )
    ax.scatter(
        negatives[:, 0],
        negatives[:, 1],
        s=48,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.7,
        label="Negative samples",
        zorder=4,
    )
    ax.scatter(
        [mistaken[0]],
        [mistaken[1]],
        s=110,
        marker="*",
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.8,
        label="Misclassified positive",
        zorder=7,
    )
    ax.plot(
        x_line,
        boundary_y(theta, x_line),
        color=COLORS["gray"],
        linewidth=2.0,
        linestyle="--",
        label="Current boundary",
    )
    ax.plot(
        x_line,
        boundary_y(theta_new, x_line),
        color=COLORS["green"],
        linewidth=2.3,
        label="Updated boundary",
    )
    add_arrow(ax, (0.0, 0.0), tuple(theta), COLORS["gray"], r"Current normal $\theta$")
    add_arrow(
        ax,
        (0.0, 0.0),
        tuple(theta_new),
        COLORS["green"],
        r"Updated normal $\theta+\alpha yx$",
    )
    add_arrow(
        ax,
        tuple(theta),
        tuple(theta_new),
        COLORS["red"],
        "Mistake correction",
        mutation_scale=11.0,
    )
    ax.axhline(0.0, color="#BBBBBB", linewidth=0.9)
    ax.axvline(0.0, color="#BBBBBB", linewidth=0.9)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Perceptron Update Rotates the Boundary Normal")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture04-perceptron-vector-update.png")


def figure_perceptron_vs_logistic_response() -> Path:
    """Compare hard Perceptron response with smooth logistic response."""
    z = np.linspace(-7.0, 7.0, 500)
    step = (z >= 0.0).astype(float)
    sig = sigmoid(z)
    grad_weight = sig * (1.0 - sig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(z, sig, color=COLORS["blue"], linewidth=2.6, label="Sigmoid probability")
    ax.step(z, step, where="post", color=COLORS["orange"], linewidth=2.0, label="Hard step decision")
    ax.fill_between(
        z,
        0.0,
        grad_weight / grad_weight.max(),
        color=COLORS["green"],
        alpha=0.15,
        label="Smooth gradient region",
    )
    ax.axvline(0.0, color=COLORS["gray"], linestyle="--", linewidth=1.5, label="Decision threshold")
    ax.axhline(0.5, color=COLORS["gray"], linestyle=":", linewidth=1.2)
    ax.annotate(
        "probability changes continuously",
        xy=(1.25, sigmoid(np.array([1.25]))[0]),
        xytext=(2.2, 0.60),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10.5,
    )
    ax.annotate(
        "mistake trigger is discrete",
        xy=(-0.05, 0.02),
        xytext=(-5.7, 0.22),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10.5,
    )
    ax.set_title("Same Linear Score, Different Response Semantics")
    ax.set_xlabel(r"Linear score $\theta^Tx$")
    ax.set_ylabel("Output")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture04-perceptron-vs-logistic-response.png")


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    """Draw a labeled rounded box."""
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


def figure_exponential_family_anatomy() -> Path:
    """Draw the main components of the exponential-family form."""
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.set_axis_off()
    positions = {
        "eta": (0.18, 0.72),
        "stat": (0.18, 0.30),
        "base": (0.50, 0.18),
        "logpart": (0.50, 0.84),
        "density": (0.82, 0.52),
    }
    draw_box(ax, positions["eta"], "natural parameter\neta", COLORS["blue"])
    draw_box(ax, positions["stat"], "sufficient statistic\nT(y)", COLORS["orange"])
    draw_box(ax, positions["base"], "base measure\nb(y)", COLORS["purple"])
    draw_box(ax, positions["logpart"], "log-partition\na(eta)", COLORS["green"])
    draw_box(ax, positions["density"], "normalized model\np(y; eta)", COLORS["red"])

    for source in ["eta", "stat", "base", "logpart"]:
        draw_connector(ax, positions[source], positions["density"])

    ax.text(
        0.50,
        0.53,
        r"$p(y;\eta)=b(y)\exp(\eta^TT(y)-a(\eta))$",
        ha="center",
        va="center",
        fontsize=13,
        color=COLORS["black"],
    )
    ax.text(
        0.50,
        0.40,
        "normalization, moments, and convex likelihood geometry",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["gray"],
    )
    ax.set_title("Anatomy of the Exponential Family")
    return save(fig, "lecture04-exponential-family-anatomy.png")


def figure_log_partition_moments() -> Path:
    """Show a Bernoulli log-partition and its first two derivatives."""
    eta = np.linspace(-6.0, 6.0, 500)
    a = np.log1p(np.exp(eta))
    mean = sigmoid(eta)
    var = mean * (1.0 - mean)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), sharex=True)
    series = [
        (a, r"$a(\eta)=\log(1+e^\eta)$", COLORS["blue"], "log partition"),
        (mean, r"$a'(\eta)=E[Y]$", COLORS["green"], "mean"),
        (var, r"$a''(\eta)=Var(Y)$", COLORS["red"], "variance"),
    ]
    for ax, (values, label, color, title) in zip(axes, series):
        ax.plot(eta, values, color=color, linewidth=2.4, label=label)
        ax.axvline(0.0, color=COLORS["gray"], linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel(r"Natural parameter $\eta$")
        ax.grid(True)
        ax.legend(frameon=True, loc="best")
    axes[0].set_ylabel("Value")
    fig.suptitle("Log-Partition Derivatives Generate Moments", y=1.03, fontsize=13, fontweight="semibold")
    return save(fig, "lecture04-log-partition-moments.png")


def figure_response_distribution_map() -> Path:
    """Map output support and semantics to candidate distributions."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_axis_off()
    rows = [
        ("real-valued\ncontinuous", "Gaussian"),
        ("binary\nevent", "Bernoulli"),
        ("multiclass\nlabel", "Categorical\nSoftmax"),
        ("event\ncount", "Poisson"),
        ("positive\ncontinuous", "Gamma\nExponential"),
        ("scalar\nprobability", "Beta"),
        ("probability\nvector", "Dirichlet"),
    ]
    y_positions = np.linspace(0.86, 0.14, len(rows))
    for y_pos, (left, right) in zip(y_positions, rows):
        draw_box(ax, (0.28, float(y_pos)), left, COLORS["blue"])
        draw_box(ax, (0.72, float(y_pos)), right, COLORS["green"])
        draw_connector(ax, (0.38, float(y_pos)), (0.62, float(y_pos)))
    ax.text(0.28, 0.96, "Response semantics and support", ha="center", fontsize=12, fontweight="semibold")
    ax.text(0.72, 0.96, "Candidate distribution family", ha="center", fontsize=12, fontweight="semibold")
    ax.text(
        0.50,
        0.05,
        "Support is necessary but not sufficient: variance, tails, zero mass, dependence, and mechanism matter.",
        ha="center",
        fontsize=10.5,
        color=COLORS["gray"],
    )
    ax.set_title("From Output Space to Distribution Choice")
    return save(fig, "lecture04-response-distribution-map.png")



def figure_why_exponential_family_emerges() -> Path:
    """Show why exponential-family GLMs emerge from modeling constraints."""
    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    ax.set_axis_off()

    steps = [
        ("ordinary linear\nmodel limits", COLORS["red"]),
        ("response\nsemantics", COLORS["blue"]),
        ("distribution\nchoice", COLORS["green"]),
        ("exponential-family\nform", COLORS["purple"]),
        ("sufficient statistic\nT(y)", COLORS["orange"]),
        ("log-partition\na(eta)", COLORS["green"]),
        ("natural parameter\neta", COLORS["blue"]),
        (r"eta = theta^T x", COLORS["orange"]),
        ("response mean\nh = grad a", COLORS["purple"]),
        ("likelihood / NLL\noptimization", COLORS["red"]),
    ]
    positions = [
        (0.10, 0.72),
        (0.30, 0.72),
        (0.50, 0.72),
        (0.70, 0.72),
        (0.90, 0.72),
        (0.90, 0.32),
        (0.70, 0.32),
        (0.50, 0.32),
        (0.30, 0.32),
        (0.10, 0.32),
    ]

    for (label, color), position in zip(steps, positions):
        draw_box(ax, position, label, color)

    for start, end in zip(positions[:-1], positions[1:]):
        draw_connector(ax, start, end)

    ax.text(
        0.50,
        0.92,
        "Why the exponential-family GLM form emerges",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="semibold",
        color=COLORS["black"],
    )
    ax.text(
        0.70,
        0.56,
        r"$p(y;\eta)=b(y)\exp(\eta^TT(y)-a(\eta))$",
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["black"],
    )
    ax.text(
        0.50,
        0.13,
        r"normalization + sufficiency + max entropy -> $h_\theta(x)=\nabla a(\theta^Tx)$ and convex-friendly NLL",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["gray"],
    )
    ax.text(
        0.50,
        0.03,
        "The response function is derived from the distribution, not selected as a generic activation.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["gray"],
    )
    return save(fig, "lecture04-why-exponential-family-emerges.png")

def figure_glm_construction_pipeline() -> Path:
    """Show forward probability modeling and inverse likelihood training for a GLM."""
    fig, ax = plt.subplots(figsize=(13.4, 6.8))
    ax.set_axis_off()

    positions = {
        "global": (0.12, 0.78),
        "x": (0.12, 0.56),
        "eta": (0.31, 0.67),
        "psi": (0.48, 0.67),
        "dist": (0.66, 0.67),
        "Y": (0.82, 0.67),
        "y": (0.82, 0.42),
        "data": (0.58, 0.23),
        "like": (0.40, 0.23),
        "mle": (0.24, 0.23),
        "update": (0.12, 0.36),
    }

    boxes = {
        "global": ("global theta\ntrainable parameter", COLORS["orange"]),
        "x": ("local x_i\ninput vector", COLORS["blue"]),
        "eta": ("local eta_i\nnatural parameter", COLORS["purple"]),
        "psi": ("local psi_i\nordinary parameter", COLORS["yellow"]),
        "dist": ("p(Y_i | x_i; theta)\nconditional distribution", COLORS["green"]),
        "Y": ("random Y_i\nbefore sampling", COLORS["blue"]),
        "y": ("observed y_i\nrealization", COLORS["red"]),
        "data": ("observed data\n(X, y)", COLORS["red"]),
        "like": ("likelihood\nfunction of theta", COLORS["purple"]),
        "mle": ("MLE\ntheta_hat", COLORS["green"]),
        "update": ("theta update\nglobal sharing", COLORS["orange"]),
    }

    for key, (label, color) in boxes.items():
        draw_box(ax, positions[key], label, color)

    forward_edges = [
        ("global", "eta"),
        ("x", "eta"),
        ("eta", "psi"),
        ("psi", "dist"),
        ("dist", "Y"),
        ("Y", "y"),
    ]
    training_edges = [
        ("y", "data"),
        ("data", "like"),
        ("like", "mle"),
        ("mle", "update"),
        ("update", "global"),
    ]
    for start_key, end_key in forward_edges:
        draw_connector(ax, positions[start_key], positions[end_key])
    for start_key, end_key in training_edges:
        draw_connector(ax, positions[start_key], positions[end_key])

    ax.text(
        0.52,
        0.91,
        "Probability direction: global mechanism plus local input defines a local distribution",
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="semibold",
        color=COLORS["black"],
    )
    ax.text(
        0.40,
        0.07,
        "Likelihood direction: observed data are fixed; theta varies during MLE",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="semibold",
        color=COLORS["black"],
    )
    ax.text(
        0.22,
        0.47,
        r"canonical: $\eta_i=x_i^T\theta$",
        ha="center",
        va="center",
        fontsize=11.8,
        color=COLORS["black"],
    )
    ax.text(
        0.50,
        0.52,
        r"$\psi_i=q^{-1}(\eta_i)$",
        ha="center",
        va="center",
        fontsize=11.8,
        color=COLORS["black"],
    )
    ax.text(
        0.72,
        0.53,
        "sampling happens\nafter the distribution is set",
        ha="center",
        va="center",
        fontsize=10.4,
        color=COLORS["gray"],
    )
    ax.text(
        0.80,
        0.30,
        "y_i is not changed by parameters;\nit is evaluated by likelihood",
        ha="center",
        va="center",
        fontsize=10.2,
        color=COLORS["gray"],
    )
    ax.set_title("GLM Pipeline: Forward Sampling and Inverse Likelihood Learning")
    return save(fig, "lecture04-glm-construction-pipeline.png")


def figure_gaussian_bernoulli_poisson_response() -> Path:
    """Plot identity, sigmoid, and exponential response functions."""
    eta = np.linspace(-4.0, 4.0, 500)
    responses = [
        (eta, "Gaussian identity", COLORS["blue"]),
        (sigmoid(eta), "Bernoulli sigmoid", COLORS["green"]),
        (np.exp(eta), "Poisson exponential", COLORS["red"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharex=True)
    for ax, (values, title, color) in zip(axes, responses):
        ax.plot(eta, values, color=color, linewidth=2.5)
        ax.axvline(0.0, color=COLORS["gray"], linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel(r"Linear predictor $\eta$")
        ax.grid(True)
    axes[0].set_ylabel("Mean response")
    axes[0].set_ylim(-4.2, 4.2)
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylim(-0.2, 18.0)
    fig.suptitle("Canonical GLM Response Functions", y=1.03, fontsize=13, fontweight="semibold")
    return save(fig, "lecture04-gaussian-bernoulli-poisson-response.png")


def figure_softmax_coupled_probabilities() -> Path:
    """Show all softmax probabilities changing as one score varies."""
    s1 = np.linspace(-5.0, 5.0, 500)
    s2 = np.full_like(s1, 0.8)
    s3 = np.full_like(s1, -0.6)
    probs = softmax(np.column_stack([s1, s2, s3]))

    fig, ax = plt.subplots(figsize=(8.1, 5.0))
    labels = ["class 1", "class 2", "class 3"]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for index, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(s1, probs[:, index], linewidth=2.4, color=color, label=label)
    ax.axvline(0.8, color=COLORS["gray"], linestyle="--", linewidth=1.1, label="score 2 fixed at 0.8")
    ax.annotate(
        "one score changes\nall probabilities",
        xy=(1.5, probs[np.searchsorted(s1, 1.5), 0]),
        xytext=(-4.6, 0.72),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10.5,
    )
    ax.set_title("Softmax Probabilities are Jointly Normalized")
    ax.set_xlabel("Class 1 score")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True)
    ax.legend(frameon=True, loc="center right")
    return save(fig, "lecture04-softmax-coupled-probabilities.png")


def simplex_to_xy(p: np.ndarray) -> np.ndarray:
    """Convert 3-class probability vectors to 2D simplex coordinates."""
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
        ]
    )
    return p @ vertices


def figure_softmax_simplex() -> Path:
    """Show softmax outputs constrained to the probability simplex."""
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
        ]
    )
    samples = RNG.dirichlet(alpha=[1.8, 1.8, 1.8], size=130)
    xy = simplex_to_xy(samples)
    t = np.linspace(-3.2, 3.2, 140)
    scores = np.column_stack([t, 0.7 * np.sin(t), -0.4 * t])
    path = simplex_to_xy(softmax(scores))

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    triangle = Polygon(vertices, closed=True, fill=False, edgecolor=COLORS["black"], linewidth=2.0)
    ax.add_patch(triangle)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=20,
        color=COLORS["blue"],
        alpha=0.42,
        edgecolor="none",
        label="valid probability vectors",
    )
    ax.plot(path[:, 0], path[:, 1], color=COLORS["red"], linewidth=2.4, label="softmax score path")
    ax.scatter(path[0, 0], path[0, 1], color=COLORS["green"], s=58, zorder=5, label="path start")
    ax.scatter(path[-1, 0], path[-1, 1], color=COLORS["orange"], s=58, zorder=5, label="path end")
    labels = [r"$p_1=1$", r"$p_2=1$", r"$p_3=1$"]
    offsets = [(-0.06, -0.06), (0.04, -0.06), (-0.03, 0.04)]
    for vertex, label, offset in zip(vertices, labels, offsets):
        ax.text(vertex[0] + offset[0], vertex[1] + offset[1], label, fontsize=11)
    ax.text(0.5, -0.12, r"$p_1+p_2+p_3=1$", ha="center", fontsize=11.5, color=COLORS["gray"])
    ax.set_title("Softmax Outputs Live on the Simplex")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.16, 0.95)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(frameon=True, loc="upper center")
    return save(fig, "lecture04-softmax-simplex.png")


def figure_newton_curvature_bridge() -> Path:
    """Compare gradient descent and Newton curvature-corrected step."""
    def f(x: np.ndarray) -> np.ndarray:
        return 0.10 * (x - 1.0) ** 4 + 0.72 * (x - 1.0) ** 2 + 0.25

    def fp(x: float) -> float:
        return 0.40 * (x - 1.0) ** 3 + 1.44 * (x - 1.0)

    def fpp(x: float) -> float:
        return 1.20 * (x - 1.0) ** 2 + 1.44

    x0 = -1.35
    alpha = 0.28
    x_gd = x0 - alpha * fp(x0)
    x_newton = x0 - fp(x0) / fpp(x0)
    grid = np.linspace(-2.1, 2.8, 500)
    tangent = f(np.array([x0]))[0] + fp(x0) * (grid - x0)
    quadratic = f(np.array([x0]))[0] + fp(x0) * (grid - x0) + 0.5 * fpp(x0) * (grid - x0) ** 2

    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    ax.plot(grid, f(grid), color=COLORS["blue"], linewidth=2.5, label="Objective")
    ax.plot(grid, tangent, color=COLORS["orange"], linewidth=1.8, linestyle="--", label="First-order tangent")
    ax.plot(grid, quadratic, color=COLORS["green"], linewidth=2.0, linestyle="-.", label="Local quadratic model")
    ax.scatter([x0], [f(np.array([x0]))[0]], s=68, color=COLORS["red"], zorder=6, label=r"Current $\theta_t$")
    ax.scatter([x_gd], [f(np.array([x_gd]))[0]], s=64, color=COLORS["orange"], zorder=6, label="Gradient step")
    ax.scatter([x_newton], [f(np.array([x_newton]))[0]], s=64, color=COLORS["green"], zorder=6, label="Newton step")
    ax.annotate(
        "uses slope only",
        xy=(x_gd, f(np.array([x_gd]))[0]),
        xytext=(-0.95, 2.2),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10.5,
    )
    ax.annotate(
        "uses slope and curvature",
        xy=(x_newton, f(np.array([x_newton]))[0]),
        xytext=(0.45, 1.45),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10.5,
    )
    ax.set_ylim(0.0, 4.8)
    ax.set_title("Newton Method as the Curvature Bridge")
    ax.set_xlabel(r"Parameter coordinate $\theta$")
    ax.set_ylabel(r"Objective $J(\theta)$")
    ax.grid(True)
    ax.legend(frameon=True, loc="upper right")
    return save(fig, "lecture04-newton-curvature-bridge.png")


def main() -> None:
    """Generate every Lecture 4 figure and print relative paths."""
    configure_matplotlib()
    generators = [
        figure_perceptron_vector_update,
        figure_perceptron_vs_logistic_response,
        figure_exponential_family_anatomy,
        figure_why_exponential_family_emerges,
        figure_log_partition_moments,
        figure_response_distribution_map,
        figure_glm_construction_pipeline,
        figure_gaussian_bernoulli_poisson_response,
        figure_softmax_coupled_probabilities,
        figure_softmax_simplex,
        figure_newton_curvature_bridge,
    ]
    paths = [generator() for generator in generators]
    print("Generated Lecture 4 figures:")
    for path in paths:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
