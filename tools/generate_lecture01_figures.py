"""Generate educational figures for CS229 Lecture 1 notes.

The figures are deterministic and intended for GitHub-rendered Markdown notes.
They use only numpy and matplotlib.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "figures"
RNG = np.random.default_rng(229)


COLORS = {
    "blue": "#2F6BBA",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#7B3F98",
    "gray": "#4D4D4D",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "grid.color": "#DDDDDD",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
        }
    )


def save(fig: plt.Figure, filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_1d_classification() -> Path:
    benign = np.array([0.9, 1.2, 1.45, 1.75, 2.05, 2.25, 2.45, 2.65])
    malignant = np.array([3.05, 3.25, 3.55, 3.85, 4.1, 4.35, 4.7, 5.05])
    y_benign = RNG.normal(0.04, 0.025, size=benign.shape)
    y_malignant = RNG.normal(-0.04, 0.025, size=malignant.shape)
    threshold = 2.85

    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    ax.axhline(0, color="#888888", linewidth=1.2)
    ax.scatter(
        benign,
        y_benign,
        s=80,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.8,
        label="Benign",
        zorder=3,
    )
    ax.scatter(
        malignant,
        y_malignant,
        s=85,
        marker="^",
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.8,
        label="Malignant",
        zorder=3,
    )
    ax.axvline(threshold, color="#222222", linestyle="--", linewidth=1.5)
    ax.annotate(
        "threshold",
        xy=(threshold, 0.18),
        xytext=(threshold + 0.25, 0.38),
        arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 1.1},
        ha="left",
    )
    ax.text(
        0.6,
        -0.35,
        'Classification asks: "Which class?"',
        fontsize=11,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#BBBBBB"},
    )
    ax.set_title("1D Classification Intuition: Tumor Size -> Benign or Malignant")
    ax.set_xlabel("Tumor size (cm)")
    ax.set_yticks([])
    ax.set_xlim(0.5, 5.4)
    ax.set_ylim(-0.48, 0.5)
    ax.legend(loc="upper right", frameon=True)
    ax.spines[["left", "right", "top"]].set_visible(False)
    return save(fig, "lecture01-1d-classification-intuition.png")


def figure_1d_regression() -> Path:
    house_size = np.linspace(650, 2450, 18)
    price = 35 + 0.23 * house_size + RNG.normal(0, 35, size=house_size.shape)
    slope, intercept = np.polyfit(house_size, price, deg=1)
    x_line = np.linspace(600, 2500, 100)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(
        house_size,
        price,
        s=58,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.8,
        label="Training examples",
        zorder=3,
    )
    ax.plot(x_line, y_line, color=COLORS["orange"], linewidth=2.2, label="Fitted line")
    ax.annotate(
        'Regression asks: "How much?"',
        xy=(2050, slope * 2050 + intercept),
        xytext=(1120, 575),
        arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 1.1},
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#BBBBBB"},
    )
    ax.set_title("1D Regression Intuition: House Size -> Price")
    ax.set_xlabel("House size (square feet)")
    ax.set_ylabel("House price ($k)")
    ax.set_xlim(560, 2540)
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True)
    return save(fig, "lecture01-1d-regression-intuition.png")


def figure_2d_classification() -> Path:
    not_spam = RNG.normal(loc=[1.3, 0.9], scale=[0.45, 0.35], size=(24, 2))
    spam = RNG.normal(loc=[3.8, 3.0], scale=[0.65, 0.55], size=(24, 2))
    not_spam = np.clip(not_spam, 0.1, None)
    spam = np.clip(spam, 0.1, None)

    x = np.linspace(0.0, 5.4, 100)
    boundary_y = (4.2 - x) / 0.85

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(
        not_spam[:, 0],
        not_spam[:, 1],
        s=58,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.8,
        label="Not spam",
        zorder=3,
    )
    ax.scatter(
        spam[:, 0],
        spam[:, 1],
        s=65,
        marker="^",
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.8,
        label="Spam",
        zorder=3,
    )
    ax.plot(x, boundary_y, color="#222222", linestyle="--", linewidth=1.8, label="Decision boundary")
    ax.annotate(
        "decision boundary",
        xy=(2.4, (4.2 - 2.4) / 0.85),
        xytext=(3.15, 1.15),
        arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 1.1},
        fontsize=10,
    )
    ax.set_title("2D Classification Intuition: Spam Detection in Feature Space")
    ax.set_xlabel("Suspicious keyword frequency")
    ax.set_ylabel("Number of suspicious links")
    ax.set_xlim(0, 5.4)
    ax.set_ylim(0, 5.0)
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True)
    return save(fig, "lecture01-2d-classification-intuition.png")


def figure_2d_regression() -> Path:
    floor_area = RNG.uniform(35, 130, size=55)
    distance = RNG.uniform(0.8, 18.0, size=55)
    price = 120 + 5.4 * floor_area - 13.5 * distance + RNG.normal(0, 45, size=55)

    fig, ax = plt.subplots(figsize=(7.7, 5.3))
    scatter = ax.scatter(
        floor_area,
        distance,
        c=price,
        cmap="viridis",
        s=70,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Apartment price ($k)")
    ax.set_title("2D Regression Intuition: Apartment Features -> Price")
    ax.set_xlabel("Floor area (square meters)")
    ax.set_ylabel("Distance to city center (km)")
    ax.grid(True)
    ax.text(
        38,
        17.0,
        "Color encodes the continuous target value",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#BBBBBB"},
    )
    return save(fig, "lecture01-2d-regression-intuition.png")


def figure_kernel_intuition() -> Path:
    n_per_cluster = 18
    centers = np.array(
        [
            [-1.25, -1.25],
            [1.25, 1.25],
            [-1.25, 1.25],
            [1.25, -1.25],
        ]
    )
    points = []
    labels = []
    for idx, center in enumerate(centers):
        cluster = RNG.normal(loc=center, scale=0.28, size=(n_per_cluster, 2))
        points.append(cluster)
        labels.extend([0 if idx < 2 else 1] * n_per_cluster)
    x = np.vstack(points)
    labels = np.array(labels)

    phi1 = x[:, 0] * x[:, 1]
    phi2 = x[:, 0] ** 2 + x[:, 1] ** 2

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    fig.suptitle("Feature mapping / kernel trick", fontsize=15, y=1.03)

    for label, color, marker, name in [
        (0, COLORS["blue"], "o", "Class A"),
        (1, COLORS["orange"], "^", "Class B"),
    ]:
        mask = labels == label
        axes[0].scatter(
            x[mask, 0],
            x[mask, 1],
            s=58,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            label=name,
            zorder=3,
        )
        axes[1].scatter(
            phi1[mask],
            phi2[mask],
            s=58,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            label=name,
            zorder=3,
        )

    axes[0].axhline(0, color="#BBBBBB", linewidth=1.0)
    axes[0].axvline(0, color="#BBBBBB", linewidth=1.0)
    axes[0].set_title("Original Space: Not Linearly Separable")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_xlim(-2.2, 2.2)
    axes[0].set_ylim(-2.2, 2.2)
    axes[0].grid(True)
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].axvline(0, color="#222222", linestyle="--", linewidth=1.8, label="Linear boundary")
    axes[1].annotate(
        "linear boundary",
        xy=(0, 2.7),
        xytext=(0.45, 3.2),
        arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 1.1},
        fontsize=10,
    )
    axes[1].set_title("Transformed Feature Space: Linearly Separable")
    axes[1].set_xlabel("phi1 = x1 * x2")
    axes[1].set_ylabel("phi2 = x1^2 + x2^2")
    axes[1].set_xlim(min(phi1) - 0.4, max(phi1) + 0.4)
    axes[1].set_ylim(min(phi2) - 0.2, max(phi2) + 0.5)
    axes[1].grid(True)
    axes[1].legend(loc="lower right", frameon=True)

    fig.text(
        0.5,
        -0.02,
        "Kernel methods can compute transformed-space inner products implicitly, without explicitly constructing every coordinate.",
        ha="center",
        fontsize=10,
        color="#222222",
    )
    fig.tight_layout()
    return save(fig, "lecture01-kernel-intuition.png")


def main() -> None:
    configure_matplotlib()
    generators = [
        figure_1d_classification,
        figure_1d_regression,
        figure_2d_classification,
        figure_2d_regression,
        figure_kernel_intuition,
    ]
    paths = [generator() for generator in generators]
    print("Generated Lecture 1 figures:")
    for path in paths:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
