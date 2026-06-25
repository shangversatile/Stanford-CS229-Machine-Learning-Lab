"""Generate deterministic educational figures for CS229 Lecture 2.

The script uses only NumPy and Matplotlib. Run it from any directory; output
paths are resolved relative to the repository root.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "figures"
RNG = np.random.default_rng(22902)

COLORS = {
    "blue": "#2F6BBA",
    "orange": "#D55E00",
    "green": "#00876C",
    "purple": "#7B3F98",
    "red": "#B23A48",
    "gray": "#555555",
    "light_gray": "#D8D8D8",
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
            "legend.fontsize": 9.5,
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


def quadratic(theta_0: np.ndarray, theta_1: np.ndarray) -> np.ndarray:
    """A positive-definite quadratic objective with elliptical contours."""
    return (
        0.5
        * (
            3.8 * theta_0**2
            + 1.8 * theta_0 * theta_1
            + 1.4 * theta_1**2
        )
    )


def quadratic_gradient(points: np.ndarray) -> np.ndarray:
    """Gradient of the quadratic objective for points shaped (..., 2)."""
    matrix = np.array([[3.8, 0.9], [0.9, 1.4]])
    return points @ matrix.T


def figure_linear_regression_fit() -> Path:
    """Plot house-size data, a fitted line, and selected residuals."""
    house_size = np.linspace(650, 2500, 22)
    true_price = 52.0 + 0.215 * house_size
    price = true_price + RNG.normal(0.0, 37.0, size=house_size.shape)
    slope, intercept = np.polyfit(house_size, price, deg=1)

    x_line = np.linspace(600, 2550, 200)
    y_line = intercept + slope * x_line
    fitted = intercept + slope * house_size
    residual_indices = np.array([2, 7, 12, 17, 20])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(
        house_size,
        price,
        s=58,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.8,
        label="Observed prices",
        zorder=3,
    )
    ax.plot(
        x_line,
        y_line,
        color=COLORS["orange"],
        linewidth=2.3,
        label=r"Fitted hypothesis $h_\theta(x)$",
        zorder=2,
    )

    for count, index in enumerate(residual_indices):
        ax.plot(
            [house_size[index], house_size[index]],
            [fitted[index], price[index]],
            color=COLORS["red"],
            linewidth=1.5,
            linestyle="--",
            label="Residual" if count == 0 else None,
            zorder=1,
        )

    ax.set_title("Linear Regression Fit: Prediction and Residuals")
    ax.set_xlabel("House size (square feet)")
    ax.set_ylabel("Price ($k)")
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    return save(fig, "lecture02-linear-regression-fit.png")


def figure_quadratic_loss_surface() -> Path:
    """Plot a two-parameter quadratic loss surface."""
    theta_0 = np.linspace(-3.0, 3.0, 120)
    theta_1 = np.linspace(-3.0, 3.0, 120)
    t0_grid, t1_grid = np.meshgrid(theta_0, theta_1)
    loss = quadratic(t0_grid, t1_grid)

    fig = plt.figure(figsize=(8.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        t0_grid,
        t1_grid,
        loss,
        cmap="viridis",
        alpha=0.92,
        linewidth=0,
        antialiased=True,
        rcount=70,
        ccount=70,
    )
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        s=50,
        color=COLORS["red"],
        label=r"Global minimum $\hat{\theta}$",
        depthshade=False,
    )
    fig.colorbar(surface, ax=ax, shrink=0.66, pad=0.08, label=r"$J(\theta)$")
    ax.set_title("Quadratic Loss Surface for Linear Regression")
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_zlabel(r"$J(\theta)$")
    ax.view_init(elev=27, azim=-56)
    ax.legend(loc="upper left", frameon=True)
    return save(fig, "lecture02-quadratic-loss-surface.png")


def figure_contour_gradient_perpendicular() -> Path:
    """Plot contours with gradient and negative-gradient arrows."""
    axis = np.linspace(-3.2, 3.2, 260)
    t0_grid, t1_grid = np.meshgrid(axis, axis)
    loss = quadratic(t0_grid, t1_grid)
    levels = [0.5, 1.2, 2.4, 4.2, 6.8, 10.0, 14.0]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    contours = ax.contour(
        t0_grid,
        t1_grid,
        loss,
        levels=levels,
        colors="#777777",
        linewidths=1.0,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    points = np.array(
        [
            [1.65, 0.15],
            [-1.45, 1.10],
            [0.55, -1.55],
        ]
    )
    gradients = quadratic_gradient(points)
    lengths = np.linalg.norm(gradients, axis=1, keepdims=True)
    unit_gradients = gradients / lengths

    ax.quiver(
        points[:, 0],
        points[:, 1],
        unit_gradients[:, 0],
        unit_gradients[:, 1],
        angles="xy",
        scale_units="xy",
        scale=0.75,
        width=0.008,
        color=COLORS["blue"],
        label=r"Gradient $\nabla J$",
        zorder=4,
    )
    ax.quiver(
        points[:, 0],
        points[:, 1],
        -unit_gradients[:, 0],
        -unit_gradients[:, 1],
        angles="xy",
        scale_units="xy",
        scale=0.75,
        width=0.008,
        color=COLORS["red"],
        label=r"Negative gradient $-\nabla J$",
        zorder=4,
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=28,
        color="#222222",
        zorder=5,
    )
    ax.scatter(
        [0.0],
        [0.0],
        s=45,
        marker="*",
        color=COLORS["green"],
        label="Minimum",
        zorder=5,
    )
    ax.set_title("Gradient Direction is Perpendicular to Contours")
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.45)
    ax.legend(loc="upper right", frameon=True)
    return save(fig, "lecture02-contour-gradient-perpendicular.png")


def surface_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Quadratic surface used for normal-projection geometry."""
    return 0.42 * x**2 + 0.85 * y**2 + 0.22 * x * y


def surface_gradient(x: float, y: float) -> tuple[float, float]:
    """Return partial derivatives of the surface function."""
    return 0.84 * x + 0.22 * y, 1.70 * y + 0.22 * x


def figure_surface_normal_projection() -> Path:
    """Show a surface normal and its projection onto the xy-plane."""
    axis = np.linspace(-2.2, 2.2, 100)
    x_grid, y_grid = np.meshgrid(axis, axis)
    z_grid = surface_function(x_grid, y_grid)

    x_0, y_0 = 1.05, 0.62
    z_0 = float(surface_function(x_0, y_0))
    f_x, f_y = surface_gradient(x_0, y_0)
    normal = np.array([f_x, f_y, -1.0])
    normal = normal / np.linalg.norm(normal)
    projected = np.array([f_x, f_y, 0.0])
    projected = projected / np.linalg.norm(projected)

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap="Blues",
        alpha=0.63,
        linewidth=0,
        rcount=55,
        ccount=55,
    )

    plane_x, plane_y = np.meshgrid(
        np.linspace(-2.0, 2.0, 8),
        np.linspace(-2.0, 2.0, 8),
    )
    plane_z = np.full_like(plane_x, z_0)
    ax.plot_surface(
        plane_x,
        plane_y,
        plane_z,
        color=COLORS["orange"],
        alpha=0.16,
        linewidth=0,
    )
    ax.contour(
        x_grid,
        y_grid,
        z_grid,
        levels=[z_0],
        colors=[COLORS["orange"]],
        linewidths=2.3,
    )

    ax.scatter(
        [x_0],
        [y_0],
        [z_0],
        color="#222222",
        s=35,
        depthshade=False,
        zorder=6,
    )
    ax.quiver(
        x_0,
        y_0,
        z_0,
        normal[0],
        normal[1],
        normal[2],
        length=1.25,
        normalize=False,
        color=COLORS["purple"],
        linewidth=2.2,
        arrow_length_ratio=0.14,
        label=r"Surface normal $\nabla F$",
    )
    ax.quiver(
        x_0,
        y_0,
        0.0,
        projected[0],
        projected[1],
        projected[2],
        length=1.35,
        normalize=False,
        color=COLORS["red"],
        linewidth=2.2,
        arrow_length_ratio=0.14,
        label=r"Projected gradient $\nabla f$",
    )
    ax.plot(
        [x_0, x_0],
        [y_0, y_0],
        [0.0, z_0],
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.5,
    )

    ax.set_title("Surface Normal Projection and Gradient Direction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z=f(x,y)$")
    ax.set_zlim(0.0, 4.5)
    ax.view_init(elev=29, azim=-58)
    ax.legend(loc="upper left", frameon=True)
    return save(fig, "lecture02-surface-normal-projection.png")


def batch_path(start: np.ndarray, steps: int = 20) -> np.ndarray:
    """Run batch gradient descent on the deterministic quadratic."""
    matrix = np.array([[3.8, 0.9], [0.9, 1.4]])
    theta = start.astype(float).copy()
    path = [theta.copy()]
    learning_rate = 0.22
    for _ in range(steps):
        theta = theta - learning_rate * (matrix @ theta)
        path.append(theta.copy())
    return np.asarray(path)


def sgd_path(start: np.ndarray, steps: int = 42) -> np.ndarray:
    """Simulate noisy unbiased stochastic gradients on the same objective."""
    matrix = np.array([[3.8, 0.9], [0.9, 1.4]])
    theta = start.astype(float).copy()
    path = [theta.copy()]
    for step in range(steps):
        learning_rate = 0.18 / (1.0 + 0.045 * step)
        noise_scale = 0.9 / np.sqrt(1.0 + 0.08 * step)
        stochastic_gradient = matrix @ theta + RNG.normal(
            0.0, noise_scale, size=2
        )
        theta = theta - learning_rate * stochastic_gradient
        path.append(theta.copy())
    return np.asarray(path)


def figure_batch_vs_sgd_paths() -> Path:
    """Compare smooth batch GD and noisy SGD trajectories."""
    axis = np.linspace(-3.1, 3.1, 260)
    t0_grid, t1_grid = np.meshgrid(axis, axis)
    loss = quadratic(t0_grid, t1_grid)
    start = np.array([-2.55, 2.65])
    batch = batch_path(start)
    stochastic = sgd_path(start)

    fig, ax = plt.subplots(figsize=(7.4, 6.1))
    ax.contour(
        t0_grid,
        t1_grid,
        loss,
        levels=[0.15, 0.4, 0.9, 1.8, 3.5, 6.5, 10.0, 15.0],
        colors="#A8A8A8",
        linewidths=0.9,
    )
    ax.plot(
        batch[:, 0],
        batch[:, 1],
        color=COLORS["blue"],
        linewidth=2.2,
        marker="o",
        markersize=3.4,
        label="Batch GD",
        zorder=4,
    )
    ax.plot(
        stochastic[:, 0],
        stochastic[:, 1],
        color=COLORS["orange"],
        linewidth=1.45,
        marker=".",
        markersize=4.0,
        alpha=0.9,
        label="SGD",
        zorder=3,
    )
    ax.scatter(
        [start[0]],
        [start[1]],
        color="#222222",
        s=42,
        label="Shared start",
        zorder=5,
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=95,
        color=COLORS["green"],
        label="Optimum",
        zorder=5,
    )
    ax.set_title("Batch Gradient Descent vs Stochastic Gradient Descent")
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-3.1, 3.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", frameon=True)
    return save(fig, "lecture02-batch-vs-sgd-paths.png")


def figure_normal_equation_projection() -> Path:
    """Draw a 2D schematic of orthogonal projection onto col(X)."""
    direction = np.array([1.0, 0.43])
    direction = direction / np.linalg.norm(direction)
    y_vector = np.array([3.0, 3.2])
    projection = direction * np.dot(y_vector, direction)
    residual = y_vector - projection

    fig, ax = plt.subplots(figsize=(7.5, 5.7))
    line_parameter = np.linspace(-0.4, 5.2, 100)
    line = line_parameter[:, None] * direction[None, :]
    ax.plot(
        line[:, 0],
        line[:, 1],
        color=COLORS["blue"],
        linewidth=2.4,
        label=r"Column space $\operatorname{col}(X)$",
    )
    ax.arrow(
        0.0,
        0.0,
        y_vector[0],
        y_vector[1],
        width=0.025,
        head_width=0.16,
        length_includes_head=True,
        color=COLORS["purple"],
        label=r"Observed vector $y$",
        zorder=4,
    )
    ax.arrow(
        0.0,
        0.0,
        projection[0],
        projection[1],
        width=0.022,
        head_width=0.15,
        length_includes_head=True,
        color=COLORS["green"],
        label=r"Projection $X\hat{\theta}$",
        zorder=5,
    )
    ax.arrow(
        projection[0],
        projection[1],
        residual[0],
        residual[1],
        width=0.018,
        head_width=0.13,
        length_includes_head=True,
        color=COLORS["red"],
        linestyle="--",
        label=r"Residual $y-X\hat{\theta}$",
        zorder=4,
    )

    perpendicular = direction * 0.22
    residual_unit = residual / np.linalg.norm(residual) * 0.22
    corner = projection - perpendicular
    right_angle = np.vstack(
        [
            corner,
            corner + residual_unit,
            corner + residual_unit + perpendicular,
        ]
    )
    ax.plot(
        right_angle[:, 0],
        right_angle[:, 1],
        color="#222222",
        linewidth=1.2,
    )
    ax.text(
        projection[0] + 0.14,
        projection[1] - 0.33,
        r"$X^T(y-X\hat{\theta})=0$",
        fontsize=11,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#BBBBBB"},
    )

    ax.set_title("Normal Equation as Orthogonal Projection")
    ax.set_xlabel("Observation-space coordinate 1")
    ax.set_ylabel("Observation-space coordinate 2")
    ax.set_xlim(-0.4, 4.9)
    ax.set_ylim(-0.3, 4.15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", frameon=True)
    return save(fig, "lecture02-normal-equation-projection.png")


def main() -> None:
    """Generate every Lecture 2 figure and print relative paths."""
    configure_matplotlib()
    generators = [
        figure_linear_regression_fit,
        figure_quadratic_loss_surface,
        figure_contour_gradient_perpendicular,
        figure_surface_normal_projection,
        figure_batch_vs_sgd_paths,
        figure_normal_equation_projection,
    ]
    paths = [generator() for generator in generators]
    print("Generated Lecture 2 figures:")
    for path in paths:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
