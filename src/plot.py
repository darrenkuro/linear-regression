import sys

def plot_results(x: list[float], y: list[float], theta0: float, theta1: float) -> None:
    """Plot actual vs predicted data points."""
    try:
        import matplotlib.pyplot as plt # type: ignore
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plot.", file=sys.stderr)
        return

    preds = [theta0 + theta1 * xi for xi in x]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="blue", label="Actual data", alpha=0.7)
    plt.plot(x, preds, color="red", label="Regression line", linewidth=2)
    plt.title("Model Evaluation")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plot.png")

def visualize(x_norm, y_norm, snapshots) -> None:
    """Animate regression line evolution."""
    try:
        import matplotlib.pyplot as plt # type: ignore
        from matplotlib.animation import FuncAnimation, PillowWriter # type: ignore
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plot.", file=sys.stderr)
        return
    print(" Generating training animation...", end="")

    fig, ax = plt.subplots()
    ax.scatter(x_norm, y_norm, color="blue", label="data", alpha=0.7)
    ax.set_xlim(min(x_norm) - 0.1, max(x_norm) + 0.1)
    ax.set_ylim(min(y_norm) - 0.1, max(y_norm) + 0.1)
    ax.legend()

    line, = ax.plot([], [], color="red")
    text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def update(i):
        t0, t1 = snapshots[i]
        y_pred = [t0 + t1 * xi for xi in x_norm]
        line.set_data(x_norm, y_pred)
        text.set_text(f"Iter {i+1}/{len(snapshots)}")
        return line, text

    ani = FuncAnimation(fig, update, frames=len(snapshots), blit=True, interval=30)
    ani.save("training.gif", writer="pillow")
    print("  ")
