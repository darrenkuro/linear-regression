#!/usr/bin/env python3

import sys
from src.utils import load_data, save_model, DataLoadError, ModelSaveError
from src.math import normalize
from src.plot import visualize

def train(
    x: list[float],
    y: list[float],
    epochs: int = 10_000,
    learning_rate: float = 0.1,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Train a simple linear regression model using gradient descent."""
    if not x or not y:
        raise ValueError("Training data cannot be empty.")
    if len(x) != len(y):
        raise ValueError("Mismatched input lengths for x and y.")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("Number of epochs must be a positive integer.")

    m = len(x)
    theta0, theta1 = 0.0, 0.0
    snapshots = []

    for i in range(epochs):
        preds = [theta0 + theta1 * xi for xi in x]
        errors = [p - yi for p, yi in zip(preds, y)]

        grad0 = sum(errors) / m
        grad1 = sum(e * xi for e, xi in zip(errors, x)) / m

        theta0 -= learning_rate * grad0
        theta1 -= learning_rate * grad1

        if i % max(1, epochs // 500) == 0:
            snapshots.append((theta0, theta1))

    return theta0, theta1, snapshots

def main(epochs: int = 10_000, learning_rate: float = 0.1) -> None:
    try:
        x, y = load_data("data.csv")
    except DataLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    x_norm, xmin, xmax = normalize(x)
    y_norm, ymin, ymax = normalize(y)

    try:
        theta0, theta1, snapshots = train(x_norm, y_norm, epochs, learning_rate)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    theta1_real = theta1 * (ymax - ymin) / (xmax - xmin)
    theta0_real = ymin + (ymax - ymin) * theta0 - theta1_real * xmin

    try:
        save_model(theta0_real, theta1_real)
    except ModelSaveError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Theta0 = {theta0_real:.6f}")
    print(f"Theta1 = {theta1_real:.6f}")
    visualize(x_norm, y_norm, snapshots)
    sys.exit(0)


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2])) # --epochs 10000 --lr 0.1
