#!/usr/bin/env python3

import sys
from src.utils import load_data, save_model, DataLoadError, ModelSaveError

def normalize(values: list[float]) -> tuple[list[float], float, float]:
    """Normalize a list of values to the range [0, 1]."""
    if not values:
        return [], 0.0, 0.0 # Empty

    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [0.0 for _ in values], vmin, vmax
    norm = [(v - vmin) / (vmax - vmin) for v in values]
    return norm, vmin, vmax

def train(
    x: list[float],
    y: list[float],
    learning_rate: float = 0.1,
    iterations: int = 10_000,
) -> tuple[float, float]:
    """Train a simple linear regression model using gradient descent."""
    if not x or not y:
        raise ValueError("Training data cannot be empty.")
    if len(x) != len(y):
        raise ValueError("Mismatched input lengths for x and y.")

    m = len(x)
    theta0, theta1 = 0.0, 0.0

    for _ in range(iterations):
        preds = [theta0 + theta1 * xi for xi in x]
        errors = [p - yi for p, yi in zip(preds, y)]

        grad0 = sum(errors) / m
        grad1 = sum(e * xi for e, xi in zip(errors, x)) / m

        theta0 -= learning_rate * grad0
        theta1 -= learning_rate * grad1

    return theta0, theta1

def main() -> None:
    # Load data
    try:
        x, y = load_data("data.csv")
    except DataLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Normalize data
    x_norm, xmin, xmax = normalize(x)
    y_norm, ymin, ymax = normalize(y)

    # Train data
    try:
        theta0, theta1 = train(x_norm, y_norm, learning_rate=0.1, iterations=10_000)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Scale back to original range
    theta1_real = theta1 * (ymax - ymin) / (xmax - xmin)
    theta0_real = ymin + (ymax - ymin) * theta0 - theta1_real * xmin

    # Save parameters
    try:
        save_model(theta0_real, theta1_real)
    except ModelSaveError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Theta0 = {theta0_real:.6f}")
    print(f"Theta1 = {theta1_real:.6f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
