#!/usr/bin/env python3

import sys

from src.utils import load_data, load_model, DataLoadError, ModelLoadError
from src.plot import plot_results

def evaluate(x: list[float], y: list[float], theta0: float, theta1: float) -> tuple[float, float]:
    """Compute precision and mean absolute error (MAE) for model predictions."""
    if not x or not y:
        raise ValueError("Evaluation data cannot be empty.")
    if len(x) != len(y):
        raise ValueError("Mismatched input lengths for x and y.")

    m = len(x)
    preds = [theta0 + theta1 * xi for xi in x]
    errors = [abs(p - yi) for p, yi in zip(preds, y)]

    mae = sum(errors) / m
    avg_y = sum(y) / m if m else 0.0

    if avg_y == 0.0:
        precision = 0.0
    else:
        precision = max(0.0, 100 - (mae / avg_y * 100))

    return precision, mae


def main() -> None:
    try:
        x, y = load_data("data.csv")
    except DataLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        theta0, theta1 = load_model("model.json")
    except ModelLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        precision, mae = evaluate(x, y, theta0, theta1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Precision: {precision:.2f}%")

    plot_results(x, y, theta0, theta1)
    sys.exit(0)

if __name__ == "__main__":
    main()
