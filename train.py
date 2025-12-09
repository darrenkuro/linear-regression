#!/usr/bin/env python3

from src.utils import load_data, save_model

def normalize(values):
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [0.0 for _ in values], vmin, vmax
    norm = [(v - vmin) / (vmax - vmin) for v in values]
    return norm, vmin, vmax

def train(x, y, learning_rate=0.1, iterations=10000):
    m = len(x)
    theta0, theta1 = 0.0, 0.0

    for epoch in range(iterations):
        preds = [theta0 + theta1 * xi for xi in x]
        errors = [p - yi for p, yi in zip(preds, y)]

        tmp0 = learning_rate * (sum(errors) / m)
        tmp1 = learning_rate * (sum(e * xi for e, xi in zip(errors, x)) / m)

        theta0 -= tmp0
        theta1 -= tmp1

    return theta0, theta1

def main():
    x, y = load_data("data.csv")

    x_norm, xmin, xmax = normalize(x)
    y_norm, ymin, ymax = normalize(y)

    theta0, theta1 = train(x_norm, y_norm, learning_rate=0.1, iterations=10000)

    theta1_real = theta1 * (ymax - ymin) / (xmax - xmin)
    theta0_real = ymin + (ymax - ymin) * theta0 - theta1_real * xmin

    save_model(theta0_real, theta1_real)
    print(f"Theta0 = {theta0_real:.6f}")
    print(f"Theta1 = {theta1_real:.6f}")


if __name__ == "__main__":
    main()
