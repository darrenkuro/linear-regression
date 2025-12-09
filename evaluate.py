#!/usr/bin/env python3

from src.utils import load_model, load_data

# plot here too!!

def compute_precision(x, y, theta0, theta1):
    m = len(x) # Sample size
    preds = [theta0 + theta1 * xi for xi in x] # All predicated prices
    errors = [abs(p - yi) for p, yi in zip(preds, y)] # All actual prices
    mae = sum(errors) / m
    avg_y = sum(y) / m
    precision = 100 - (mae / avg_y * 100)
    return precision, mae

def main():
    x, y = load_data()
    theta0, theta1 = load_model()
    precision, mae = compute_precision(x, y, theta0, theta1)

    print(f"Mean absolute error: {mae:.2f}")
    print(f"Precision: {precision:.2f}%")

if __name__ == "__main__":
    main()
