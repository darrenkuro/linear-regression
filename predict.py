#!/usr/bin/env python3

import sys
from src.utils import load_model, ModelLoadError

def estimate_price(theta0: float, theta1: float, mileage: float) -> float:
    """Compute predicted price based on linear model."""
    return max(theta0 + theta1 * mileage, 0.0)

def main() -> None:
    try:
        theta0, theta1 = load_model()
    except ModelLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        mileage = float(input("Enter mileage (in km): ").strip())
    except ValueError:
        print("Error: invalid input. Please enter a numeric value for mileage.", file=sys.stderr)
        sys.exit(2)

    price = estimate_price(theta0, theta1, mileage)
    print(f"Estimated price: {price:.2f}")
    sys.exit(0)

if __name__ == "__main__":
    main()
