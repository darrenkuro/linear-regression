#!/usr/bin/env python3

from src.utils import load_model

def main():
    theta0, theta1 = load_model()
    try:
        mileage = float(input("Enter mileage (in km): "))
    except ValueError:
        print("Invalid input!")
        return
    price = theta0 + theta1 * mileage
    if price < 0:
        price = 0
    print(f"Estimated price: {price:.2f}")

if __name__ == "__main__":
    main()
