#!/usr/bin/env python3

import json
import os
import sys

def load_model(path="model.json"):
    if not os.path.exists(path):
        print("Cannot find model file.", file=sys.stderr)
        exit(1)
    with open(path) as f: # what happens if no read access
        data = json.load(f)
    if not data["trained"]:
        print("Model is untrained. Please run train.py first.", file=sys.stderr)
        exit(1)
    return data["theta0"], data["theta1"]

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
