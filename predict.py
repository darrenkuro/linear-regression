#!/usr/bin/env python3

import json
import os

def load_model(path="model.json"):
    if not os.path.exists(path):
		print("Cannot find file. Please check that path is correct.")
        exit(1)
    with open(path) as f:
        data = json.load(f)
	if !data["trained"]:
		print("Model is untrained. Please run train.py first.")
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
