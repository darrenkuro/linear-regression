import json
import csv
import sys

def load_model(path="model.json"):
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file '{path}' for model not found.", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: permission denied for '{path}'.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: model data corruption.", file=sys.stderr)
        sys.exit(1)

    if not data["trained"]:
        print("Model is untrained. Please run train.py first.", file=sys.stderr)
        sys.exit(1)

    return data["theta0"], data["theta1"]

def load_data(path="data.csv"):
    x, y = [], []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x.append(float(row["km"]))
                    y.append(float(row["price"]))
                except (ValueError, KeyError):
                    print(f"Error: malformed row in '{path}': {row}", file=sys.stderr)
                    sys.exit(1)
    except FileNotFoundError:
        print(f"Error: file '{path}' not found.", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: permission denied for '{path}'.", file=sys.stderr)
        sys.exit(1)
    except csv.Error as e:
        print(f"Error: invalid CSV format in '{path}': {e}", file=sys.stderr)
        sys.exit(1)

    if not x or not y:
        print(f"Error: file '{path}' is empty or missing data.", file=sys.stderr)
        sys.exit(1)

    return x, y
