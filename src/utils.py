import json
import csv

class DataLoadError(Exception): pass
class ModelLoadError(Exception): pass
class ModelSaveError(Exception): pass

def load_data(path: str = "data.csv") -> tuple[list[float], list[float]]:
    """Load 'km' and 'price' data from a CSV file. Raises DataLoadError if failed."""
    x, y = [], []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x.append(float(row["km"]))
                    y.append(float(row["price"]))
                except (ValueError, KeyError) as e:
                    raise DataLoadError(f"Malformed row in '{path}': {row}") from e
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        msg = getattr(e, "strerror", str(e))
        raise DataLoadError(f"Failed to read '{path}': {msg}") from e

    if not x or not y:
        raise DataLoadError(f"File '{path}' contains no valid rows.")

    return x, y

def save_model(theta0: float, theta1: float, path: str = "model.json") -> None:
    """Save model parameters to a JSON file. Raises ModelSaveError if failed."""
    data = {"theta0": theta0, "theta1": theta1, "trained": True}
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except OSError as e:
        raise ModelSaveError(f"Failed to write model to '{path}': {e.strerror}") from e

def load_model(path: str = "model.json") -> tuple[float, float]:
    """Load model parameters from a JSON file. Raises ModelLoadError if failed."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        msg = getattr(e, "strerror", str(e))
        raise ModelLoadError(f"Failed to load model from '{path}': {msg}") from e

    if not data.get("trained", False):
        raise ModelLoadError(f"Model file '{path}' is untrained. Run train.py first.")

    try:
        theta0 = float(data["theta0"])
        theta1 = float(data["theta1"])
    except (KeyError, ValueError, TypeError) as e:
        raise ModelLoadError(f"Model file '{path}' is malformed or missing parameters.") from e

    return theta0, theta1
