def normalize(values: list[float]) -> tuple[list[float], float, float]:
    """Normalize a list of values to the range [0, 1]."""
    if not values:
        return [], 0.0, 0.0 # Empty

    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [0.0 for _ in values], vmin, vmax
    norm = [(v - vmin) / (vmax - vmin) for v in values]
    return norm, vmin, vmax
