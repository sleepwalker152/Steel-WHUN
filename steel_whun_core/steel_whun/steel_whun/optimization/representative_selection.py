from __future__ import annotations

import numpy as np


def select_balanced(objectives: np.ndarray) -> int:
    """Return the point nearest the normalized ideal point."""

    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Objectives must be a nonempty two-dimensional array.")
    lower = values.min(axis=0)
    span = values.max(axis=0) - lower
    normalized = (values - lower) / np.where(span > 0.0, span, 1.0)
    return int(np.argmin(np.linalg.norm(normalized, axis=1)))
