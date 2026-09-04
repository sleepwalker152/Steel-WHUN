from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


def latin_hypercube(
    bounds: Mapping[str, tuple[float, float]],
    sample_size: int,
    seed: int,
) -> list[dict[str, float]]:
    """Generate a reproducible Latin hypercube design over independent bounds."""

    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    names = list(bounds)
    rng = np.random.default_rng(seed)
    design = np.empty((sample_size, len(names)), dtype=float)
    for column, name in enumerate(names):
        lower, upper = bounds[name]
        if upper < lower:
            raise ValueError(f"Invalid bound for {name!r}.")
        unit = (np.arange(sample_size) + rng.random(sample_size)) / sample_size
        rng.shuffle(unit)
        design[:, column] = lower + unit * (upper - lower)
    return [dict(zip(names, row, strict=True)) for row in design]


def propagate_uncertainty(
    evaluator: Callable[[Mapping[str, float]], Sequence[float]],
    samples: Sequence[Mapping[str, float]],
) -> np.ndarray:
    """Evaluate a user function for every parameter sample."""

    return np.asarray([evaluator(sample) for sample in samples], dtype=float)
