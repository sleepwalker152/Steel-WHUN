from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence


def one_at_a_time(
    evaluator: Callable[[Mapping[str, float]], Sequence[float]],
    reference: Mapping[str, float],
    relative_steps: Sequence[float],
) -> dict[str, list[tuple[float, Sequence[float]]]]:
    """Evaluate local one-at-a-time perturbations around a reference point."""

    output: dict[str, list[tuple[float, Sequence[float]]]] = {}
    for name, reference_value in reference.items():
        entries: list[tuple[float, Sequence[float]]] = []
        for step in relative_steps:
            trial = dict(reference)
            trial[name] = reference_value * (1.0 + float(step))
            entries.append((float(step), evaluator(trial)))
        output[name] = entries
    return output
