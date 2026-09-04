from __future__ import annotations

from typing import Any


def _validate_budget(population_size: int, generations: int) -> None:
    if population_size <= 1:
        raise ValueError("Population size must exceed one.")
    if generations <= 0:
        raise ValueError("The number of generations must be positive.")


def _minimize(problem: Any, algorithm: Any, generations: int, seed: int, verbose: bool) -> Any:
    try:
        from pymoo.optimize import minimize
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise ImportError("Install the dependencies in requirements.txt to run optimization.") from exc
    return minimize(
        problem,
        algorithm,
        ("n_gen", int(generations)),
        seed=int(seed),
        verbose=verbose,
        save_history=False,
    )


def run_nsga2(
    problem: Any,
    population_size: int,
    generations: int,
    seed: int,
    verbose: bool = False,
    **algorithm_options: Any,
) -> Any:
    """Run NSGA-II with a user-defined budget and random seed."""

    _validate_budget(population_size, generations)
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise ImportError("Install the dependencies in requirements.txt to run optimization.") from exc
    algorithm = NSGA2(pop_size=int(population_size), **algorithm_options)
    return _minimize(problem, algorithm, generations, seed, verbose)


def run_spea2(
    problem: Any,
    population_size: int,
    generations: int,
    seed: int,
    verbose: bool = False,
    **algorithm_options: Any,
) -> Any:
    """Run SPEA2 with a user-defined budget and random seed."""

    _validate_budget(population_size, generations)
    try:
        from pymoo.algorithms.moo.spea2 import SPEA2
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise ImportError("Install the dependencies in requirements.txt to run optimization.") from exc
    algorithm = SPEA2(pop_size=int(population_size), **algorithm_options)
    return _minimize(problem, algorithm, generations, seed, verbose)
