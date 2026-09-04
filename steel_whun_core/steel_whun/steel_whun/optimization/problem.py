"""Generic decision encoding and objective evaluation for WHUN models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AllocationVariable:
    """Continuous allocation from one configured source to one target."""

    source_id: str
    target_id: str
    upper_bound: float
    lower_bound: float = 0.0


@dataclass(frozen=True)
class ContinuousVariable:
    """One continuous operating decision with explicit bounds."""

    name: str
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class TopologySlot:
    """One topology position encoded by an integer choice index."""

    name: str
    choices: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("A topology slot must contain at least one choice.")


class DecisionCodec:
    """Encode and decode allocation, operating and topology decisions.

    Allocation and operating variables are continuous. Each topology slot is
    represented by a bounded real value and decoded by its integer part. A
    choice named ``Bypass``, ``None`` or ``Empty`` is omitted from the active
    route sequence.
    """

    _BYPASS = {"bypass", "none", "empty", "0"}

    def __init__(
        self,
        allocations: Sequence[AllocationVariable] = (),
        operating_variables: Sequence[ContinuousVariable] = (),
        topology_slots: Sequence[TopologySlot] = (),
    ) -> None:
        self.allocations = tuple(allocations)
        self.operating_variables = tuple(operating_variables)
        self.topology_slots = tuple(topology_slots)
        for variable in (*self.allocations, *self.operating_variables):
            if variable.upper_bound < variable.lower_bound:
                raise ValueError(f"Invalid bounds for {variable}.")

    @property
    def n_variables(self) -> int:
        return len(self.allocations) + len(self.operating_variables) + len(self.topology_slots)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = [item.lower_bound for item in self.allocations]
        upper = [item.upper_bound for item in self.allocations]
        lower.extend(item.lower_bound for item in self.operating_variables)
        upper.extend(item.upper_bound for item in self.operating_variables)
        lower.extend(0.0 for _ in self.topology_slots)
        upper.extend(float(len(item.choices)) - 1e-9 for item in self.topology_slots)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def decode(self, vector: Iterable[float]) -> dict[str, Any]:
        values = np.asarray(tuple(vector), dtype=float)
        if values.shape != (self.n_variables,):
            raise ValueError(f"Expected {self.n_variables} decision values, received {values.size}.")
        lower, upper = self.bounds()
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("The decision vector is outside the configured bounds.")

        cursor = 0
        allocation_matrix: dict[str, dict[str, float]] = {}
        for item in self.allocations:
            allocation_matrix.setdefault(item.source_id, {})[item.target_id] = float(values[cursor])
            cursor += 1

        operating_values: dict[str, float] = {}
        for item in self.operating_variables:
            operating_values[item.name] = float(values[cursor])
            cursor += 1

        topology_choices: dict[str, str] = {}
        sequence: list[str] = []
        for item in self.topology_slots:
            choice = item.choices[int(np.floor(values[cursor]))]
            topology_choices[item.name] = choice
            if choice.casefold() not in self._BYPASS:
                sequence.append(choice)
            cursor += 1

        return {
            "allocation_matrix": allocation_matrix,
            "operating_values": operating_values,
            "topology_choices": topology_choices,
            "sequence": sequence,
        }


@dataclass(frozen=True)
class OptimizationEvaluation:
    """Objectives and inequality constraints for one decision vector."""

    objectives: np.ndarray
    constraints: np.ndarray
    result: Mapping[str, Any]
    decoded_decision: Mapping[str, Any]

    @property
    def feasible(self) -> bool:
        return bool(np.all(self.constraints <= 0.0))


class WHUNProblem:
    """Connect a generic decision codec to a user-supplied WHUN evaluator."""

    def __init__(
        self,
        codec: DecisionCodec,
        evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        objective_function: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[float]],
        constraint_function: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[float]] | None = None,
        decision_builder: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        objective_count: int = 2,
        constraint_count: int = 0,
    ) -> None:
        self.codec = codec
        self.evaluator = evaluator
        self.objective_function = objective_function
        self.constraint_function = constraint_function or (lambda result, decoded: ())
        self.decision_builder = decision_builder or (lambda decoded: decoded)
        self.objective_count = int(objective_count)
        self.constraint_count = int(constraint_count)

    def evaluate_vector(self, vector: Iterable[float]) -> OptimizationEvaluation:
        decoded = self.codec.decode(vector)
        result = self.evaluator(self.decision_builder(decoded))
        objectives = np.asarray(self.objective_function(result, decoded), dtype=float)
        constraints = np.asarray(self.constraint_function(result, decoded), dtype=float)
        if objectives.shape != (self.objective_count,):
            raise ValueError("The objective function returned an unexpected number of values.")
        if constraints.shape != (self.constraint_count,):
            raise ValueError("The constraint function returned an unexpected number of values.")
        return OptimizationEvaluation(objectives, constraints, result, decoded)

    def as_pymoo_problem(self) -> Any:
        """Return an elementwise ``pymoo`` problem for NSGA-II or SPEA2."""

        try:
            from pymoo.core.problem import ElementwiseProblem
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError("Install the dependencies in requirements.txt to run optimization.") from exc

        owner = self
        lower, upper = self.codec.bounds()

        class _PymooAdapter(ElementwiseProblem):
            def __init__(self) -> None:
                super().__init__(
                    n_var=owner.codec.n_variables,
                    n_obj=owner.objective_count,
                    n_ieq_constr=owner.constraint_count,
                    xl=lower,
                    xu=upper,
                )

            def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
                evaluation = owner.evaluate_vector(x)
                out["F"] = evaluation.objectives
                out["G"] = evaluation.constraints

        return _PymooAdapter()


def nondominated_indices(objectives: np.ndarray) -> np.ndarray:
    """Return indices of mutually nondominated minimization objectives."""

    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("Objectives must be a two-dimensional array.")
    keep = np.ones(values.shape[0], dtype=bool)
    for index, point in enumerate(values):
        dominated = np.all(values <= point, axis=1) & np.any(values < point, axis=1)
        dominated[index] = False
        keep[index] = not np.any(dominated)
    return np.flatnonzero(keep)
