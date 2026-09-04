def test_public_imports() -> None:
    import steel_whun
    from steel_whun.analysis import latin_hypercube, one_at_a_time
    from steel_whun.optimization import run_nsga2, run_spea2, select_balanced

    assert steel_whun.EnergyNetwork is not None
    assert callable(latin_hypercube)
    assert callable(one_at_a_time)
    assert callable(run_nsga2)
    assert callable(run_spea2)
    assert callable(select_balanced)


def test_representative_selection_interface() -> None:
    import numpy as np

    from steel_whun.optimization import select_balanced

    objectives = np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0]])
    assert select_balanced(objectives) == 1


def test_optimizer_budget_contracts_without_loading_backend() -> None:
    import pytest

    from steel_whun.optimization import run_nsga2, run_spea2

    with pytest.raises(ValueError, match="Population size"):
        run_nsga2(problem=None, population_size=1, generations=10, seed=1)
    with pytest.raises(ValueError, match="generations"):
        run_spea2(problem=None, population_size=10, generations=0, seed=1)


def test_generic_decision_codec_and_objective_evaluation() -> None:
    import numpy as np

    from steel_whun.optimization import (
        AllocationVariable,
        ContinuousVariable,
        DecisionCodec,
        TopologySlot,
        WHUNProblem,
        nondominated_indices,
    )

    codec = DecisionCodec(
        allocations=[AllocationVariable("heat", "WHPG", 100.0)],
        operating_variables=[ContinuousVariable("outlet_temperature_K", 400.0, 600.0)],
        topology_slots=[TopologySlot("slot_1", ("Bypass", "WHPG", "SCR"))],
    )
    decoded = codec.decode([40.0, 520.0, 1.0])
    assert decoded["allocation_matrix"] == {"heat": {"WHPG": 40.0}}
    assert decoded["sequence"] == ["WHPG"]

    problem = WHUNProblem(
        codec=codec,
        evaluator=lambda decision: {"energy": decision["allocation_matrix"]["heat"]["WHPG"]},
        objective_function=lambda result, decoded_decision: [result["energy"], -result["energy"]],
        constraint_function=lambda result, decoded_decision: [result["energy"] - 50.0],
        constraint_count=1,
    )
    evaluation = problem.evaluate_vector([40.0, 520.0, 1.0])
    assert evaluation.feasible
    assert np.array_equal(evaluation.objectives, np.array([40.0, -40.0]))
    assert np.array_equal(nondominated_indices(np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])), np.array([0, 1]))
