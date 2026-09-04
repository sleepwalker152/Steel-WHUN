from __future__ import annotations

import pytest

from steel_whun import (
    EmissionChecker,
    EnergyNetwork,
    EnergySource,
    FlueGasState,
    ProcessRoute,
    SystemEvaluator,
    TransmissionEdge,
)
from steel_whun.modules import CCPP


def _build_test_system() -> tuple[SystemEvaluator, FlueGasState]:
    network = EnergyNetwork()
    network.add_source(
        EnergySource(
            source_id="COG",
            carrier="fuel_gas",
            temperature=298.15,
            pressure=101325.0,
            max_energy_rate=6000.0,
        )
    )
    network.add_edge(
        TransmissionEdge(
            source_id="COG",
            target_id="CCPP",
            exergy_efficiency=0.80,
            capacity=6000.0,
        )
    )
    ccpp = CCPP(power_design_kW=3000.0, eta_design=0.50)
    evaluator = SystemEvaluator(
        energy_network=network,
        process_route=ProcessRoute([ccpp], name="test_route_pool"),
        emission_checker=EmissionChecker({}),
    )
    inlet = FlueGasState(
        mass_flow=1.0,
        temperature=373.15,
        pressure=101325.0,
    )
    return evaluator, inlet


def _evaluate_test_system(allocation_matrix: dict, operation_map: dict):
    evaluator, inlet = _build_test_system()
    return evaluator.evaluate(
        {
            "flue_gas_initial": inlet,
            "allocation_matrix": allocation_matrix,
            "operation_map": operation_map,
            "material_map": {},
        }
    )


def _ccpp_backend_result(system_result):
    return next(
        result
        for result in system_result["backend_results"]
        if result.module_name == "CCPP_backend"
    )


def test_only_cog_allocation_does_not_create_bfg_or_ldg_fuel() -> None:
    result = _evaluate_test_system(
        allocation_matrix={"COG": {"CCPP": 3000.0}},
        operation_map={
            "backend_assets": ["CCPP"],
            "CCPP": {
            "BFG_flow_m3_s": 25.0,
            "LDG_flow_m3_s": 5.0,
            },
        },
    )
    ccpp = _ccpp_backend_result(result)

    assert ccpp.material_consumption["BFG_consumed_m3_h"] == pytest.approx(0.0)
    assert ccpp.material_consumption["COG_consumed_m3_h"] > 0.0
    assert ccpp.material_consumption["LDG_consumed_m3_h"] == pytest.approx(0.0)


def test_zero_gas_allocation_produces_zero_ccpp_output() -> None:
    result = _evaluate_test_system(
        allocation_matrix={"COG": {"CCPP": 0.0}},
        operation_map={"backend_assets": ["CCPP"]},
    )
    ccpp = _ccpp_backend_result(result)

    assert ccpp.material_consumption["BFG_consumed_m3_h"] == pytest.approx(0.0)
    assert ccpp.material_consumption["COG_consumed_m3_h"] == pytest.approx(0.0)
    assert ccpp.material_consumption["LDG_consumed_m3_h"] == pytest.approx(0.0)
    assert ccpp.energy_consumption["gross_generated_kW"] == pytest.approx(0.0)
    assert ccpp.energy_consumption["electricity_output_kW"] == pytest.approx(0.0)


def test_ccpp_missing_flow_keys_default_to_zero() -> None:
    _, inlet = _build_test_system()
    ccpp = CCPP(name="CCPP")

    result = ccpp.evaluate(inlet, operation_params={})

    assert all(value == pytest.approx(0.0) for value in result.material_consumption.values())
    assert result.energy_consumption["gross_generated_kW"] == pytest.approx(0.0)
    assert result.energy_consumption["electricity_output_kW"] == pytest.approx(0.0)


def test_ccpp_uses_network_delivered_exergy_without_reconstruction() -> None:
    result = _evaluate_test_system(
        allocation_matrix={"COG": {"CCPP": 3000.0}},
        operation_map={"backend_assets": ["CCPP"]},
    )
    delivered_stream = result["energy_map"]["CCPP"][0]
    ccpp = _ccpp_backend_result(result)

    assert delivered_stream.exergy_rate == pytest.approx(3000.0 * 1.04 * 0.80)
    assert ccpp.metadata["fuel_exergy_input_kW"] == pytest.approx(
        delivered_stream.exergy_rate
    )
