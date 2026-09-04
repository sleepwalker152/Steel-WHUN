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
from steel_whun.modules import HeatBufferSink, Reheater


def test_framework_components_execute_end_to_end() -> None:
    network = EnergyNetwork()
    network.add_source(
        EnergySource(
            source_id="test_heat",
            carrier="hot_water",
            temperature=393.15,
            pressure=101325.0,
            max_energy_rate=1000.0,
        )
    )
    network.add_edge(
        TransmissionEdge(
            source_id="test_heat",
            target_id="HeatBufferSink",
            capacity=1000.0,
        )
    )
    sink = HeatBufferSink(
        max_heat_input_kw=1000.0,
        useful_output_factor=0.90,
        service_temperature_K=353.15,
        auxiliary_fraction=0.0,
    )
    evaluator = SystemEvaluator(
        energy_network=network,
        process_route=ProcessRoute([sink], name="test_route_pool"),
        emission_checker=EmissionChecker({}),
    )
    inlet = FlueGasState(
        mass_flow=1.0,
        temperature=373.15,
        pressure=101325.0,
    )
    result = evaluator.evaluate(
        {
            "flue_gas_initial": inlet,
            "allocation_matrix": {"test_heat": {"HeatBufferSink": 500.0}},
            "operation_map": {"backend_assets": ["HeatBufferSink"]},
            "material_map": {},
        }
    )

    assert result["system_feasible"]
    assert result["network_result"]["total_energy_delivered"] > 0.0
    assert any(item.module_name == "HeatBufferSink_backend" for item in result["backend_results"])


def test_source_over_dispatch_is_scaled_proportionally() -> None:
    network = EnergyNetwork()
    network.add_source(
        EnergySource(
            source_id="limited_source",
            carrier="hot_water",
            temperature=393.15,
            pressure=101325.0,
            max_energy_rate=100.0,
            exergy_factor=0.25,
        )
    )
    for target_id in ("target_a", "target_b"):
        network.add_edge(
            TransmissionEdge(
                source_id="limited_source",
                target_id=target_id,
                capacity=200.0,
            )
        )

    energy_map, summary = network.dispatch(
        {"limited_source": {"target_a": 80.0, "target_b": 80.0}}
    )

    assert summary["total_energy_requested"] == pytest.approx(160.0)
    assert summary["source_requested"]["limited_source"] == pytest.approx(160.0)
    assert summary["source_usage"]["limited_source"] == pytest.approx(100.0)
    assert summary["source_dispatch_scale"]["limited_source"] == pytest.approx(0.625)
    assert summary["source_capacity_violation"]["limited_source"] == pytest.approx(60.0)
    assert energy_map["target_a"][0].energy_rate == pytest.approx(50.0)
    assert energy_map["target_b"][0].energy_rate == pytest.approx(50.0)
    assert summary["total_energy_delivered"] == pytest.approx(100.0)
    assert not summary["feasible"]


def test_edge_capacity_truncation_is_reported_by_network() -> None:
    network = EnergyNetwork()
    network.add_source(
        EnergySource(
            source_id="source",
            carrier="hot_water",
            temperature=393.15,
            pressure=101325.0,
            max_energy_rate=200.0,
            exergy_factor=0.25,
        )
    )
    network.add_edge(
        TransmissionEdge(
            source_id="source",
            target_id="target",
            capacity=100.0,
        )
    )

    energy_map, summary = network.dispatch({"source": {"target": 150.0}})

    assert energy_map["target"][0].energy_rate == pytest.approx(100.0)
    assert summary["edge_capacity_violation"]["source->target"] == pytest.approx(50.0)
    assert summary["edge_results"][0]["edge_sent_energy_rate"] == pytest.approx(100.0)
    assert summary["edge_results"][0]["edge_capacity_violation"] == pytest.approx(50.0)
    assert not summary["feasible"]


def test_transport_cost_and_carbon_are_counted_once_at_system_level() -> None:
    network = EnergyNetwork()
    network.add_source(
        EnergySource(
            source_id="priced_heat",
            carrier="hot_water",
            temperature=393.15,
            pressure=101325.0,
            max_energy_rate=1000.0,
            unit_cost=0.01,
            carbon_factor=0.02,
            exergy_factor=0.25,
        )
    )
    network.add_edge(
        TransmissionEdge(
            source_id="priced_heat",
            target_id="Reheater",
            distance_km=1.0,
            operation_cost_per_kw_km=0.001,
            carbon_per_kw_km=0.001,
            capacity=1000.0,
        )
    )

    evaluator = SystemEvaluator(
        energy_network=network,
        process_route=ProcessRoute([Reheater()], name="reheater_cost_ledger"),
        emission_checker=EmissionChecker({}),
    )
    inlet = FlueGasState(
        mass_flow=1.0,
        temperature=373.15,
        pressure=101325.0,
    )

    result = evaluator.evaluate(
        {
            "flue_gas_initial": inlet,
            "allocation_matrix": {"priced_heat": {"Reheater": 1000.0}},
            "operation_map": {},
            "material_map": {},
        }
    )

    reheater_result = result["module_results"][0]
    delivered_stream = result["energy_map"]["Reheater"][0]

    assert delivered_stream.cost_rate == pytest.approx(10.0)
    assert delivered_stream.carbon_rate == pytest.approx(20.0)
    assert delivered_stream.metadata["transmission_cost"] == pytest.approx(1.0)
    assert delivered_stream.metadata["transmission_carbon"] == pytest.approx(1.0)
    assert reheater_result.cost == pytest.approx(10.0)
    assert reheater_result.carbon_emission == pytest.approx(20.0)
    assert result["details"]["network_cost"] == pytest.approx(1.0)
    assert result["details"]["network_carbon"] == pytest.approx(1.0)
    assert result["total_cost"] == pytest.approx(11.0)
    assert result["total_carbon"] == pytest.approx(21.0)
