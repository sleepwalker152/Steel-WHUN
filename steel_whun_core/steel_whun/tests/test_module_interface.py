import pytest

from steel_whun import EnergyStream, FlueGasState, MaterialInput
from steel_whun.modules import ESP, HeatBufferSink, SCR


def _flue_gas() -> FlueGasState:
    return FlueGasState(
        mass_flow=100.0,
        temperature=420.0,
        pressure=101325.0,
        composition={"N2": 0.74, "O2": 0.08, "CO2": 0.13, "H2O": 0.05},
        pollutants={"PM": 300.0, "SO2": 500.0, "NOx": 300.0},
    )


def test_pollution_module_uses_common_result_interface() -> None:
    inlet = _flue_gas()
    result = ESP(name="ESP").evaluate(inlet, operation_params={"load_factor": 1.0})

    assert result.flue_gas_out is not inlet
    assert 0.0 <= result.flue_gas_out.pollutants["PM"] <= inlet.pollutants["PM"]
    assert result.removed_pollutants["PM"] >= 0.0
    assert isinstance(result.feasible, bool)


def test_heat_buffer_is_a_static_terminal_sink() -> None:
    inlet = _flue_gas()
    stream = EnergyStream(
        name="residual_heat_to_buffer",
        carrier="hot_water",
        energy_type="heat",
        flow_rate=750.0,
        temperature=400.0,
        pressure=101325.0,
        energy_rate=750.0,
        exergy_rate=200.0,
    )
    module = HeatBufferSink(
        max_heat_input_kw=1000.0,
        useful_output_factor=0.70,
        service_temperature_K=360.0,
        auxiliary_fraction=0.01,
    )
    result = module.evaluate(inlet, energy_inputs=[stream])

    assert result.metadata["static_sink"] is True
    assert result.metadata["heat_input_kw"] == pytest.approx(750.0)
    assert result.metadata["useful_energy_output_kw"] == pytest.approx(525.0)
    assert result.feasible


def test_scr_uses_material_input_cost_rate() -> None:
    inlet = _flue_gas()
    inlet.temperature = 618.15
    inlet.pollutants["SO2"] = 0.0
    reagent = MaterialInput(
        name="ammonia_reagent",
        flow_rate=2.0,
        cost_rate=37.5,
    )

    result = SCR().evaluate(
        inlet,
        material_inputs=[reagent],
        operation_params={"ammonia_ratio": 1.0, "electricity_price": 0.5},
    )

    electricity_cost = result.energy_consumption["electricity_total_kW"] * 0.5
    assert result.cost == pytest.approx(electricity_cost + reagent.cost_rate)
