# steel_whun/modules/reheater.py

from __future__ import annotations

from typing import List, Optional, Dict, Any

from steel_whun.core.states import (
    FlueGasState,
    EnergyStream,
    MaterialInput,
    ModuleResult,
    ConstraintReport,
)
from steel_whun.core.thermo import ThermoCore
from steel_whun.modules.base import PollutionModule


class Reheater(PollutionModule):
    'Reheater component used by the Steel-WHUN core framework.'

    def __init__(
        self,
        name: str = "Reheater",
        heat_transfer_efficiency: float = 0.85,
        target_temperature: Optional[float] = None,
        max_outlet_temperature: Optional[float] = None,
        enabled: bool = True,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, module_type="reheater", enabled=enabled, parameters=parameters)
        self.heat_transfer_efficiency = heat_transfer_efficiency
        self.target_temperature = target_temperature
        self.max_outlet_temperature = max_outlet_temperature

    def evaluate(
        self,
        flue_gas_in: FlueGasState,
        energy_inputs: Optional[List[EnergyStream]] = None,
        material_inputs: Optional[List[MaterialInput]] = None,
        operation_params: Optional[Dict[str, Any]] = None,
    ) -> ModuleResult:

        if not self.enabled:
            return self.bypass_result(flue_gas_in)

        operation_params = operation_params or {}

        fg_in = flue_gas_in.copy()
        fg_out = flue_gas_in.copy()

        Q_in = self.total_energy_input(energy_inputs)
        Ex_in = self.total_exergy_input(energy_inputs)

        Q_to_gas = Q_in * self.heat_transfer_efficiency

        target_T = operation_params.get("target_temperature", self.target_temperature)

        if target_T is not None:
            required_Q = (
                fg_in.mass_flow
                * ThermoCore.gas_cp(fg_in.temperature, fg_in.composition)
                * max(target_T - fg_in.temperature, 0.0)
            )
            
            Q_to_gas = min(Q_to_gas, required_Q)

        delta_T = ThermoCore.temperature_change_by_heat(
            heat_rate=Q_to_gas,
            mass_flow=fg_in.mass_flow,
            temperature=fg_in.temperature,
            composition=fg_in.composition,
        )

        fg_out.temperature = fg_in.temperature + delta_T

        constraints = []
        if self.max_outlet_temperature is not None:
            constraints.append(
                ConstraintReport.upper_bound(
                    name=f"{self.name}_max_outlet_temperature",
                    value=fg_out.temperature,
                    limit=self.max_outlet_temperature,
                    message="Reheater outlet temperature exceeds upper bound.",
                )
            )

        Ex_fg_in = ThermoCore.flue_gas_physical_exergy(
            fg_in.mass_flow, fg_in.temperature, fg_in.pressure, fg_in.composition
        )
        Ex_fg_out = ThermoCore.flue_gas_physical_exergy(
            fg_out.mass_flow, fg_out.temperature, fg_out.pressure, fg_out.composition
        )

        useful_exergy_gain = max(Ex_fg_out - Ex_fg_in, 0.0)
        exergy_destruction = max(Ex_in - useful_exergy_gain, 0.0)

        result = ModuleResult(
            module_name=self.name,
            flue_gas_in=fg_in,
            flue_gas_out=fg_out,
            removal_efficiency={},
            removed_pollutants={},
            energy_consumption={
                "heat_input_kW": Q_in,
                "heat_to_gas_kW": Q_to_gas,
            },
            material_consumption={},
            exergy_destruction=exergy_destruction,
            cost=self.total_energy_cost(energy_inputs),
            carbon_emission=self.total_energy_carbon(energy_inputs),
            constraints=constraints,
            messages=[
                f"{self.name}: inlet T = {fg_in.temperature - 273.15:.2f} C, "
                f"outlet T = {fg_out.temperature - 273.15:.2f} C."
            ],
            metadata={
                "delta_T": delta_T,
                "Q_in": Q_in,
                "Q_to_gas": Q_to_gas,
                "Ex_in": Ex_in,
                "useful_exergy_gain": useful_exergy_gain,
            },
        )
        result.update_feasibility()
        return result