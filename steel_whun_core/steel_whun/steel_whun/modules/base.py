# steel_whun/modules/base.py

from __future__ import annotations

from typing import List, Dict, Optional, Any

from steel_whun.core.states import (
    FlueGasState,
    EnergyStream,
    MaterialInput,
    ModuleResult,
)


class PollutionModule:
    'PollutionModule component used by the Steel-WHUN core framework.'

    def __init__(
        self,
        name: str,
        module_type: str,
        enabled: bool = True,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.module_type = module_type
        self.enabled = enabled
        self.parameters = parameters or {}

    def evaluate(
        self,
        flue_gas_in: FlueGasState,
        energy_inputs: Optional[List[EnergyStream]] = None,
        material_inputs: Optional[List[MaterialInput]] = None,
        operation_params: Optional[Dict[str, Any]] = None,
    ) -> ModuleResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate()."
        )

    def total_energy_input(
        self,
        energy_inputs: Optional[List[EnergyStream]],
    ) -> float:
        if not energy_inputs:
            return 0.0
        return sum(max(e.energy_rate, 0.0) for e in energy_inputs)

    def total_exergy_input(
        self,
        energy_inputs: Optional[List[EnergyStream]],
    ) -> float:
        if not energy_inputs:
            return 0.0
        return sum(max(e.exergy_rate, 0.0) for e in energy_inputs)

    def total_energy_cost(
        self,
        energy_inputs: Optional[List[EnergyStream]],
    ) -> float:
        if not energy_inputs:
            return 0.0
        return sum(max(e.cost_rate, 0.0) for e in energy_inputs)

    def total_energy_carbon(
        self,
        energy_inputs: Optional[List[EnergyStream]],
    ) -> float:
        if not energy_inputs:
            return 0.0
        return sum(max(e.carbon_rate, 0.0) for e in energy_inputs)

    def total_material_cost(
        self,
        material_inputs: Optional[List[MaterialInput]],
    ) -> float:
        if not material_inputs:
            return 0.0
        return sum(max(m.cost_rate, 0.0) for m in material_inputs)

    def total_material_carbon(
        self,
        material_inputs: Optional[List[MaterialInput]],
    ) -> float:
        if not material_inputs:
            return 0.0
        return sum(max(m.carbon_rate, 0.0) for m in material_inputs)

    def bypass_result(self, flue_gas_in: FlueGasState) -> ModuleResult:
        'Execute the bypass result calculation.'
        return ModuleResult(
            module_name=self.name,
            flue_gas_in=flue_gas_in.copy(),
            flue_gas_out=flue_gas_in.copy(),
            feasible=True,
            messages=[f"{self.name} is disabled and bypassed."],
        )