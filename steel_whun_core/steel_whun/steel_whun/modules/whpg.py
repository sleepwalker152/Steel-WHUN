# steel_whun/modules/whpg.py
'Core component of the Steel-WHUN computational framework.'

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


class WHPG(PollutionModule):
    'WHPG component used by the Steel-WHUN core framework.'

    def __init__(
            self,
            name: str = "WHPG",
            enabled: bool = True,
            sink_temperature: float = 298.15,
            turbine_exergy_efficiency: float = 0.30,  # configurable default
            auxiliary_power: float = 100.0,  # configurable default, kW
            max_extraction_temperature_drop: Optional[float] = None,
            parameters: Optional[Dict[str, Any]] = None,
    ):
        'Execute the init calculation.'
        super().__init__(
            name=name,
            module_type="whpg",
            enabled=enabled,
            parameters=parameters,
        )
        self.sink_temperature = sink_temperature
        self.turbine_exergy_efficiency = turbine_exergy_efficiency
        self.auxiliary_power = auxiliary_power
        self.max_extraction_temperature_drop = max_extraction_temperature_drop

    def _flue_exergy(self, fg: FlueGasState) -> float:
        'Execute the flue exergy calculation.'
        return ThermoCore.flue_gas_physical_exergy(
            fg.mass_flow, fg.temperature, fg.pressure, fg.composition
        )

    def _estimate_rankine_efficiency(self, T_hot: float, T_cold: float) -> float:
        'Execute the estimate rankine efficiency calculation.'
        if T_hot <= T_cold or T_hot <= 0:
            return 0.0
        eta_carnot = 1.0 - (T_cold / T_hot)
        return max(0.0, min(eta_carnot * 0.5, 1.0))

    def evaluate(
            self,
            flue_gas_in: FlueGasState,
            energy_inputs: Optional[List[EnergyStream]] = None,  
            material_inputs: Optional[List[MaterialInput]] = None,  
            operation_params: Optional[Dict[str, Any]] = None,
    ) -> ModuleResult:
        'Execute the evaluate calculation.'
        
        if not self.enabled:
            return self.bypass_result(flue_gas_in)

        operation_params = operation_params or {}

        fg_in = flue_gas_in.copy()
        fg_out = flue_gas_in.copy()

        
        next_T_min = operation_params.get("next_inlet_temperature_min", None)
        next_T_max = operation_params.get("next_inlet_temperature_max", None)
        next_module_name = operation_params.get("next_module_name", None)

        
        if (next_T_min is None or next_T_max is None) and next_module_name is not None:
            FALLBACK_WINDOWS = {
                "scr": (533.15, 723.15),
                "wfgd": (323.15, 473.15),
                "esp": (323.15, 673.15),
                "reheater": (300.0, 800.0),
            }
            module_key = next_module_name.lower()
            for key, (T_min, T_max) in FALLBACK_WINDOWS.items():
                if key in module_key:
                    next_T_min = next_T_min if next_T_min is not None else T_min
                    next_T_max = next_T_max if next_T_max is not None else T_max
                    break

        desired_outlet_temperature = operation_params.get("desired_outlet_temperature", None)

        
        if desired_outlet_temperature is None:
            fg_out.temperature = fg_in.temperature
            fg_out.pressure = max(fg_in.pressure, 1.0)

            ex_in = self._flue_exergy(fg_in)
            ex_out = self._flue_exergy(fg_out)

            
            exergy_destruction = self.auxiliary_power

            electricity_price = operation_params.get("electricity_price", 0.5)
            grid_carbon_factor = operation_params.get("grid_carbon_factor", 0.6)

            
            cost = self.auxiliary_power * electricity_price
            
            carbon_emission = self.auxiliary_power * grid_carbon_factor

            
            constraints: List[ConstraintReport] = []
            if next_T_min is not None or next_T_max is not None:
                lower = next_T_min if next_T_min is not None else -1e18
                upper = next_T_max if next_T_max is not None else 1e18
                constraints.append(
                    ConstraintReport.range_bound(
                        name=f"{self.name}_outlet_temperature_for_next_device",
                        value=fg_out.temperature,
                        lower=lower,
                        upper=upper,
                        message="WHPG outlet temperature not compatible with next device inlet window.",
                    )
                )

            result = ModuleResult(
                module_name=self.name,
                flue_gas_in=fg_in,
                flue_gas_out=fg_out,
                removal_efficiency={},
                removed_pollutants={},
                energy_consumption={
                    "electricity_aux_kW": self.auxiliary_power,
                    "electricity_output_kW": 0.0,
                    "Q_extracted_kW": 0.0,
                },
                material_consumption={},
                exergy_destruction=exergy_destruction,
                cost=cost,
                carbon_emission=carbon_emission,
                constraints=constraints,
                messages=[f"{self.name}: no extraction (desired_outlet_temperature not specified)."],
                metadata={
                    "T_in": fg_in.temperature,
                    "T_out": fg_out.temperature,
                    "mode": "bypass",
                    "turbine_exergy_efficiency": self.turbine_exergy_efficiency,
                },
            )
            result.update_feasibility()
            return result

        
        T_out_tentative = float(desired_outlet_temperature)

        
        if T_out_tentative > fg_in.temperature:
            T_out_tentative = fg_in.temperature

        
        if self.max_extraction_temperature_drop is not None:
            T_out_tentative = max(
                T_out_tentative,
                fg_in.temperature - self.max_extraction_temperature_drop,
            )

        
        clamped = False
        clamp_reason = None

        if next_T_min is not None or next_T_max is not None:
            effective_T_min = next_T_min if next_T_min is not None else -1e18
            effective_T_max = next_T_max if next_T_max is not None else 1e18

            
            
            if fg_in.temperature < effective_T_min:
                T_out_tentative = fg_in.temperature
                clamped = True
                clamp_reason = "Inlet temperature already below next device lower bound; WHPG cannot heat flue gas."
            else:
                
                if T_out_tentative > effective_T_max:
                    T_out_tentative = effective_T_max
                    clamped = True
                    clamp_reason = "clamped to next_inlet_temperature_max."
                elif T_out_tentative < effective_T_min:
                    T_out_tentative = effective_T_min
                    clamped = True
                    clamp_reason = "clamped to next_inlet_temperature_min."

        fg_out.temperature = T_out_tentative
        fg_out.pressure = max(fg_in.pressure, 1.0)

        
        delta_T = max(fg_in.temperature - fg_out.temperature, 0.0)
        cp = ThermoCore.gas_cp((fg_in.temperature + fg_out.temperature) / 2.0, fg_in.composition)
        Q_extracted_kW = fg_in.mass_flow * cp * delta_T

        
        Ex_in = self._flue_exergy(fg_in)
        Ex_out = self._flue_exergy(fg_out)
        exergy_removed = max(Ex_in - Ex_out, 0.0)

        
        eta_rankine = self._estimate_rankine_efficiency(fg_in.temperature, self.sink_temperature)

        
        heat_limited_power_kW = Q_extracted_kW * eta_rankine
        exergy_limited_power_kW = exergy_removed * self.turbine_exergy_efficiency
        electricity_output_kW = max(
            min(heat_limited_power_kW, exergy_limited_power_kW),
            0.0,
        )

        
        exergy_destruction = max(exergy_removed - electricity_output_kW, 0.0) + self.auxiliary_power

        
        electricity_price = operation_params.get("electricity_price", 0.5)
        grid_carbon_factor = operation_params.get("grid_carbon_factor", 0.6)

        
        cost = (-electricity_output_kW * electricity_price) + (self.auxiliary_power * electricity_price)

        
        carbon_emission = (self.auxiliary_power - electricity_output_kW) * grid_carbon_factor

        
        constraints: List[ConstraintReport] = []

        
        if next_T_min is not None or next_T_max is not None:
            lower = next_T_min if next_T_min is not None else -1e18
            upper = next_T_max if next_T_max is not None else 1e18
            constraints.append(
                ConstraintReport.range_bound(
                    name=f"{self.name}_outlet_temperature_for_next_device",
                    value=fg_out.temperature,
                    lower=lower,
                    upper=upper,
                    message="WHPG outlet temperature not compatible with next device inlet window.",
                )
            )

        
        result = ModuleResult(
            module_name=self.name,
            flue_gas_in=fg_in,
            flue_gas_out=fg_out,
            removal_efficiency={},
            removed_pollutants={},
            energy_consumption={
                "electricity_output_kW": electricity_output_kW,
                "electricity_aux_kW": self.auxiliary_power,
                "Q_extracted_kW": Q_extracted_kW,
            },
            material_consumption={},
            exergy_destruction=exergy_destruction,
            cost=cost,
            carbon_emission=carbon_emission,
            constraints=constraints,
            messages=[
                f"{self.name}: Tin={fg_in.temperature:.2f}K, Tout={fg_out.temperature:.2f}K, "
                f"exergy_removed={exergy_removed:.2f}kW, Pel={electricity_output_kW:.2f}kW."
                + (f" {clamp_reason}" if clamped else "")
            ],
            metadata={
                "T_out_tentative": T_out_tentative,
                "T_next_min": next_T_min,
                "T_next_max": next_T_max,
                "clamped": clamped,
                "clamp_reason": clamp_reason,
                "delta_T_K": delta_T,
                "exergy_removed_kW": exergy_removed,
                "eta_rankine": eta_rankine,
                "heat_limited_power_kW": heat_limited_power_kW,
                "exergy_limited_power_kW": exergy_limited_power_kW,
                "turbine_exergy_efficiency": self.turbine_exergy_efficiency,
            },
        )
        result.update_feasibility()
        return result
