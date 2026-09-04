"""
Physics-informed grey-box model of selective catalytic reduction. The model
includes temperature activation, NSR saturation, GHSV effects, ammonia slip and
an inlet-SO2 applicability constraint. Units are K, Pa, kg/s, mg/Nm3 and kW.
"""

from __future__ import annotations

import math
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


class SCR(PollutionModule):
    'SCR component used by the Steel-WHUN core framework.'

    def __init__(
            self,
            name: str = "SCR",
            enabled: bool = True,
            eta_max: float = 0.985,  
            T_opt: float = 618.15,  
            sigma_T: float = 95.0,  
            beta_nsr: float = 3.3,  
            ghsv_design: float = 5000.0,  
            gamma_ghsv: float = 1.5e-4,  
            catalyst_volume_m3: float = 120.0,  
            pressure_drop_base: float = 800.0,  
            auxiliary_power_base: float = 100.0,  
            min_temperature: float = 553.15,  
            max_temperature: float = 693.15,  
            max_inlet_so2: float = 35.0,  
            parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, module_type="scr", enabled=enabled, parameters=parameters)
        self.eta_max = eta_max
        self.T_opt = T_opt
        self.sigma_T = sigma_T
        self.beta_nsr = beta_nsr
        self.ghsv_design = ghsv_design
        self.gamma_ghsv = gamma_ghsv
        self.catalyst_volume_m3 = catalyst_volume_m3
        self.pressure_drop_base = pressure_drop_base
        self.auxiliary_power_base = auxiliary_power_base
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.max_inlet_so2 = max_inlet_so2

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

        fg_in = flue_gas_in
        fg_out = fg_in.copy()
        operation_params = operation_params or {}
        messages = []
        constraints: List[ConstraintReport] = []

        # ==========================================
        
        # ==========================================
        T_in = fg_in.temperature
        is_temp_feasible = self.min_temperature <= T_in <= self.max_temperature
        violation_temp = 0.0
        if T_in < self.min_temperature:
            violation_temp = (self.min_temperature - T_in) / self.min_temperature
        elif T_in > self.max_temperature:
            violation_temp = (T_in - self.max_temperature) / self.max_temperature

        constraints.append(
            ConstraintReport(
                name=f"{self.name}_temperature_window",
                value=T_in,
                limit=self.max_temperature,
                feasible=is_temp_feasible,
                violation=violation_temp * 10.0,
                message=(
                    f"SCR inlet flue-gas temperature ({T_in - 273.15:.1f} °C) "
                    "is outside the design operating window."
                ),
            )
        )

        # ==========================================
        
        # ==========================================
        so2_in = fg_in.pollutants.get("SO2", 0.0)
        is_so2_safe = so2_in <= self.max_inlet_so2
        violation_so2 = 0.0
        if not is_so2_safe:
            violation_so2 = (so2_in - self.max_inlet_so2) / self.max_inlet_so2

        constraints.append(
            ConstraintReport(
                name=f"{self.name}_inlet_so2_limit",
                value=so2_in,
                limit=self.max_inlet_so2,
                feasible=is_so2_safe,
                violation=violation_so2 * 30.0,  
                message=(
                    f"SCR inlet SO2 concentration ({so2_in:.1f} mg/Nm3) exceeds "
                    f"the catalyst sulfur-tolerance limit ({self.max_inlet_so2} "
                    "mg/Nm3), increasing the risk of poisoning and plugging."
                ),
            )
        )

        # ==========================================
        
        # ==========================================
        std_vol_flow_m3_h = (fg_in.mass_flow / 1.34) * 3600.0
        ghsv = std_vol_flow_m3_h / self.catalyst_volume_m3

        f_ghsv = 1.0
        if ghsv > self.ghsv_design:
            f_ghsv = max(1.0 - self.gamma_ghsv * (ghsv - self.ghsv_design), 0.4)
            if ghsv > self.ghsv_design * 1.6:
                constraints.append(
                    ConstraintReport(
                        name=f"{self.name}_ghsv_upper_limit",
                        value=ghsv,
                        limit=self.ghsv_design * 1.6,
                        feasible=False,
                        violation=(ghsv - self.ghsv_design * 1.6) / self.ghsv_design,
                        message=(
                            "Flue-gas space velocity exceeds the upper limit and "
                            "may cause catalyst breakthrough."
                        ),
                    )
                )

        # ==========================================
        
        # ==========================================
        nsr = operation_params.get("ammonia_ratio", 1.0)

        f_T = math.exp(-((T_in - self.T_opt) ** 2) / (2 * (self.sigma_T ** 2)))
        f_NSR = 1.0 - math.exp(-self.beta_nsr * nsr)
        eta_nox = self.eta_max * f_T * f_NSR * f_ghsv
        eta_nox = min(max(eta_nox, 0.0), 0.99)

        nox_removed = fg_out.update_pollutant_by_efficiency("NOx", eta_nox)
        nox_out = fg_out.pollutants.get("NOx", 0.0)

        # ==========================================
        
        # ==========================================
        ammonia_slip_ppm = 0.0
        if nsr > 1.08:
            ammonia_slip_ppm = 1.2 * math.exp(4.2 * (nsr - 1.08))
            if ammonia_slip_ppm > 6.0:
                constraints.append(
                    ConstraintReport(
                        name=f"{self.name}_ammonia_slip_bound",
                        value=ammonia_slip_ppm,
                        limit=6.0,
                        feasible=False,
                        violation=(ammonia_slip_ppm - 6.0) / 6.0,
                        message=(
                            "Excess reductant causes elevated ammonia-slip risk "
                            f"({ammonia_slip_ppm:.2f} ppm > 6 ppm)."
                        ),
                    )
                )

        # ==========================================
        
        # ==========================================
        flow_ratio = fg_in.mass_flow / 200.0
        actual_pressure_drop = self.pressure_drop_base * (flow_ratio ** 2)
        fg_out.pressure = max(fg_in.pressure - actual_pressure_drop, 1.0)

        fan_efficiency = 0.78
        vol_flow_actual = fg_in.mass_flow / 1.0
        fan_power_kW = (actual_pressure_drop * vol_flow_actual / 1000.0) / fan_efficiency
        total_electricity_kW = self.auxiliary_power_base + fan_power_kW

        # ==========================================
        
        # ==========================================
        Ex_fg_in = ThermoCore.flue_gas_physical_exergy(
            fg_in.mass_flow, fg_in.temperature, fg_in.pressure, fg_in.composition
        )
        Ex_fg_out = ThermoCore.flue_gas_physical_exergy(
            fg_out.mass_flow, fg_out.temperature, fg_out.pressure, fg_out.composition
        )
        exergy_destruction = max(Ex_fg_in - Ex_fg_out, 0.0) + total_electricity_kW

        electricity_price = operation_params.get("electricity_price", 0.5)
        grid_carbon_factor = operation_params.get("grid_carbon_factor", 0.6)

        cost = total_electricity_kW * electricity_price
        carbon = total_electricity_kW * grid_carbon_factor

        ammonia_consumption_kg_s = nsr * (fg_in.mass_flow * (fg_in.pollutants.get("NOx", 0.0) * 1e-6) * 0.37)
        if material_inputs:
            cost += self.total_material_cost(material_inputs)
        else:
            cost += ammonia_consumption_kg_s * 3.5

        messages.append(
            f"{self.name}: NOx removal efficiency={eta_nox * 100:.2f}%, "
            f"outlet NOx={nox_out:.1f} mg/Nm3"
        )

        result = ModuleResult(
            module_name=self.name,
            flue_gas_in=fg_in,
            flue_gas_out=fg_out,
            removal_efficiency={"NOx": eta_nox},
            removed_pollutants={"NOx": nox_removed},
            energy_consumption={"electricity_total_kW": total_electricity_kW, "fan_power_kW": fan_power_kW},
            material_consumption={"ammonia_agent_kg_s": ammonia_consumption_kg_s},
            exergy_destruction=exergy_destruction,
            cost=cost,
            carbon_emission=carbon,
            constraints=constraints,
            messages=messages,
            metadata={
                "temperature_in_K": T_in,
                "ghsv_h1": ghsv,
                "ammonia_slip_ppm": ammonia_slip_ppm,
                "pressure_drop_Pa": actual_pressure_drop,
                "temperature_activation_factor": f_T,
                "nsr_saturation_factor": f_NSR,
            },
        )
        result.update_feasibility()
        if any(not c.feasible for c in constraints):
            result.feasible = False
        return result

