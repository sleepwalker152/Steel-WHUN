# steel_whun/modules/ccpp.py
'Core component of the Steel-WHUN computational framework.'

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


class CCPP(PollutionModule):
    'CCPP component used by the Steel-WHUN core framework.'

    def __init__(
            self,
            name: str = "CCPP",
            enabled: bool = True,
            power_design_kW: float = 100000.0,  # configurable default
            eta_design: float = 0.45,  # configurable default
            t_ref_ambient: float = 298.15,  
            beta_temp: float = 0.0035,  
            aux_power_rate: float = 0.05,  # configurable default
            parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, module_type="ccpp", enabled=enabled, parameters=parameters)
        self.power_design_kW = power_design_kW
        self.eta_design = eta_design
        self.t_ref_ambient = t_ref_ambient
        self.beta_temp = beta_temp
        self.aux_power_rate = aux_power_rate

        
        self._gas_properties = {
            "BFG": {"LHV": 3300.0, "exergy_factor": 1.02, "fg_ratio": 1.45, "density": 1.30},  
            "COG": {"LHV": 17500.0, "exergy_factor": 1.04, "fg_ratio": 5.40, "density": 0.45},  
            "LDG": {"LHV": 8400.0, "exergy_factor": 1.03, "fg_ratio": 1.95, "density": 1.25},  
        }

        
        for gas_id, overrides in (parameters or {}).get("gas_properties", {}).items():
            if gas_id not in self._gas_properties:
                continue
            self._gas_properties[gas_id].update(
                {
                    key: float(value)
                    for key, value in overrides.items()
                    if key in self._gas_properties[gas_id]
                }
            )

        self._part_load_coeffs = {
            "a0": 0.65,
            "a1": 0.45,
            "a2": -0.10,
        }

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
        # A missing gas-flow key means that the network delivered none of that
        # gas to the CCPP.  Do not substitute an implicit operating point here:
        # doing so would create fuel outside the network allocation balance.
        v_bfg = max(float(operation_params.get("BFG_flow_m3_s", 0.0)), 0.0)
        v_cog = max(float(operation_params.get("COG_flow_m3_s", 0.0)), 0.0)
        v_ldg = max(float(operation_params.get("LDG_flow_m3_s", 0.0)), 0.0)

        
        q_in_bfg = v_bfg * self._gas_properties["BFG"]["LHV"]
        q_in_cog = v_cog * self._gas_properties["COG"]["LHV"]
        q_in_ldg = v_ldg * self._gas_properties["LDG"]["LHV"]
        total_heat_input_kW = q_in_bfg + q_in_cog + q_in_ldg

        # ==========================================
        
        # ==========================================
        
        q_design_input_kW = self.power_design_kW / self.eta_design
        
        lr = total_heat_input_kW / q_design_input_kW if q_design_input_kW > 0 else 0.0

        
        is_lr_feasible = 0.3 <= lr <= 1.1
        violation_lr = 0.0
        if lr < 0.3:
            violation_lr = (0.3 - lr) / 0.3
        elif lr > 1.1:
            violation_lr = (lr - 1.1) / 1.1

        constraints.append(
            ConstraintReport(
                name=f"{self.name}_load_rate_window",
                value=lr,
                limit=1.1,
                feasible=is_lr_feasible,
                violation=violation_lr * 15.0,  
                message=(
                    f"CCPP unit load ratio ({lr * 100:.1f}%) is outside the "
                    "stable operating window [30%, 110%]."
                ),
            )
        )

        
        a0 = self._part_load_coeffs["a0"]
        a1 = self._part_load_coeffs["a1"]
        a2 = self._part_load_coeffs["a2"]
        f_lr = a0 + a1 * lr + a2 * (lr ** 2)
        f_lr = min(max(f_lr, 0.2), 1.05)

        
        t_amb = operation_params.get("ambient_temperature_K", 298.15)
        f_temp = 1.0 - self.beta_temp * (t_amb - self.t_ref_ambient)

        
        eta_ccpp = self.eta_design * f_lr * f_temp
        eta_ccpp = min(max(eta_ccpp, 0.05), 0.55)

        
        gross_electricity_kW = total_heat_input_kW * eta_ccpp
        auxiliary_power_kW = gross_electricity_kW * self.aux_power_rate
        net_electricity_output_kW = gross_electricity_kW - auxiliary_power_kW

        # ==========================================
        
        # ==========================================
        
        m_fg_generated = (
                v_bfg * self._gas_properties["BFG"]["fg_ratio"] * 1.34 / 1.0 +  
                v_cog * self._gas_properties["COG"]["fg_ratio"] * 1.34 / 1.0 +
                v_ldg * self._gas_properties["LDG"]["fg_ratio"] * 1.34 / 1.0
        )

        
        fg_out.mass_flow = fg_in.mass_flow + m_fg_generated

        
        
        base_outlet_T = 403.15  
        if lr > 0:
            fg_out.temperature = base_outlet_T + 15.0 * ((1.0 - lr) ** 2)
        else:
            fg_out.temperature = t_amb  

        
        
        fg_out.pollutants["NOx"] = 120.0  
        fg_out.pollutants["SO2"] = 80.0  
        fg_out.pollutants["PM"] = 15.0  

        # ==========================================
        
        # ==========================================
        
        
        carbon_combustion = v_bfg * 0.95 + v_cog * 0.45 + v_ldg * 1.15
        carbon_combustion_h = carbon_combustion * 3600.0

        
        grid_carbon_factor = operation_params.get("grid_carbon_factor", 0.61)
        carbon_credit_h = net_electricity_output_kW * grid_carbon_factor
        total_net_carbon_emission = carbon_combustion_h - carbon_credit_h

        
        elec_price = operation_params.get("electricity_price", 0.65)
        revenue_h = net_electricity_output_kW * elec_price

        
        if energy_inputs:
            # Preserve the delivered exergy ledger supplied by the network.
            # Reconstructing chemical exergy from delivered LHV here would
            # apply the gas exergy factor a second time and break continuity.
            ex_fuel = self.total_exergy_input(energy_inputs)
        else:
            ex_fuel = (
                q_in_bfg * self._gas_properties["BFG"]["exergy_factor"]
                + q_in_cog * self._gas_properties["COG"]["exergy_factor"]
                + q_in_ldg * self._gas_properties["LDG"]["exergy_factor"]
            )
        ex_fg_out = ThermoCore.flue_gas_physical_exergy(
            fg_out.mass_flow, fg_out.temperature, fg_out.pressure, fg_out.composition
        )
        
        exergy_destruction = max(ex_fuel - net_electricity_output_kW - ex_fg_out, 0.0)

        messages.append(
            f"{self.name}: load ratio={lr * 100:.1f}%, "
            f"thermal efficiency={eta_ccpp * 100:.2f}%, "
            f"net electricity={net_electricity_output_kW:.1f} kW, "
            f"generated flue gas={m_fg_generated:.1f} kg/s"
        )

        result = ModuleResult(
            module_name=self.name,
            flue_gas_in=fg_in,
            flue_gas_out=fg_out,
            removal_efficiency={},
            removed_pollutants={},
            
            energy_consumption={
                "electricity_output_kW": -net_electricity_output_kW,
                "auxiliary_power_kW": auxiliary_power_kW,
                "gross_generated_kW": gross_electricity_kW
            },
            material_consumption={
                "BFG_consumed_m3_h": v_bfg * 3600.0,
                "COG_consumed_m3_h": v_cog * 3600.0,
                "LDG_consumed_m3_h": v_ldg * 3600.0,
            },
            exergy_destruction=exergy_destruction,
            cost=-revenue_h,  
            carbon_emission=total_net_carbon_emission,
            constraints=constraints,
            messages=messages,
            metadata={
                "load_rate": lr,
                "thermal_efficiency": eta_ccpp,
                "generated_flue_gas_kg_s": m_fg_generated,
                "exhaust_temperature_K": fg_out.temperature,
                "carbon_combustion_kg_h": carbon_combustion_h,
                "fuel_exergy_input_kW": ex_fuel,
            },
        )

        
        result.update_feasibility()
        if any(not c.feasible for c in constraints):
            result.feasible = False

        return result
