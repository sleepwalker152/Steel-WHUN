'Core component of the Steel-WHUN computational framework.'

from __future__ import annotations

from typing import List, Optional, Dict, Any
import math

from steel_whun.core.states import (
    FlueGasState,
    EnergyStream,
    MaterialInput,
    ModuleResult,
    ConstraintReport,
)
from steel_whun.core.thermo import ThermoCore
from steel_whun.modules.base import PollutionModule


class WFGD(PollutionModule):
    'WFGD component used by the Steel-WHUN core framework.'

    def __init__(
            self,
            name: str = "WFGD",
            base_efficiency: float = 0.98,
            pressure_drop: float = 1500.0,  
            pump_power: float = 500.0,  
            fan_power: float = 200.0,  
            tower_diameter_m: float = 5.0,  
            tower_height_m: float = 29.0,  
            min_temperature: float = 323.15,  
            max_temperature: float = 473.15,  
            min_outlet_temperature: float = 328.15,  
            enabled: bool = True,
            parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, module_type="wfgd", enabled=enabled, parameters=parameters)

        self.base_efficiency = base_efficiency
        self.pressure_drop = pressure_drop
        self.pump_power = pump_power
        self.fan_power = fan_power
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.min_outlet_temperature = min_outlet_temperature

        self.tower_diameter_m = tower_diameter_m
        self.tower_height_m = tower_height_m
        self.tower_volume_m3 = self._tower_volume()

        
        self._efficiency_model_defaults = {
            "K_mass": 101.0,  
            "E0": 1.5e-3,  
            "k_z": 5.0e-3,  
            "E_min": 1.0e-4,
            "E_max": 8.0e-3,
            "CaS_ref": 1.0,
        }

        self._carbon_emission_factors = {
            "EF_electricity_kgCO2_per_kWh": 0.624,
            "EF_sorbent_kgCO2_per_kg": 1.20769,
            "EF_water_supply_kgCO2_per_m3": 0.211,
            "EF_wastewater_kgCO2_per_m3": 0.346,
        }

        self._economic_factors = {
            "sorbent_price_CNY_per_kg": 0.8,
            "water_price_CNY_per_m3": 2.2,
        }

    def _calculate_enhancement_factor(self, ca_s: float, operation_params: Dict[str, Any]) -> float:
        'Execute the calculate enhancement factor calculation.'
        E0 = operation_params.get("E0", self._efficiency_model_defaults["E0"])
        k_z = operation_params.get("k_z", self._efficiency_model_defaults["k_z"])
        E_min = operation_params.get("E_min", self._efficiency_model_defaults["E_min"])
        E_max = operation_params.get("E_max", self._efficiency_model_defaults["E_max"])
        CaS_ref = operation_params.get("CaS_ref", self._efficiency_model_defaults["CaS_ref"])

        E = E0 + k_z * (ca_s - CaS_ref)
        return max(E_min, min(E, E_max))

    def _calculate_liquid_content_ratio(self, liquid_flow_kg_h: float) -> float:
        'Execute the calculate liquid content ratio calculation.'
        return (liquid_flow_kg_h / 9126.0) * 1.0

    def _tower_volume(self) -> float:
        return 0.25 * math.pi * (float(self.tower_diameter_m) ** 2) * float(self.tower_height_m)

    @staticmethod
    def _composition_with_h2o_fraction(
            composition: Dict[str, float],
            target_h2o_fraction: float,
    ) -> Dict[str, float]:
        target = min(max(float(target_h2o_fraction), 0.0), 0.30)
        non_water = {
            key: max(float(value), 0.0)
            for key, value in composition.items()
            if key != "H2O"
        }
        total_non_water = sum(non_water.values())
        if total_non_water <= 0.0:
            return {"H2O": target}
        scale = (1.0 - target) / total_non_water
        updated = {key: value * scale for key, value in non_water.items()}
        updated["H2O"] = target
        return updated

    def _calculate_residence_time(self, flue_gas_flow_Nm3_h: float) -> float:
        'Execute the calculate residence time calculation.'
        Qg_Nm3_s = flue_gas_flow_Nm3_h / 3600.0
        if Qg_Nm3_s <= 0:
            return 1e-6
        self.tower_volume_m3 = self._tower_volume()
        return self.tower_volume_m3 / Qg_Nm3_s

    def _efficiency_dual_film_model(
            self,
            flue_gas: FlueGasState,
            ca_s: float,
            liquid_flow_kg_h: float,
            operation_params: Dict[str, Any],
    ) -> float:
        'Execute the efficiency dual film model calculation.'
        Qg_Nm3_h = operation_params.get("Qg_Nm3_h", flue_gas.mass_flow / 1.34 * 3600.0)

        E = self._calculate_enhancement_factor(ca_s, operation_params)
        epsilon_g = self._calculate_liquid_content_ratio(liquid_flow_kg_h)
        t0 = self._calculate_residence_time(Qg_Nm3_h)

        K_mass = operation_params.get("K_mass", self._efficiency_model_defaults["K_mass"])

        
        T_ref_K = 413.15
        f_T_damping = max(1.0 - 0.015 * (flue_gas.temperature - T_ref_K), 0.3)

        exponent = K_mass * E * epsilon_g * t0 * f_T_damping
        eta = 1.0 - math.exp(-min(exponent, 15.0))
        return min(max(eta, 0.0), 0.995)

    def _calculate_sorbent_consumption(self, so2_removed_kg_h: float, ca_s: float) -> float:
        Us = 0.92
        sorbent_purity = 0.95
        return (so2_removed_kg_h * ((56.0 * ca_s) / 64.0)) / (Us * sorbent_purity)

    def efficiency_model(
            self,
            flue_gas: FlueGasState,
            material_inputs: Optional[List[MaterialInput]] = None,
            operation_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        operation_params = operation_params or {}
        if "CaS" in operation_params and "liquid_flow_kg_h" in operation_params:
            return self._efficiency_dual_film_model(
                flue_gas, operation_params["CaS"], operation_params["liquid_flow_kg_h"], operation_params
            )

        liquid_gas_ratio = operation_params.get("liquid_gas_ratio", 1.0)
        eta = self.base_efficiency * min(max(liquid_gas_ratio, 0.0), 1.2)
        if flue_gas.temperature < self.min_temperature or flue_gas.temperature > self.max_temperature:
            eta *= 0.9
        return min(max(eta, 0.0), 0.99)

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
        fg_in = flue_gas_in
        fg_out = fg_in.copy()
        constraints: List[ConstraintReport] = []
        messages = []

        
        ca_s = operation_params.get("CaS", 1.2)
        liquid_flow_kg_h = operation_params.get("liquid_flow_kg_h", 9126.0)

        
        epsilon_g = self._calculate_liquid_content_ratio(liquid_flow_kg_h)

        
        eta_so2 = self.efficiency_model(fg_in, material_inputs, operation_params)
        so2_removed_dict = fg_out.update_pollutant_by_efficiency("SO2", eta_so2)

        vol_flow_Nm3_h = (fg_in.mass_flow / 1.34) * 3600.0
        so2_removed_kg_h = fg_in.pollutants.get("SO2", 0.0) * vol_flow_Nm3_h * eta_so2 * 1e-6
        so2_removed_kg_h = max(so2_removed_kg_h, 0.0)

        # =====================================================================
        
        # =====================================================================
        
        
        
        delta_T = (liquid_flow_kg_h / 9126.0) * 70.0
        outlet_T = fg_in.temperature - delta_T
        outlet_T = max(outlet_T, 313.15)  

        fg_out.temperature = outlet_T
        fg_out.pressure = max(fg_in.pressure - self.pressure_drop, 1.0)
        outlet_h2o = max(
            float(fg_in.composition.get("H2O", 0.0) or 0.0),
            float(operation_params.get("outlet_h2o_mol_fraction", 0.10)),
        )
        fg_out.composition = self._composition_with_h2o_fraction(
            fg_in.composition,
            outlet_h2o,
        )
        fg_out.flow_basis = "wet"
        fg_out.metadata["wfgd_wet_exit"] = True

        
        t_in = fg_in.temperature
        is_inlet_temp_safe = self.min_temperature <= t_in <= self.max_temperature
        constraints.append(
            ConstraintReport(
                name=f"{self.name}_temperature_window",
                value=t_in,
                limit=self.max_temperature,
                feasible=is_inlet_temp_safe,
                violation=(
                    max(self.min_temperature - t_in, 0.0) / self.min_temperature
                    + max(t_in - self.max_temperature, 0.0) / self.max_temperature
                ),
                message=(
                    "WFGD inlet flue-gas temperature is outside the recommended "
                    "operating window."
                )
            )
        )

        
        is_dew_safe = outlet_T >= self.min_outlet_temperature
        violation_dew = max(self.min_outlet_temperature - outlet_T, 0.0) / self.min_outlet_temperature
        constraints.append(
            ConstraintReport(
                name=f"{self.name}_dew_point_safety",
                value=outlet_T,
                limit=self.min_outlet_temperature,
                feasible=is_dew_safe,
                violation=violation_dew * 25.0,
                message=(
                    f"Absorber outlet flue-gas temperature "
                    f"({outlet_T - 273.15:.1f} °C) is below the "
                    "corrosion-control dew-point limit."
                )
            )
        )

        
        electricity_total_kW = self.pump_power + self.fan_power
        sorbent_consumption_kg_h = self._calculate_sorbent_consumption(so2_removed_kg_h, ca_s)

        ef_elec = operation_params.get(
            "grid_carbon_factor",
            self._carbon_emission_factors["EF_electricity_kgCO2_per_kWh"],
        )
        ef_sorb = self._carbon_emission_factors["EF_sorbent_kgCO2_per_kg"]
        carbon_elec = electricity_total_kW * ef_elec
        carbon_sorb = sorbent_consumption_kg_h * ef_sorb
        carbon_water = (liquid_flow_kg_h / 1000.0) * self._carbon_emission_factors["EF_water_supply_kgCO2_per_m3"]
        total_carbon = carbon_elec + carbon_sorb + carbon_water

        cost_sorb = sorbent_consumption_kg_h * self._economic_factors["sorbent_price_CNY_per_kg"]
        electricity_price = operation_params.get("electricity_price", 0.65)
        cost_elec = electricity_total_kW * electricity_price
        total_cost = cost_sorb + cost_elec

        
        ex_fg_in = ThermoCore.flue_gas_physical_exergy(fg_in.mass_flow, fg_in.temperature, fg_in.pressure,
                                                       fg_in.composition)
        ex_fg_out = ThermoCore.flue_gas_physical_exergy(fg_in.mass_flow, fg_out.temperature, fg_out.pressure,
                                                        fg_out.composition)
        exergy_destruction = max(ex_fg_in - ex_fg_out, 0.0) + electricity_total_kW

        messages.append(
            f"{self.name}: SO2 removal efficiency={eta_so2 * 100:.2f}%, "
            f"outlet temperature={outlet_T - 273.15:.1f} °C"
        )

        result = ModuleResult(
            module_name=self.name,
            flue_gas_in=fg_in,
            flue_gas_out=fg_out,
            removal_efficiency={"SO2": eta_so2},
            removed_pollutants=so2_removed_dict,
            energy_consumption={"electricity_total_kW": electricity_total_kW, "pump_power_kW": self.pump_power},
            material_consumption={"sorbent_CaO_kg_h": sorbent_consumption_kg_h, "water_makeup_kg_h": liquid_flow_kg_h},
            exergy_destruction=exergy_destruction,
            cost=total_cost,
            carbon_emission=total_carbon,
            constraints=constraints,
            messages=messages,
            metadata={
                "outlet_temperature_K": outlet_T,
                "CaS_mol_mol": ca_s,
                "liquid_content_ratio_percent": epsilon_g,
                "carbon_sorbent_kgCO2_h": carbon_sorb,
                "carbon_electricity_kgCO2_h": carbon_elec,
            }
        )

        
        result.update_feasibility()
        if any(not c.feasible for c in constraints):
            result.feasible = False

        return result

