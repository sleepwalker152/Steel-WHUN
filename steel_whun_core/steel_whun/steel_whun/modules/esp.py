"""
Electrostatic-precipitator model based on the modified Deutsch-Anderson/NDe
relationship, partial particle charging and reverse-corona corrections. The
implemented coefficients and operating limits are declared in the module class.
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


class ESP(PollutionModule):
    """Electrostatic-precipitator module for steelworks flue gas."""

    
    
    NANOPARTICLE_COEFF_LOW_NDe = {  # NDe < 10
        "A": 1.4018,
        "B": 0.7601,
        "C": -0.0059,
    }
    NANOPARTICLE_COEFF_HIGH_NDe = {  # NDe >= 10
        "A": 3.28e-7,
        "B": 7.113,
        "C": -8.51e-4,
    }

    
    MICROPARTICLE_COEFF_LOW_NDe = {  # NDe < 0.15
        "A": 0.0023,
        "B": -0.5058,
        "C": 3.8389,
    }
    MICROPARTICLE_COEFF_MID_NDe = {  # 0.15 <= NDe <= 2.20
        "A": 2.273,
        "B": 0.471,
        "C": 0.0168,
    }

    
    LAWLESS_COEFF = {
        "a": 1.91588,
        "b": -0.1425,
        "c": 1.296e-5,
        "d": -1.2671,
    }

    
    REVERSE_CORONA_THRESHOLD = 8000.0  
    REVERSE_CORONA_SEVERE = 25000.0  

    def __init__(
            self,
            name: str = "ESP",
            enabled: bool = True,
            
            electrode_area_m2: float = 1400.0,
            collection_plate_length_m: float = 8.0,
            electrode_spacing_m: float = 0.12,
            number_of_wires: int = 5,
            wire_radius_mm: float = 0.70,
            
            base_efficiency: float = 0.98,
            operating_voltage_kV: float = 57.0,
            min_temperature: float = 383.15,
            max_temperature: float = 573.15,
            optimal_temperature: float = 573.15,
            temperature_sensitivity: float = 0.001,
            
            electricity_power: float = 800.0,
            fan_power: float = 100.0,
            pressure_drop: float = 150.0,
            temperature_drop: float = 5.0,  
            
            min_h2o_mol_fraction: Optional[float] = None,
            max_h2o_mol_fraction: Optional[float] = 0.18,
            reverse_corona_threshold_kg_m3: Optional[float] = None,
            reverse_corona_severe_kg_m3: Optional[float] = None,
            parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            module_type="esp",
            enabled=enabled,
            parameters=parameters or {},
        )

        
        self.electrode_area_m2 = float(electrode_area_m2)
        self.collection_plate_length_m = float(collection_plate_length_m)
        self.electrode_spacing_m = float(electrode_spacing_m)
        self.number_of_wires = int(number_of_wires)
        self.wire_radius_mm = float(wire_radius_mm)

        
        self.base_efficiency = float(base_efficiency)
        self.operating_voltage_kV = float(operating_voltage_kV)
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.optimal_temperature = float(optimal_temperature)
        self.temperature_sensitivity = float(temperature_sensitivity)

        
        self.electricity_power = float(electricity_power)
        self.fan_power = float(fan_power)
        self.pressure_drop = float(pressure_drop)
        self.temperature_drop = float(temperature_drop)  

        
        self.min_h2o_mol_fraction = (
            None if min_h2o_mol_fraction is None else float(min_h2o_mol_fraction)
        )
        self.max_h2o_mol_fraction = (
            None if max_h2o_mol_fraction is None else float(max_h2o_mol_fraction)
        )

        if reverse_corona_threshold_kg_m3 is not None:
            self.REVERSE_CORONA_THRESHOLD = float(reverse_corona_threshold_kg_m3) * 1e6
        if reverse_corona_severe_kg_m3 is not None:
            self.REVERSE_CORONA_SEVERE = float(reverse_corona_severe_kg_m3) * 1e6

    

    def _calculate_nde(
            self,
            flue_gas: FlueGasState,
            operation_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        'Execute the calculate nde calculation.'
        operation_params = operation_params or {}

        
        V_kV = float(operation_params.get("operating_voltage", self.operating_voltage_kV))
        V_ref = 50.0

        w_ref_m_s = 0.05  
        w_m_s = w_ref_m_s * math.sqrt(max(V_kV / V_ref, 0.1))

        
        try:
            rho = ThermoCore.gas_density(
                flue_gas.temperature, flue_gas.pressure, flue_gas.composition
            )
        except Exception:
            rho = 1.2

        rho = max(rho, 0.001)

        
        cross_section_m2 = max(
            self.electrode_area_m2 / max(self.number_of_wires, 1), 1.0
        )
        u_ave = flue_gas.mass_flow / (rho * cross_section_m2)
        u_ave = max(u_ave, 0.01)

        
        NDe = (w_m_s * self.collection_plate_length_m) / (
                self.electrode_spacing_m * u_ave
        )

        return max(NDe, 0.01)

    def _calculate_nre(self, flue_gas: FlueGasState) -> float:
        'Execute the calculate nre calculation.'
        try:
            rho = ThermoCore.gas_density(
                flue_gas.temperature, flue_gas.pressure, flue_gas.composition
            )
            mu = ThermoCore.gas_viscosity(
                flue_gas.temperature, flue_gas.composition
            )
        except Exception:
            rho = 1.2
            mu = 1.8e-5

        rho = max(rho, 0.001)
        mu = max(mu, 1e-7)

        
        cross_section_m2 = max(
            self.electrode_area_m2 / max(self.number_of_wires, 1), 1.0
        )
        v = flue_gas.mass_flow / (rho * cross_section_m2)
        v = max(v, 0.01)

        
        D_h = 2.0 * self.electrode_spacing_m

        NRe = (rho * v * D_h) / mu

        return max(NRe, 1.0)

    def _calculate_particle_charge_lawless(
            self,
            particle_diameter_nm: float,
            ion_concentration_m3: float,
            residence_time_s: float,
            electric_field_V_m: float,
    ) -> float:
        'Execute the calculate particle charge lawless calculation.'
        d_m = max(particle_diameter_nm * 1e-9, 1e-10)

        
        ndiff = max(0.01, ion_concentration_m3 * residence_time_s * (d_m * 1e6) * 1e-9)
        nfield = max(
            0.01, electric_field_V_m * ion_concentration_m3 * residence_time_s * d_m * 1e-12
        )

        a = self.LAWLESS_COEFF["a"]
        b = self.LAWLESS_COEFF["b"]
        c = self.LAWLESS_COEFF["c"]
        d = self.LAWLESS_COEFF["d"]

        try:
            n_p_t = ndiff + a * (ndiff ** b) * (c * nfield - d)
        except Exception:
            n_p_t = ndiff

        return max(n_p_t, 0.0)

    def _partial_charging_factor(self, n_p_t: float) -> float:
        'Execute the partial charging factor calculation.'
        return min(max(n_p_t, 0.0), 1.0)

    def _lin_collection_efficiency(
            self,
            NDe: float,
            alpha: float,
            particle_diameter_nm: Optional[float] = None,
    ) -> float:
        'Execute the lin collection efficiency calculation.'
        
        if particle_diameter_nm is not None and particle_diameter_nm > 100.0:
            
            if NDe < 0.15:
                coeff = self.MICROPARTICLE_COEFF_LOW_NDe
            else:
                coeff = self.MICROPARTICLE_COEFF_MID_NDe
        else:
            
            if NDe < 10.0:
                coeff = self.NANOPARTICLE_COEFF_LOW_NDe
            else:
                coeff = self.NANOPARTICLE_COEFF_HIGH_NDe

        A = coeff["A"]
        B = coeff["B"]
        C = coeff["C"]

        
        try:
            exp_term = math.exp(-A * (NDe ** B))
        except Exception:
            exp_term = 0.0

        eta = (1.0 - exp_term) + C * NDe - (1.0 - alpha)
        eta = max(min(eta, 1.0), 0.0)

        return eta

    

    def _temperature_factor(self, T: float) -> float:
        'Execute the temperature factor calculation.'
        if T < self.min_temperature or T > self.max_temperature:
            return 0.5

        delta_T = T - self.optimal_temperature
        alpha = self.temperature_sensitivity
        f_T = 1.0 - alpha * (delta_T ** 2) / (self.optimal_temperature ** 2)

        return max(min(f_T, 1.0), 0.5)

    def _load_factor_correction(self, load_factor: float) -> float:
        'Execute the load factor correction calculation.'
        if load_factor <= 0 or load_factor > 2.0:
            return 0.0

        
        deviation = abs(load_factor - 1.0)
        k_L = 0.15
        gamma = 1.5

        f_L = 1.0 - k_L * (deviation ** gamma)

        return max(min(f_L, 1.0), 0.6)

    def _voltage_factor(self, operation_params: Optional[Dict[str, Any]] = None) -> float:
        'Execute the voltage factor calculation.'
        operation_params = operation_params or {}
        V_kV = float(operation_params.get("operating_voltage", self.operating_voltage_kV))
        V_ref = 50.0

        if V_kV <= 0:
            return 0.6

        
        NV = math.sqrt(max(V_kV / V_ref, 0.1))

        return max(min(NV, 1.4), 0.5)  

    def _reverse_corona_factor(self, pm_concentration_mg_Nm3: float) -> float:
        'Execute the reverse corona factor calculation.'
        if pm_concentration_mg_Nm3 is None:
            return 1.0

        if pm_concentration_mg_Nm3 <= self.REVERSE_CORONA_THRESHOLD:
            return 1.0

        excess = pm_concentration_mg_Nm3 - self.REVERSE_CORONA_THRESHOLD
        denominator = max(
            self.REVERSE_CORONA_SEVERE - self.REVERSE_CORONA_THRESHOLD, 1.0
        )
        severity = excess / denominator
        severity = max(min(severity, 1.0), 0.0)

        k_rev = 1.5
        f_rev = 0.5 * (1.0 + math.tanh(-k_rev * (severity - 0.5)))

        return max(min(f_rev, 1.0), 0.25)

    def _concentration_factor(self, pm_inlet_mg_Nm3: float) -> float:
        'Execute the concentration factor calculation.'
        C_ref = 2000.0
        k_C = 0.05

        if pm_inlet_mg_Nm3 is None or pm_inlet_mg_Nm3 <= 0:
            return 1.0

        f_C = 1.0 - k_C * math.sqrt(pm_inlet_mg_Nm3 / C_ref)

        return max(min(f_C, 1.0), 0.7)

    
    

    def _average_electric_field(self, operation_params: Optional[Dict[str, Any]] = None) -> float:
        'Execute the average electric field calculation.'
        operation_params = operation_params or {}
        V_kV = float(operation_params.get("operating_voltage", self.operating_voltage_kV))
        V_volt = V_kV * 1000.0  # V
        sy = max(self.electrode_spacing_m, 1e-4)  # m
        E_avg = V_volt / sy  # V/m
        return float(E_avg)

    def _corona_onset_field(self, operation_params: Optional[Dict[str, Any]] = None) -> float:
        'Execute the corona onset field calculation.'
        operation_params = operation_params or {}
        E_c = float(operation_params.get("corona_onset_field_V_m", 3.0e6))
        return max(E_c, 1e3)

    def _plate_current_density_cooperman(self, E_avg: float,
                                         operation_params: Optional[Dict[str, Any]] = None) -> float:
        'Execute the plate current density cooperman calculation.'
        operation_params = operation_params or {}
        E_c = self._corona_onset_field(operation_params)
        if E_avg <= E_c:
            return 0.0

        J0 = float(operation_params.get("J0_A_m2", 1e-6))  # A/m2 baseline
        m = float(operation_params.get("J_exponent_m", 2.0))

        ratio = max(E_avg / E_c - 1.0, 0.0)
        Jp = J0 * (ratio ** m)

        ptype = str(operation_params.get("power_supply_type", "")).lower()
        if ptype == "igbt":
            Jp *= float(operation_params.get("igbt_current_boost", 1.8))
        elif ptype == "pulsed":
            Jp *= float(operation_params.get("pulsed_current_boost", 1.3))

        return float(max(Jp, 0.0))

    def _total_current_and_power(self, flue_gas: FlueGasState, operation_params: Optional[Dict[str, Any]] = None):
        'Execute the total current and power calculation.'
        operation_params = operation_params or {}
        V_kV = float(operation_params.get("operating_voltage", self.operating_voltage_kV))
        V_volt = V_kV * 1000.0

        E_avg = self._average_electric_field(operation_params)
        Jp = self._plate_current_density_cooperman(E_avg, operation_params)

        # total current (A): scale Jp by electrode area (A = A/m2 * m2)
        I_total = Jp * max(self.electrode_area_m2, 1e-6)
        I_per_wire = I_total / max(self.number_of_wires, 1)

        # instantaneous power (W)
        P_instant_W = V_volt * I_total

        supply_type = str(operation_params.get("power_supply_type", "thyristor")).lower()
        duty = float(operation_params.get("pulse_duty_cycle", 1.0))
        duty = max(min(duty, 1.0), 0.0)

        if supply_type == "igbt":
            supply_eff = float(operation_params.get("power_supply_efficiency", 0.95))
        elif supply_type == "pulsed":
            supply_eff = float(operation_params.get("power_supply_efficiency", 0.92))
        else:
            supply_eff = float(operation_params.get("power_supply_efficiency", 0.90))

        E_flash_J = float(operation_params.get("flashover_energy_J", 0.0))
        flash_freq = float(operation_params.get("flashover_frequency_Hz", 0.0))
        P_flash_avg_W = E_flash_J * flash_freq

        P_avg_W = P_instant_W * duty + P_flash_avg_W

        # grid power consumed accounting for supply losses
        P_grid_W = P_avg_W / max(supply_eff, 1e-6)
        power_kW = float(P_grid_W / 1000.0)

        return {
            "E_avg_V_m": E_avg,
            "Jp_A_m2": Jp,
            "I_total_A": float(I_total),
            "I_per_wire_A": float(I_per_wire),
            "power_elec_kW": power_kW,
            "power_instant_kW": float(P_instant_W / 1000.0),
            "supply_efficiency": supply_eff,
            "duty_cycle": duty,
            "flashover_power_kW": float(P_flash_avg_W / 1000.0),
            "supply_type": supply_type,
        }

    

    def _combined_efficiency(
            self,
            flue_gas: FlueGasState,
            operation_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        'Execute the combined efficiency calculation.'
        operation_params = operation_params or {}

        
        pm_inlet = float(
            operation_params.get(
                "pm_inlet_concentration",
                flue_gas.get_pollutant("PM", 2000.0) or 2000.0,
            )
        )

        
        particle_diameter_nm = float(
            operation_params.get("particle_diameter_nm", 50.0)
        )

        
        NDe = self._calculate_nde(flue_gas, operation_params)

        
        try:
            rho = ThermoCore.gas_density(
                flue_gas.temperature, flue_gas.pressure, flue_gas.composition
            )
        except Exception:
            rho = 1.2

        cross_section_m2 = max(
            self.electrode_area_m2 / max(self.number_of_wires, 1), 1.0
        )
        u_avg = max(flue_gas.mass_flow / (rho * cross_section_m2), 0.01)
        residence_time = max(self.collection_plate_length_m / u_avg, 0.01)

        
        ion_concentration = float(
            operation_params.get("ion_concentration_m3", 1e14)
        )
        V_operating = float(operation_params.get("operating_voltage", self.operating_voltage_kV))
        E_avg = (V_operating * 1000.0) / max(self.electrode_spacing_m, 1e-3)

        
        n_p_t = self._calculate_particle_charge_lawless(
            particle_diameter_nm,
            ion_concentration,
            residence_time,
            E_avg,
        )
        alpha = self._partial_charging_factor(n_p_t)

        
        eta_lin = self._lin_collection_efficiency(NDe, alpha, particle_diameter_nm)

        
        f_T = self._temperature_factor(flue_gas.temperature)
        f_L = self._load_factor_correction(float(operation_params.get("load_factor", 1.0)))
        f_V = self._voltage_factor(operation_params)
        f_C = self._concentration_factor(pm_inlet)
        f_rev = self._reverse_corona_factor(pm_inlet)

        
        eta = eta_lin * f_T * f_L * f_V * f_C * f_rev

        
        eta = max(min(eta, 0.99), 0.10)

        return eta

    

    @staticmethod
    def _h2o_mol_fraction(flue_gas: FlueGasState) -> float:
        return float(flue_gas.composition.get("H2O", 0.0) or 0.0)

    def _generate_constraints(
            self,
            flue_gas_in: FlueGasState,
            pm_inlet: float,
            flue_gas_out: Optional[FlueGasState] = None  
    ) -> List[ConstraintReport]:
        'Execute the generate constraints calculation.'
        constraints: List[ConstraintReport] = []

        
        check_temp = flue_gas_out.temperature if flue_gas_out else flue_gas_in.temperature

        
        constraints.append(
            ConstraintReport.range_bound(
                name=f"{self.name}_temperature_window_safe",
                value=check_temp,
                lower=self.min_temperature,
                upper=self.max_temperature,
                message=f"ESP outlet temperature {check_temp:.2f}K outside window [{self.min_temperature:.2f}K, {self.max_temperature:.2f}K]",
            )
        )

        
        h2o_fraction = self._h2o_mol_fraction(flue_gas_in)
        if self.min_h2o_mol_fraction is not None:
            constraints.append(
                ConstraintReport.lower_bound(
                    name=f"{self.name}_minimum_h2o_fraction",
                    value=h2o_fraction,
                    limit=self.min_h2o_mol_fraction,
                    message=(
                        f"{self.name} inlet H2O fraction {h2o_fraction:.3f} below "
                        f"minimum {self.min_h2o_mol_fraction:.3f}."
                    ),
                )
            )
        if self.max_h2o_mol_fraction is not None:
            constraints.append(
                ConstraintReport.upper_bound(
                    name=f"{self.name}_maximum_h2o_fraction",
                    value=h2o_fraction,
                    limit=self.max_h2o_mol_fraction,
                    message=(
                        f"{self.name} inlet H2O fraction {h2o_fraction:.3f} above "
                        f"maximum {self.max_h2o_mol_fraction:.3f}."
                    ),
                )
            )

        if pm_inlet > self.REVERSE_CORONA_SEVERE:
            constraints.append(
                ConstraintReport.upper_bound(
                    name=f"{self.name}_reverse_corona_critical",
                    value=pm_inlet,
                    limit=self.REVERSE_CORONA_SEVERE,
                    message=f"PM {pm_inlet:.0f} mg/Nm³ > severe limit {self.REVERSE_CORONA_SEVERE:.0f}. Efficiency compromised.",
                )
            )
        elif pm_inlet > self.REVERSE_CORONA_THRESHOLD:
            constraints.append(
                ConstraintReport.upper_bound(
                    name=f"{self.name}_reverse_corona_warning",
                    value=pm_inlet,
                    limit=self.REVERSE_CORONA_THRESHOLD,
                    message=f"PM {pm_inlet:.0f} mg/Nm³ in reverse corona region. Monitor.",
                )
            )

        return constraints

    

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

        
        pm_inlet_concentration = float(
            operation_params.get(
                "pm_inlet_concentration",
                fg_in.get_pollutant("PM", 2000.0) or 2000.0,
            )
        )

        
        eta = self._combined_efficiency(fg_in, operation_params)

        
        pm_removed = fg_out.update_pollutant_by_efficiency("PM", eta)
        fg_out.pressure = max(fg_in.pressure - self.pressure_drop / 1000.0, 1.0)
        fg_out.temperature = max(fg_in.temperature - self.temperature_drop, 273.15)

        
        try:
            Ex_in = ThermoCore.flue_gas_physical_exergy(
                fg_in.mass_flow,
                fg_in.temperature,
                fg_in.pressure,
                fg_in.composition,
            )
            Ex_out = ThermoCore.flue_gas_physical_exergy(
                fg_out.mass_flow,
                fg_out.temperature,
                fg_out.pressure,
                fg_out.composition,
            )
            exergy_loss_pressure = max(Ex_in - Ex_out, 0.0)
        except Exception:
            exergy_loss_pressure = 0.0

        
        vi = self._total_current_and_power(fg_in, operation_params)
        electrical_field_power_kW = vi["power_elec_kW"] if vi is not None else float(self.electricity_power)

        # total grid electricity (kW) = electrical field (computed) + fan
        total_electricity_power = electrical_field_power_kW + float(self.fan_power)

        # exergy destruction: pressure loss (kW) + electricity (kW)
        exergy_destruction = exergy_loss_pressure + total_electricity_power

        electricity_price = float(operation_params.get("electricity_price", 0.0))
        grid_carbon_factor = float(operation_params.get("grid_carbon_factor", 0.0))

        cost = total_electricity_power * electricity_price
        carbon_emission = total_electricity_power * grid_carbon_factor

        
        constraints = self._generate_constraints(fg_in, pm_inlet_concentration, fg_out)

        pm_outlet_concentration = pm_inlet_concentration * (1.0 - eta)

        
        NDe = self._calculate_nde(fg_in, operation_params)
        NRe = self._calculate_nre(fg_in)
        f_T = self._temperature_factor(fg_in.temperature)
        f_L = self._load_factor_correction(operation_params.get("load_factor", 1.0))
        f_V = self._voltage_factor(operation_params)
        f_C = self._concentration_factor(pm_inlet_concentration)
        f_rev = self._reverse_corona_factor(pm_inlet_concentration)

        messages = [
            f"{self.name}: Model=Lin2012_NDe, Inlet={pm_inlet_concentration:.0f} mg/Nm³, "
            f"Outlet={pm_outlet_concentration:.0f} mg/Nm³, Efficiency={eta:.4f}.",
            f"  Dimensionless: NDe={NDe:.3f}, NRe={NRe:.0f}",
            f"  Factors: f_T={f_T:.3f}, f_L={f_L:.3f}, f_V={f_V:.3f}, f_C={f_C:.3f}, f_rev={f_rev:.3f}",
        ]

        # add electrical diagnostics to metadata ( added keys)
        metadata_extra = {
            "electrical_field_power_kW": electrical_field_power_kW,
            "electrical_power_instant_kW": vi.get("power_instant_kW", None) if vi else None,
            "I_total_A": vi.get("I_total_A", None) if vi else None,
            "I_per_wire_A": vi.get("I_per_wire_A", None) if vi else None,
            "Jp_A_m2": vi.get("Jp_A_m2", None) if vi else None,
            "E_avg_V_m": vi.get("E_avg_V_m", None) if vi else None,
            "supply_type": vi.get("supply_type", None) if vi else None,
            "supply_efficiency": vi.get("supply_efficiency", None) if vi else None,
            "flashover_power_kW": vi.get("flashover_power_kW", None) if vi else None,
            "pulse_duty_cycle": vi.get("duty_cycle", None) if vi else None,
        }

        result = ModuleResult(
            module_name=self.name,
            flue_gas_in=fg_in,
            flue_gas_out=fg_out,
            removal_efficiency={"PM": eta},
            removed_pollutants={"PM": pm_removed},
            energy_consumption={
                "electricity_total_kW": total_electricity_power,
                "electrical_field_kW": self.electricity_power,
                "fan_power_kW": self.fan_power,
            },
            material_consumption={},
            exergy_destruction=exergy_destruction,
            cost=cost,
            carbon_emission=carbon_emission,
            constraints=constraints,
            messages=messages,
            metadata={
                "pm_inlet_mg_Nm3": pm_inlet_concentration,
                "pm_outlet_mg_Nm3": pm_outlet_concentration,
                "pm_removed_mg_Nm3": pm_removed,
                "efficiency": eta,
                "NDe": NDe,
                "NRe": NRe,
                "temperature_factor": f_T,
                "load_factor": operation_params.get("load_factor", 1.0),
                "load_factor_correction": f_L,
                "voltage_factor": f_V,
                "concentration_factor": f_C,
                "reverse_corona_factor": f_rev,
                #  include electrical diagnostics in metadata
                **metadata_extra,
            },
        )

        result.update_feasibility()
        return result


class WESP(ESP):
    """Wet electrostatic precipitator used for wet, low-temperature polishing."""

    def __init__(
            self,
            name: str = "WESP",
            enabled: bool = True,
            base_efficiency: float = 0.995,
            operating_voltage_kV: float = 45.0,
            min_temperature: float = 313.15,
            max_temperature: float = 373.15,
            optimal_temperature: float = 333.15,
            temperature_sensitivity: float = 0.0005,
            electricity_power: float = 650.0,
            fan_power: float = 90.0,
            water_pump_power: float = 45.0,
            pressure_drop: float = 250.0,
            temperature_drop: float = 1.0,
            spray_water_kg_h: float = 2500.0,
            min_h2o_mol_fraction: float = 0.08,
            water_price_CNY_m3: float = 2.2,
            water_carbon_kg_m3: float = 0.211,
            parameters: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            base_efficiency=base_efficiency,
            operating_voltage_kV=operating_voltage_kV,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            optimal_temperature=optimal_temperature,
            temperature_sensitivity=temperature_sensitivity,
            electricity_power=electricity_power,
            fan_power=fan_power,
            pressure_drop=pressure_drop,
            temperature_drop=temperature_drop,
            min_h2o_mol_fraction=min_h2o_mol_fraction,
            max_h2o_mol_fraction=None,
            parameters=parameters,
            **kwargs,
        )
        self.module_type = "wesp"
        self.water_pump_power = float(water_pump_power)
        self.spray_water_kg_h = float(spray_water_kg_h)
        self.water_price_CNY_m3 = float(water_price_CNY_m3)
        self.water_carbon_kg_m3 = float(water_carbon_kg_m3)

    def _combined_efficiency(
            self,
            flue_gas: FlueGasState,
            operation_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        operation_params = operation_params or {}
        eta_dry_model = super()._combined_efficiency(flue_gas, operation_params)
        h2o_fraction = self._h2o_mol_fraction(flue_gas)
        humidity_boost = 1.0 + 1.2 * max(h2o_fraction - self.min_h2o_mol_fraction, 0.0)
        wet_boost = float(operation_params.get("wet_collection_boost", 1.12))
        floor_eta = float(operation_params.get("wesp_floor_efficiency", self.base_efficiency * 0.985))
        eta = max(eta_dry_model * wet_boost * humidity_boost, floor_eta)
        return max(min(eta, 0.995), 0.30)

    def evaluate(
            self,
            flue_gas_in: FlueGasState,
            energy_inputs: Optional[List[EnergyStream]] = None,
            material_inputs: Optional[List[MaterialInput]] = None,
            operation_params: Optional[Dict[str, Any]] = None,
    ) -> ModuleResult:
        operation_params = operation_params or {}
        result = super().evaluate(
            flue_gas_in=flue_gas_in,
            energy_inputs=energy_inputs,
            material_inputs=material_inputs,
            operation_params=operation_params,
        )

        water_flow = float(operation_params.get("spray_water_kg_h", self.spray_water_kg_h))
        water_pump_power = float(operation_params.get("water_pump_power", self.water_pump_power))
        electricity_price = float(operation_params.get("electricity_price", 0.0))
        grid_carbon_factor = float(operation_params.get("grid_carbon_factor", 0.0))
        water_m3_h = max(water_flow, 0.0) / 1000.0

        result.energy_consumption["water_pump_power_kW"] = water_pump_power
        result.energy_consumption["electricity_total_kW"] = (
            result.energy_consumption.get("electricity_total_kW", 0.0)
            + water_pump_power
        )
        result.material_consumption["spray_water_kg_h"] = water_flow
        result.exergy_destruction += water_pump_power
        result.cost += water_pump_power * electricity_price + water_m3_h * self.water_price_CNY_m3
        result.carbon_emission += water_pump_power * grid_carbon_factor + water_m3_h * self.water_carbon_kg_m3
        result.metadata.update(
            {
                "wet_polishing": True,
                "spray_water_kg_h": water_flow,
                "water_pump_power_kW": water_pump_power,
                "inlet_h2o_mol_fraction": self._h2o_mol_fraction(flue_gas_in),
            }
        )
        result.messages.append(
            f"{self.name}: wet ESP polishing with spray water {water_flow:.0f} kg/h."
        )
        result.update_feasibility()
        return result

