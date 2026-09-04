# steel_whun/core/thermo.py

from __future__ import annotations

from typing import Dict, Optional, Tuple
import math
import numpy as np


class ThermoCore:
    'ThermoCore component used by the Steel-WHUN core framework.'

    T0 = 298.15
    P0 = 101325.0
    R_u = 8.314  # J/(mol K)

    DEFAULT_CP_GAS = 1.05  # kJ/(kg K)

    MOLAR_MASS = {
        "N2": 28.01,
        "O2": 32.00,
        "CO2": 44.01,
        "H2O": 18.02,
        "CO": 28.01,
        "H2": 2.016,
        "CH4": 16.04,
        "SO2": 64.06,
        "NO": 30.01,
        "NO2": 46.01,
        "LDG": 29.0,
        "BFG": 30.0,
        "COG": 12.0,
    }

    @staticmethod
    def _normalize_name(gas_name: str) -> str:
        return gas_name.strip().upper()

    @staticmethod
    def calc_cp_component(gas_name: str, T: float) -> float:
        'Execute the calc cp component calculation.'
        if T <= 0:
            return ThermoCore.DEFAULT_CP_GAS

        t = T / 1000.0
        name = ThermoCore._normalize_name(gas_name)

        A = B = C = D = E = 0.0

        if name == "N2":
            if 100.0 <= T <= 500.0:
                A, B, C, D, E = 28.98641, 1.853978, -9.647459, 16.63537, 0.000117
            else:
                A, B, C, D, E = 19.50583, 19.88705, -8.598535, 1.369784, 0.527601

        elif name == "O2":
            if 100.0 <= T <= 700.0:
                A, B, C, D, E = 31.32234, -20.23531, 57.86644, -36.50624, -0.007374
            else:
                A, B, C, D, E = 30.03235, 8.772972, -3.988133, 0.788313, -0.741599

        elif name == "CO2":
            if 298.0 <= T <= 1200.0:
                A, B, C, D, E = 24.99735, 55.18696, -33.69137, 7.948387, -0.136638
            else:
                A, B, C, D, E = 58.16639, 2.720074, -0.492289, 0.038844, -6.447293

        elif name == "H2O":
            if 500.0 < T <= 1700.0:
                A, B, C, D, E = 30.09200, 6.832514, 6.793435, -2.534480, 0.082139
            else:
                A, B, C, D, E = -203.6060, 1523.290, -3196.413, 2474.455, 3.855326

        elif name == "CO":
            if 298.0 <= T <= 1300.0:
                A, B, C, D, E = 25.56759, 6.096130, 4.054656, -2.671301, 0.131021
            else:
                A, B, C, D, E = 35.15070, 1.300095, -0.205921, 0.013550, -3.282780

        elif name == "H2":
            if 298.0 <= T <= 1000.0:
                A, B, C, D, E = 33.066178, -11.363417, 11.432816, -2.772874, -0.158558
            else:
                A, B, C, D, E = 18.563083, 12.257357, -2.859786, 0.268238, 1.977990

        elif name == "CH4":
            if 298.0 <= T <= 1300.0:
                A, B, C, D, E = -0.703029, 108.4773, -42.52157, 5.862788, 0.678565
            else:
                A, B, C, D, E = 85.81217, 11.26467, -2.114146, 0.138190, -26.42221

        else:
            return ThermoCore.DEFAULT_CP_GAS

        Cp_molar = A + B * t + C * t ** 2 + D * t ** 3 + E / (t ** 2)

        molar_mass = ThermoCore.MOLAR_MASS.get(name, 29.0)

        cp_mass = Cp_molar / molar_mass

        if not np.isfinite(cp_mass) or cp_mass <= 0:
            return ThermoCore.DEFAULT_CP_GAS

        return float(cp_mass)

    @staticmethod
    def calc_mixture_properties(
        composition_vol: Optional[Dict[str, float]],
        T: float,
        P: float = 101325.0,
    ) -> Tuple[float, float]:
        'Execute the calc mixture properties calculation.'
        if not composition_vol:
            return ThermoCore.DEFAULT_CP_GAS, 1.2

        total = sum(max(v, 0.0) for v in composition_vol.values())
        if total <= 0:
            return ThermoCore.DEFAULT_CP_GAS, 1.2

        
        P_pa = P * 1000.0 if P < 2000.0 else P

        M_mix = 0.0

        for gas, value in composition_vol.items():
            y_i = max(value, 0.0) / total
            M_i = ThermoCore.MOLAR_MASS.get(ThermoCore._normalize_name(gas), 29.0)
            M_mix += y_i * M_i

        if M_mix <= 0:
            M_mix = 29.0

        cp_mix = 0.0

        for gas, value in composition_vol.items():
            y_i = max(value, 0.0) / total
            name = ThermoCore._normalize_name(gas)
            M_i = ThermoCore.MOLAR_MASS.get(name, 29.0)

            
            w_i = y_i * M_i / M_mix

            cp_i = ThermoCore.calc_cp_component(name, T)
            cp_mix += w_i * cp_i

        
        # M_mix: g/mol -> kg/mol
        rho_mix = P_pa * (M_mix / 1000.0) / (ThermoCore.R_u * T)

        if not np.isfinite(cp_mix) or cp_mix <= 0:
            cp_mix = ThermoCore.DEFAULT_CP_GAS

        if not np.isfinite(rho_mix) or rho_mix <= 0:
            rho_mix = 1.2

        return float(cp_mix), float(rho_mix)

    @classmethod
    def gas_cp(
        cls,
        temperature: float,
        composition: Optional[Dict[str, float]] = None,
    ) -> float:
        'Execute the gas cp calculation.'
        cp, _ = cls.calc_mixture_properties(
            composition_vol=composition,
            T=temperature,
            P=cls.P0,
        )
        return cp

    @staticmethod
    def calc_physical_exergy(
        mass_flow: float,
        cp: float,
        T: float,
        T0: Optional[float] = None,
    ) -> float:
        'Execute the calc physical exergy calculation.'
        if T0 is None:
            T0 = ThermoCore.T0

        if mass_flow <= 0 or cp <= 0 or T <= 0:
            return 0.0

        if abs(T - T0) < 1e-9:
            return 0.0

        specific_exergy = cp * ((T - T0) - T0 * math.log(T / T0))

        return float(max(mass_flow * specific_exergy, 0.0))

    @classmethod
    def flue_gas_physical_exergy(
        cls,
        mass_flow: float,
        temperature: float,
        pressure: float,
        composition: Optional[Dict[str, float]] = None,
        ambient_temperature: Optional[float] = None,
        ambient_pressure: Optional[float] = None,
    ) -> float:
        'Execute the flue gas physical exergy calculation.'
        if ambient_temperature is None:
            ambient_temperature = cls.T0

        cp = cls.gas_cp(temperature, composition)

        return cls.calc_physical_exergy(
            mass_flow=mass_flow,
            cp=cp,
            T=temperature,
            T0=ambient_temperature,
        )

    @classmethod
    def physical_exergy_heat(
        cls,
        heat_rate: float,
        source_temperature: float,
        ambient_temperature: Optional[float] = None,
    ) -> float:
        'Execute the physical exergy heat calculation.'
        if ambient_temperature is None:
            ambient_temperature = cls.T0

        if heat_rate <= 0:
            return 0.0

        if source_temperature <= ambient_temperature:
            return 0.0

        return float(heat_rate * (1.0 - ambient_temperature / source_temperature))

    @staticmethod
    def calc_chemical_exergy(
        dispatch_amount: float,
        LHV: float,
        gamma: float,
    ) -> float:
        'Execute the calc chemical exergy calculation.'
        return float(max(dispatch_amount, 0.0) * max(LHV, 0.0) * max(gamma, 0.0))

    @classmethod
    def temperature_change_by_heat(
        cls,
        heat_rate: float,
        mass_flow: float,
        temperature: float,
        composition: Optional[Dict[str, float]] = None,
    ) -> float:
        'Execute the temperature change by heat calculation.'
        if mass_flow <= 0:
            return 0.0

        cp = cls.gas_cp(temperature, composition)

        if cp <= 0:
            return 0.0

        return float(heat_rate / (mass_flow * cp))

    @classmethod
    def heat_required_for_temperature_change(
        cls,
        mass_flow: float,
        T_in: float,
        T_out: float,
        composition: Optional[Dict[str, float]] = None,
    ) -> float:
        'Execute the heat required for temperature change calculation.'
        if mass_flow <= 0:
            return 0.0

        if T_out <= T_in:
            return 0.0

        T_mean = 0.5 * (T_in + T_out)
        cp = cls.gas_cp(T_mean, composition)

        return float(mass_flow * cp * (T_out - T_in))

    @classmethod
    def electricity_exergy(cls, power_rate: float) -> float:
        'Execute the electricity exergy calculation.'
        return float(max(power_rate, 0.0))

    @staticmethod
    def gas_density(temperature: float, pressure: float, composition: Optional[Dict[str, float]] = None) -> float:
        'Execute the gas density calculation.'
        _, rho_mix = ThermoCore.calc_mixture_properties(composition_vol=composition, T=temperature, P=pressure)
        return float(rho_mix)

    @staticmethod
    def gas_viscosity(temperature: float, composition: Optional[Dict[str, float]] = None) -> float:
        'Execute the gas viscosity calculation.'
        if temperature <= 0:
            return 1.81e-5
        
        T_ref = 273.15
        mu_ref = 1.716e-5
        S = 111.0

        mu = mu_ref * ((temperature / T_ref) ** 1.5) * ((T_ref + S) / (temperature + S))
        return float(max(mu, 1e-7))
