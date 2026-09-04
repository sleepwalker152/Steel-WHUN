# steel_whun/modules/backend_utilization.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from steel_whun.core.states import (
    ConstraintReport,
    EnergyStream,
    FlueGasState,
    MaterialInput,
    ModuleResult,
)
from steel_whun.core.thermo import ThermoCore
from steel_whun.modules.base import PollutionModule


BASE_OPERATION_HOURS_PER_YEAR = 4000.0  # configurable default
NATURAL_GAS_CARBON_FACTOR_KG_PER_KWH = 0.0  # supplied by user configuration
ELECTRIC_CHILLER_COP = 3.5  # configurable default


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


class BackendUtilizationModule(PollutionModule):
    """Backend energy utilization asset evaluated by the system evaluator.

    These modules behave like CCPP in the framework: they receive dispatched
    ``EnergyStream`` objects from the network, return a ``ModuleResult``, and
    are merged into system accounting totals by ``SystemEvaluator``. They do
    not mutate the main flue gas treatment state.
    """

    def __init__(
        self,
        *,
        name: str,
        module_type: str,
        asset_type: str,
        max_heat_input_kw: float,
        annual_service_hours: float,
        useful_energy_efficiency: float,
        capex_per_kw: float,
        fixed_om_fraction: float,
        variable_om_CNY_per_kWh: float,
        revenue_CNY_per_kWh: float,
        output_temperature_K: Optional[float] = None,
        carbon_mode: str = "none",
        asset_metadata: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        parameters: Optional[Dict[str, Any]] = None,
        **model_parameters: Any,
    ):
        merged_parameters = {
            **dict(model_parameters),
            **dict(parameters or {}),
        }
        super().__init__(
            name=name,
            module_type=module_type,
            enabled=enabled,
            parameters=merged_parameters,
        )
        self.asset_type = asset_type
        self.max_heat_input_kw = max_heat_input_kw
        self.annual_service_hours = annual_service_hours
        self.useful_energy_efficiency = useful_energy_efficiency
        self.capex_per_kw = capex_per_kw
        self.fixed_om_fraction = fixed_om_fraction
        self.variable_om_CNY_per_kWh = variable_om_CNY_per_kWh
        self.revenue_CNY_per_kWh = revenue_CNY_per_kWh
        self.output_temperature_K = output_temperature_K
        self.carbon_mode = carbon_mode
        self.asset_metadata = asset_metadata or {}

    @staticmethod
    def _crf(discount_rate: float, lifetime_years: float) -> float:
        if lifetime_years <= 0.0:
            return 0.0
        if abs(discount_rate) < 1e-12:
            return 1.0 / lifetime_years
        factor = (1.0 + discount_rate) ** lifetime_years
        return discount_rate * factor / (factor - 1.0)

    def _runtime_params(self, operation_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        op = operation_params or {}
        data: Dict[str, Any] = {
            "asset_type": self.asset_type,
            "max_heat_input_kw": self.max_heat_input_kw,
            "annual_service_hours": self.annual_service_hours,
            "useful_energy_efficiency": self.useful_energy_efficiency,
            "capex_per_kw": self.capex_per_kw,
            "fixed_om_fraction": self.fixed_om_fraction,
            "variable_om_CNY_per_kWh": self.variable_om_CNY_per_kWh,
            "revenue_CNY_per_kWh": self.revenue_CNY_per_kWh,
            "output_temperature_K": self.output_temperature_K,
            "carbon_mode": self.carbon_mode,
            "discount_rate": 0.08,
            "lifetime_years": 20.0,
            "grid_carbon_factor_kg_kWh": 0.0,
            "gas_boiler_efficiency": 0.90,
            "natural_gas_carbon_factor_kg_kWh": NATURAL_GAS_CARBON_FACTOR_KG_PER_KWH,
            "electric_chiller_COP": ELECTRIC_CHILLER_COP,
            "electricity_price_CNY_per_kWh": 0.0,
            "ambient_temperature_K": ThermoCore.T0,
            "minimum_useful_output_kw": 0.0,
            "enforce_min_part_load": False,
        }
        data.update(self.parameters or {})
        for key, value in op.items():
            if key in {"asset_metadata", "metadata"}:
                continue
            if value is not None:
                data[key] = value
        data["asset_metadata"] = {
            **self.asset_metadata,
            **dict(op.get("asset_metadata", {})),
            **dict(op.get("metadata", {})),
        }
        return data

    @staticmethod
    def _weighted_stream_temperature(
        streams: List[EnergyStream],
        fallback: Optional[float] = None,
    ) -> Optional[float]:
        weighted = 0.0
        total = 0.0
        for stream in streams:
            if stream.temperature is None:
                continue
            energy = max(stream.energy_rate, 0.0)
            if energy <= 0.0:
                continue
            weighted += stream.temperature * energy
            total += energy
        if total <= 0.0:
            return fallback
        return weighted / total

    def _source_temperature(
        self,
        streams: List[EnergyStream],
        params: Dict[str, Any],
        fallback: float,
    ) -> float:
        explicit = params.get("source_temperature_K", params.get("heat_source_temperature_K"))
        if explicit is not None:
            return float(explicit)
        weighted = self._weighted_stream_temperature(streams)
        if weighted is not None:
            return float(weighted)
        return fallback

    def _resource_inputs(
        self,
        streams: List[EnergyStream],
        params: Dict[str, Any],
    ) -> Tuple[float, float, float, float, float, float, float]:
        heat_input_kw = self.total_energy_input(streams)
        exergy_input_kw = self.total_exergy_input(streams)
        max_heat_input_kw = max(float(params["max_heat_input_kw"]), 0.0)
        heat_used_kw = min(heat_input_kw, max_heat_input_kw)
        overflow_kw = max(heat_input_kw - max_heat_input_kw, 0.0)
        used_fraction = heat_used_kw / heat_input_kw if heat_input_kw > 0.0 else 0.0
        exergy_used_kw = exergy_input_kw * used_fraction
        overflow_exergy_kw = exergy_input_kw * (1.0 - used_fraction)
        return (
            heat_input_kw,
            exergy_input_kw,
            max_heat_input_kw,
            heat_used_kw,
            overflow_kw,
            exergy_used_kw,
            overflow_exergy_kw,
        )

    @staticmethod
    def _target_output_exergy(
        asset_type: str,
        useful_energy_kw: float,
        output_temperature_K: Optional[float],
    ) -> float:
        if useful_energy_kw <= 0.0:
            return 0.0
        if asset_type == "orc_power":
            return useful_energy_kw
        if asset_type == "absorption_cooling":
            cold_temperature = output_temperature_K or 280.15
            return useful_energy_kw * max(ThermoCore.T0 / cold_temperature - 1.0, 0.0)
        output_temperature = output_temperature_K or 333.15
        return ThermoCore.physical_exergy_heat(useful_energy_kw, output_temperature)

    @staticmethod
    def _avoided_carbon(params: Dict[str, Any], useful_energy_kw: float) -> float:
        annual_useful_energy_kWh = useful_energy_kw * float(params["annual_service_hours"])
        carbon_mode = str(params["carbon_mode"])
        if carbon_mode == "grid_electricity":
            annual_avoided = annual_useful_energy_kWh * float(
                params["grid_carbon_factor_kg_kWh"]
            )
        elif carbon_mode == "cooling_grid_displacement":
            electric_kWh = annual_useful_energy_kWh / max(
                float(params["electric_chiller_COP"]),
                1e-9,
            )
            annual_avoided = electric_kWh * float(params["grid_carbon_factor_kg_kWh"])
        elif carbon_mode == "gas_heat":
            boiler_eff = max(float(params["gas_boiler_efficiency"]), 1e-9)
            annual_gas_input_kWh = annual_useful_energy_kWh / boiler_eff
            annual_avoided = annual_gas_input_kWh * float(
                params["natural_gas_carbon_factor_kg_kWh"]
            )
        else:
            annual_avoided = 0.0
        return annual_avoided / BASE_OPERATION_HOURS_PER_YEAR

    def _economic_cost(
        self,
        params: Dict[str, Any],
        heat_used_kw: float,
        useful_energy_output_kw: float,
        auxiliary_power_kw: float = 0.0,
    ) -> float:
        crf = self._crf(
            discount_rate=float(params["discount_rate"]),
            lifetime_years=float(params["lifetime_years"]),
        )
        capex_basis_kw = max(
            heat_used_kw,
            float(params.get("installed_capacity_kw", 0.0)),
            float(params.get("minimum_capex_basis_kw", 0.0)),
        )
        annual_capital_cost = capex_basis_kw * float(params["capex_per_kw"]) * crf
        annual_fixed_om = (
            capex_basis_kw
            * float(params["capex_per_kw"])
            * float(params["fixed_om_fraction"])
        )
        annual_variable_om = (
            useful_energy_output_kw
            * float(params["annual_service_hours"])
            * float(params["variable_om_CNY_per_kWh"])
        )
        annual_aux_power_cost = (
            auxiliary_power_kw
            * float(params["annual_service_hours"])
            * float(params["electricity_price_CNY_per_kWh"])
        )
        annual_revenue = (
            useful_energy_output_kw
            * float(params["annual_service_hours"])
            * float(params["revenue_CNY_per_kWh"])
        )
        return (
            annual_capital_cost
            + annual_fixed_om
            + annual_variable_om
            + annual_aux_power_cost
            - annual_revenue
        ) / BASE_OPERATION_HOURS_PER_YEAR

    def _net_carbon(
        self,
        params: Dict[str, Any],
        useful_energy_output_kw: float,
        auxiliary_power_kw: float = 0.0,
    ) -> Tuple[float, float, float]:
        avoided_carbon_kg_per_h = self._avoided_carbon(
            params=params,
            useful_energy_kw=useful_energy_output_kw,
        )
        aux_carbon_kg_per_h = (
            auxiliary_power_kw
            * float(params["annual_service_hours"])
            * float(params["grid_carbon_factor_kg_kWh"])
            / BASE_OPERATION_HOURS_PER_YEAR
        )
        return (
            aux_carbon_kg_per_h - avoided_carbon_kg_per_h,
            avoided_carbon_kg_per_h,
            aux_carbon_kg_per_h,
        )

    def _capacity_constraint(
        self,
        heat_input_kw: float,
        max_heat_input_kw: float,
    ) -> ConstraintReport:
        return ConstraintReport.upper_bound(
            name=f"{self.name}_heat_input_capacity",
            value=heat_input_kw,
            limit=max_heat_input_kw,
            message=f"{self.name} receives more heat than installed capacity.",
        )

    def _build_result(
        self,
        flue_gas_in: FlueGasState,
        params: Dict[str, Any],
        *,
        heat_input_kw: float,
        heat_used_kw: float,
        useful_energy_output_kw: float,
        useful_exergy_output_kw: float,
        conversion_exergy_destruction_kw: float,
        overflow_kw: float,
        net_cost_CNY_per_h: float,
        carbon_emission_kg_per_h: float,
        avoided_carbon_kg_per_h: float,
        constraints: List[ConstraintReport],
        metadata: Dict[str, Any],
        auxiliary_power_kw: float = 0.0,
        messages: Optional[List[str]] = None,
    ) -> ModuleResult:
        base_metadata = {
            "target_id": self.name,
            "asset_type": str(params["asset_type"]),
            "heat_input_kw": heat_input_kw,
            "heat_used_kw": heat_used_kw,
            "useful_energy_output_kw": useful_energy_output_kw,
            "useful_exergy_output_kw": useful_exergy_output_kw,
            "conversion_exergy_destruction_kw": conversion_exergy_destruction_kw,
            "overflow_kw": overflow_kw,
            "net_cost_CNY_per_h": net_cost_CNY_per_h,
            "avoided_carbon_kg_per_h": avoided_carbon_kg_per_h,
            "annual_service_hours": float(params["annual_service_hours"]),
            "max_heat_input_kw": float(params["max_heat_input_kw"]),
            "auxiliary_power_kw": auxiliary_power_kw,
            **dict(params.get("asset_metadata", {})),
            **metadata,
        }
        result = ModuleResult(
            module_name=self.name,
            flue_gas_in=flue_gas_in.copy(),
            flue_gas_out=flue_gas_in.copy(),
            removal_efficiency={},
            removed_pollutants={},
            energy_consumption={
                "heat_input_kW": heat_input_kw,
                "heat_used_kW": heat_used_kw,
                "useful_energy_output_kW": useful_energy_output_kw,
                "useful_exergy_output_kW": useful_exergy_output_kw,
                "overflow_kW": overflow_kw,
                "auxiliary_power_kW": auxiliary_power_kw,
            },
            material_consumption={},
            exergy_destruction=conversion_exergy_destruction_kw,
            cost=net_cost_CNY_per_h,
            carbon_emission=carbon_emission_kg_per_h,
            constraints=constraints,
            messages=messages
            or [
                (
                    f"{self.name}: input={heat_input_kw:.1f} kW, "
                    f"useful={useful_energy_output_kw:.1f} kW, "
                    f"overflow={overflow_kw:.1f} kW."
                )
            ],
            metadata=base_metadata,
        )
        result.update_feasibility()
        if useful_energy_output_kw < float(params.get("minimum_useful_output_kw", 0.0)):
            result.feasible = False
        return result

    def evaluate(
        self,
        flue_gas_in: FlueGasState,
        energy_inputs: Optional[List[EnergyStream]] = None,
        material_inputs: Optional[List[MaterialInput]] = None,
        operation_params: Optional[Dict[str, Any]] = None,
    ) -> ModuleResult:
        if not self.enabled:
            return self.bypass_result(flue_gas_in)

        params = self._runtime_params(operation_params)
        streams = energy_inputs or []
        (
            heat_input_kw,
            _,
            max_heat_input_kw,
            heat_used_kw,
            overflow_kw,
            exergy_used_kw,
            overflow_exergy_kw,
        ) = self._resource_inputs(streams, params)

        useful_energy_output_kw = heat_used_kw * max(
            float(params["useful_energy_efficiency"]),
            0.0,
        )
        useful_exergy_output_kw = self._target_output_exergy(
            asset_type=str(params["asset_type"]),
            useful_energy_kw=useful_energy_output_kw,
            output_temperature_K=params.get("output_temperature_K"),
        )
        conversion_exergy_destruction_kw = (
            max(exergy_used_kw - useful_exergy_output_kw, 0.0) + overflow_exergy_kw
        )
        net_cost_CNY_per_h = self._economic_cost(
            params=params,
            heat_used_kw=heat_used_kw,
            useful_energy_output_kw=useful_energy_output_kw,
        )
        carbon_emission, avoided_carbon, _ = self._net_carbon(
            params=params,
            useful_energy_output_kw=useful_energy_output_kw,
        )
        constraints = [self._capacity_constraint(heat_input_kw, max_heat_input_kw)]
        return self._build_result(
            flue_gas_in,
            params,
            heat_input_kw=heat_input_kw,
            heat_used_kw=heat_used_kw,
            useful_energy_output_kw=useful_energy_output_kw,
            useful_exergy_output_kw=useful_exergy_output_kw,
            conversion_exergy_destruction_kw=conversion_exergy_destruction_kw,
            overflow_kw=overflow_kw,
            net_cost_CNY_per_h=net_cost_CNY_per_h,
            carbon_emission_kg_per_h=carbon_emission,
            avoided_carbon_kg_per_h=avoided_carbon,
            constraints=constraints,
            metadata={},
        )


class DistrictHeating(BackendUtilizationModule):
    def __init__(self, name: str = "DistrictHeating", enabled: bool = True, **kwargs: Any):
        params = {
            "asset_type": "district_heating",
            "max_heat_input_kw": 12_000.0,
            "annual_service_hours": 2000.0,
            "useful_energy_efficiency": 0.88,
            "capex_per_kw": 0.0,
            "fixed_om_fraction": 0.0,
            "variable_om_CNY_per_kWh": 0.0,
            "revenue_CNY_per_kWh": 0.0,
            "output_temperature_K": 338.15,
            "carbon_mode": "gas_heat",
            "return_temperature_K": 313.15,
            "minimum_approach_K": 8.0,
            "pipe_heat_loss_fraction": 0.035,
            "pump_power_fraction": 0.0035,
            "default_source_temperature_K": 363.15,
        }
        params.update(kwargs)
        super().__init__(
            name=name,
            module_type="district_heating",
            enabled=enabled,
            **params,
        )

    def evaluate(
        self,
        flue_gas_in: FlueGasState,
        energy_inputs: Optional[List[EnergyStream]] = None,
        material_inputs: Optional[List[MaterialInput]] = None,
        operation_params: Optional[Dict[str, Any]] = None,
    ) -> ModuleResult:
        if not self.enabled:
            return self.bypass_result(flue_gas_in)

        params = self._runtime_params(operation_params)
        streams = energy_inputs or []
        (
            heat_input_kw,
            _,
            max_heat_input_kw,
            heat_used_kw,
            overflow_kw,
            exergy_used_kw,
            overflow_exergy_kw,
        ) = self._resource_inputs(streams, params)
        source_T = self._source_temperature(
            streams,
            params,
            fallback=float(params.get("default_source_temperature_K", 363.15)),
        )
        supply_T = float(params.get("supply_temperature_K", params["output_temperature_K"]))
        return_T = float(params.get("return_temperature_K", 313.15))
        minimum_approach_K = float(params.get("minimum_approach_K", 8.0))
        pipe_loss_fraction = _clamp(float(params.get("pipe_heat_loss_fraction", 0.035)), 0.0, 0.50)
        hx_efficiency = _clamp(float(params["useful_energy_efficiency"]), 0.0, 1.0)

        required_source_T = supply_T + minimum_approach_K
        if heat_used_kw > 0.0:
            temperature_factor = _clamp(
                (source_T - return_T) / max(required_source_T - return_T, 1e-9),
                0.0,
                1.0,
            )
        else:
            temperature_factor = 0.0

        heat_after_hx_kw = heat_used_kw * hx_efficiency * temperature_factor
        useful_energy_output_kw = heat_after_hx_kw * (1.0 - pipe_loss_fraction)
        auxiliary_power_kw = useful_energy_output_kw * max(
            float(params.get("pump_power_fraction", 0.0035)),
            0.0,
        )
        useful_exergy_output_kw = ThermoCore.physical_exergy_heat(
            useful_energy_output_kw,
            supply_T,
            ambient_temperature=float(params["ambient_temperature_K"]),
        )
        conversion_exergy_destruction_kw = (
            max(exergy_used_kw + auxiliary_power_kw - useful_exergy_output_kw, 0.0)
            + overflow_exergy_kw
        )

        net_cost_CNY_per_h = self._economic_cost(
            params=params,
            heat_used_kw=heat_used_kw,
            useful_energy_output_kw=useful_energy_output_kw,
            auxiliary_power_kw=auxiliary_power_kw,
        )
        carbon_emission, avoided_carbon, aux_carbon = self._net_carbon(
            params=params,
            useful_energy_output_kw=useful_energy_output_kw,
            auxiliary_power_kw=auxiliary_power_kw,
        )

        constraints = [self._capacity_constraint(heat_input_kw, max_heat_input_kw)]
        if heat_used_kw > 0.0:
            constraints.extend(
                [
                    ConstraintReport.lower_bound(
                        name=f"{self.name}_source_temperature_for_supply",
                        value=source_T,
                        limit=required_source_T,
                        message="Source temperature cannot satisfy supply-water pinch.",
                    ),
                    ConstraintReport.lower_bound(
                        name=f"{self.name}_supply_return_delta_T",
                        value=supply_T - return_T,
                        limit=10.0,
                        message="Supply/return temperature difference is too small.",
                    ),
                    ConstraintReport.upper_bound(
                        name=f"{self.name}_pipe_heat_loss_fraction",
                        value=pipe_loss_fraction,
                        limit=0.20,
                        message="Pipe heat loss fraction is too high.",
                    ),
                ]
            )

        cp_water = 4.186
        water_mass_flow_kg_s = useful_energy_output_kw / max(
            cp_water * max(supply_T - return_T, 1e-9),
            1e-9,
        )
        metadata = {
            "source_temperature_K": source_T,
            "supply_temperature_K": supply_T,
            "return_temperature_K": return_T,
            "required_source_temperature_K": required_source_T,
            "temperature_factor": temperature_factor,
            "heat_exchanger_efficiency": hx_efficiency,
            "pipe_heat_loss_fraction": pipe_loss_fraction,
            "district_water_mass_flow_kg_s": water_mass_flow_kg_s,
            "auxiliary_carbon_kg_per_h": aux_carbon,
        }
        return self._build_result(
            flue_gas_in,
            params,
            heat_input_kw=heat_input_kw,
            heat_used_kw=heat_used_kw,
            useful_energy_output_kw=useful_energy_output_kw,
            useful_exergy_output_kw=useful_exergy_output_kw,
            conversion_exergy_destruction_kw=conversion_exergy_destruction_kw,
            overflow_kw=overflow_kw,
            net_cost_CNY_per_h=net_cost_CNY_per_h,
            carbon_emission_kg_per_h=carbon_emission,
            avoided_carbon_kg_per_h=avoided_carbon,
            constraints=constraints,
            metadata=metadata,
            auxiliary_power_kw=auxiliary_power_kw,
        )


class AbsorptionCooling(BackendUtilizationModule):
    def __init__(self, name: str = "AbsorptionCooling", enabled: bool = True, **kwargs: Any):
        params = {
            "asset_type": "absorption_cooling",
            "max_heat_input_kw": 10_000.0,
            "annual_service_hours": 2000.0,
            "useful_energy_efficiency": 0.70,
            "capex_per_kw": 0.0,
            "fixed_om_fraction": 0.0,
            "variable_om_CNY_per_kWh": 0.0,
            "revenue_CNY_per_kWh": 0.0,
            "output_temperature_K": 280.15,
            "carbon_mode": "cooling_grid_displacement",
            "minimum_generator_temperature_K": 358.15,
            "nominal_generator_temperature_K": 373.15,
            "cooling_water_temperature_K": 303.15,
            "maximum_cooling_water_temperature_K": 315.15,
            "min_part_load": 0.20,
            "pump_power_fraction_of_cooling": 0.006,
            "default_source_temperature_K": 368.15,
        }
        params.update(kwargs)
        super().__init__(
            name=name,
            module_type="absorption_cooling",
            enabled=enabled,
            **params,
        )

    def evaluate(
        self,
        flue_gas_in: FlueGasState,
        energy_inputs: Optional[List[EnergyStream]] = None,
        material_inputs: Optional[List[MaterialInput]] = None,
        operation_params: Optional[Dict[str, Any]] = None,
    ) -> ModuleResult:
        if not self.enabled:
            return self.bypass_result(flue_gas_in)

        params = self._runtime_params(operation_params)
        streams = energy_inputs or []
        (
            heat_input_kw,
            _,
            max_heat_input_kw,
            heat_used_kw,
            overflow_kw,
            exergy_used_kw,
            overflow_exergy_kw,
        ) = self._resource_inputs(streams, params)
        generator_T = self._source_temperature(
            streams,
            params,
            fallback=float(params.get("default_source_temperature_K", 368.15)),
        )
        evaporator_T = float(params["output_temperature_K"] or 280.15)
        cooling_water_T = float(params.get("cooling_water_temperature_K", 303.15))
        min_generator_T = float(params.get("minimum_generator_temperature_K", 358.15))
        nominal_generator_T = float(params.get("nominal_generator_temperature_K", 373.15))
        min_part_load = max(float(params.get("min_part_load", 0.20)), 1e-9)
        plr = heat_used_kw / max_heat_input_kw if max_heat_input_kw > 0.0 else 0.0

        if heat_used_kw > 0.0:
            generator_factor = _clamp(
                (generator_T - min_generator_T) / max(nominal_generator_T - min_generator_T, 1e-9),
                0.0,
                1.15,
            )
            cooling_penalty = _clamp(1.0 - 0.018 * max(cooling_water_T - 303.15, 0.0), 0.70, 1.05)
            plr_factor = 1.0 if plr >= min_part_load else _clamp(0.45 + 0.55 * plr / min_part_load, 0.0, 1.0)
        else:
            generator_factor = 0.0
            cooling_penalty = 1.0
            plr_factor = 0.0

        nominal_cop = max(float(params["useful_energy_efficiency"]), 0.0)
        cop = nominal_cop * (0.78 + 0.22 * generator_factor) * cooling_penalty * plr_factor
        cop = _clamp(cop, 0.0, float(params.get("maximum_COP", 0.85)))
        useful_energy_output_kw = heat_used_kw * cop
        auxiliary_power_kw = useful_energy_output_kw * max(
            float(params.get("pump_power_fraction_of_cooling", 0.006)),
            0.0,
        )
        useful_exergy_output_kw = useful_energy_output_kw * max(
            float(params["ambient_temperature_K"]) / max(evaporator_T, 1e-9) - 1.0,
            0.0,
        )
        conversion_exergy_destruction_kw = (
            max(exergy_used_kw + auxiliary_power_kw - useful_exergy_output_kw, 0.0)
            + overflow_exergy_kw
        )
        net_cost_CNY_per_h = self._economic_cost(
            params=params,
            heat_used_kw=heat_used_kw,
            useful_energy_output_kw=useful_energy_output_kw,
            auxiliary_power_kw=auxiliary_power_kw,
        )
        carbon_emission, avoided_carbon, aux_carbon = self._net_carbon(
            params=params,
            useful_energy_output_kw=useful_energy_output_kw,
            auxiliary_power_kw=auxiliary_power_kw,
        )

        constraints = [self._capacity_constraint(heat_input_kw, max_heat_input_kw)]
        if heat_used_kw > 0.0:
            constraints.extend(
                [
                    ConstraintReport.lower_bound(
                        name=f"{self.name}_generator_temperature",
                        value=generator_T,
                        limit=min_generator_T,
                        message="Generator heat-source temperature is below single-effect LiBr-H2O window.",
                    ),
                    ConstraintReport.upper_bound(
                        name=f"{self.name}_cooling_water_temperature",
                        value=cooling_water_T,
                        limit=float(params["maximum_cooling_water_temperature_K"]),
                        message="Cooling-water temperature is too high for stable absorption cooling.",
                    ),
                    ConstraintReport.upper_bound(
                        name=f"{self.name}_crystallization_risk_proxy",
                        value=max(min_generator_T - generator_T, 0.0)
                        + max(cooling_water_T - float(params["maximum_cooling_water_temperature_K"]), 0.0),
                        limit=0.0,
                        message="LiBr crystallization proxy violated.",
                    ),
                ]
            )
            if bool(params.get("enforce_min_part_load", False)):
                constraints.append(
                    ConstraintReport.lower_bound(
                        name=f"{self.name}_part_load_ratio",
                        value=plr,
                        limit=min_part_load,
                        message="Absorption chiller part-load ratio is below stable operation.",
                    )
                )

        rejected_heat_kw = heat_used_kw + useful_energy_output_kw + auxiliary_power_kw
        metadata = {
            "generator_temperature_K": generator_T,
            "evaporator_temperature_K": evaporator_T,
            "cooling_water_temperature_K": cooling_water_T,
            "COP": cop,
            "nominal_COP": nominal_cop,
            "part_load_ratio": plr,
            "generator_temperature_factor": generator_factor,
            "cooling_water_penalty": cooling_penalty,
            "rejected_heat_kw": rejected_heat_kw,
            "auxiliary_carbon_kg_per_h": aux_carbon,
        }
        return self._build_result(
            flue_gas_in,
            params,
            heat_input_kw=heat_input_kw,
            heat_used_kw=heat_used_kw,
            useful_energy_output_kw=useful_energy_output_kw,
            useful_exergy_output_kw=useful_exergy_output_kw,
            conversion_exergy_destruction_kw=conversion_exergy_destruction_kw,
            overflow_kw=overflow_kw,
            net_cost_CNY_per_h=net_cost_CNY_per_h,
            carbon_emission_kg_per_h=carbon_emission,
            avoided_carbon_kg_per_h=avoided_carbon,
            constraints=constraints,
            metadata=metadata,
            auxiliary_power_kw=auxiliary_power_kw,
        )


class ORC(BackendUtilizationModule):
    def __init__(self, name: str = "ORC", enabled: bool = True, **kwargs: Any):
        params = {
            "asset_type": "orc_power",
            "max_heat_input_kw": 8_000.0,
            "annual_service_hours": BASE_OPERATION_HOURS_PER_YEAR,
            "useful_energy_efficiency": 0.12,
            "capex_per_kw": 0.0,
            "fixed_om_fraction": 0.0,
            "variable_om_CNY_per_kWh": 0.0,
            "revenue_CNY_per_kWh": 0.0,
            "output_temperature_K": None,
            "carbon_mode": "grid_electricity",
            "condensing_temperature_K": 308.15,
            "evaporator_pinch_K": 10.0,
            "condenser_pinch_K": 5.0,
            "minimum_evaporation_temperature_K": 333.15,
            "minimum_source_temperature_K": 353.15,
            "maximum_evaporation_temperature_K": 423.15,
            "carnot_fraction": 0.48,
            "generator_efficiency": 0.96,
            "evaporator_effectiveness": 0.92,
            "maximum_thermal_efficiency": 0.18,
            "minimum_thermal_efficiency": 0.03,
            "auxiliary_power_fraction": 0.015,
            "min_part_load": 0.20,
            "default_source_temperature_K": 423.15,
        }
        params.update(kwargs)
        super().__init__(
            name=name,
            module_type="orc",
            enabled=enabled,
            **params,
        )

    def evaluate(
        self,
        flue_gas_in: FlueGasState,
        energy_inputs: Optional[List[EnergyStream]] = None,
        material_inputs: Optional[List[MaterialInput]] = None,
        operation_params: Optional[Dict[str, Any]] = None,
    ) -> ModuleResult:
        if not self.enabled:
            return self.bypass_result(flue_gas_in)

        params = self._runtime_params(operation_params)
        streams = energy_inputs or []
        (
            heat_input_kw,
            _,
            max_heat_input_kw,
            heat_used_kw,
            overflow_kw,
            exergy_used_kw,
            overflow_exergy_kw,
        ) = self._resource_inputs(streams, params)
        source_T = self._source_temperature(
            streams,
            params,
            fallback=float(params.get("default_source_temperature_K", 423.15)),
        )
        condensing_T = float(params.get("condensing_temperature_K", 308.15))
        pinch_ev = float(params.get("evaporator_pinch_K", 10.0))
        min_evap_T = float(params.get("minimum_evaporation_temperature_K", 333.15))
        max_evap_T = float(params.get("maximum_evaporation_temperature_K", 423.15))
        desired_evap_T = float(params.get("evaporation_temperature_K", source_T - pinch_ev))
        evaporation_T = _clamp(desired_evap_T, min_evap_T, min(source_T - pinch_ev, max_evap_T))

        min_source_T = max(
            float(params.get("minimum_source_temperature_K", 353.15)),
            min_evap_T + pinch_ev,
        )
        min_part_load = max(float(params.get("min_part_load", 0.20)), 1e-9)
        plr = heat_used_kw / max_heat_input_kw if max_heat_input_kw > 0.0 else 0.0

        if heat_used_kw > 0.0 and source_T > condensing_T and evaporation_T > condensing_T:
            carnot_efficiency = max(1.0 - condensing_T / evaporation_T, 0.0)
            thermal_efficiency = (
                carnot_efficiency
                * float(params.get("carnot_fraction", 0.48))
                * float(params.get("generator_efficiency", 0.96))
            )
            thermal_efficiency = _clamp(
                thermal_efficiency,
                float(params.get("minimum_thermal_efficiency", 0.03)),
                float(params.get("maximum_thermal_efficiency", 0.18)),
            )
            if plr < min_part_load:
                thermal_efficiency *= _clamp(0.50 + 0.50 * plr / min_part_load, 0.0, 1.0)
            if source_T < min_source_T:
                thermal_efficiency *= _clamp(source_T / max(min_source_T, 1e-9), 0.0, 1.0)
        else:
            carnot_efficiency = 0.0
            thermal_efficiency = 0.0

        evaporator_heat_kw = heat_used_kw * _clamp(
            float(params.get("evaporator_effectiveness", 0.92)),
            0.0,
            1.0,
        )
        gross_power_kw = evaporator_heat_kw * thermal_efficiency
        auxiliary_power_kw = gross_power_kw * max(
            float(params.get("auxiliary_power_fraction", 0.015)),
            0.0,
        )
        useful_energy_output_kw = max(gross_power_kw - auxiliary_power_kw, 0.0)
        useful_exergy_output_kw = useful_energy_output_kw
        rejected_heat_kw = max(evaporator_heat_kw - gross_power_kw, 0.0)
        rejected_heat_exergy_kw = ThermoCore.physical_exergy_heat(
            rejected_heat_kw,
            condensing_T + float(params.get("condenser_pinch_K", 5.0)),
            ambient_temperature=float(params["ambient_temperature_K"]),
        )
        conversion_exergy_destruction_kw = (
            max(
                exergy_used_kw
                - useful_exergy_output_kw
                - rejected_heat_exergy_kw,
                0.0,
            )
            + overflow_exergy_kw
        )
        net_cost_CNY_per_h = self._economic_cost(
            params=params,
            heat_used_kw=heat_used_kw,
            useful_energy_output_kw=useful_energy_output_kw,
        )
        carbon_emission, avoided_carbon, _ = self._net_carbon(
            params=params,
            useful_energy_output_kw=useful_energy_output_kw,
            auxiliary_power_kw=0.0,
        )

        constraints = [self._capacity_constraint(heat_input_kw, max_heat_input_kw)]
        if heat_used_kw > 0.0:
            constraints.extend(
                [
                    ConstraintReport.lower_bound(
                        name=f"{self.name}_source_temperature",
                        value=source_T,
                        limit=min_source_T,
                        message="ORC source temperature is below the evaporation and pinch window.",
                    ),
                    ConstraintReport.lower_bound(
                        name=f"{self.name}_evaporation_temperature_lift",
                        value=evaporation_T - condensing_T,
                        limit=20.0,
                        message="ORC evaporation-condensation lift is too small.",
                    ),
                ]
            )
            if bool(params.get("enforce_min_part_load", False)):
                constraints.append(
                    ConstraintReport.lower_bound(
                        name=f"{self.name}_part_load_ratio",
                        value=plr,
                        limit=min_part_load,
                        message="ORC part-load ratio is below stable operation.",
                    )
                )

        metadata = {
            "source_temperature_K": source_T,
            "evaporation_temperature_K": evaporation_T,
            "condensing_temperature_K": condensing_T,
            "carnot_efficiency": carnot_efficiency,
            "orc_thermal_efficiency": thermal_efficiency,
            "gross_power_kw": gross_power_kw,
            "net_power_kw": useful_energy_output_kw,
            "part_load_ratio": plr,
            "evaporator_heat_kw": evaporator_heat_kw,
            "rejected_heat_kw": rejected_heat_kw,
            "rejected_heat_exergy_kw": rejected_heat_exergy_kw,
        }
        return self._build_result(
            flue_gas_in,
            params,
            heat_input_kw=heat_input_kw,
            heat_used_kw=heat_used_kw,
            useful_energy_output_kw=useful_energy_output_kw,
            useful_exergy_output_kw=useful_exergy_output_kw,
            conversion_exergy_destruction_kw=conversion_exergy_destruction_kw,
            overflow_kw=overflow_kw,
            net_cost_CNY_per_h=net_cost_CNY_per_h,
            carbon_emission_kg_per_h=carbon_emission,
            avoided_carbon_kg_per_h=avoided_carbon,
            constraints=constraints,
            metadata=metadata,
            auxiliary_power_kw=auxiliary_power_kw,
        )


class HeatBufferSink(BackendUtilizationModule):
    """Static sink for residual heat with no inter-period storage state."""

    def __init__(
        self,
        *,
        max_heat_input_kw: float,
        useful_output_factor: float,
        service_temperature_K: float,
        auxiliary_fraction: float,
        name: str = "HeatBufferSink",
        enabled: bool = True,
        **configuration: Any,
    ):
        params = {
            "asset_type": "heat_buffer_sink",
            "max_heat_input_kw": max_heat_input_kw,
            "annual_service_hours": 1.0,
            "useful_energy_efficiency": useful_output_factor,
            "capex_per_kw": 0.0,
            "fixed_om_fraction": 0.0,
            "variable_om_CNY_per_kWh": 0.0,
            "revenue_CNY_per_kWh": 0.0,
            "output_temperature_K": service_temperature_K,
            "carbon_mode": "none",
            "auxiliary_fraction": auxiliary_fraction,
        }
        params.update(configuration)
        super().__init__(
            name=name,
            module_type="heat_buffer_sink",
            asset_type=params.pop("asset_type"),
            enabled=enabled,
            **params,
        )

    def evaluate(
        self,
        flue_gas_in: FlueGasState,
        energy_inputs: Optional[List[EnergyStream]] = None,
        material_inputs: Optional[List[MaterialInput]] = None,
        operation_params: Optional[Dict[str, Any]] = None,
    ) -> ModuleResult:
        if not self.enabled:
            return self.bypass_result(flue_gas_in)

        params = self._runtime_params(operation_params)
        streams = energy_inputs or []
        (
            heat_input_kw,
            _,
            max_heat_input_kw,
            heat_used_kw,
            overflow_kw,
            exergy_used_kw,
            overflow_exergy_kw,
        ) = self._resource_inputs(streams, params)
        useful_heat_kw = heat_used_kw * max(float(params["useful_energy_efficiency"]), 0.0)
        auxiliary_power_kw = heat_used_kw * max(float(params.get("auxiliary_fraction", 0.0)), 0.0)
        useful_exergy_kw = ThermoCore.physical_exergy_heat(
            useful_heat_kw,
            float(params["output_temperature_K"]),
            ambient_temperature=float(params["ambient_temperature_K"]),
        )
        destruction_kw = (
            max(exergy_used_kw + auxiliary_power_kw - useful_exergy_kw, 0.0)
            + overflow_exergy_kw
        )
        cost_CNY_h = self._economic_cost(
            params=params,
            heat_used_kw=heat_used_kw,
            useful_energy_output_kw=useful_heat_kw,
            auxiliary_power_kw=auxiliary_power_kw,
        )
        carbon, avoided, _ = self._net_carbon(
            params=params,
            useful_energy_output_kw=useful_heat_kw,
            auxiliary_power_kw=auxiliary_power_kw,
        )
        return self._build_result(
            flue_gas_in,
            params,
            heat_input_kw=heat_input_kw,
            heat_used_kw=heat_used_kw,
            useful_energy_output_kw=useful_heat_kw,
            useful_exergy_output_kw=useful_exergy_kw,
            conversion_exergy_destruction_kw=destruction_kw,
            overflow_kw=overflow_kw,
            net_cost_CNY_per_h=cost_CNY_h,
            carbon_emission_kg_per_h=carbon,
            avoided_carbon_kg_per_h=avoided,
            constraints=[self._capacity_constraint(heat_input_kw, max_heat_input_kw)],
            metadata={"static_sink": True},
            auxiliary_power_kw=auxiliary_power_kw,
            messages=["Static residual-heat sink evaluated without a storage state or inter-period balance."],
        )
