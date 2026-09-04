# steel_whun/core/states.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import copy


@dataclass
class FlueGasState:
    'FlueGasState component used by the Steel-WHUN core framework.'

    mass_flow: float
    temperature: float
    pressure: float
    composition: Dict[str, float] = field(default_factory=dict)
    pollutants: Dict[str, float] = field(default_factory=dict)
    flow_basis: str = "wet"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "FlueGasState":
        return copy.deepcopy(self)

    def get_pollutant(self, name: str, default: float = 0.0) -> float:
        return self.pollutants.get(name, default)

    def set_pollutant(self, name: str, value: float) -> None:
        self.pollutants[name] = max(value, 0.0)

    def update_pollutant_by_efficiency(self, name: str, efficiency: float) -> float:
        'Execute the update pollutant by efficiency calculation.'
        efficiency = min(max(efficiency, 0.0), 1.0)
        c_in = self.get_pollutant(name, 0.0)
        c_out = c_in * (1.0 - efficiency)
        self.set_pollutant(name, c_out)
        return c_in - c_out

    def add_temperature_change(self, delta_T: float) -> None:
        self.temperature += delta_T

    def summary(self) -> Dict[str, Any]:
        return {
            "mass_flow_kg_s": self.mass_flow,
            "temperature_K": self.temperature,
            "temperature_C": self.temperature - 273.15,
            "pressure_Pa": self.pressure,
            "composition": self.composition,
            "pollutants": self.pollutants,
            "flow_basis": self.flow_basis,
            "metadata": self.metadata,
        }


@dataclass
class EnergyStream:
    'EnergyStream component used by the Steel-WHUN core framework.'

    name: str
    carrier: str
    energy_type: str
    flow_rate: float
    temperature: Optional[float]
    pressure: Optional[float]
    energy_rate: float
    exergy_rate: float
    cost_rate: float = 0.0
    carbon_rate: float = 0.0
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "EnergyStream":
        return copy.deepcopy(self)

    @property
    def is_electricity(self) -> bool:
        return self.carrier.lower() in ["electricity", "power", "elec"]

    @property
    def available_energy(self) -> float:
        return max(self.energy_rate, 0.0)

    @property
    def available_exergy(self) -> float:
        return max(self.exergy_rate, 0.0)

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "carrier": self.carrier,
            "flow_rate": self.flow_rate,
            "temperature_K": self.temperature,
            "temperature_C": None if self.temperature is None else self.temperature - 273.15,
            "pressure_Pa": self.pressure,
            "energy_rate_kW": self.energy_rate,
            "exergy_rate_kW": self.exergy_rate,
            "cost_rate": self.cost_rate,
            "carbon_rate": self.carbon_rate,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "metadata": self.metadata,
        }


@dataclass
class MaterialInput:
    'MaterialInput component used by the Steel-WHUN core framework.'

    name: str
    flow_rate: float
    unit: str = "kg/s"
    cost_rate: float = 0.0
    carbon_rate: float = 0.0
    composition: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "MaterialInput":
        return copy.deepcopy(self)


@dataclass
class ConstraintReport:
    'ConstraintReport component used by the Steel-WHUN core framework.'

    name: str
    value: float
    limit: float
    sense: str = "<="
    violation: float = 0.0
    feasible: bool = True
    message: str = ""

    @classmethod
    def upper_bound(cls, name: str, value: float, limit: float, message: str = ""):
        violation = max(value - limit, 0.0)
        return cls(
            name=name,
            value=value,
            limit=limit,
            sense="<=",
            violation=violation,
            feasible=violation <= 0.0,
            message=message,
        )

    @classmethod
    def lower_bound(cls, name: str, value: float, limit: float, message: str = ""):
        violation = max(limit - value, 0.0)
        return cls(
            name=name,
            value=value,
            limit=limit,
            sense=">=",
            violation=violation,
            feasible=violation <= 0.0,
            message=message,
        )

    @classmethod
    def range_bound(
        cls,
        name: str,
        value: float,
        lower: float,
        upper: float,
        message: str = "",
    ):
        violation = max(lower - value, 0.0) + max(value - upper, 0.0)
        return cls(
            name=name,
            value=value,
            limit=float("nan"),
            sense=f"{lower} <= x <= {upper}",
            violation=violation,
            feasible=violation <= 0.0,
            message=message,
        )


@dataclass
class ModuleResult:
    'ModuleResult component used by the Steel-WHUN core framework.'

    module_name: str
    flue_gas_in: FlueGasState
    flue_gas_out: FlueGasState

    removal_efficiency: Dict[str, float] = field(default_factory=dict)
    removed_pollutants: Dict[str, float] = field(default_factory=dict)

    energy_consumption: Dict[str, float] = field(default_factory=dict)
    material_consumption: Dict[str, float] = field(default_factory=dict)

    exergy_destruction: float = 0.0
    cost: float = 0.0
    carbon_emission: float = 0.0

    constraints: List[ConstraintReport] = field(default_factory=list)
    feasible: bool = True
    messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_feasibility(self) -> None:
        self.feasible = all(c.feasible for c in self.constraints)

    def total_violation(self) -> float:
        return sum(c.violation for c in self.constraints)

    def summary(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "flue_gas_in": self.flue_gas_in.summary(),
            "flue_gas_out": self.flue_gas_out.summary(),
            "removal_efficiency": self.removal_efficiency,
            "removed_pollutants": self.removed_pollutants,
            "energy_consumption": self.energy_consumption,
            "material_consumption": self.material_consumption,
            "exergy_destruction": self.exergy_destruction,
            "cost": self.cost,
            "carbon_emission": self.carbon_emission,
            "feasible": self.feasible,
            "total_violation": self.total_violation(),
            "messages": self.messages,
            "metadata": self.metadata,
        }