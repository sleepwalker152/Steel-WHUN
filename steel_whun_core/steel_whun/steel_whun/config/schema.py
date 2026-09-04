from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Scenario:
    """Generic boundary conditions supplied by a user or external data source."""

    name: str
    source_availability: Mapping[str, float] = field(default_factory=dict)
    pollutant_load_mg_Nm3: Mapping[str, float] = field(default_factory=dict)
    flue_gas_state: Mapping[str, Any] = field(default_factory=dict)
    terminal_capacity_kW: Mapping[str, float] = field(default_factory=dict)
    prices: Mapping[str, float] = field(default_factory=dict)
    emission_limits_mg_Nm3: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Scenario":
        allowed = {
            "name",
            "source_availability",
            "pollutant_load_mg_Nm3",
            "flue_gas_state",
            "terminal_capacity_kW",
            "prices",
            "emission_limits_mg_Nm3",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown scenario fields: {sorted(unknown)}")
        return cls(**data)
