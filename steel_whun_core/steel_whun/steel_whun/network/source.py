# steel_whun/network/source.py

from __future__ import annotations

from typing import Optional, Dict, Any

from steel_whun.core.states import EnergyStream
from steel_whun.core.thermo import ThermoCore


class EnergySource:
    """Configurable source of waste heat, electricity or secondary fuel."""

    DEFAULT_CHEMICAL_EXERGY_FACTORS = {
        "BFG": 1.02,
        "COG": 1.04,
        "LDG": 1.03,
    }

    def __init__(
        self,
        source_id: str,
        carrier: str,
        temperature: Optional[float],
        pressure: Optional[float],
        max_energy_rate: float,
        unit_cost: float = 0.0,
        carbon_factor: float = 0.0,
        exergy_factor: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.source_id = source_id
        self.carrier = carrier
        self.temperature = temperature
        self.pressure = pressure
        self.max_energy_rate = max_energy_rate
        self.unit_cost = unit_cost
        self.carbon_factor = carbon_factor
        self.exergy_factor = self._resolve_exergy_factor(exergy_factor)
        self.metadata = metadata or {}

        
        self.name = source_id
        self.capacity = max_energy_rate

    def _resolve_exergy_factor(self, exergy_factor: Optional[float]) -> Optional[float]:
        if exergy_factor is not None:
            return exergy_factor
        if self.carrier.lower() not in ["fuel_gas", "gas", "cog", "bfg", "ldg"]:
            return None
        return self.DEFAULT_CHEMICAL_EXERGY_FACTORS.get(self.source_id.upper())

    def generate_stream(
        self,
        energy_rate: float,
        target_id: Optional[str] = None,
    ) -> EnergyStream:

        energy_rate = min(max(energy_rate, 0.0), self.max_energy_rate)

        if self.carrier.lower() in ["electricity", "power", "elec"]:
            exergy_rate = ThermoCore.electricity_exergy(energy_rate)
        elif self.exergy_factor is not None:
            exergy_rate = energy_rate * self.exergy_factor
        else:
            exergy_rate = ThermoCore.physical_exergy_heat(
                heat_rate=energy_rate,
                source_temperature=self.temperature or ThermoCore.T0,
            )

        return EnergyStream(
            name=f"{self.source_id}_to_{target_id}",
            carrier=self.carrier,
            flow_rate=energy_rate,
            temperature=self.temperature,
            pressure=self.pressure,
            energy_rate=energy_rate,
            exergy_rate=exergy_rate,
            cost_rate=energy_rate * self.unit_cost,
            carbon_rate=energy_rate * self.carbon_factor,
            source_id=self.source_id,
            target_id=target_id,
            metadata=self.metadata.copy(),
            energy_type=self.carrier,

        )
