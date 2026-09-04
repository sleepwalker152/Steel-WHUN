# steel_whun/network/transmission.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
import math

from steel_whun.core.states import EnergyStream


@dataclass
class TransmissionEdge:
    'TransmissionEdge component used by the Steel-WHUN core framework.'

    source_id: str
    target_id: str

    energy_efficiency: float = 1.0
    exergy_efficiency: float = 1.0

    distance_km: float = 0.0

    energy_loss_rate_per_km: float = 0.0
    exergy_loss_rate_per_km: float = 0.0

    temperature_drop_per_km: float = 0.0
    pressure_drop_per_km: float = 0.0

    capital_cost_per_km: float = 0.0
    lifetime_years: float = 15.0
    annual_operation_hours: float = 8000.0

    operation_cost_per_kw_km: float = 0.0
    fixed_om_cost_rate: float = 0.0

    carbon_per_kw_km: float = 0.0

    capacity: float = float("inf")

    metadata: Dict[str, Any] = field(default_factory=dict)

    def effective_energy_efficiency(self) -> float:
        distance_factor = math.exp(
            -max(self.energy_loss_rate_per_km, 0.0) * max(self.distance_km, 0.0)
        )
        eta = self.energy_efficiency * distance_factor
        return min(max(eta, 0.0), 1.0)

    def effective_exergy_efficiency(self) -> float:
        distance_factor = math.exp(
            -max(self.exergy_loss_rate_per_km, 0.0) * max(self.distance_km, 0.0)
        )
        eta = self.exergy_efficiency * distance_factor
        return min(max(eta, 0.0), 1.0)

    def annualized_capital_cost_rate(self) -> float:
        if self.capital_cost_per_km <= 0:
            return 0.0

        if self.lifetime_years <= 0 or self.annual_operation_hours <= 0:
            return 0.0

        total_capital_cost = self.capital_cost_per_km * max(self.distance_km, 0.0)

        return total_capital_cost / self.lifetime_years / self.annual_operation_hours

    def operation_cost_rate(self, requested_energy_rate: float) -> float:
        q = max(requested_energy_rate, 0.0)
        l = max(self.distance_km, 0.0)

        variable_cost = q * l * max(self.operation_cost_per_kw_km, 0.0)
        fixed_cost = max(self.fixed_om_cost_rate, 0.0)
        capex_rate = self.annualized_capital_cost_rate()

        return variable_cost + fixed_cost + capex_rate

    def carbon_rate(self, requested_energy_rate: float) -> float:
        q = max(requested_energy_rate, 0.0)
        l = max(self.distance_km, 0.0)

        return q * l * max(self.carbon_per_kw_km, 0.0)

    def transmit(
        self,
        stream: EnergyStream,
        requested_energy_rate: float,
    ) -> EnergyStream:
        'Execute the transmit calculation.'

        q_requested = max(float(requested_energy_rate), 0.0)
        edge_capacity = max(float(self.capacity), 0.0)
        q_sent = min(q_requested, edge_capacity)
        edge_capacity_violation = max(q_requested - edge_capacity, 0.0)

        eta_e = self.effective_energy_efficiency()
        eta_ex = self.effective_exergy_efficiency()

        q_delivered = q_sent * eta_e

        if stream.energy_rate > 0:
            exergy_sent = stream.exergy_rate * q_sent / stream.energy_rate
            cost_sent = stream.cost_rate * q_sent / stream.energy_rate
            carbon_sent = stream.carbon_rate * q_sent / stream.energy_rate
        else:
            exergy_sent = 0.0
            cost_sent = 0.0
            carbon_sent = 0.0

        exergy_delivered = exergy_sent * eta_ex

        old_temperature = stream.temperature if stream.temperature is not None else 298.15
        old_pressure = stream.pressure if stream.pressure is not None else 101325.0

        new_temperature = old_temperature - self.temperature_drop_per_km * self.distance_km
        new_pressure = old_pressure - self.pressure_drop_per_km * self.distance_km

        new_temperature = max(new_temperature, 298.15)
        new_pressure = max(new_pressure, 101325.0 * 0.5)

        transmission_cost = self.operation_cost_rate(q_sent)
        transmission_carbon = self.carbon_rate(q_sent)

        metadata = dict(stream.metadata) if stream.metadata is not None else {}
        metadata.update(
            {
                "source_id": self.source_id,
                "target_id": self.target_id,
                "distance_km": self.distance_km,
                "q_requested": q_requested,
                "q_sent": q_sent,
                "q_delivered": q_delivered,
                "edge_capacity_violation": edge_capacity_violation,
                "energy_efficiency_effective": eta_e,
                "exergy_efficiency_effective": eta_ex,
                "energy_loss": q_sent - q_delivered,
                "exergy_loss": exergy_sent - exergy_delivered,
                "transmission_cost": transmission_cost,
                "transmission_carbon": transmission_carbon,
                "annualized_capital_cost_rate": self.annualized_capital_cost_rate(),
            }
        )

        return EnergyStream(
            name=stream.name,

            
            energy_type=getattr(stream, "energy_type", getattr(stream, "carrier", "heat")),
            carrier=getattr(stream, "carrier", getattr(stream, "energy_type", "heat")),

            flow_rate=q_delivered,
            temperature=new_temperature,
            pressure=new_pressure,
            energy_rate=q_delivered,
            exergy_rate=exergy_delivered,
            cost_rate=cost_sent,
            carbon_rate=carbon_sent,
            source_id=self.source_id,
            target_id=self.target_id,
            metadata=metadata,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "distance_km": self.distance_km,
            "energy_efficiency": self.energy_efficiency,
            "exergy_efficiency": self.exergy_efficiency,
            "effective_energy_efficiency": self.effective_energy_efficiency(),
            "effective_exergy_efficiency": self.effective_exergy_efficiency(),
            "capital_cost_per_km": self.capital_cost_per_km,
            "annualized_capital_cost_rate": self.annualized_capital_cost_rate(),
            "operation_cost_per_kw_km": self.operation_cost_per_kw_km,
            "capacity": self.capacity,
            "metadata": self.metadata,
        }
