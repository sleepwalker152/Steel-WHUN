# steel_whun/network/energy_network.py

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from steel_whun.core.states import EnergyStream
from steel_whun.network.source import EnergySource
from steel_whun.network.transmission import TransmissionEdge


@dataclass
class EnergyNetworkResult:
    energy_map: Dict[str, List[EnergyStream]] = field(default_factory=dict)

    total_energy_requested: float = 0.0
    total_energy_delivered: float = 0.0
    total_exergy_delivered: float = 0.0

    total_energy_loss: float = 0.0
    total_exergy_loss: float = 0.0

    total_transmission_cost: float = 0.0
    total_transmission_carbon: float = 0.0

    source_requested: Dict[str, float] = field(default_factory=dict)
    source_usage: Dict[str, float] = field(default_factory=dict)
    source_dispatch_scale: Dict[str, float] = field(default_factory=dict)
    source_capacity_violation: Dict[str, float] = field(default_factory=dict)
    edge_capacity_violation: Dict[str, float] = field(default_factory=dict)

    edge_results: List[Dict[str, Any]] = field(default_factory=list)

    feasible: bool = True
    messages: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_energy_requested": self.total_energy_requested,
            "total_energy_delivered": self.total_energy_delivered,
            "total_exergy_delivered": self.total_exergy_delivered,
            "total_energy_loss": self.total_energy_loss,
            "total_exergy_loss": self.total_exergy_loss,
            "total_transmission_cost": self.total_transmission_cost,
            "total_transmission_carbon": self.total_transmission_carbon,
            "source_requested": self.source_requested,
            "source_usage": self.source_usage,
            "source_dispatch_scale": self.source_dispatch_scale,
            "source_capacity_violation": self.source_capacity_violation,
            "edge_capacity_violation": self.edge_capacity_violation,
            "edge_results": self.edge_results,
            "feasible": self.feasible,
            "messages": self.messages,
        }


class EnergyNetwork:
    'EnergyNetwork component used by the Steel-WHUN core framework.'

    def __init__(self):
        self.sources: Dict[str, EnergySource] = {}
        self.edges: Dict[Tuple[str, str], TransmissionEdge] = {}

    def add_source(self, source: EnergySource):
        source_id = getattr(source, "source_id", None)

        if source_id is None:
            source_id = getattr(source, "name", None)

        if source_id is None:
            raise ValueError("EnergySource must have source_id or name.")

        self.sources[source_id] = source

    def add_edge(self, edge: TransmissionEdge):
        self.edges[(edge.source_id, edge.target_id)] = edge

    def get_edge(self, source_id: str, target_id: str) -> Optional[TransmissionEdge]:
        return self.edges.get((source_id, target_id), None)

    def dispatch(
        self,
        allocation_matrix: Dict[str, Dict[str, float]],
    ) -> Tuple[Dict[str, List[EnergyStream]], Dict[str, Any]]:

        result = EnergyNetworkResult()

        for source_id, target_allocations in allocation_matrix.items():

            if source_id not in self.sources:
                result.feasible = False
                result.messages.append(f"Unknown energy source: {source_id}")
                continue

            source = self.sources[source_id]

            requested_by_target = {
                target_id: max(float(value), 0.0)
                for target_id, value in target_allocations.items()
            }
            source_total_requested = sum(requested_by_target.values())

            source_capacity = getattr(
                source,
                "max_energy_rate",
                getattr(source, "capacity", float("inf")),
            )

            source_capacity = max(float(source_capacity), 0.0)
            violation = max(source_total_requested - source_capacity, 0.0)
            dispatch_scale = 1.0
            if (
                source_total_requested > 0.0
                and math.isfinite(source_capacity)
                and source_total_requested > source_capacity
            ):
                dispatch_scale = source_capacity / source_total_requested

            result.source_requested[source_id] = source_total_requested
            result.source_usage[source_id] = source_total_requested * dispatch_scale
            result.source_dispatch_scale[source_id] = dispatch_scale
            result.source_capacity_violation[source_id] = violation

            if violation > 0:
                result.feasible = False
                result.messages.append(
                    f"Source capacity violation: {source_id}, "
                    f"requested={source_total_requested}, capacity={source_capacity}"
                )

            for target_id, q_req in requested_by_target.items():

                if q_req <= 0:
                    continue

                q_dispatched = q_req * dispatch_scale

                edge = self.get_edge(source_id, target_id)

                if edge is None:
                    result.feasible = False
                    result.messages.append(
                        f"No transmission edge from {source_id} to {target_id}"
                    )
                    continue

                stream = source.generate_stream(
                    energy_rate=q_dispatched,
                    target_id=target_id,
                )

                delivered_stream = edge.transmit(
                    stream=stream,
                    requested_energy_rate=q_dispatched,
                )

                result.energy_map.setdefault(target_id, [])
                result.energy_map[target_id].append(delivered_stream)

                result.total_energy_requested += q_req
                result.total_energy_delivered += delivered_stream.energy_rate
                result.total_exergy_delivered += delivered_stream.exergy_rate

                metadata = delivered_stream.metadata or {}

                energy_loss = metadata.get("energy_loss", 0.0)
                exergy_loss = metadata.get("exergy_loss", 0.0)
                transmission_cost = metadata.get("transmission_cost", 0.0)
                transmission_carbon = metadata.get("transmission_carbon", 0.0)
                edge_violation = metadata.get("edge_capacity_violation", 0.0)
                edge_key = f"{source_id}->{target_id}"
                result.edge_capacity_violation[edge_key] = edge_violation

                if edge_violation > 0.0:
                    result.feasible = False
                    result.messages.append(
                        f"Edge capacity violation: {edge_key}, "
                        f"requested={q_dispatched}, capacity={edge.capacity}"
                    )

                result.total_energy_loss += energy_loss
                result.total_exergy_loss += exergy_loss
                result.total_transmission_cost += transmission_cost
                result.total_transmission_carbon += transmission_carbon

                result.edge_results.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "requested_energy_rate": q_req,
                        "source_dispatched_energy_rate": q_dispatched,
                        "edge_sent_energy_rate": metadata.get("q_sent", 0.0),
                        "edge_capacity_violation": edge_violation,
                        "delivered_energy_rate": delivered_stream.energy_rate,
                        "delivered_exergy_rate": delivered_stream.exergy_rate,
                        "energy_loss": energy_loss,
                        "exergy_loss": exergy_loss,
                        "transmission_cost": transmission_cost,
                        "transmission_carbon": transmission_carbon,
                        "distance_km": metadata.get("distance_km", None),
                        "energy_efficiency_effective": metadata.get(
                            "energy_efficiency_effective", None
                        ),
                        "exergy_efficiency_effective": metadata.get(
                            "exergy_efficiency_effective", None
                        ),
                    }
                )

        return result.energy_map, result.summary()
