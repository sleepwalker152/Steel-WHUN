from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AccountingLedger:
    """Keep physical loss, opportunity loss and economic terms separate."""

    module_exergy_destruction_kW: float = 0.0
    transport_exergy_loss_kW: float = 0.0
    unused_source_opportunity_loss_kW: float = 0.0
    costs_CNY_h: dict[str, float] = field(default_factory=dict)
    credits_CNY_h: dict[str, float] = field(default_factory=dict)

    @property
    def physical_exergy_loss_kW(self) -> float:
        return self.module_exergy_destruction_kW + self.transport_exergy_loss_kW

    @property
    def augmented_system_exergy_loss_kW(self) -> float:
        return self.physical_exergy_loss_kW + self.unused_source_opportunity_loss_kW

    @property
    def net_economic_cost_CNY_h(self) -> float:
        return sum(self.costs_CNY_h.values()) - sum(self.credits_CNY_h.values())

    def as_dict(self) -> dict[str, float]:
        return {
            "module_exergy_destruction_kW": self.module_exergy_destruction_kW,
            "transport_exergy_loss_kW": self.transport_exergy_loss_kW,
            "unused_source_opportunity_loss_kW": self.unused_source_opportunity_loss_kW,
            "augmented_system_exergy_loss_kW": self.augmented_system_exergy_loss_kW,
            "net_economic_cost_CNY_h": self.net_economic_cost_CNY_h,
        }
