# steel_whun/process/emission.py

from __future__ import annotations

from typing import Dict, List, Any

from steel_whun.core.states import FlueGasState, ConstraintReport


class EmissionChecker:
    'EmissionChecker component used by the Steel-WHUN core framework.'

    def __init__(
        self,
        limits: Dict[str, float],
        name: str = "stack_emission_checker",
    ):
        self.name = name
        self.limits = limits

    def check(self, flue_gas_final: FlueGasState) -> tuple[bool, Dict[str, float], List[ConstraintReport]]:
        violations = {}
        reports = []

        for pollutant, limit in self.limits.items():
            value = flue_gas_final.pollutants.get(pollutant, 0.0)
            report = ConstraintReport.upper_bound(
                name=f"stack_{pollutant}_limit",
                value=value,
                limit=limit,
                message=f"Stack emission limit for {pollutant}.",
            )
            reports.append(report)
            violations[pollutant] = report.violation

        feasible = all(r.feasible for r in reports)
        return feasible, violations, reports

    def summary(self) -> Dict[str, Any]:
        return {
            "checker_name": self.name,
            "limits": self.limits,
        }