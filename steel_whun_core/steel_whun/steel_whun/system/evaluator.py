# steel_whun/system/evaluator.py

from __future__ import annotations

from typing import Any, Dict, List

from steel_whun.core.states import EnergyStream, FlueGasState, ModuleResult
from steel_whun.network.energy_network import EnergyNetwork
from steel_whun.process.emission import EmissionChecker
from steel_whun.process.route import ProcessRoute


class SystemEvaluator:
    """System-level evaluator.

    The main process route handles the flue gas treatment train. Energy assets
    that do not belong in the flue gas train, such as CCPP, are evaluated as
    backend assets and then merged into the system accounting totals.
    """

    def __init__(
        self,
        energy_network: EnergyNetwork,
        process_route: ProcessRoute,
        emission_checker: EmissionChecker,
        name: str = "system_evaluator",
    ):
        self.name = name
        self.energy_network = energy_network
        self.process_route = process_route
        self.emission_checker = emission_checker

    @staticmethod
    def _base_module_name(module_name: str) -> str:
        base, sep, suffix = module_name.rpartition("_")
        if sep and suffix.isdigit():
            return base
        return module_name

    def _derive_backend_operation_params(
        self,
        module_name: str,
        module: Any,
        energy_inputs: List[EnergyStream],
        operation_map: Dict[str, Any],
    ) -> Dict[str, Any]:
        op_params = dict(operation_map.get(module_name, {}))

        if module_name.lower() == "ccpp":
            gas_properties = getattr(module, "_gas_properties", {})

            # The network delivery ledger is authoritative for CCPP fuel use.
            # Initialize every supported gas explicitly so that an absent
            # source-to-CCPP allocation is represented by zero flow, even if a
            # caller supplied a stale value in the generic operation map.
            for source_id in ("BFG", "COG", "LDG"):
                op_params[f"{source_id}_flow_m3_s"] = 0.0

            for stream in energy_inputs:
                source_id = (
                    stream.source_id
                    or stream.metadata.get("source_id")
                    or ""
                ).upper()

                if source_id not in gas_properties:
                    continue

                lhv = gas_properties[source_id].get("LHV", 0.0)
                if lhv <= 0.0:
                    continue

                # m3/s = kW / (kJ/m3). This uses the energy actually delivered
                # by the network after transmission losses.
                flow_key = f"{source_id}_flow_m3_s"
                op_params[flow_key] += max(stream.energy_rate, 0.0) / lhv

        return op_params

    def _evaluate_backend_assets(
        self,
        flue_gas_initial: FlueGasState,
        energy_map: Dict[str, List[EnergyStream]],
        operation_map: Dict[str, Any],
        front_module_results: List[ModuleResult],
    ) -> List[ModuleResult]:
        modules_pool = getattr(self.process_route, "modules_pool", {})
        front_names = {
            self._base_module_name(result.module_name).lower()
            for result in front_module_results
        }

        explicit_assets = operation_map.get("backend_assets", [])
        if isinstance(explicit_assets, dict):
            explicit_asset_names = set(explicit_assets.keys())
        elif isinstance(explicit_assets, (list, tuple, set)):
            explicit_asset_names = set(explicit_assets)
        else:
            explicit_asset_names = set()

        candidate_names = set(energy_map.keys()) | explicit_asset_names
        backend_results: List[ModuleResult] = []

        for target_name in sorted(candidate_names):
            module = modules_pool.get(str(target_name).lower())
            if module is None:
                continue
            if module.name.lower() in front_names:
                continue
            if (
                not self.process_route._is_backend_module(module)
                and module.name not in explicit_asset_names
                and str(target_name) not in explicit_asset_names
            ):
                continue

            energy_inputs = energy_map.get(module.name, energy_map.get(target_name, []))
            op_params = self._derive_backend_operation_params(
                module.name,
                module,
                energy_inputs,
                operation_map,
            )

            dummy_fg = flue_gas_initial.copy()
            dummy_fg.mass_flow = 0.0

            result = module.evaluate(
                flue_gas_in=dummy_fg,
                energy_inputs=energy_inputs,
                operation_params=op_params,
            )
            result.module_name = f"{module.name}_backend"
            backend_results.append(result)

        return backend_results

    def evaluate(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        flue_gas_initial: FlueGasState = decision["flue_gas_initial"]
        allocation_matrix = decision.get("allocation_matrix", {})
        material_map = decision.get("material_map", {})
        operation_map = decision.get("operation_map", {})

        energy_map, network_result = self.energy_network.dispatch(allocation_matrix)

        module_results, final_flue_gas = self.process_route.evaluate(
            flue_gas_initial=flue_gas_initial,
            energy_map=energy_map,
            material_map=material_map,
            operation_map=operation_map,
        )

        backend_results = self._evaluate_backend_assets(
            flue_gas_initial=flue_gas_initial,
            energy_map=energy_map,
            operation_map=operation_map,
            front_module_results=module_results,
        )

        accounting_results = module_results + backend_results

        emission_feasible, emission_violations, emission_reports = (
            self.emission_checker.check(final_flue_gas)
        )

        total_module_exergy_destruction = sum(
            r.exergy_destruction for r in accounting_results
        )
        total_module_cost = sum(r.cost for r in accounting_results)
        total_module_carbon = sum(r.carbon_emission for r in accounting_results)

        network_exergy_destruction = network_result.get("total_exergy_loss", 0.0)
        network_cost = network_result.get("total_transmission_cost", 0.0)
        network_carbon = network_result.get("total_transmission_carbon", 0.0)

        total_exergy_destruction = (
            network_exergy_destruction + total_module_exergy_destruction
        )
        total_cost = network_cost + total_module_cost
        total_carbon = network_carbon + total_module_carbon

        source_capacity_violation = sum(
            network_result.get("source_capacity_violation", {}).values()
        )
        edge_capacity_violation = sum(
            network_result.get("edge_capacity_violation", {}).values()
        )
        explicit_capacity_violation = (
            source_capacity_violation + edge_capacity_violation
        )
        network_structure_violation = (
            0.0
            if network_result.get("feasible", True) or explicit_capacity_violation > 0.0
            else 500.0
        )
        network_violation = explicit_capacity_violation + network_structure_violation

        process_feasible = all(r.feasible for r in accounting_results)
        network_feasible = (
            bool(network_result.get("feasible", True))
            and network_violation <= 1e-6
        )
        system_feasible = process_feasible and emission_feasible and network_feasible

        process_violation = sum(r.total_violation() for r in accounting_results)
        emission_violation = sum(emission_violations.values())
        total_violation = process_violation + emission_violation + network_violation

        return {
            "system_feasible": system_feasible,
            "process_feasible": process_feasible,
            "network_feasible": network_feasible,
            "emission_feasible": emission_feasible,
            "total_violation": total_violation,
            "process_violation": process_violation,
            "network_violation": network_violation,
            "emission_violation": emission_violation,
            "initial_flue_gas": flue_gas_initial,
            "final_flue_gas": final_flue_gas,
            "energy_map": energy_map,
            "network_result": network_result,
            "module_results": module_results,
            "backend_results": backend_results,
            "accounting_results": accounting_results,
            "emission_violations": emission_violations,
            "emission_reports": emission_reports,
            "total_exergy_destruction": total_exergy_destruction,
            "total_cost": total_cost,
            "total_carbon": total_carbon,
            "details": {
                "total_module_exergy_destruction": total_module_exergy_destruction,
                "total_module_cost": total_module_cost,
                "total_module_carbon": total_module_carbon,
                "front_module_count": len(module_results),
                "backend_module_count": len(backend_results),
                "network_exergy_destruction": network_exergy_destruction,
                "network_cost": network_cost,
                "network_carbon": network_carbon,
                "source_capacity_violation": source_capacity_violation,
                "edge_capacity_violation": edge_capacity_violation,
            },
        }
