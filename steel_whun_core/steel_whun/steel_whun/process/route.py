"""
Process route evaluator for dynamically ordered pollution control and heat
recovery modules. Repeated module instances receive independent operating
parameters; when no dynamic sequence is supplied, the initialized order is used.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any

from steel_whun.core.states import (
    FlueGasState,
    EnergyStream,
    MaterialInput,
    ModuleResult,
)
from steel_whun.modules.base import PollutionModule


class ProcessRoute:
    'ProcessRoute component used by the Steel-WHUN core framework.'

    BACKEND_MODULE_TYPES = {
        "ccpp",
        "orc",
        "district_heating",
        "absorption_cooling",
        "heat_buffer_sink",
    }

    MODULE_TEMPERATURE_WINDOWS = {
        "scr": {"min_inlet_K": 533.15, "max_inlet_K": 723.15, "module_type": "SCR"},
        "wfgd": {"min_inlet_K": 323.15, "max_inlet_K": 473.15, "module_type": "WFGD"},
        "esp": {"min_inlet_K": 323.15, "max_inlet_K": 673.15, "module_type": "ESP"},
        "wesp": {"min_inlet_K": 313.15, "max_inlet_K": 373.15, "module_type": "WESP"},
        "reheater": {"min_inlet_K": 300.0, "max_inlet_K": 800.0, "module_type": "Reheater"},
        "whpg": {"min_inlet_K": 300.0, "max_inlet_K": 1100.0, "module_type": "WHPG"},
    }

    def __init__(self, modules: List[PollutionModule], name: str = "process_route"):
        self.name = name
        self.modules = modules
        self.module_index = {m.name: i for i, m in enumerate(modules)}

        
        self.modules_pool = {m.name.lower(): m for m in modules}

    def _is_backend_module(self, module: PollutionModule) -> bool:
        module_type = getattr(module, "module_type", "").lower()
        module_name = getattr(module, "name", "").lower()
        return (
            module_type in self.BACKEND_MODULE_TYPES
            or module_name in self.BACKEND_MODULE_TYPES
        )

    def _inject_downstream_info(
            self,
            module: PollutionModule,
            index: int,
            operation_params: Dict[str, Any],
            active_modules: List[PollutionModule],  
    ) -> Dict[str, Any]:
        'Execute the inject downstream info calculation.'
        op_params_run = operation_params.copy()

        if module.name.lower() == "whpg" or module.module_type == "whpg":
            
            if 0 <= index < len(active_modules) - 1:
                next_module = active_modules[index + 1]
                next_module_name = next_module.name.lower()

                if hasattr(next_module, "min_temperature"):
                    op_params_run.setdefault("next_inlet_temperature_min", next_module.min_temperature)
                if hasattr(next_module, "max_temperature"):
                    op_params_run.setdefault("next_inlet_temperature_max", next_module.max_temperature)

                if "next_inlet_temperature_min" not in op_params_run or "next_inlet_temperature_max" not in op_params_run:
                    for key, window_info in self.MODULE_TEMPERATURE_WINDOWS.items():
                        if key in next_module_name or next_module_name in key:
                            op_params_run.setdefault("next_inlet_temperature_min", window_info["min_inlet_K"])
                            op_params_run.setdefault("next_inlet_temperature_max", window_info["max_inlet_K"])
                            break

                op_params_run.setdefault("next_module_name", next_module.name)

        return op_params_run

    def evaluate(
            self,
            flue_gas_initial: FlueGasState,
            energy_map: Optional[Dict[str, List[EnergyStream]]] = None,
            material_map: Optional[Dict[str, List[MaterialInput]]] = None,
            operation_map: Optional[Dict[str, Dict[str, Any]]] = None,
            sequence: Optional[List[str]] = None,  
    ) -> tuple[List[ModuleResult], FlueGasState]:
        'Execute the evaluate calculation.'
        energy_map = energy_map or {}
        material_map = material_map or {}
        operation_map = operation_map or {}

        
        active_sequence_names = sequence or operation_map.get("route_control", {}).get("sequence", None)

        if active_sequence_names:
            active_modules = []
            for name in active_sequence_names:
                name_lower = name.lower()
                if name_lower in ["empty", "bypass", "none", "0"]:
                    continue  
                if name_lower in self.modules_pool:
                    active_modules.append(self.modules_pool[name_lower])
                else:
                    
                    matched = False
                    for p_name, p_obj in self.modules_pool.items():
                        if p_name in name_lower or name_lower in p_name:
                            active_modules.append(p_obj)
                            matched = True
                            break
                    if not matched:
                        raise ValueError(
                            f"Module name '{name}' specified by the superstructure "
                            "is not registered in the initialized module pool."
                        )
        else:
            # If no dynamic sequence is supplied, use the initialized fixed order.
            active_modules = [
                module for module in self.modules
                if not self._is_backend_module(module)
            ]

        current_state = flue_gas_initial.copy()
        results: List[ModuleResult] = []

        
        instance_counts: Dict[str, int] = {}

        for i, module in enumerate(active_modules):
            m_name = module.name
            instance_counts[m_name] = instance_counts.get(m_name, 0) + 1
            
            inst_key = f"{m_name}_{instance_counts[m_name]}"

            
            energy_inputs = energy_map.get(inst_key, energy_map.get(m_name, []))
            material_inputs = material_map.get(inst_key, material_map.get(m_name, []))
            operation_params = operation_map.get(inst_key, operation_map.get(m_name, {}))

            
            operation_params = self._inject_downstream_info(module, i, operation_params, active_modules)

            
            result = module.evaluate(
                flue_gas_in=current_state,
                energy_inputs=energy_inputs,
                material_inputs=material_inputs,
                operation_params=operation_params,
            )

            
            result.module_name = inst_key

            results.append(result)
            current_state = result.flue_gas_out.copy()

        return results, current_state

    def module_names(self) -> List[str]:
        return [m.name for m in self.modules]

    def summary(self) -> Dict[str, Any]:
        return {
            "route_name": self.name,
            "pool_modules": self.module_names(),
        }

