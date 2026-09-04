# Steel-WHUN

## Overview

Steel-WHUN is a modular computational framework for optimization of waste heat and secondary energy networks in integrated steelworks subject to pollution control constraints.

This repository provides the reusable computational framework and modular interfaces associated with the Steel-WHUN methodology. It includes the source, transport, treatment, conversion, accounting, feasibility and optimization architecture. Numerical reference-case definitions, parameters and reported results are documented in the associated manuscript and Supplementary Methodology. This repository does not contain a paper-specific demonstration case or a frozen numerical reproduction package.

## What is included

- configurable source and stream representations;
- ESP, WFGD and SCR pollution control modules;
- reheater, WHPG, CCPP, ORC, district heating, absorption cooling and heat-buffer sink interfaces;
- serial treatment route state propagation;
- source–sink transport and allocation;
- separate energy, chemical-exergy, physical-exergy, transport-loss, unused-source and economic accounts;
- generic decision encoding and topology decoding;
- adapters for NSGA-II and SPEA2 through `pymoo`;
- nondominated archive filtering;
- balanced-solution selection;
- Latin hypercube uncertainty sampling and one-at-a-time sensitivity utilities;
- framework-level interface and software-consistency tests.

## Scope

The release is limited to the reusable framework and interfaces. Paper-specific source capacities, route definitions, pollutant inputs, tariffs, scenario configurations, Pareto datasets and numerical outputs are documented in the manuscript and supporting materials and are not distributed as an executable case in this repository.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Module API

Every module returns a `ModuleResult` containing the outlet state, useful outputs, auxiliary requirements, removal quantities, exergy destruction, cost contribution and feasibility checks. Constructor arguments expose configurable model coefficients and operating domains.

The heat-buffer module is a static residual-heat sink. It has no state of charge, charging or discharging schedule, standing loss, storage duration or inter-period balance.

## Network API

`ProcessRoute` propagates a `FlueGasState` sequentially through active treatment and recovery modules. `EnergyNetwork` connects configurable `EnergySource` objects to target modules through `TransmissionEdge` objects and reports delivered energy, delivered exergy, losses and costs.

When allocations exceed a source capacity, the network preserves the requested allocation ratios while scaling the physical dispatch to the available source capacity. Source- and edge-capacity violations are reported separately in the network result. Backend CCPP accounting consumes the delivered network energy and exergy ledgers; unallocated fuels are never introduced implicitly.

Delivered `EnergyStream.cost_rate` and `EnergyStream.carbon_rate` retain source-side contributions only. Transmission cost and carbon are carried in edge metadata and aggregated once by `EnergyNetwork`, preventing terminal modules and `SystemEvaluator` from counting transport contributions twice.

## Optimization API

`DecisionCodec` maps continuous source-to-target allocations, operating variables and discrete topology slots to a network decision. `WHUNProblem` connects that decision to user-supplied objective and constraint functions and can create a `pymoo` problem. `run_nsga2` and `run_spea2` accept a user-defined population size, generation count and random seed. No optimizer budget or seed is presented as universally preferred. `nondominated_indices` filters a minimization archive, and `select_balanced` provides normalized-distance post-processing.

## Uncertainty and sensitivity

`latin_hypercube` generates bounded parameter designs, `propagate_uncertainty` evaluates a user callback and `one_at_a_time` performs a local parameter sweep. Users are responsible for defining scientifically appropriate bounds and interpretations.

## Tests

```bash
python -m pytest -q
```

The tests check framework imports, accounting closure, module input/output consistency and network execution. They enforce proportional source-capacity scaling, explicit edge-capacity diagnostics, delivered-exergy continuity, non-overlapping source/transport cost and carbon ledgers, `MaterialInput.cost_rate` accounting and CCPP fuel-identity conservation. In particular, unallocated by-product gases have zero consumption, and zero delivered gas produces zero electrical output. These are framework-level software tests and do not use manuscript results as regression targets.

## Citation

See `CITATION.cff`. Full article metadata can be added after publication without changing the software interfaces.
