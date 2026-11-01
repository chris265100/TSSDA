# Phase Objective

P0 initializes the global SQL generation process by making high-level semantic decisions, including task family, task subtype.

It provides an initial semantic blueprint and coarse generation direction for downstream phases, determining the overall preference of the current SQL generation trajectory.

# Registration Information

Basic registration information for this skill package is declared in `package.yaml`, including phase identity, package metadata, action-level skills, execution order, required inputs, and produced SPO/flag fields.

This package contains the following action-level skills:

- `TaskFamilySkill`: decides the main task family.
- `RetrievalSubtypeSkill`: decides the retrieval-oriented task subtype.
- `ComputationSubtypeSkill`: decides the computation-oriented task subtype.
- `AggregationSubtypeSkill`: decides the aggregation-oriented task subtype.
- `MaterializationSubtypeSkill`: decides the materialization-oriented task subtype.

# Grammar Context

P0 is an initialization phase and does not directly correspond to any concrete SQL grammar clause.

It only produces high-level semantic control signals for downstream grammar-related phases.

# Decision Principles

Each action-level skill makes one local semantic decision from its masked candidate space.

Subtype skills are conditionally activated according to the selected task family and the current generation target.

P0 should only generate coarse-grained initialization results and should not instantiate concrete schema objects, temporal boundaries, operators, predicates, or SQL clauses.