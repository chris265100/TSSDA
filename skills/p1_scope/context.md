# Phase Objective

P1 determines the query scope over the hierarchical time-series schema.

It makes structured decisions about the query object, device scope, measurement scope, and tag context, providing the working schema boundary for downstream phases.

# Registration Information

Package registration is declared in `package.yaml`, including phase metadata, dependencies, outputs, mask policy, and execution policy.

This package contains four action-level skills:

- `ScopePatternSkill`: decides the query scope pattern.
- `DeviceScopeSkill`: decides the device-level scope.
- `MeasurementScopeSkill`: decides the measurement-level scope.
- `TagScopeSkill`: decides the tag-related scope context.

# Input Description

The decision input contains:

- `phase_id`, `skill_id`, `action_id`, `decision_name`: current skill identity.
- `schema_view`: active working schema if available, otherwise compact database
  meta schema with devices and measurements.
- `query_frame`: current flat QueryFrame with upstream intent and generated
  partial artifacts.
- `state`: compact global state with run metadata, phase progress, and SPO.
- `natural_language_semantics`: generated semantic notes, provided once as a
  top-level field when available.
- `action_space`: available candidates after masking.
- `candidate_descriptions`: candidate meanings and generation requirements.

P1-specific optional fields:

- `candidate_routing`: downstream skills activated by each scope pattern.
- `tag_schema`: database tag metadata, when available.
- `working_tag`: tag metadata restricted to the current working schema.
- `expected_output_format`: optional structured fields for scope decisions.

# Grammar Context

P1 mainly corresponds to the scope semantics behind the `FROM` clause.

It does not directly generate the final SQL clause, but determines schema objects used by later phases for `SELECT`, `GROUP BY`, `WHERE`,... .

# Decision Principles

Each action-level skill makes one local semantic decision from its masked candidate space.

P1 should localize the global schema into a valid working scope and preserve structured information for downstream decisions.

P1 only binds schema-level scope information and should not instantiate time ranges, operators, predicates, aggregation modes, or final SQL clauses.
