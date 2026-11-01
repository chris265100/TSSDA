# Phase Objective

P2 determines the temporal semantics of the query.

It decides whether the query uses no time, point time, range time, window time, or latest time, and generates the required time artifacts for downstream phases.

# Registration Information

Package registration is declared in `package.yaml`, including phase metadata, dependencies, outputs, mask policy, and execution policy.

This package contains four action-level skills:

* `TemporalModeSkill`: decides the main temporal mode.
* `TimeScopeSkill`: decides the time range or boundary.
* `WindowSizeSkill`: decides the fixed window interval.
* `WindowStepSkill`: decides the optional sliding step.

# Input Description

The decision input contains:

- `phase_id`, `skill_id`, `action_id`, `decision_name`: current skill identity.
- `schema_view`: active working schema from P1, or compact database meta schema
  if no working schema exists.
- `query_frame`: current flat QueryFrame, including prior scope selections and
  any existing `time` artifact.
- `state`: compact global state with run metadata, phase progress, and SPO.
- `natural_language_semantics`: generated semantic notes, especially
  `temporal_semantic` for an existing window.
- `action_space`: available candidates after masking.
- `candidate_descriptions`: candidate meanings and generation requirements.

# Grammar Context

P2 mainly corresponds to:

```sql
WHERE time ...
GROUP BY ([startTime, endTime), interval [, slidingStep])
```

Time literals should use the standard `YYYY-MM-DDTHH:MM:SS` form, for example
`2017-11-01T00:00:00`, rather than epoch-millisecond timestamps. It provides
temporal artifacts for later `GROUP BY`, `WHERE`, and `FILL` decisions.
