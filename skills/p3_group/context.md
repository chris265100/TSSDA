# Phase Objective

P3 determines the GROUP BY semantics of the query.

It decides whether to use no grouping, segmentation aggregation, cross-series grouping, or hybrid grouping, and generates the required GROUP BY artifacts for downstream phases.

# Registration Information

Package registration is declared in `package.yaml`, including phase metadata, dependencies, outputs, mask policy, and execution policy.

This package contains three action-level skills:

* `GroupParadigmSkill`: decides the overall GROUP BY mode.
* `SegmentAxisSkill`: decides the time/event segmentation axis.
* `GroupAxisSkill`: decides the cross-series grouping axis.

# Input Description

The decision input contains:

- `phase_id`, `skill_id`, `action_id`, `decision_name`: current skill identity.
- `schema_view`: active working schema from P1, or compact database meta schema
  if no working schema exists.
- `query_frame`: current flat QueryFrame, including scope selections, tag
  context, and any P2 `time` artifact.
- `state`: compact global state with run metadata, phase progress, and SPO.
- `natural_language_semantics`: generated semantic notes, especially temporal
  window meaning from P2.
- `action_space`: available candidates after masking.
- `candidate_descriptions`: candidate meanings and generation requirements.

# Grammar Context

P3 mainly corresponds to:

```sql
GROUP BY ([startTime, endTime), interval [, slidingStep])
GROUP BY VARIATION(...)
GROUP BY CONDITION(...)
GROUP BY SESSION(...)
GROUP BY COUNT(...)
GROUP BY LEVEL = ...
GROUP BY TAGS(...)
```
