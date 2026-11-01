# Phase Objective

P5 forms one or more SELECT expression groups from the operator expressions
produced by P4, then decides whether selected expressions should receive
readable aliases.

# Registration Information

`package.yaml` registers two skills:

* `SelectExpressionSkill` (P5A1): selects compatible P4 expressions and calls
  the SQL generation agent to form one or more groups in `query_frame.select`.
* `AliasModeSkill` (P5A2): chooses `no_alias` or `use_alias`; alias generation
  runs only for `use_alias`.

# Input Description

Common inputs are:

- skill identity: `phase_id`, `skill_id`, `action_id`, `decision_name`;
- `schema_view`: active working schema, otherwise compact meta schema;
- `query_frame`: current flat QueryFrame;
- `state`: compact run metadata, phase progress, and SPO;
- `natural_language_semantics`: semantic notes, provided once when available;
- `action_space` and `candidate_descriptions`: available decisions and meanings.

P5A1 additionally receives `select_expression_candidates`. It uses `AggExpr`
when that pool is non-empty; otherwise it uses `SeriesExpr`. The generation
agent returns a list of one or more SELECT expression groups. Each group is one
comma-separated string whose expressions are internally compatible:

```json
{
  "select": [
    "AVG(temperature), MAX_VALUE(humidity)",
    "COUNT(temperature)"
  ]
}
```

P5A2 reads the completed `query_frame.select`; it does not receive the P4
operator store. When aliases are enabled, the generation agent preserves every
expression, group boundary, and order, adds aliases to suitable expressions,
and overwrites the same `select` field. If the active working schema contains
more than one target device, P5A2 deterministically selects `no_alias` before
model decision.

# Decision Principles

Select expressions that are semantically compatible within each query. Preserve
P4 expressions exactly during composition. Multiple groups should be
meaningfully different when the candidate pool supports diversity. Add aliases
only when they improve human readability, using concise names that reflect
expression semantics.
