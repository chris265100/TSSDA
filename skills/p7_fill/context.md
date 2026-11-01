# Phase Objective

P7 decides whether the query should use an IoTDB `FILL` clause and, when needed,
instantiates the selected fill mode with legal parameters.

# Background

IoTDB `FILL` is used to fill null values in query results. Supported modes are:

* `PREVIOUS`: fill nulls with the previous value. In this phase it uses a time
  threshold, such as `FILL(PREVIOUS, 2m)`.
* `LINEAR`: fill nulls through linear interpolation. It is valid only for
  numeric result columns.
* constant: fill nulls with a literal constant, such as `FILL(0)` or
  `FILL('unknown')`.

Only one fill method is used for the whole query.

# Registration Information

`package.yaml` registers two skills:

* `FillModeSkill` (P7A1): chooses `none`, `previous`, `linear`, or `constant`.
* `FillInstantiationSkill` (P7A2): directly calls the SQL generation agent to
  generate the concrete FILL clause.

# Input Description

Common inputs are:

- skill identity: `phase_id`, `skill_id`, `action_id`, `decision_name`;
- `schema_view`: active working schema, otherwise compact meta schema;
- `query_frame`: current flat QueryFrame;
- `state`: compact run metadata, phase progress, and SPO;
- `natural_language_semantics`: semantic notes, provided once when available;
- `action_space` and `candidate_descriptions`: available decisions and meanings.

P7A1 additionally receives selected SELECT expression type information and the
fill-mode support table. If `P2.temporal_mode` is `point_time` or `latest_time`,
P7A1 deterministically selects `none`. Otherwise, unsupported fill modes are
masked according to selected expression data types:

| Data type | Supported fill modes |
| --- | --- |
| BOOLEAN | PREVIOUS, constant |
| INT32 | PREVIOUS, LINEAR, constant |
| INT64 | PREVIOUS, LINEAR, constant |
| FLOAT | PREVIOUS, LINEAR, constant |
| DOUBLE | PREVIOUS, LINEAR, constant |
| TEXT | PREVIOUS, constant |

P7A2 receives `selected_fill_mode` and expression type information, then returns:

```json
{
  "fill": {
    "mode": "PREVIOUS",
    "clause": "FILL(PREVIOUS, 2m)",
    "parameter": "2m"
  }
}
```

# Decision Principles

Use `none` when the query is point-like or latest-point oriented. Use `linear`
only when all selected result expressions are numeric. Use `previous` for
time-continuity semantics. Use `constant` when a stable default value is
natural for the selected expression data types.
