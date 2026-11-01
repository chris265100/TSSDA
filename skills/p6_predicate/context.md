# Phase Objective

P6 decides whether the query should use predicate clauses and, when needed,
generates value-side `WHERE` predicates and aggregate-side `HAVING` predicates.

# Registration Information

`package.yaml` registers three skills:

* `PredicateModeSkill` (P6A1): chooses `none`, `where_only`,
  `having_only`, or `both`.
* `WhereValuePredicateSkill` (P6A2): directly calls the SQL generation agent
  to generate value-related WHERE predicates.
* `HavingPredicateSkill` (P6A3): directly calls the SQL generation agent to
  generate HAVING predicates over level-1 aggregate expressions.

# Input Description

Common inputs are:

- skill identity: `phase_id`, `skill_id`, `action_id`, `decision_name`;
- `schema_view`: active working schema, otherwise compact meta schema;
- `query_frame`: current flat QueryFrame;
- `state`: compact run metadata, phase progress, and SPO;
- `natural_language_semantics`: semantic notes, provided once when available;
- `action_space` and `candidate_descriptions`: available decisions and meanings.

Before the model call, the runner uses flags from earlier phases to mask P6A1.
If both `P2.where_time` and
`P3.where_value` are false, WHERE modes are masked. If either `P3.having_flag`
or `P4.having_flag2` is false, HAVING modes are masked.

P6A2 additionally receives `working_measurements` and returns:

```json
{
  "where": "temperature > 25 AND humidity < 80"
}
```

The generated predicate must contain only value-related conditions. Existing
time predicates from `query_frame.time` are merged by the runner.

P6A3 additionally receives `level_one_aggregate_expressions` and returns:

```json
{
  "having": "AVG(temperature) > 20 OR COUNT(humidity) >= 10"
}
```

# Decision Principles

Use predicates only when they are supported by prior phase flags. WHERE
predicates should be grounded in the working schema measurements and avoid time
conditions. HAVING predicates should be grounded in available level-1 aggregate
expressions and should remain meaningful for the current SELECT and GROUP BY
context.
