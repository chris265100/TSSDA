# Phase Objective

P4 determines the SELECT-side operator semantics of the query.

It chooses the main operator regime and incrementally constructs meaningful
time-series or aggregate expressions from the scoped measurements. P4 records
the regime decision in SPO; intermediate and completed expressions are carried
by the state operator store for downstream phases.

# Registration Information

Package registration is declared in `package.yaml`, including phase metadata,
dependencies, outputs, mask policy, and execution policy.

This package contains nine action-level skills:

* `OperatorRegimeSkill`: decides the main operator regime.
* `SeriesExpressionSkill`: extends or completes a SeriesExpr.
* `AggregateAtomSkill`: creates an initial AggAtom from a SeriesExpr.
* `AggregateExpressionSkill`: extends or completes an AggExpr.
* `FunctionSourceSkill`: selects built-in, UDF, or mixed function sources.
* `AggregateFunctionFamilySkill`: selects an aggregate function family.
* `TimeSeriesFunctionFamilySkill`: selects a time-series function family.
* `OperatorFamilySkill`: selects a concrete arithmetic operator.
* `ComponentInstantiationSkill`: instantiates the selected function or operator.

# Input Description

The decision input contains:

- `phase_id`, `skill_id`, `action_id`, `decision_name`: current skill identity.
- `schema_view`: active working schema from P1, or compact database meta schema
  if no working schema exists.
- `query_frame`: current flat QueryFrame with upstream scope, time, and grouping
  artifacts.
- `state`: compact global state with run metadata, phase progress, and SPO.
- `natural_language_semantics`: generated semantic notes, provided once as a
  top-level field when available.
- `action_space`: available candidates after masking and compatibility filtering.
- `candidate_descriptions`: candidate meanings, compatible functions, and
  generation requirements.

P4 skills may additionally receive `SeriesExpr`, `AggExpr`, or the current
`constructing_expression`, together with the relevant expression and output
instructions. These fields describe the local expression being processed and
are not duplicated inside `state`.

# Expression Context

P4 distinguishes two expression layers:

```text
SeriesExpr :=
    SeriesPath
  | TSGF(SeriesExpr, attrs)
  | UnaryOp(SeriesExpr)
  | BinaryOp(SeriesExpr, SeriesExpr | Constant)

AggAtom := AGG(SeriesExpr)

AggExpr :=
    AggAtom
  | ScalarCompatibleTSGF(AggExpr, attrs)
  | UnaryOp(AggExpr)
  | BinaryOp(AggExpr, AggExpr | Constant)
```

An aggregate function may wrap a valid SeriesExpr. After aggregation, only a
scalar-compatible time-series function may wrap an AggExpr.

# Decision Principles

Construct expressions that are semantically consistent with the working
schema and upstream time and grouping decisions. Respect data-type and function
compatibility, avoid meaningless nesting, and stop extending an expression when
no useful operation remains. P4 builds expression candidates; final SELECT
composition and rendering are handled by later phases.
