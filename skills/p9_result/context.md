# Phase Objective

P9 organizes the final IoTDB result through explicit ordering and optional row
or value-column pagination. Result alignment has already been selected in P8.

# Ordering Semantics

- Time-aligned results may be ordered by `TIME` or a valid result expression.
- Device-aligned results may use `DEVICE`, `TIME`, `DEVICE,TIME`,
  `TIME,DEVICE`, or result-expression ordering.
- Every ordering key uses `ASC` or `DESC`. Expression ordering may additionally
  use `NULLS FIRST` or `NULLS LAST`.
- `SELECT LAST` has a special four-column result and may be ordered by `TIME`,
  `TIMESERIES`, `VALUE`, or `DATATYPE`.
- SELECT INTO does not support ORDER BY, so writeback queries use `no_order`.

# Pagination Semantics

- `LIMIT` controls returned rows and `OFFSET` controls the starting row.
- `SLIMIT` controls returned value columns and `SOFFSET` controls the starting
  value column. Time and Device columns are not counted.
- Limits must be positive integers. Offsets are non-negative integers.
- Row and column pagination may be combined.
- SELECT INTO does not support `SLIMIT` or `SOFFSET`, so P9 masks every
  column-pagination policy when P8 enabled INTO.
- The dedicated SELECT LAST grammar does not include pagination, so latest
  queries use `no_pagination`.

# Input Description

Common decision input contains skill identity, compact state with SPO, working
schema, flat QueryFrame, masked actions, and candidate meanings. Runtime flags
are applied before the model call and are not sent to the model.

P9A1 additionally receives P8 alignment, latest-query status, and SELECT
expressions shared by every P5 SELECT group. P9A2 receives the selected ordering
mode and the exact sortable expressions.

P9A3 receives `result_shape`, including per-group value-column counts, minimum
and maximum column counts, source-device count, alignment, and INTO status.
P9A4 receives the selected pagination policy and the same result shape.

# Decision Principles

Prefer ordering that exposes a meaningful chronology, device comparison, or
ranked result. Avoid explicit ordering when the default order already expresses
the intended semantics. Use compact page sizes and conservative offsets so the
query remains likely to return data. Use column pagination only when several
value columns exist and no INTO clause is active.
