# Phase Objective

P8 decides the IoTDB result alignment and, for materialization tasks, builds a
valid `INTO` mapping from query result columns to target time series.

# INTO Semantics

An INTO item has the form `[ALIGNED] targetDevice(m1, m2, ...)`.

- With default time alignment, all target measurements across all INTO items
  must correspond one-to-one, left-to-right, with the expanded result value
  columns.
- With `ALIGN BY DEVICE`, target-device count must equal source-device count.
  Every target device must contain one target measurement for each SELECT value
  column, in result-column order.
- `ALIGNED` describes target storage layout. It is independent of whether the
  query result is time-aligned or device-aligned.
- Target series must be unique. Existing targets require compatible data
  types; new target series may be created automatically by IoTDB.

INTO is unsupported with `LAST`, `GROUP BY TAGS`, `SLIMIT`, or `SOFFSET`.

# Input Description

Common decision input contains skill identity, compact state with SPO, active
schema, flat QueryFrame, masked actions, and candidate meanings. Runtime flags
are consumed by masks and routing before the model call and are not sent to the
model.

P8A3 additionally receives `selected_alignment`, `selected_writeback_mode`,
`writeback_cardinality_plan`, `working_schema`, and the full `schema_view`.
This is the explicit exception that needs both the active source scope and the
database view to avoid conflicting writeback targets. The plan gives every
SELECT group, its result shape, source devices, and the exact number of target
devices and measurements that must be generated.

# Decision Principles

Use device alignment when several compatible devices should retain separate
device identities in the result. Use time alignment when result columns should
be mapped globally by column order. Enable INTO only for the P0 materialization
trajectory and only when no unsupported upstream clause is present. Derive
clear target names from source schema and result semantics, while avoiding
accidental writes back into source series.
