# Apache IoTDB tree-model SQL generation knowledge

This knowledge pack is a curated generation-time summary of the official
Apache IoTDB tree-model SQL syntax. It targets the
tree model used by this repository. The table model is a different SQL dialect
and must not be mixed into generated queries.

## 1. Data model and paths

- A tree-model database is a hierarchical path beginning with `root`, for
  example `root.factory`.
- A device is a path below the database, for example
  `root.factory.workshop1.machine1`.
- A time series is a device path plus a measurement, for example
  `root.factory.workshop1.machine1.temperature`.
- In `SELECT`, a measurement suffix such as `temperature` is resolved against
  the `FROM` prefix. A full path may be used where the grammar permits it.
- `*` matches one path node and `**` matches zero or more path nodes. Use
  wildcards only when the schema context makes the resulting columns and types
  meaningful.
- Tree-model queries usually align result columns by time. `ALIGN BY DEVICE`
  instead returns `Time` and `Device` as leading columns followed by
  measurement columns shared across devices.
- Never invent a device, measurement, tag key, alias, or data type. Use only
  items in the supplied schema snapshot.

## 2. Read-only SELECT shape and clause order

Generate exactly one read-only query. This workflow excludes `SELECT INTO`
because it mutates the database during validation.

Typical tree-model shape:

```sql
SELECT [LAST] select_expression [, ...]
FROM prefix_path [, ...]
[WHERE predicate]
[GROUP BY grouping_clause]
[HAVING aggregate_predicate]
[ORDER BY TIME|DEVICE|expression [ASC|DESC] [, ...]]
[FILL(PREVIOUS [, duration] | LINEAR | constant)]
[SLIMIT positive_integer [SOFFSET nonnegative_integer]]
[LIMIT positive_integer [OFFSET nonnegative_integer]]
[WITHOUT NULL ANY|ALL [(output_expression [, ...])]]
[ALIGN BY DEVICE]
```

Use the ordering accepted by the installed server. Do not copy relational SQL
features into tree SQL: no arbitrary relational `JOIN`, CTE, `UNION`, or table
subquery should be generated for this tree-model workflow.

## 3. Raw and aggregate query regimes

- A raw query selects measurements, constants, scalar expressions, or
  time-series-generating functions and retains timestamps.
- An aggregate query uses aggregate functions such as `COUNT`, `SUM`, `AVG`,
  `MIN_VALUE`, `MAX_VALUE`, `EXTREME`, `FIRST_VALUE`, `LAST_VALUE`,
  `MIN_TIME`, and `MAX_TIME`.
- Do not mix a raw series expression and an aggregate expression in the same
  SELECT list unless the installed IoTDB grammar explicitly supports that
  exact combination.
- Nested expressions over aggregate results are allowed in supported forms,
  for example `SIN(AVG(temperature))` or `AVG(temperature) + 1`.
- Aggregation input types must be compatible: `SUM` and `AVG` require numeric
  measurements; boolean or text measurements are suitable for `COUNT`,
  `FIRST_VALUE`, `LAST_VALUE`, and matching predicates.
- Aliases use `AS`, for example `AVG(temperature) AS avg_temperature`.

## 4. WHERE predicates

- Time is an implicit column. Use bounded ranges whenever possible:

```sql
WHERE time >= 2024-01-01T00:00:00
  AND time < 2024-02-01T00:00:00
```

- Preserve interval boundary semantics. Prefer half-open ranges `[start,end)`
  for downsampling consistency.
- Value predicates may use comparison operators, `BETWEEN`, `IN`, `LIKE`,
  `REGEXP`, `IS NULL`, and `IS NOT NULL` when compatible with the measurement
  type.
- Combine predicates with explicit parentheses when mixing `AND`, `OR`, and
  `NOT`.
- Numeric constants must be used with numeric series, booleans with boolean
  series, and quoted strings with text/string series.
- Predicate paths must resolve unambiguously against the `FROM` scope.

## 5. Time-window aggregation

Use fixed or sliding windows:

```sql
SELECT AVG(temperature), MAX_VALUE(humidity)
FROM root.weather.station
GROUP BY([2024-01-01T00:00:00, 2024-02-01T00:00:00), 1d)
```

Sliding windows supply an interval and a sliding step:

```sql
GROUP BY([2024-01-01T00:00:00, 2024-01-08T00:00:00), 6h, 1h)
```

- Interval and sliding-step durations must be positive.
- Duration units include `ms`, `s`, `m`, `h`, `d`, `w`, and natural calendar
  units such as `mo` where supported.
- Pick a window size appropriate for the supplied data span; avoid producing
  millions of groups.
- The left/right boundary brackets are semantic and must be reflected in the
  natural-language question.
- `__endTime` can be selected for event-segmented groups when an end timestamp
  is useful.

## 6. Event and count segmentation

These clauses create variable-length groups and require aggregate SELECT
expressions.

Variation grouping starts a new group when a control value changes beyond a
threshold:

```sql
GROUP BY VARIATION(control_expression [, delta] [, ignoreNull=true|false])
```

- The control expression must resolve to one series/expression, not a wildcard
  expansion.
- `delta` must be meaningful for the control type.

Condition grouping forms groups controlled by a boolean predicate and a keep
condition. The latest manual's grammar block spells the null option
`ignoreNull`, while examples on the same page may spell it `ignoringNull`.
Use the syntax form below and rely on execution validation for the
installed server version:

```sql
GROUP BY CONDITION(boolean_predicate, keep_condition [, ignoreNull=true|false])
```

Session grouping separates points when the gap exceeds the supplied duration:

```sql
GROUP BY SESSION(30m)
```

Count grouping creates groups by a number of non-null control points:

```sql
GROUP BY COUNT(control_expression, positive_size [, ignoreNull=true|false])
```

- Do not combine incompatible segmentation clauses in one query.
- `GROUP BY VARIATION` is not combined with `GROUP BY LEVEL` in the documented
  tree-model syntax.
- With `ALIGN BY DEVICE`, each device is segmented independently.

## 7. Cross-series grouping

Group same-name measurements by path level:

```sql
SELECT COUNT(status)
FROM root.factory.**
GROUP BY LEVEL = 2
```

Time and level grouping may be combined:

```sql
GROUP BY([start, end), 1h), LEVEL = 2
```

Group by schema tags only when the supplied tag schema proves that the tag keys
exist:

```sql
SELECT AVG(temperature)
FROM root.factory.**
GROUP BY TAGS(city, workshop)
```

Time and tag grouping may be combined in supported releases:

```sql
GROUP BY([start, end), 1h), TAGS(city)
```

- LEVEL values refer to concrete path positions and must be valid for the
  supplied paths.
- GROUP BY TAGS requires aggregates and tagged time series.
- Some installed releases impose restrictions on `HAVING`, pagination, or
  `ALIGN BY DEVICE` together with GROUP BY TAGS. Prefer simpler legal forms;
  execution validation is authoritative.

## 8. HAVING

- `HAVING` filters aggregate results and is used with aggregate queries and a
  `GROUP BY` clause.
- Its expressions must be aggregate expressions legal for the grouping mode.
- Do not compare an aggregate to a raw series in HAVING.
- With GROUP BY LEVEL, avoid device-qualified aggregate paths that conflict
  with the grouped output path.

Example:

```sql
SELECT COUNT(temperature), AVG(temperature)
FROM root.weather.station
GROUP BY([start, end), 1h)
HAVING COUNT(temperature) > 10
```

## 9. Missing-value fill

Supported fill forms are:

```sql
FILL(PREVIOUS)
FILL(PREVIOUS, 2m)
FILL(LINEAR)
FILL(0)
FILL(false)
FILL('unknown')
```

- `LINEAR` is appropriate only for numeric value columns.
- A constant must be compatible with every selected value column it fills.
- Point-time and `LAST` queries generally should not receive an unrelated fill
  clause.
- Fill is applied to query results; describe it explicitly in the question.

## 10. Ordering and pagination

- In time-aligned output, use `ORDER BY TIME ASC|DESC`.
- Under `ALIGN BY DEVICE`, ordering may include `DEVICE` and `TIME`, for
  example `ORDER BY DEVICE ASC, TIME DESC ALIGN BY DEVICE`.
- Arbitrary result expressions may be ordered where the installed release
  supports them; `NULLS FIRST` or `NULLS LAST` can make null ordering explicit.
- `LIMIT`/`OFFSET` paginate rows.
- `SLIMIT`/`SOFFSET` paginate selected result series/columns.
- All limits must be positive. Offsets are nonnegative and should be small
  enough to leave meaningful results.
- A question must mention ordering, top/bottom intent, skipped rows, and series
  pagination whenever those clauses affect the answer.

## 11. Null handling and alignment

- The versioned tree-model manual documents `WITHOUT NULL ANY` (remove a row
  when any checked output is null) and `WITHOUT NULL ALL` (remove a row when
  all checked outputs are null). An optional parenthesized output-column or
  output-expression list narrows the check. Verify availability and exact
  output names on the installed version.
- Use `ALIGN BY DEVICE` when applying the same measurement-level query to
  multiple devices and a compact device-column result is desired.
- Do not use suffix measurements in ALIGN BY DEVICE unless they exist with
  compatible types across the selected device scope.

## 12. LAST queries

`SELECT LAST measurement FROM device` returns the latest point for a series.
It may include a lower time bound. Do not combine LAST with unrelated grouping
or fill clauses.

## 13. Safety and generation-quality rules

1. Output one statement only.
2. Output a read-only `SELECT`; never generate `INSERT`, `DELETE`, `DROP`,
   `CREATE`, `ALTER`, `LOAD`, `FLUSH`, `SELECT INTO`, or administrative SQL.
3. Use only schema paths and types supplied in the prompt.
4. Keep time filters inside the real data bounds supplied in the prompt.
5. Prefer predicates grounded in supplied sample values. If no sample values
   are available, choose conservative type-correct predicates rather than
   inventing domain facts.
6. Every requested temporal operation must be represented in the SQL and later
   verbalized in the question.
7. The installed server parser and execution engine are the final authority;
   a generated query is admitted only after it executes successfully.
