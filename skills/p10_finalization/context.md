# Phase Objective

P10 converts the completed generation state into final IoTDB NL2SQL data
samples. The runner executes each SQL on IoTDB and admits only passing samples.

# Input Description

The SQL generation input primarily contains:

- `query_frame`: the complete SQL construction plan produced by upstream phases;
- `assembly_contract`: the normalized executable view of QueryFrame, including
  SQL query mode, SELECT/INTO pairings, final clauses, alignment, ordering, and
  pagination;
- `working_schema`: the selected IoTDB tree-model devices, measurements, paths,
  data types, and metadata available to the query;
- `base_info`: the P10 skill identity and deterministic assembly action;
- `output_instruction`: the exact structured response contract.
- Repair context, when present at the end: relevant successful rules by error
  code, followed by failed SQL and errors. Correct only execution problems and
  preserve existing semantic decisions. Return replacements in target order,
  adding a top-level one-sentence `rule` alongside `samples` for repair responses.

# Generation Requirements

Generate complete SQL statements that faithfully realize QueryFrame and use
only schema elements available in working schema. Generate matching user-style
natural-language requests that describe the same intents and constraints.

The request should be natural, meaningful, unambiguous, and not needlessly
verbose. Prioritize reasonable, executable, schema-grounded SQL over query
length or complexity.

Use the current input to identify different semantic scenarios that can be
supported by the available QueryFrame and working schema. When
`max_sample_count` is provided, treat it as an upper bound rather than a fixed
target count. Prefer to use the full allowance (normally three samples) when
the input supports that many reasonable, legal, and meaningfully different
NL2SQL pairs. If it does not, return fewer high-quality samples instead of
duplicates or invalid SQL. Preserve SQL validity, schema grounding, data
quality, and natural-language quality. For multiple samples, prefer
meaningfully different supported SELECT variants and semantic combinations
rather than superficial formatting changes or paraphrases of the same SQL.

Strictly follow the supplied `query_frame`. Do not generate SQL clauses,
expressions, filters, aliases, limits, paths, or semantic constraints that are
not present in QueryFrame. If `query_frame.select` is a list of groups, each
generated SQL must choose exactly one group from that list as its SELECT list;
do not combine expressions across groups or invent new SELECT expressions.

Use `assembly_contract.sql_query_mode` to distinguish `SELECT` from `SELECT
LAST`. `query_frame.query_type` stores the P0 intent description and must not be
interpreted as the SQL query keyword. Each sample must use one complete
`assembly_contract.select_variants` entry.

If multiple samples are generated, every sample must have a different SQL
statement and a different natural-language request. Do not create multiple
samples by only paraphrasing the same SQL, and do not reuse the same NL request
for different SQL statements.

If `query_frame.into.mappings` is present, each mapping belongs to the SELECT
group with the same zero-based `select_index`. Use that mapping's complete
`clause` and never pair it with another SELECT group. Time-aligned INTO uses
the default layout without an ALIGN BY clause; device-aligned INTO must end
with `ALIGN BY DEVICE`.

Treat `query_frame.order_by` as the ordered list of items after `ORDER BY`.
Render every non-null `limit`, `offset`, `slimit`, and `soffset` value with its
corresponding SQL keyword. Keep clause order as `ORDER BY`, `FILL`, `SLIMIT`,
`SOFFSET`, `LIMIT`, `OFFSET`, and then `ALIGN BY`. Do not add or omit ordering
and pagination clauses.

The final SELECT clause must not contain path-qualified measurements. If the
input `query_frame.select` contains full paths, strip the path prefix in SELECT
expressions and keep only the measurement name. Path-qualified devices or
patterns should remain in FROM when needed.

# Output Format

Return JSON only:

```json
{
  "samples": [
    {
      "sql": "complete IoTDB SQL",
      "nl": "matching natural-language request"
    }
  ]
}
```
