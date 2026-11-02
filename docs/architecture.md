# Generation and reuse

## Dependency graph and constraints

`tssda/constraints.py` builds the phase DAG from each package's
`required_phases`, `required_spo` and `required_flags`. The runner uses
`graphlib.TopologicalSorter` to obtain the execution order and rejects cycles.
The bundled dependencies currently yield P0 through P10. The graph and rules
are manually curated; this repository does not automatically mine rules from
documentation or DBMS source code.

The executable rules are grouped as `R_c = R_T + R_D + R_A`. P2 and P5 use
these predicates directly in their enabled action masks. Other phases retain
their existing specialized masks and type checks.

| Rule | Group | Decision | Predicate |
| --- | --- | --- | --- |
| T01 | Temporal semantics | P2A2 | Range-scope decisions require range-time mode when the mode is known. |
| T02 | Temporal semantics | P2A3 | Window-size decisions require window-time mode when the mode is known. |
| T03 | Temporal semantics | P2A4 | Window-step decisions require window-time mode and a resolved window size. |
| D01 | SQL compatibility | P2A1 | Aggregation/computation intent cannot select the raw SELECT LAST route. |
| A01 | Runtime/Analyzer compatibility guard | P5A1 | The operator regime must have a nonempty compatible expression pool. |
| A02 | Runtime/Analyzer compatibility guard | P5A2 | Alias generation requires a SELECT list and a single-device scope. |

The source groups classify the rationale for the manually encoded predicates;
they do not claim automatic extraction or exhaustive Analyzer coverage.

## Incremental Skill routing

Each branching action exposes only its immediate next Skill. After execution
updates SPOs, flags and the query frame, the runner resolves the next edge
again. For example, `window_time` activates `WindowSizeSkill`; only after that
decision does window-time State activate `WindowStepSkill`. Hybrid grouping,
scope binding, combined predicates and ordering/pagination work the same way.

Every branching step can receive contextual routing guidance and has a unique
trace ID, including repeated P4 exploration decisions. Failed model-output
attempts are discarded from the committed routing trace. The typed P4
component construction pipeline remains an atomic runner operation. Function
selection and expression construction still use the existing action/type
filters. Distribution priors cover the P4 continue/stop decisions as well as
function-family/function choices after validated data activates learning.

## Temporal-aware hierarchical memory

Both memory levels share feature-wise profile similarity, validation confidence
and the reuse penalty:

```
Conf = (Ns + 1) / (Ns + Nf + 2)
Score = Sim(current, remembered) * Conf / (1 + lambda * Nr)
P(entry) = Score(entry) / sum(Score for entries in TopK)
```

State memory retrieves one semantic decision. Operator memory retrieves one
typed expression template and adds its bound expression to the State operator
pool used by P4/P5. Both retain Top-K candidates and sample one by normalized
score. Existing provisional admission and final-validation settlement remain
in place; unused expressions are not admitted as validated experience.

Temporal profiles include intent, span, range overlap, query/window shape and:

| Feature | Available evidence |
| --- | --- |
| Sampling interval | Median adjacent timestamp interval, bucketed from subsecond to day-or-more. |
| Sampling regularity | Whether observed gaps stay within 1.5 times their median. |
| Timestamp alignment | Minimum pairwise timestamp overlap, classified as aligned, partial or asynchronous. |
| Point distribution | Variation of per-series counts in four equal subwindows of the common sampled interval. |

P2 reads bounds for at most eight measurements and at most 64 timestamps per
measurement by default (at most 24 SELECT probes). These observations are
cached in trajectory State and reused for temporal generation and memory.
Timestamp probes share a common time interval; prefixes are trimmed to a common
observed horizon before comparison. These are bounded sample characteristics,
not exact statistics for an entire database. If probes fail, bounds still
work. Missing sampling information, or a query outside the sampled horizon,
remains `UNKNOWN` and does not become invented alignment evidence. Known
alignment, regularity or distribution conflicts reject reuse; very different
sampling-interval buckets also reject reuse.

Expression binding requires exact measurement names and types. The framework
does not substitute alphabetically chosen same-type measurements. Multi-input
expressions also reject observed operands with no common timestamps within a
device. Names/types are the available semantic identity; no external ontology
or unit-conversion system is assumed. Missing observations remain unknown.

A memory hit seeds the P4 expression pool. P4 then makes its normal next
exploration decision and can extend or stop; an aggregate hit can enter the
aggregate loop directly without regenerating its seed. Depth limits remain in
force. Profile version 2 prevents state records from older profiles from being
reused blindly. Older expression records without a temporal profile remain
loadable but are excluded from retrieval; new validated records accumulate
normally.

## Execution experience before generation

Successful repair rules retain up to eight contexts consisting of clause,
operator/function and query-mode signatures. Before generating an artifact,
the shared knowledge base retrieves up to three relevant validated rules by
signature overlap and confidence, then includes them as `execution_knowledge`
in the generation context. Rules are reference advice and do not override the
schema, intent or output contract.

P10 still retrieves error-code-specific advice during repair. A new rule and
its context are admitted only after successful validation of the complete
repair batch. Old error-code-only rules remain usable during repair and gain
contexts after subsequent successful use. No model call is added for retrieval.

## Checks

`python -m pytest` exercises bounded timestamp reads, overlapping ranges with
asynchronous samples, both memory levels, probability normalization,
persistence, topology/cycle checks, action masks, incremental routes, proactive
experience and complete P0-P10 trajectories with mocked model/IoTDB services.
A live run still requires the user's configured model endpoint and IoTDB.
