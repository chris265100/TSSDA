"""Global state and query-frame structures for the TSSDA data agent."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .memory import MemoryRuntimeContext
from .schema import DatabaseSchema


PHASE_IDS = [f"P{index}" for index in range(11)]


@dataclass
class SkillExecutionRecord:
    """One action-level skill execution result kept for traceability."""

    phase_id: str
    skill_id: str | None
    action_id: str | None
    decision_name: str | None
    selected_action: Any
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseOutput:
    """Uniform phase output used by runners and adapters."""

    phase_id: str
    slot_name: str | None = None
    selected_action: dict[str, Any] = field(default_factory=dict)
    spo: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    query_frame_patch: dict[str, Any] = field(default_factory=dict)
    natural_language_semantics: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryFrame:
    """Phase-organized SQL candidate frame.

    `phase_frames` is the agent-facing storage. The legacy clause fields are
    kept as a flat rendering view so existing renderers can still consume this
    object while skills gradually move to phase-scoped patches.
    """

    phase_frames: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_phase: str | None = None

    query_type: str = "select"
    sql_query_mode: str = "select"
    time: dict[str, Any] | str = field(default_factory=dict)
    measurement: list[str] = field(default_factory=list)
    tag: dict[str, Any] = field(default_factory=dict)
    select: str | list[str] | list[dict[str, Any]] = ""
    from_paths: list[str] = field(default_factory=list)
    where: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    having: list[str] = field(default_factory=list)
    fill: dict[str, Any] | None = None
    into: dict[str, Any] | None = None
    align_by: str | None = "time"
    order_by: list[str] = field(default_factory=list)
    limit: int | None = None
    offset: int | None = None
    slimit: int | None = None
    soffset: int | None = None

    def start_phase(self, phase_id: str) -> None:
        """Mark the frame as currently receiving updates for a phase."""

        self.current_phase = phase_id
        self.phase_frames.setdefault(phase_id, {})

    def get_phase_frame(self, phase_id: str) -> dict[str, Any]:
        """Return a copy of one phase frame."""

        return deepcopy(self.phase_frames.get(phase_id, {}))

    def get_value(
        self,
        phase_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Read a value from a phase frame."""

        return deepcopy(self.phase_frames.get(phase_id, {}).get(key, default))

    def set_value(self, phase_id: str, key: str, value: Any) -> None:
        """Write a value into a phase frame."""

        self.phase_frames.setdefault(phase_id, {})[key] = deepcopy(value)

    def update_phase(
        self,
        phase_id: str,
        patch: dict[str, Any],
        merge_lists: bool = True,
    ) -> None:
        """Merge a phase-scoped query-frame patch."""

        if not patch:
            return

        phase_frame = self.phase_frames.setdefault(phase_id, {})
        for key, value in patch.items():
            if (
                merge_lists
                and key in phase_frame
                and isinstance(phase_frame[key], list)
                and isinstance(value, list)
            ):
                phase_frame[key].extend(deepcopy(value))
            else:
                phase_frame[key] = deepcopy(value)

        self._update_flat_view(patch)

    def append_value(self, phase_id: str, key: str, value: Any) -> None:
        """Append one value to a list field in a phase frame."""

        phase_frame = self.phase_frames.setdefault(phase_id, {})
        phase_frame.setdefault(key, [])
        if not isinstance(phase_frame[key], list):
            phase_frame[key] = [phase_frame[key]]
        phase_frame[key].append(deepcopy(value))
        self._update_flat_view({key: [value]})

    def apply_patch(
        self,
        patch: dict[str, Any],
        phase_id: str | None = None,
    ) -> None:
        """Apply a query-frame patch to a phase or the current phase."""

        target_phase = phase_id or self.current_phase or "GLOBAL"
        self.update_phase(target_phase, patch)

    def to_phase_dict(self) -> dict[str, dict[str, Any]]:
        """Return all phase-organized query-frame data."""

        return deepcopy(self.phase_frames)

    def to_flat_dict(self) -> dict[str, Any]:
        """Return the flat SQL rendering view."""

        return {
            "query_type": self.query_type,
            "sql_query_mode": self.sql_query_mode,
            "time": deepcopy(self.time),
            "measurement": deepcopy(self.measurement),
            "tag": deepcopy(self.tag),
            "select": deepcopy(self.select),
            "from_paths": deepcopy(self.from_paths),
            "where": deepcopy(self.where),
            "group_by": deepcopy(self.group_by),
            "having": deepcopy(self.having),
            "fill": deepcopy(self.fill),
            "into": deepcopy(self.into),
            "align_by": self.align_by,
            "order_by": deepcopy(self.order_by),
            "limit": self.limit,
            "offset": self.offset,
            "slimit": self.slimit,
            "soffset": self.soffset,
        }

    def _update_flat_view(self, patch: dict[str, Any]) -> None:
        """Update the legacy flat rendering view when known fields appear."""

        append_fields = {"where", "having", "group_by"}
        overwrite_fields = {
            "query_type",
            "sql_query_mode",
            "time",
            "measurement",
            "tag",
            "select",
            "from_paths",
            "fill",
            "into",
            "align_by",
            "order_by",
            "limit",
            "offset",
            "slimit",
            "soffset",
        }

        for key, value in patch.items():
            if key in append_fields:
                current = getattr(self, key)
                if value is None:
                    continue
                if isinstance(value, list):
                    current.extend(deepcopy(value))
                else:
                    current.append(deepcopy(value))
            elif key in overwrite_fields:
                setattr(self, key, deepcopy(value))


@dataclass
class OperatorStore:
    """State-scoped storage for P4 operator expression components."""

    series_expr: list[dict[str, Any]] = field(default_factory=list)
    agg_expr: list[dict[str, Any]] = field(default_factory=list)
    constructing_expressions: list[dict[str, Any]] = field(default_factory=list)
    processed_constructing_expressions: list[dict[str, Any]] = field(
        default_factory=list
    )
    current_constructing_expression: dict[str, Any] | None = None
    current_constructing_expression_index: int | None = None

    @classmethod
    def from_schema(cls, schema: DatabaseSchema) -> "OperatorStore":
        """Initialize operator inputs from database meta measurements."""

        store = cls()
        store.initialize_from_schema(schema)
        return store

    def initialize_from_schema(self, schema: DatabaseSchema) -> None:
        """Seed SeriesExpr from meta-schema measurements and clear AggExpr."""

        self.series_expr = self._extract_measurements(
            schema.meta_schema,
        )
        self._clear_derived_expressions()

    def initialize_from_working_schema(
        self,
        working_schema: dict[str, Any],
    ) -> None:
        """Seed SeriesExpr exclusively from the final P1 working schema."""

        self.series_expr = self._extract_measurements(working_schema)
        self._clear_derived_expressions()

    def _clear_derived_expressions(self) -> None:
        """Clear expressions that are invalid after changing schema scope."""

        self.agg_expr = []
        self.constructing_expressions = []
        self.processed_constructing_expressions = []
        self.current_constructing_expression = None
        self.current_constructing_expression_index = None

    def to_dict(self) -> dict[str, Any]:
        """Return the operator store using expression names from the P4 design."""

        return {
            "SeriesExpr": deepcopy(self.series_expr),
            "AggExpr": deepcopy(self.agg_expr),
            "constructing_expressions": deepcopy(self.constructing_expressions),
            "processed_constructing_expressions": deepcopy(
                self.processed_constructing_expressions
            ),
            "current_constructing_expression": deepcopy(
                self.current_constructing_expression
            ),
            "current_constructing_expression_index": (
                self.current_constructing_expression_index
            ),
        }

    def set_series_expr(self, values: list[dict[str, Any]]) -> None:
        """Replace the current SeriesExpr pool."""

        self.series_expr = deepcopy(values or [])

    def set_agg_expr(self, values: list[dict[str, Any]]) -> None:
        """Replace the current AggExpr pool."""

        self.agg_expr = deepcopy(values or [])

    def add_series_expr(self, value: dict[str, Any]) -> None:
        """Append one SeriesExpr descriptor."""

        if value:
            self.series_expr.append(deepcopy(value))

    def add_agg_expr(self, value: dict[str, Any]) -> None:
        """Append one AggExpr descriptor."""

        if value:
            self.agg_expr.append(deepcopy(value))

    def set_constructing_expressions(self, values: list[dict[str, Any]]) -> None:
        """Replace the expressions currently being constructed in P4."""

        self.constructing_expressions = deepcopy(values or [])

    def add_constructing_expression(self, value: dict[str, Any]) -> None:
        """Append one expression-under-construction descriptor."""

        if value:
            self.constructing_expressions.append(deepcopy(value))

    def clear_constructing_expressions(self) -> None:
        """Clear the P4 expression construction workspace."""

        self.constructing_expressions = []

    def set_processed_constructing_expressions(
        self,
        values: list[dict[str, Any]],
    ) -> None:
        """Replace the expressions completed by the current component loop."""

        self.processed_constructing_expressions = deepcopy(values or [])

    def add_processed_constructing_expression(
        self,
        value: dict[str, Any],
    ) -> None:
        """Append one expression completed by the current component loop."""

        if value:
            self.processed_constructing_expressions.append(deepcopy(value))

    def clear_processed_constructing_expressions(self) -> None:
        """Clear component-loop results before processing a new expression batch."""

        self.processed_constructing_expressions = []

    def set_current_constructing_expression(
        self,
        value: dict[str, Any] | None,
        index: int | None = None,
    ) -> None:
        """Select one expression for the current P4 component pipeline."""

        self.current_constructing_expression = deepcopy(value)
        self.current_constructing_expression_index = index

    def clear_current_constructing_expression(self) -> None:
        """Clear the expression selected for component processing."""

        self.current_constructing_expression = None
        self.current_constructing_expression_index = None

    def _extract_measurements(
        self,
        schema_view: dict[str, Any],
    ) -> list[dict[str, Any]]:
        seen = set()
        measurements = []

        for device in schema_view.get("devices", []):
            if not isinstance(device, dict):
                continue
            for measurement in device.get("measurements", []):
                if not isinstance(measurement, dict):
                    continue
                name = measurement.get("name") or measurement.get("measurement")
                if not name:
                    continue
                data_type = measurement.get("data_type")
                key = (name, data_type)
                if key in seen:
                    continue
                seen.add(key)
                measurements.append(
                    {
                        "name": name,
                        "data_type": data_type,
                        "op_level": 1,
                    }
                )

        if measurements:
            return measurements

        for item in schema_view.get("measurements", []):
            if isinstance(item, dict):
                name = item.get("name") or item.get("measurement")
                data_type = item.get("data_type")
            else:
                name = item
                data_type = None
            if not name:
                continue
            key = (name, data_type)
            if key in seen:
                continue
            seen.add(key)
            measurements.append(
                {
                    "name": name,
                    "data_type": data_type,
                    "op_level": 1,
                }
            )
        return measurements


@dataclass
class State:
    """Mutable global state for one TSSDA data-generation trajectory."""

    run_id: str
    schema: DatabaseSchema
    query_frame: QueryFrame = field(default_factory=QueryFrame)
    operator_store: OperatorStore = field(default_factory=OperatorStore)
    input_request: dict[str, Any] = field(default_factory=dict)
    current_phase: str | None = None
    phase_progress: list[str] = field(default_factory=list)
    spo: dict[str, dict[str, Any]] = field(default_factory=dict)
    flags: dict[str, dict[str, Any]] = field(default_factory=dict)
    natural_language_semantics: dict[str, Any] = field(default_factory=dict)
    memory_context: MemoryRuntimeContext = field(default_factory=MemoryRuntimeContext)
    skill_records: list[SkillExecutionRecord] = field(default_factory=list)
    phase_outputs: list[PhaseOutput] = field(default_factory=list)
    final_sql: str | None = None
    final_nl: str | None = None
    final_samples: list[dict[str, str]] = field(default_factory=list)
    sql_validation: dict[str, Any] = field(default_factory=dict)
    skill_routing_decisions: list[dict[str, Any]] = field(default_factory=list)
    skill_routing_validation: dict[str, Any] = field(default_factory=dict)
    failed_skill_count: int = 0
    skill_failures: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        schema: DatabaseSchema,
        input_request: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> "State":
        """Create a fresh state object."""

        return cls(
            run_id=run_id or f"run_{uuid4().hex[:8]}",
            schema=schema,
            operator_store=OperatorStore.from_schema(schema),
            input_request=deepcopy(input_request or {}),
        )

    def start_phase(self, phase_id: str) -> None:
        """Set the active phase and initialize phase-scoped stores."""

        self.current_phase = phase_id
        self.spo.setdefault(phase_id, {})
        self.flags.setdefault(phase_id, {})
        self.query_frame.start_phase(phase_id)

    def complete_phase(self, phase_id: str | None = None) -> None:
        """Mark a phase as completed."""

        target_phase = phase_id or self.current_phase
        if target_phase and target_phase not in self.phase_progress:
            self.phase_progress.append(target_phase)

    def update_spo(self, phase_id: str, values: dict[str, Any]) -> None:
        """Merge SPO values under a phase id."""

        if values:
            self.spo.setdefault(phase_id, {}).update(deepcopy(values))
            if phase_id == "P0":
                self.memory_context.capture_p0_spo(values)

    def set_spo_value(self, phase_id: str, key: str, value: Any) -> None:
        """Set one SPO value under a phase id."""

        self.update_spo(phase_id, {key: value})

    def get_phase_spo(self, phase_id: str) -> dict[str, Any]:
        """Return all SPO values for one phase."""

        return deepcopy(self.spo.get(phase_id, {}))

    def get_spo_value(
        self,
        phase_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Read one SPO value from a phase."""

        return deepcopy(self.spo.get(phase_id, {}).get(key, default))

    def update_flags(self, phase_id: str, values: dict[str, Any]) -> None:
        """Merge Flag values under a phase id."""

        if values:
            self.flags.setdefault(phase_id, {}).update(deepcopy(values))

    def set_flag(self, phase_id: str, key: str, value: Any) -> None:
        """Set one Flag value under a phase id."""

        self.flags.setdefault(phase_id, {})[key] = deepcopy(value)

    def get_phase_flags(self, phase_id: str) -> dict[str, Any]:
        """Return all Flag values for one phase."""

        return deepcopy(self.flags.get(phase_id, {}))

    def get_flag(
        self,
        key: str,
        phase_id: str | None = None,
        default: Any = None,
    ) -> Any:
        """Read a flag by phase, or search phases from newest to oldest."""

        if phase_id is not None:
            return deepcopy(self.flags.get(phase_id, {}).get(key, default))

        for completed_phase in reversed(self.phase_progress):
            phase_flags = self.flags.get(completed_phase, {})
            if key in phase_flags:
                return deepcopy(phase_flags[key])
        if self.current_phase and key in self.flags.get(self.current_phase, {}):
            return deepcopy(self.flags[self.current_phase][key])
        return default

    def update_query_frame(
        self,
        phase_id: str,
        patch: dict[str, Any],
    ) -> None:
        """Merge query-frame candidate data under a phase id."""

        self.query_frame.update_phase(phase_id, patch)

    def set_memory_temporal_observations(self, values: dict[str, Any]) -> None:
        """Cache one bounded DB temporal-profile observation for memory keys."""

        self.memory_context.set_temporal_observations(values)

    def get_memory_context(self) -> dict[str, Any]:
        """Return trajectory-local memory inputs without exposing them to agents."""

        return self.memory_context.to_dict()

    def initialize_operator_store_from_schema(self) -> None:
        """Reset operator expression inputs from the current schema."""

        self.operator_store.initialize_from_schema(self.schema)

    def initialize_operator_store_from_working_schema(self) -> None:
        """Reset P4 inputs from the final schema scope selected by P1."""

        working_schema = self.schema.get_active_working_schema()
        if not working_schema:
            raise RuntimeError(
                "Cannot initialize SeriesExpr before P1 creates a working schema"
            )
        self.operator_store.initialize_from_working_schema(working_schema)

    def get_operator_store(self) -> dict[str, Any]:
        """Return the current operator expression store."""

        return self.operator_store.to_dict()

    def update_operator_store(self, patch: dict[str, Any]) -> None:
        """Merge a patch into the operator expression store."""

        if not patch:
            return

        if "SeriesExpr" in patch:
            self.operator_store.set_series_expr(patch["SeriesExpr"])
        if "series_expr" in patch:
            self.operator_store.set_series_expr(patch["series_expr"])
        if "AggExpr" in patch:
            self.operator_store.set_agg_expr(patch["AggExpr"])
        if "agg_expr" in patch:
            self.operator_store.set_agg_expr(patch["agg_expr"])
        if "constructing_expressions" in patch:
            self.operator_store.set_constructing_expressions(
                patch["constructing_expressions"]
            )
        if "processed_constructing_expressions" in patch:
            self.operator_store.set_processed_constructing_expressions(
                patch["processed_constructing_expressions"]
            )
        if "current_constructing_expression" in patch:
            self.operator_store.set_current_constructing_expression(
                patch["current_constructing_expression"],
                patch.get("current_constructing_expression_index"),
            )

        for value in patch.get("append_SeriesExpr", []) or []:
            self.operator_store.add_series_expr(value)
        for value in patch.get("append_series_expr", []) or []:
            self.operator_store.add_series_expr(value)
        for value in patch.get("append_AggExpr", []) or []:
            self.operator_store.add_agg_expr(value)
        for value in patch.get("append_agg_expr", []) or []:
            self.operator_store.add_agg_expr(value)
        for value in patch.get("append_constructing_expressions", []) or []:
            self.operator_store.add_constructing_expression(value)
        for value in patch.get(
            "append_processed_constructing_expressions", []
        ) or []:
            self.operator_store.add_processed_constructing_expression(value)
        if patch.get("clear_constructing_expressions"):
            self.operator_store.clear_constructing_expressions()
        if patch.get("clear_processed_constructing_expressions"):
            self.operator_store.clear_processed_constructing_expressions()
        if patch.get("clear_current_constructing_expression"):
            self.operator_store.clear_current_constructing_expression()

    def update_natural_language_semantics(self, values: dict[str, Any]) -> None:
        """Merge generated natural-language semantic descriptions."""

        if values:
            self.natural_language_semantics.update(deepcopy(values))

    def set_natural_language_semantic(self, key: str, value: Any) -> None:
        """Set one natural-language semantic description."""

        self.natural_language_semantics[key] = deepcopy(value)

    def get_natural_language_semantic(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Read one natural-language semantic description."""

        return deepcopy(self.natural_language_semantics.get(key, default))

    def apply_skill_result(
        self,
        result: dict[str, Any],
        default_phase_id: str | None = None,
    ) -> None:
        """Apply one action-level skill result into SPO, Flags, and trace."""

        phase_id = default_phase_id or self._phase_from_result(result)
        if phase_id is None:
            raise ValueError("Cannot apply skill result without a phase id")

        for spo_phase_id, values in result.get("spo_delta", {}).items():
            self.update_spo(spo_phase_id, values)
            if spo_phase_id == "P10" and isinstance(values, dict):
                if values.get("final_sql"):
                    self.final_sql = values["final_sql"]
                if values.get("natural_language_query"):
                    self.final_nl = values["natural_language_query"]
                if isinstance(values.get("final_samples"), list):
                    self.final_samples = deepcopy(values["final_samples"])
                elif isinstance(values.get("nl2sql_data"), dict) and isinstance(
                    values["nl2sql_data"].get("samples"),
                    list,
                ):
                    self.final_samples = deepcopy(
                        values["nl2sql_data"]["samples"]
                    )

        flag_delta = result.get("flag_delta", {})
        if flag_delta and all(not isinstance(value, dict) for value in flag_delta.values()):
            self.update_flags(phase_id, flag_delta)
        else:
            for flag_phase_id, values in flag_delta.items():
                if isinstance(values, dict):
                    self.update_flags(flag_phase_id, values)

        query_frame_patch = result.get("query_frame_delta") or result.get("query_frame_patch", {})
        if query_frame_patch:
            self.update_query_frame(phase_id, query_frame_patch)

        nl_semantics = result.get("natural_language_semantics", {})
        if nl_semantics:
            self.update_natural_language_semantics(nl_semantics)

        working_schema_patch = result.get("working_schema") or result.get("working_schema_patch")
        if working_schema_patch:
            self.schema.update_working_schema(working_schema_patch, phase_id=phase_id)

        operator_store_patch = result.get("operator_store_delta") or result.get(
            "operator_store_patch"
        )
        if operator_store_patch:
            self.update_operator_store(operator_store_patch)

        self.skill_records.append(
            SkillExecutionRecord(
                phase_id=phase_id,
                skill_id=result.get("skill_id"),
                action_id=result.get("action_id"),
                decision_name=result.get("decision_name"),
                selected_action=result.get("selected_action"),
                result=deepcopy(result),
            )
        )

    def record_skill_failure(self, failure: dict[str, Any]) -> None:
        """Record one skill that exhausted its model-output retries."""

        self.failed_skill_count += 1
        self.skill_failures.append(deepcopy(failure))

    def apply_phase_output(self, output: PhaseOutput) -> None:
        """Apply a phase-level output object."""

        self.update_spo(output.phase_id, output.spo)
        self.update_flags(output.phase_id, output.flags)
        self.update_query_frame(output.phase_id, output.query_frame_patch)
        self.update_natural_language_semantics(output.natural_language_semantics)
        self.phase_outputs.append(output)
        if output.phase_id == "P10" and "final_sql" in output.spo:
            self.final_sql = output.spo["final_sql"]
            if isinstance(output.spo.get("final_samples"), list):
                self.final_samples = deepcopy(output.spo["final_samples"])

    def to_skill_state(self) -> dict[str, Any]:
        """Expose a dict interface that existing skills can read directly."""

        return {
            "run_id": self.run_id,
            "input_request": deepcopy(self.input_request),
            "current_phase": self.current_phase,
            "phase_progress": deepcopy(self.phase_progress),
            "spo": deepcopy(self.spo),
            "flags": deepcopy(self.flags),
            "natural_language_semantics": deepcopy(self.natural_language_semantics),
            "failed_skill_count": self.failed_skill_count,
            "skill_failures": deepcopy(self.skill_failures),
            "operator_store": self.operator_store.to_dict(),
            "schema": self.schema.to_agent_context(include_detail=False),
            "query_frame": self.query_frame.to_phase_dict(),
            "query_frame_flat": self.query_frame.to_flat_dict(),
        }

    def to_decision_state(self) -> dict[str, Any]:
        """Expose a compact state view for semantic decision prompts."""

        return {
            "run_id": self.run_id,
            "input_request": deepcopy(self.input_request),
            "current_phase": self.current_phase,
            "phase_progress": deepcopy(self.phase_progress),
            "spo": deepcopy(self.spo),
            "natural_language_semantics": deepcopy(self.natural_language_semantics),
        }

    def _phase_from_result(self, result: dict[str, Any]) -> str | None:
        action_id = result.get("action_id")
        if isinstance(action_id, str) and action_id.startswith("P"):
            return action_id.split("A", 1)[0]
        return self.current_phase

    @property
    def semantic_slots(self) -> dict[str, dict[str, Any]]:
        """Backward-compatible alias for phase-scoped SPO."""

        return self.spo


def create_initial_state(
    input_request: dict[str, Any],
    schema: DatabaseSchema,
) -> State:
    """Create a fresh state object for one run."""

    return State.create(schema=schema, input_request=input_request)


def merge_query_frame_patch(
    query_frame: QueryFrame,
    patch: dict[str, Any],
    phase_id: str | None = None,
) -> None:
    """Backward-compatible helper for merging query-frame patches."""

    query_frame.apply_patch(patch=patch, phase_id=phase_id)


