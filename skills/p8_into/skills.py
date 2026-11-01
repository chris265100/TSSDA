"""Action-level skills for P8 result alignment and INTO writeback."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from . import outputs, tool
except ImportError:
    outputs = None
    tool = None


WRITEBACK_TARGET_OUTPUT_INSTRUCTION = """
Return JSON only with exactly one field named `into`.

`into` must be an object with these fields:
- `alignment`: `time_alignment` or `device_alignment`;
- `target_storage_mode`: `non_aligned_writeback` or `aligned_writeback`;
- `mappings`: one mapping for every entry in
  `writeback_cardinality_plan.select_groups`.

Each mapping must contain:
- `select_index`: the matching SELECT group index;
- `into_items`: a list of target items. Each item contains a full
  `target_device` path and a `measurements` list;
- `clause`: the complete INTO clause for that SELECT group.

For `aligned_writeback`, every target item must also contain `aligned: true`
and the clause must prefix every item with `ALIGNED`. For
`non_aligned_writeback`, use `aligned: false` and omit `ALIGNED`.

Follow every target count in `writeback_cardinality_plan` exactly. Target
device paths and measurement names must be unique within one mapping and must
be semantically grounded in working_schema. Prefer a distinct derived target
namespace so source data is not overwritten. Do not return SELECT, FROM, a
complete SQL statement, explanations, markdown, comments, or extra fields.
"""


class BaseActionSkill:
    skill_id = "BaseActionSkill"
    action_id = ""
    decision_name = ""
    candidate_descriptions: dict[str, dict[str, Any]] = {}

    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state: dict, context: str, action_space: list[str]) -> dict[str, Any]:
        action_space = self._normalize_action_space(action_space)
        if not action_space:
            raise ValueError(f"{self.skill_id} received an empty action space")
        skill_input = self.build_skill_input(state, context, action_space)
        decision = self._parse_decision(self.decide(skill_input, action_space))
        selected_action = decision["selected_action"]
        if selected_action not in action_space:
            raise ValueError(
                f"{self.skill_id} selected invalid action {selected_action!r}; "
                f"valid candidates are {action_space}"
            )
        return self._build_skill_output(selected_action, decision, state)

    def build_skill_input(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> dict[str, Any]:
        schema = state.get("schema", {})
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "context": context,
            "schema_view": schema.get("working_schema") or schema.get("meta_schema") or {},
            "query_frame": deepcopy(state.get("query_frame_flat", {})),
            "state": self._compact_state(state),
            "action_space": list(action_space),
            "candidate_descriptions": {
                action: deepcopy(self.candidate_descriptions[action])
                for action in action_space
                if action in self.candidate_descriptions
            },
            **self.build_additional_skill_input(state),
        }

    @staticmethod
    def _compact_state(state: dict) -> dict[str, Any]:
        return {
            "run_id": state.get("run_id"),
            "input_request": deepcopy(state.get("input_request", {})),
            "current_phase": state.get("current_phase"),
            "phase_progress": deepcopy(state.get("phase_progress", [])),
            "spo": deepcopy(state.get("spo", {})),
        }

    def build_additional_skill_input(self, state: dict) -> dict[str, Any]:
        return {}

    def decide(self, skill_input: dict[str, Any], action_space: list[str]):
        if self.llm is not None and hasattr(self.llm, "select_action"):
            return self.llm.select_action(
                skill_id=self.skill_id,
                action_id=self.action_id,
                decision_name=self.decision_name,
                skill_input=skill_input,
            )
        return action_space[0]

    def build_action_effect(
        self,
        selected_action: str,
        decision: dict[str, Any],
        state: dict,
    ) -> dict[str, Any]:
        candidate = self.candidate_descriptions.get(selected_action, {})
        return {
            "activate_next_skills": list(candidate.get("required_next_skills", []))
        }

    def _build_skill_output(
        self,
        selected_action: str,
        decision: dict[str, Any],
        state: dict,
        *,
        semantic_decision_skipped: bool = False,
    ) -> dict[str, Any]:
        candidate = self.candidate_descriptions.get(selected_action, {})
        spo_delta = (
            outputs.build_spo_delta(self.action_id, selected_action)
            if outputs is not None and outputs.has_spo_update(self.action_id)
            else {}
        )
        flag_delta = (
            outputs.build_flag_delta(self.action_id, selected_action)
            if outputs is not None and outputs.has_flag_update(self.action_id)
            else {}
        )
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "selected_action": selected_action,
            "structured_decision": deepcopy(decision),
            "action_meaning": candidate.get("meaning", ""),
            "action_effect": self.build_action_effect(selected_action, decision, state),
            "requires_sql_generation": candidate.get("requires_sql_generation", False),
            "generation_target": deepcopy(candidate.get("generation_target")),
            "spo_delta": spo_delta,
            "flag_delta": flag_delta,
            "query_frame_delta": {},
            "trace": f"{self.skill_id} selected {selected_action}.",
            "confidence": decision.get("confidence", 0.0),
            "reason": decision.get("reason", ""),
            "semantic_decision_skipped": semantic_decision_skipped,
        }

    @staticmethod
    def _parse_decision(raw_decision: Any) -> dict[str, Any]:
        if isinstance(raw_decision, str):
            return {"selected_action": raw_decision}
        if isinstance(raw_decision, dict) and raw_decision.get("selected_action"):
            return dict(raw_decision)
        raise ValueError("P8 decision must contain selected_action")

    @staticmethod
    def _normalize_action_space(action_space: list[str]) -> list[str]:
        return list(dict.fromkeys(str(action).strip() for action in action_space if action))


class ResultAlignmentSkill(BaseActionSkill):
    """P8A1: decide the query result alignment used by INTO mapping."""

    skill_id = "ResultAlignmentSkill"
    action_id = "P8A1"
    decision_name = "result_alignment"
    memory_enabled = True
    memory_required_spo = (
        "P3.group_paradigm",
        "P4.operator_regime",
    )
    memory_profile_fields = (
        "query.select_shape",
        "query.select_layout.group_count",
        "query.select_layout.expressions_per_group",
        "query.clause_mask",
        "schema.device_count",
        "schema.timeseries_count",
        "schema.measurement_sharing",
        "temporal.intent_pattern",
        "temporal.coverage_overlap",
        "temporal.query_scope.data_overlap",
    )

    candidate_descriptions = {
        "time_alignment": {
            "meaning": (
                "Use IoTDB's default time-aligned result layout. INTO target "
                "series correspond one-to-one with all expanded value columns."
            ),
            "requires_sql_generation": False,
            "required_next_skills": ["WritebackModeSkill"],
        },
        "device_alignment": {
            "meaning": (
                "Use ALIGN BY DEVICE. Every source device maps to one target "
                "device with one target measurement per SELECT value column."
            ),
            "requires_sql_generation": False,
            "required_next_skills": ["WritebackModeSkill"],
        },
    }


    def _build_skill_output(self, selected_action, decision, state, **kwargs):
        output = super()._build_skill_output(
            selected_action, decision, state, **kwargs
        )
        output["query_frame_delta"] = {
            "align_by": "device" if selected_action == "device_alignment" else "time"
        }
        return output


class WritebackModeSkill(BaseActionSkill):
    """P8A2: decide whether and how query results are written back."""

    skill_id = "WritebackModeSkill"
    action_id = "P8A2"
    decision_name = "writeback_mode"
    memory_enabled = True
    memory_required_spo = (
        "P4.operator_regime",
        "P5.alias_mode",
        "P8.result_alignment",
    )
    memory_profile_fields = (
        "query.select_shape",
        "query.select_layout.group_count",
        "query.select_layout.expressions_per_group",
        "query.select_type_family",
        "query.clause_mask",
        "schema.device_count",
        "schema.timeseries_count",
        "temporal.intent_pattern",
    )

    candidate_descriptions = {
        "no_writeback": {
            "meaning": "Do not generate an INTO clause for this query.",
            "requires_sql_generation": False,
        },
        "non_aligned_writeback": {
            "meaning": (
                "Write result columns into ordinary non-aligned target series."
            ),
            "requires_sql_generation": False,
            "required_next_skills": ["WritebackTargetSkill"],
        },
        "aligned_writeback": {
            "meaning": (
                "Write result columns into aligned target devices by prefixing "
                "each INTO item with ALIGNED."
            ),
            "requires_sql_generation": False,
            "required_next_skills": ["WritebackTargetSkill"],
        },
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict[str, Any]:
        action_space = self._normalize_action_space(action_space)
        if action_space == ["no_writeback"]:
            reasons = tool.into_incompatibility_reasons(state) if tool is not None else []
            output = self._build_skill_output(
                "no_writeback",
                {
                    "selected_action": "no_writeback",
                    "confidence": 1.0,
                    "reason": (
                        "; ".join(reasons)
                        or "P0 did not request a materialization trajectory."
                    ),
                },
                state,
                semantic_decision_skipped=True,
            )
            output["query_frame_delta"] = {"into": None}
            return output
        return super().run(state, context, action_space)

    def build_additional_skill_input(self, state: dict) -> dict[str, Any]:
        if tool is None:
            return {}
        return {
            "writeback_requested": tool.writeback_requested(state),
            "into_incompatibility_reasons": tool.into_incompatibility_reasons(state),
        }


class WritebackTargetSkill(BaseActionSkill):
    """P8A3: generate schema-grounded INTO mappings without another decision."""

    skill_id = "WritebackTargetSkill"
    action_id = "P8A3"
    decision_name = "writeback_target"

    candidate_descriptions = {
        "generate_writeback_targets": {
            "meaning": (
                "Generate one cardinality-correct INTO target mapping for each "
                "SELECT expression group."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "into",
                "artifact_type": "into_clause_mappings",
                "requires_full_schema_view": True,
                "generation_instruction": (
                    "Generate schema-grounded IoTDB INTO target mappings for "
                    "every SELECT group. Follow writeback_cardinality_plan "
                    "exactly, preserve result-column order, keep target series "
                    "unique, and apply selected_writeback_mode. Prefer derived "
                    "target paths that describe the materialized result and do "
                    "not overwrite source series. Follow output_instruction exactly."
                ),
            },
        }
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict[str, Any]:
        expected_action = "generate_writeback_targets"
        if self._normalize_action_space(action_space) != [expected_action]:
            raise ValueError(
                f"{self.skill_id} requires the single action {expected_action!r}"
            )
        alignment = self._selected_alignment(state)
        writeback_mode = self._selected_writeback_mode(state)
        if writeback_mode not in {"non_aligned_writeback", "aligned_writeback"}:
            raise ValueError(
                f"{self.skill_id} requires an enabled writeback mode, got "
                f"{writeback_mode!r}"
            )
        if tool is None:
            raise RuntimeError("P8 tool module is required")
        cardinality_plan = tool.build_writeback_cardinality_plan(state, alignment)
        output = self._build_skill_output(
            expected_action,
            {
                "selected_action": expected_action,
                "confidence": 1.0,
                "reason": "P8A3 directly generates cardinality-correct INTO mappings.",
            },
            state,
            semantic_decision_skipped=True,
        )
        output["selected_alignment"] = alignment
        output["selected_writeback_mode"] = writeback_mode
        output["writeback_cardinality_plan"] = cardinality_plan
        output["generation_output_instruction"] = WRITEBACK_TARGET_OUTPUT_INSTRUCTION
        return output

    def build_additional_skill_input(self, state: dict) -> dict[str, Any]:
        if tool is None:
            return {}
        alignment = self._selected_alignment(state)
        return {
            "selected_alignment": alignment,
            "selected_writeback_mode": self._selected_writeback_mode(state),
            "writeback_cardinality_plan": tool.build_writeback_cardinality_plan(
                state, alignment
            ),
            "output_instruction": WRITEBACK_TARGET_OUTPUT_INSTRUCTION,
        }

    @staticmethod
    def _selected_alignment(state: dict) -> str:
        return (
            state.get("spo", {}).get("P8", {}).get("result_alignment")
            or "time_alignment"
        )

    @staticmethod
    def _selected_writeback_mode(state: dict) -> str | None:
        return state.get("spo", {}).get("P8", {}).get("writeback_mode")


SKILL_REGISTRY = {
    "ResultAlignmentSkill": ResultAlignmentSkill,
    "WritebackModeSkill": WritebackModeSkill,
    "WritebackTargetSkill": WritebackTargetSkill,
}
