"""Action-level skills for P7 Fill Semantics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from . import outputs
except ImportError:
    outputs = None

try:
    from . import tool
except ImportError:
    tool = None


FILL_INSTANTIATION_OUTPUT_INSTRUCTION = """
Return JSON only with exactly one field named `fill`.

The `fill` value must be an object with a complete IoTDB FILL clause:

{
  "fill": {
    "mode": "PREVIOUS | LINEAR | constant",
    "clause": "FILL(PREVIOUS, 2m)",
    "parameter": "2m"
  }
}

Rules:
- For `previous`, generate a reasonable time threshold and return a clause such
  as `FILL(PREVIOUS, 2m)`. Store the threshold in `parameter`. Do not return
  bare `FILL(PREVIOUS)` for this phase.
- For `linear`, return `FILL(LINEAR)` and do not add a threshold parameter.
- For `constant`, generate one constant value compatible with the selected
  expression data types. Return a clause such as `FILL(0)`, `FILL(false)`, or
  `FILL('unknown')`, and store the literal in `value`.

Do not return explanations, markdown, comments, a complete SQL query, or any
fields outside `fill`.
"""


class BaseActionSkill:
    """Base interface for local P7 fill skills."""

    skill_id: str = "BaseActionSkill"
    action_id: str = ""
    decision_name: str = ""
    candidate_descriptions: dict[str, dict[str, Any]] = {}

    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state: dict, context: str, action_space: list[str]) -> dict[str, Any]:
        action_space = self._normalize_action_space(action_space)
        self._validate_action_space(action_space)
        skill_input = self.build_skill_input(state, context, action_space)
        raw_decision = self.decide(skill_input, action_space)
        decision = self.parse_decision(raw_decision)
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
        skill_input = {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "state": self._compact_state(state),
            "context": context,
            "schema_view": state.get("schema", {}),
            "query_frame": state.get("query_frame_flat", {}),
            "action_space": action_space,
            "candidate_descriptions": {
                action: deepcopy(self.candidate_descriptions[action])
                for action in action_space
                if action in self.candidate_descriptions
            },
        }
        skill_input.update(self.build_additional_skill_input(state))
        return skill_input

    @staticmethod
    def _compact_state(state: dict) -> dict[str, Any]:
        return {
            "run_id": state.get("run_id"),
            "input_request": deepcopy(state.get("input_request", {})),
            "current_phase": state.get("current_phase"),
            "phase_progress": deepcopy(state.get("phase_progress", [])),
            "spo": deepcopy(state.get("spo", {})),
            "natural_language_semantics": deepcopy(
                state.get("natural_language_semantics", {})
            ),
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

    @staticmethod
    def parse_decision(raw_decision) -> dict[str, Any]:
        if isinstance(raw_decision, str):
            return {"selected_action": raw_decision}
        if isinstance(raw_decision, dict) and "selected_action" in raw_decision:
            return dict(raw_decision)
        raise ValueError("P7 decision must contain selected_action")

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
        spo_delta = {}
        flag_delta = {}
        if outputs is not None and outputs.has_spo_update(self.action_id):
            spo_delta = outputs.build_spo_delta(self.action_id, selected_action)
        if outputs is not None and outputs.has_flag_update(self.action_id):
            flag_delta = outputs.build_flag_delta(self.action_id, selected_action)
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "selected_action": selected_action,
            "action_meaning": candidate.get("meaning", ""),
            "requires_sql_generation": candidate.get(
                "requires_sql_generation", False
            ),
            "generation_target": deepcopy(candidate.get("generation_target")),
            "structured_decision": decision,
            "action_effect": self.build_action_effect(
                selected_action, decision, state
            ),
            "spo_delta": spo_delta,
            "flag_delta": flag_delta,
            "query_frame_delta": {},
            "trace": f"{self.skill_id} selected {selected_action}.",
            "confidence": decision.get("confidence", 0.0),
            "reason": decision.get("reason", ""),
            "semantic_decision_skipped": semantic_decision_skipped,
        }

    @staticmethod
    def _normalize_action_space(action_space: list[str]) -> list[str]:
        return list(dict.fromkeys(str(action).strip() for action in action_space if action))

    def _validate_action_space(self, action_space: list[str]) -> None:
        if not action_space:
            raise ValueError(f"{self.skill_id} received an empty action space")


class FillModeSkill(BaseActionSkill):
    """P7A1: decide the FILL mode."""

    skill_id = "FillModeSkill"
    action_id = "P7A1"
    decision_name = "fill_mode"
    memory_enabled = True
    memory_required_spo = (
        "P2.temporal_mode",
        "P3.group_paradigm",
        "P4.operator_regime",
    )
    memory_profile_fields = (
        "query.select_shape",
        "query.select_type_family",
        "query.expression_depth",
        "query.clause_mask",
        "schema.timeseries_count",
        "schema.data_type_family",
        "temporal.intent_pattern",
        "temporal.data_span",
        "temporal.coverage_overlap",
        "temporal.query_scope.data_overlap",
        "temporal.query_scope.span_ratio",
        "temporal.query_scope.position",
        "temporal.window_shape.window_count",
        "temporal.window_shape.step_relation",
    )

    candidate_descriptions = {
        "none": {
            "meaning": "Do not use a FILL clause.",
            "requires_sql_generation": False,
        },
        "previous": {
            "meaning": (
                "Use previous-value filling for null result cells. This mode "
                "may use a time threshold, for example FILL(PREVIOUS, 2m)."
            ),
            "requires_sql_generation": False,
            "required_next_skills": ["FillInstantiationSkill"],
        },
        "linear": {
            "meaning": (
                "Use linear interpolation filling. This mode is valid only for "
                "numeric result columns."
            ),
            "requires_sql_generation": False,
            "required_next_skills": ["FillInstantiationSkill"],
        },
        "constant": {
            "meaning": (
                "Use a constant literal to fill null result cells. The literal "
                "must be compatible with selected expression data types."
            ),
            "requires_sql_generation": False,
            "required_next_skills": ["FillInstantiationSkill"],
        },
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict[str, Any]:
        if tool is not None and tool.temporal_mode_disables_fill(state):
            output = self._build_skill_output(
                selected_action="none",
                decision={
                    "selected_action": "none",
                    "confidence": 1.0,
                    "reason": (
                        "Rule-selected none because P2 temporal mode is "
                        "point_time or latest_time."
                    ),
                },
                state=state,
                semantic_decision_skipped=True,
            )
            output["query_frame_delta"] = {"fill": None}
            return output
        return super().run(state=state, context=context, action_space=action_space)

    def build_additional_skill_input(self, state: dict) -> dict[str, Any]:
        if tool is None:
            return {}
        return {
            "selected_expression_type_summary": (
                tool.selected_expression_type_summary(state)
            ),
            "fill_mode_support": tool.fill_mode_support_table(),
        }

    def build_action_effect(
        self,
        selected_action: str,
        decision: dict[str, Any],
        state: dict,
    ) -> dict[str, Any]:
        if selected_action in {"previous", "linear", "constant"}:
            return {
                "activate_next_skills": ["FillInstantiationSkill"],
                "fill_route": selected_action,
            }
        return {"activate_next_skills": [], "fill_route": selected_action}


class FillInstantiationSkill(BaseActionSkill):
    """P7A2: instantiate the selected FILL mode into a concrete clause."""

    skill_id = "FillInstantiationSkill"
    action_id = "P7A2"
    decision_name = "fill_instantiation"

    candidate_descriptions = {
        "instantiate_fill_clause": {
            "meaning": (
                "Generate the complete FILL clause and any required parameter "
                "for the selected fill mode."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "fill",
                "artifact_type": "fill_clause",
                "generation_instruction": (
                    "Generate a complete IoTDB FILL clause for "
                    "`selected_fill_mode`. Use the selected expression data "
                    "types to choose legal parameters. For previous, generate "
                    "a reasonable time threshold and include it in the FILL "
                    "clause. For constant, choose a literal compatible with "
                    "the selected expression types. "
                    "For linear, generate FILL(LINEAR). Follow "
                    "`output_instruction` exactly."
                ),
            },
        }
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict[str, Any]:
        expected_action = "instantiate_fill_clause"
        action_space = self._normalize_action_space(action_space)
        if action_space != [expected_action]:
            raise ValueError(
                f"{self.skill_id} requires the single action {expected_action!r}"
            )

        selected_fill_mode = self._selected_fill_mode(state)
        if selected_fill_mode not in {"previous", "linear", "constant"}:
            raise ValueError(
                f"{self.skill_id} requires a concrete fill mode, got "
                f"{selected_fill_mode!r}"
            )

        output = self._build_skill_output(
            selected_action=expected_action,
            decision={
                "selected_action": expected_action,
                "confidence": 1.0,
                "reason": "P7A2 directly calls SQL generation for FILL instantiation.",
            },
            state=state,
            semantic_decision_skipped=True,
        )
        output["selected_fill_mode"] = selected_fill_mode
        if tool is not None:
            output["selected_expression_type_summary"] = (
                tool.selected_expression_type_summary(state)
            )
            output["fill_mode_support"] = tool.fill_mode_support_table()
        output["generation_output_instruction"] = FILL_INSTANTIATION_OUTPUT_INSTRUCTION
        return output

    def build_additional_skill_input(self, state: dict) -> dict[str, Any]:
        payload = {
            "selected_fill_mode": self._selected_fill_mode(state),
            "output_instruction": FILL_INSTANTIATION_OUTPUT_INSTRUCTION,
        }
        if tool is not None:
            payload["selected_expression_type_summary"] = (
                tool.selected_expression_type_summary(state)
            )
            payload["fill_mode_support"] = tool.fill_mode_support_table()
        return payload

    @staticmethod
    def _selected_fill_mode(state: dict) -> str | None:
        return (
            state.get("spo", {})
            .get("P7", {})
            .get("fill_mode")
        )


SKILL_REGISTRY = {
    "FillModeSkill": FillModeSkill,
    "FillInstantiationSkill": FillInstantiationSkill,
}
