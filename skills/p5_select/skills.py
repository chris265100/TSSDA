"""Action-level skills for P5 Select Semantics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from . import outputs
except ImportError:
    outputs = None


SELECT_EXPRESSION_OUTPUT_INSTRUCTION = """
Return JSON only with exactly one field named `select`.

The `select` value must be a non-empty list of SELECT expression groups. Each
group must be one non-empty comma-separated SQL expression string, for example:

{
  "select": [
    "AVG(temperature), MAX_VALUE(humidity)",
    "COUNT(temperature), MIN_VALUE(humidity)"
  ]
}

Generate one or more semantically reasonable SELECT expression groups. Each
group may contain one or more compatible expressions copied exactly from
`select_expression_candidates`. Choose expressions that serve the query's
meaning. Different groups should represent distinct SELECT choices when the
candidate pool supports that diversity.

Do not add aliases, the SELECT keyword, explanations, markdown, comments, or
any other fields.
"""


ALIAS_DECISION_INSTRUCTION = """
Inspect the current `query_frame.select` value, which may be either one SELECT
expression string or a list of SELECT expression groups. Decide whether aliases
would make the selected expressions clearer to a human reader. Choose
`use_alias` when one or more expressions benefit from concise, semantically
meaningful names; otherwise choose `no_alias`. Do not use operator_store for
this decision.
"""


ALIAS_GENERATION_OUTPUT_INSTRUCTION = """
Return JSON only with exactly one field named `select`.

If the input `query_frame.select` is a string, return `select` as one complete
SELECT expression string. If the input `query_frame.select` is a list, return
`select` as a list with the same number of groups and the same group order.
Preserve every existing expression and its order inside each group, and add
human-readable, semantically meaningful aliases to one or more suitable
expressions using `AS`.

Example:
{
  "select": "AVG(temperature) as avg_temperature, MAX_VALUE(humidity)"
}

For multi-group input:
{
  "select": [
    "AVG(temperature) as avg_temperature, MAX_VALUE(humidity)",
    "COUNT(temperature) as temperature_count"
  ]
}

Do not return the SELECT keyword, explanations, markdown, comments, or any
other fields.
"""


class BaseActionSkill:
    """Base interface for a local P5 select decision."""

    skill_id: str = "BaseActionSkill"
    action_id: str = ""
    decision_name: str = ""
    candidate_descriptions: dict[str, dict[str, Any]] = {}

    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
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

    def decide(self, skill_input: dict, action_space: list[str]):
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
        raise ValueError("P5 decision must contain selected_action")

    def build_action_effect(
        self,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> dict[str, Any]:
        candidate = self.candidate_descriptions.get(selected_action, {})
        return {
            "activate_next_skills": list(
                candidate.get("required_next_skills", [])
            )
        }

    def _build_skill_output(
        self,
        selected_action: str,
        decision: dict,
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


class SelectExpressionSkill(BaseActionSkill):
    """P5A1: select a coherent SELECT expression list from P4 results."""

    skill_id = "SelectExpressionSkill"
    action_id = "P5A1"
    decision_name = "select_expressions"

    candidate_descriptions = {
        "select_from_operator_store": {
            "meaning": (
                "Choose a semantically coherent expression set from the P4 "
                "operator expression pool for the SELECT list."
            ),
            "requires_sql_generation": True,
            "required_next_skills": ["AliasModeSkill"],
            "generation_target": {
                "query_frame_key": "select",
                "artifact_type": "select_expression_groups",
                "generation_instruction": (
                    "Generate one or more semantically reasonable SELECT "
                    "expression groups from `select_expression_candidates`. "
                    "Within each group, select semantically compatible "
                    "expressions that can be queried together in one SELECT "
                    "list. Each group may contain one or more expressions. "
                    "Copy every expression exactly without adding aliases or "
                    "the SELECT keyword. Choose distinct, coherent groups. "
                    "Follow `output_instruction` exactly."
                ),
            },
        }
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        action_space = self._normalize_action_space(action_space)
        self._validate_action_space(action_space)
        selected_action = "select_from_operator_store"
        if action_space != [selected_action]:
            raise ValueError(
                f"{self.skill_id} requires the single action {selected_action!r}"
            )
        candidates, source_pool = self._select_expression_pool(state)
        output = self._build_skill_output(
            selected_action=selected_action,
            decision={
                "selected_action": selected_action,
                "confidence": 1.0,
                "reason": f"Rule-selected the {source_pool} expression pool.",
            },
            state=state,
            semantic_decision_skipped=True,
        )
        output["select_expression_candidates"] = candidates
        output["select_expression_source"] = source_pool
        output["generation_output_instruction"] = (
            SELECT_EXPRESSION_OUTPUT_INSTRUCTION
        )
        output["trace"] = (
            f"{self.skill_id} prepared {len(candidates)} {source_pool} "
            "candidates for SELECT generation."
        )
        return output

    def build_additional_skill_input(self, state: dict) -> dict[str, Any]:
        candidates, source_pool = self._select_expression_pool(state)
        return {
            "select_expression_candidates": candidates,
            "select_expression_source": source_pool,
            "output_instruction": SELECT_EXPRESSION_OUTPUT_INSTRUCTION,
        }

    @staticmethod
    def _select_expression_pool(state: dict) -> tuple[list[dict[str, Any]], str]:
        operator_store = state.get("operator_store", {})
        agg_expr = operator_store.get("AggExpr", [])
        series_expr = operator_store.get("SeriesExpr", [])
        if isinstance(agg_expr, list) and agg_expr:
            candidates = agg_expr
            source_pool = "AggExpr"
        else:
            candidates = series_expr if isinstance(series_expr, list) else []
            source_pool = "SeriesExpr"
        normalized = [
            deepcopy(item)
            for item in candidates
            if isinstance(item, dict) and (item.get("name") or item.get("expression"))
        ]
        if not normalized:
            raise ValueError(
                "SelectExpressionSkill requires at least one P4 operator expression"
            )
        return normalized, source_pool


class AliasModeSkill(BaseActionSkill):
    """P5A2: decide whether selected expressions use aliases."""

    skill_id = "AliasModeSkill"
    action_id = "P5A2"
    decision_name = "alias_mode"

    candidate_descriptions = {
        "no_alias": {
            "meaning": "Keep the selected expressions without aliases.",
            "requires_sql_generation": False,
        },
        "use_alias": {
            "meaning": "Assign meaningful aliases to the selected expressions.",
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "select",
                "artifact_type": "aliased_select_expression_list",
                "generation_instruction": (
                    "Use the current `query_frame.select` value as the source. "
                    "It may be a string or a list of SELECT expression groups. "
                    "Choose one or more suitable expressions and add aliases "
                    "that follow natural human naming conventions and preserve "
                    "the expressions' semantics. Keep all original expressions, "
                    "group boundaries, and order, then return the complete "
                    "updated SELECT value under the `select` key. Follow "
                    "`output_instruction` exactly."
                ),
            },
        },
    }

    def build_additional_skill_input(self, state: dict) -> dict[str, Any]:
        return {"alias_decision_instruction": ALIAS_DECISION_INSTRUCTION}

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        action_space = self._normalize_action_space(action_space)
        self._validate_action_space(action_space)
        device_count = self._working_schema_device_count(state)
        if device_count > 1 and "no_alias" in action_space:
            selected_action = "no_alias"
            decision = {
                "selected_action": selected_action,
                "confidence": 1.0,
                "reason": (
                    "Rule-selected no_alias because the current working schema "
                    f"contains {device_count} target devices."
                ),
            }
            output = self._build_skill_output(
                selected_action=selected_action,
                decision=decision,
                state=state,
                semantic_decision_skipped=True,
            )
            output["trace"] = (
                f"{self.skill_id} selected no_alias because multiple target "
                "devices are present in working_schema."
            )
            return output

        output = super().run(state, context, action_space)
        if output.get("selected_action") == "use_alias":
            output["generation_output_instruction"] = (
                ALIAS_GENERATION_OUTPUT_INSTRUCTION
            )
        return output

    @staticmethod
    def _working_schema_device_count(state: dict) -> int:
        schema_context = state.get("schema", {})
        if not isinstance(schema_context, dict):
            return 0
        working_schema = schema_context.get("working_schema")
        if not isinstance(working_schema, dict) or not working_schema:
            return 0
        device_count = working_schema.get("device_count")
        if isinstance(device_count, int) and not isinstance(device_count, bool):
            return device_count
        devices = working_schema.get("devices")
        if isinstance(devices, list):
            return len(devices)
        return 0


SKILL_REGISTRY = {
    "SelectExpressionSkill": SelectExpressionSkill,
    "AliasModeSkill": AliasModeSkill,
}
