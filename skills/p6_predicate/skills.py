"""Action-level skills for P6 Predicate Semantics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from . import outputs
except ImportError:
    outputs = None


WHERE_VALUE_OUTPUT_INSTRUCTION = """
Return JSON only with exactly one field named `where`.

The `where` value must be one SQL predicate string containing only value-related
conditions over measurements from `working_measurements`. Do not include the
WHERE keyword, time predicates, GROUP BY, HAVING, explanations, markdown, or
comments.

You may combine multiple value predicates using AND / OR and may use NOT when
it is semantically useful. Use IoTDB-compatible predicate syntax.

Example:
{
  "where": "(temperature > 25 OR humidity < 60) AND NOT status == 'offline'"
}
"""


HAVING_OUTPUT_INSTRUCTION = """
Return JSON only with exactly one field named `having`.

The `having` value must be one SQL predicate string over expressions from
`level_one_aggregate_expressions`. Do not include the HAVING keyword, WHERE,
GROUP BY, explanations, markdown, or comments.

You may combine multiple aggregate predicates using AND / OR. Copy or adapt the
provided aggregate expressions exactly enough that the predicate remains
schema-grounded and syntactically meaningful.

Example:
{
  "having": "AVG(temperature) > 20 AND COUNT(humidity) >= 10"
}
"""


class BaseActionSkill:
    """Base interface for local P6 predicate skills."""

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
        raise ValueError("P6 decision must contain selected_action")

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


class PredicateModeSkill(BaseActionSkill):
    """P6A1: choose whether and where predicate clauses are used."""

    skill_id = "PredicateModeSkill"
    action_id = "P6A1"
    decision_name = "predicate_mode"
    memory_enabled = True
    memory_required_spo = (
        "P2.temporal_mode",
        "P3.group_paradigm",
        "P4.operator_regime",
    )
    memory_profile_fields = (
        "query.select_shape",
        "query.select_layout.group_count",
        "query.select_layout.expressions_per_group",
        "query.select_type_family",
        "query.expression_depth",
        "query.clause_mask",
        "schema.timeseries_count",
        "schema.data_type_family",
        "temporal.intent_pattern",
        "temporal.coverage_overlap",
        "temporal.query_scope.data_overlap",
        "temporal.query_scope.span_ratio",
        "temporal.window_shape.step_relation",
    )

    candidate_descriptions = {
        "none": {
            "meaning": "Do not use predicate conditions in this query.",
            "requires_sql_generation": False,
        },
        "where_only": {
            "meaning": (
                "Use only a WHERE predicate. It may combine value predicates "
                "with an existing time predicate from the query frame."
            ),
            "requires_sql_generation": False,
            "required_next_skills": ["WhereValuePredicateSkill"],
        },
        "having_only": {
            "meaning": (
                "Use only a HAVING predicate over level-1 aggregate "
                "expressions."
            ),
            "requires_sql_generation": False,
            "required_next_skills": ["HavingPredicateSkill"],
        },
        "both": {
            "meaning": (
                "Use both WHERE and HAVING predicates: WHERE for time/value "
                "filtering before aggregation and HAVING for aggregate "
                "filtering after aggregation."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "WhereValuePredicateSkill",
                "HavingPredicateSkill",
            ],
        },
    }

    def build_action_effect(
        self,
        selected_action: str,
        decision: dict[str, Any],
        state: dict,
    ) -> dict[str, Any]:
        route_table = {
            "where_only": ["WhereValuePredicateSkill"],
            "having_only": ["HavingPredicateSkill"],
            "both": ["WhereValuePredicateSkill", "HavingPredicateSkill"],
        }
        return {
            "activate_next_skills": route_table.get(selected_action, []),
            "predicate_route": selected_action,
        }


class WhereValuePredicateSkill(BaseActionSkill):
    """P6A2: generate value-side WHERE predicates."""

    skill_id = "WhereValuePredicateSkill"
    action_id = "P6A2"
    decision_name = "where_value_predicates"

    candidate_descriptions = {
        "generate_where_value_predicates": {
            "meaning": (
                "Generate value-related WHERE predicates over measurements in "
                "the current working schema."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "where",
                "artifact_type": "where_value_predicate",
                "generation_instruction": (
                    "Read `working_measurements` from the input. Select "
                    "semantically suitable measurements and generate value-side "
                    "IoTDB WHERE predicates over them. Multiple predicates may "
                    "be connected with AND or OR, and NOT may be used when it "
                    "is meaningful. Generate only value-related predicates; do "
                    "not include time predicates, because P2 already owns time "
                    "conditions and the runner will merge them if present. "
                    "Return exactly one JSON field named `where`."
                ),
            },
        }
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict[str, Any]:
        expected_action = "generate_where_value_predicates"
        action_space = self._normalize_action_space(action_space)
        if action_space != [expected_action]:
            raise ValueError(
                f"{self.skill_id} requires the single action {expected_action!r}"
            )
        output = self._build_skill_output(
            selected_action=expected_action,
            decision={
                "selected_action": expected_action,
                "confidence": 1.0,
                "reason": "P6A2 directly calls SQL generation for WHERE predicates.",
            },
            state=state,
            semantic_decision_skipped=True,
        )
        output["working_measurements"] = self._extract_working_measurements(state)
        output["generation_output_instruction"] = WHERE_VALUE_OUTPUT_INSTRUCTION
        return output

    def build_additional_skill_input(self, state: dict) -> dict[str, Any]:
        return {
            "working_measurements": self._extract_working_measurements(state),
            "output_instruction": WHERE_VALUE_OUTPUT_INSTRUCTION,
        }

    @staticmethod
    def _extract_working_measurements(state: dict) -> list[dict[str, Any]]:
        schema_context = state.get("schema", {})
        if not isinstance(schema_context, dict):
            return []
        schema_view = (
            schema_context.get("working_schema")
            or schema_context.get("meta_schema")
            or {}
        )
        devices = schema_view.get("devices", [])
        measurements: list[dict[str, Any]] = []
        if not isinstance(devices, list):
            return measurements
        for device in devices:
            if not isinstance(device, dict):
                continue
            device_name = device.get("name")
            device_path = device.get("path")
            for measurement in device.get("measurements", []) or []:
                if not isinstance(measurement, dict):
                    continue
                name = measurement.get("name")
                path = measurement.get("path")
                if not path and device_path and name:
                    path = f"{device_path}.{name}"
                measurements.append(
                    {
                        "name": name,
                        "path": path,
                        "data_type": measurement.get("data_type")
                        or measurement.get("datatype")
                        or measurement.get("type"),
                        "encoding": measurement.get("encoding"),
                        "device": device_name,
                        "device_path": device_path,
                    }
                )
        return [item for item in measurements if item.get("name") or item.get("path")]


class HavingPredicateSkill(BaseActionSkill):
    """P6A3: generate HAVING predicates over aggregate expressions."""

    skill_id = "HavingPredicateSkill"
    action_id = "P6A3"
    decision_name = "having_predicates"

    candidate_descriptions = {
        "generate_having_predicates": {
            "meaning": (
                "Generate HAVING predicates over level-1 aggregate expressions "
                "constructed by P4."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "having",
                "artifact_type": "having_predicate",
                "generation_instruction": (
                    "Read `level_one_aggregate_expressions` from the input. "
                    "Generate one or more semantically suitable HAVING "
                    "predicates over these aggregate expressions. Multiple "
                    "predicates may be connected with AND or OR. Return exactly "
                    "one JSON field named `having`."
                ),
            },
        }
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict[str, Any]:
        expected_action = "generate_having_predicates"
        action_space = self._normalize_action_space(action_space)
        if action_space != [expected_action]:
            raise ValueError(
                f"{self.skill_id} requires the single action {expected_action!r}"
            )
        output = self._build_skill_output(
            selected_action=expected_action,
            decision={
                "selected_action": expected_action,
                "confidence": 1.0,
                "reason": "P6A3 directly calls SQL generation for HAVING predicates.",
            },
            state=state,
            semantic_decision_skipped=True,
        )
        output["level_one_aggregate_expressions"] = (
            self._level_one_aggregate_expressions(state)
        )
        output["generation_output_instruction"] = HAVING_OUTPUT_INSTRUCTION
        return output

    def build_additional_skill_input(self, state: dict) -> dict[str, Any]:
        return {
            "level_one_aggregate_expressions": (
                self._level_one_aggregate_expressions(state)
            ),
            "output_instruction": HAVING_OUTPUT_INSTRUCTION,
        }

    @staticmethod
    def _level_one_aggregate_expressions(state: dict) -> list[dict[str, Any]]:
        operator_store = state.get("operator_store", {})
        if not isinstance(operator_store, dict):
            return []
        agg_expr = operator_store.get("AggExpr", [])
        if not isinstance(agg_expr, list):
            return []
        return [
            deepcopy(expression)
            for expression in agg_expr
            if isinstance(expression, dict)
            and _expression_level(expression) == 1
            and (expression.get("name") or expression.get("expression"))
        ]


def _expression_level(expression: dict[str, Any]) -> int | None:
    try:
        return int(expression.get("op_level", expression.get("level")))
    except (TypeError, ValueError):
        return None


SKILL_REGISTRY = {
    "PredicateModeSkill": PredicateModeSkill,
    "WhereValuePredicateSkill": WhereValuePredicateSkill,
    "HavingPredicateSkill": HavingPredicateSkill,
}
