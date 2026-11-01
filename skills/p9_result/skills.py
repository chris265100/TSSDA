"""Action-level skills for P9 result ordering and pagination."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

try:
    from . import outputs, tool
except ImportError:
    outputs = None
    tool = None


ORDER_OUTPUT_INSTRUCTION = """
Return JSON only with exactly one field named `order_by`.
The value must be a non-empty list of ORDER BY items without the `ORDER BY`
keyword, for example:
{"order_by": ["DEVICE DESC", "TIME ASC"]}

Follow `selected_ordering_mode` exactly. Use ASC or DESC for every key. For
expression ordering, use only expressions from `sortable_expressions` and you
may add NULLS FIRST or NULLS LAST. Return no explanations or extra fields.
"""


PAGINATION_OUTPUT_INSTRUCTION = """
Return JSON only with exactly these four fields:
`limit`, `offset`, `slimit`, and `soffset`.

Use positive integers for LIMIT and SLIMIT. Use non-negative integers for
OFFSET and SOFFSET, but offset policies must use an offset greater than zero.
Set unused fields to null. Follow `selected_pagination_policy` exactly. For
column pagination, keep SLIMIT and SOFFSET within
`result_shape.minimum_value_column_count` so every SELECT group retains at
least one value column. Return no explanations or extra fields.
"""


_PAGINATION_POLICY_FIELDS = {
    "row_limit": {"limit"},
    "row_limit_offset": {"limit", "offset"},
    "column_limit": {"slimit"},
    "column_limit_offset": {"slimit", "soffset"},
    "row_column_limit": {"limit", "slimit"},
    "row_column_limit_offset": {"limit", "offset", "slimit", "soffset"},
}


def build_pagination_output_instruction(
    policy: str,
    result_shape: dict[str, Any],
) -> str:
    """Build a policy-specific P9A4 schema and a valid matching example."""

    required_fields = _PAGINATION_POLICY_FIELDS.get(policy)
    if required_fields is None:
        return PAGINATION_OUTPUT_INSTRUCTION

    minimum_columns = result_shape.get("minimum_value_column_count")
    if not isinstance(minimum_columns, int) or minimum_columns < 1:
        minimum_columns = 1
    example = {
        "limit": 20 if "limit" in required_fields else None,
        "offset": 5 if "offset" in required_fields else None,
        "slimit": None,
        "soffset": None,
    }
    if "slimit" in required_fields:
        example_soffset = (
            1 if "soffset" in required_fields and minimum_columns > 1 else 0
        )
        example["soffset"] = (
            example_soffset if "soffset" in required_fields else None
        )
        example["slimit"] = max(
            1,
            min(5, minimum_columns - example_soffset),
        )

    required_text = ", ".join(sorted(required_fields))
    null_fields = sorted(
        {"limit", "offset", "slimit", "soffset"} - required_fields
    )
    null_text = ", ".join(null_fields) if null_fields else "none"
    return (
        PAGINATION_OUTPUT_INSTRUCTION
        + "\nThe selected policy is `"
        + policy
        + "`. The following fields MUST be non-null positive integers: "
        + required_text
        + ". The fields that MUST be null are: "
        + null_text
        + ". Return an object matching this policy-specific example exactly "
        + "in shape:\n"
        + json.dumps(example, ensure_ascii=False)
    )


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

    def build_skill_input(self, state, context, action_space):
        schema = state.get("schema", {})
        skill_input = {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "context": context,
            "schema_view": schema.get("working_schema") or schema.get("meta_schema") or {},
            "query_frame": deepcopy(state.get("query_frame_flat", {})),
            "state": {
                "run_id": state.get("run_id"),
                "input_request": deepcopy(state.get("input_request", {})),
                "current_phase": state.get("current_phase"),
                "phase_progress": deepcopy(state.get("phase_progress", [])),
                "spo": deepcopy(state.get("spo", {})),
            },
            "action_space": list(action_space),
            "candidate_descriptions": {
                action: deepcopy(self.candidate_descriptions[action])
                for action in action_space
                if action in self.candidate_descriptions
            },
        }
        skill_input.update(self.build_additional_skill_input(state))
        return skill_input

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

    def build_action_effect(self, selected_action, decision, state):
        return {
            "activate_next_skills": list(
                self.candidate_descriptions.get(selected_action, {}).get(
                    "required_next_skills", []
                )
            )
        }

    def _build_skill_output(
        self,
        selected_action,
        decision,
        state,
        *,
        semantic_decision_skipped=False,
    ):
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
    def _parse_decision(raw_decision):
        if isinstance(raw_decision, str):
            return {"selected_action": raw_decision}
        if isinstance(raw_decision, dict) and raw_decision.get("selected_action"):
            return dict(raw_decision)
        raise ValueError("P9 decision must contain selected_action")

    @staticmethod
    def _normalize_action_space(action_space):
        return list(dict.fromkeys(str(action).strip() for action in action_space if action))


class OrderingModeSkill(BaseActionSkill):
    """P9A1: choose a result ordering mode."""

    skill_id = "OrderingModeSkill"
    action_id = "P9A1"
    decision_name = "ordering_mode"
    memory_enabled = True
    memory_required_spo = (
        "P2.temporal_mode",
        "P4.operator_regime",
        "P8.result_alignment",
    )
    memory_profile_fields = (
        "query.select_shape",
        "query.select_layout.group_count",
        "query.select_layout.expressions_per_group",
        "query.select_type_family",
        "query.expression_depth",
        "query.clause_mask",
        "schema.device_count",
        "temporal.intent_pattern",
        "temporal.query_scope.data_overlap",
        "temporal.query_scope.position",
        "temporal.window_shape.step_relation",
    )

    candidate_descriptions = {
        "no_order": {
            "meaning": "Use the result layout's default ordering.",
            "requires_sql_generation": False,
            "required_next_skills": ["PaginationPolicySkill"],
        },
        "time_order": {
            "meaning": "Order result rows by timestamp.",
            "requires_sql_generation": False,
            "required_next_skills": [
                "OrderingInstantiationSkill",
                "PaginationPolicySkill",
            ],
        },
        "device_order": {
            "meaning": "Order a device-aligned result primarily by device name.",
            "requires_sql_generation": False,
            "required_next_skills": [
                "OrderingInstantiationSkill",
                "PaginationPolicySkill",
            ],
        },
        "device_time_order": {
            "meaning": "Order first by device and then by timestamp.",
            "requires_sql_generation": False,
            "required_next_skills": [
                "OrderingInstantiationSkill",
                "PaginationPolicySkill",
            ],
        },
        "time_device_order": {
            "meaning": "Order first by timestamp and then by device.",
            "requires_sql_generation": False,
            "required_next_skills": [
                "OrderingInstantiationSkill",
                "PaginationPolicySkill",
            ],
        },
        "expression_order": {
            "meaning": (
                "Order by one or more selected result expressions, optionally "
                "controlling null placement."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "OrderingInstantiationSkill",
                "PaginationPolicySkill",
            ],
        },
        "latest_metadata_order": {
            "meaning": (
                "Order a SELECT LAST result by TIME, TIMESERIES, VALUE, or DATATYPE."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "OrderingInstantiationSkill",
                "PaginationPolicySkill",
            ],
        },
    }

    def build_additional_skill_input(self, state):
        return {
            "result_alignment": tool.result_alignment(state) if tool else "time_alignment",
            "latest_query": tool.is_latest_query(state) if tool else False,
            "sortable_expressions": (
                tool.common_sortable_expressions(state) if tool else []
            ),
        }

    def _build_skill_output(self, selected_action, decision, state, **kwargs):
        output = super()._build_skill_output(
            selected_action, decision, state, **kwargs
        )
        if selected_action == "no_order":
            output["query_frame_delta"] = {"order_by": []}
        return output


class OrderingInstantiationSkill(BaseActionSkill):
    """P9A2: generate concrete ORDER BY items for the selected mode."""

    skill_id = "OrderingInstantiationSkill"
    action_id = "P9A2"
    decision_name = "ordering_instantiation"

    candidate_descriptions = {
        "instantiate_ordering": {
            "meaning": "Generate concrete ORDER BY keys and directions.",
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "order_by",
                "artifact_type": "order_by_items",
                "generation_instruction": (
                    "Generate ORDER BY items for selected_ordering_mode. "
                    "Respect result_alignment and latest-query rules. For "
                    "expression_order, use only sortable_expressions shared "
                    "by every SELECT group. Choose directions and null "
                    "placement that are semantically useful. Follow "
                    "output_instruction exactly."
                ),
            },
        }
    }

    def run(self, state, context, action_space):
        expected = "instantiate_ordering"
        if self._normalize_action_space(action_space) != [expected]:
            raise ValueError(f"{self.skill_id} requires the single action {expected!r}")
        mode = tool.selected_ordering_mode(state) if tool else None
        if mode in {None, "no_order"}:
            raise ValueError(f"{self.skill_id} requires an explicit ordering mode")
        output = self._build_skill_output(
            expected,
            {
                "selected_action": expected,
                "confidence": 1.0,
                "reason": "P9A2 directly instantiates the selected ordering mode.",
            },
            state,
            semantic_decision_skipped=True,
        )
        output["selected_ordering_mode"] = mode
        output["result_alignment"] = tool.result_alignment(state)
        output["sortable_expressions"] = tool.common_sortable_expressions(state)
        output["generation_output_instruction"] = ORDER_OUTPUT_INSTRUCTION
        return output


class PaginationPolicySkill(BaseActionSkill):
    """P9A3: choose row, column, or combined pagination semantics."""

    skill_id = "PaginationPolicySkill"
    action_id = "P9A3"
    decision_name = "pagination_policy"
    memory_enabled = True
    memory_required_spo = (
        "P8.result_alignment",
        "P9.ordering_mode",
    )
    memory_profile_fields = (
        "query.select_shape",
        "query.select_layout.group_count",
        "query.select_layout.expressions_per_group",
        "query.clause_mask",
        "schema.device_count",
        "schema.timeseries_count",
        "temporal.intent_pattern",
    )

    _generated_next = ["PaginationInstantiationSkill"]
    candidate_descriptions = {
        "no_pagination": {
            "meaning": "Return the result without LIMIT or SLIMIT pagination.",
            "requires_sql_generation": False,
        },
        "row_limit": {
            "meaning": "Limit the number of returned rows with LIMIT.",
            "requires_sql_generation": False,
            "required_next_skills": _generated_next,
        },
        "row_limit_offset": {
            "meaning": "Select a row page with LIMIT and a non-zero OFFSET.",
            "requires_sql_generation": False,
            "required_next_skills": _generated_next,
        },
        "column_limit": {
            "meaning": "Limit returned value columns with SLIMIT.",
            "requires_sql_generation": False,
            "required_next_skills": _generated_next,
        },
        "column_limit_offset": {
            "meaning": "Select a value-column page with SLIMIT and SOFFSET.",
            "requires_sql_generation": False,
            "required_next_skills": _generated_next,
        },
        "row_column_limit": {
            "meaning": "Apply both LIMIT and SLIMIT without offsets.",
            "requires_sql_generation": False,
            "required_next_skills": _generated_next,
        },
        "row_column_limit_offset": {
            "meaning": (
                "Apply row and value-column pagination with LIMIT/OFFSET and "
                "SLIMIT/SOFFSET."
            ),
            "requires_sql_generation": False,
            "required_next_skills": _generated_next,
        },
    }

    def build_additional_skill_input(self, state):
        return {
            "result_shape": tool.result_shape_summary(state) if tool else {},
            "pagination_semantics": (
                "LIMIT/OFFSET paginate rows; SLIMIT/SOFFSET paginate value "
                "columns only and do not count Time or Device columns."
            ),
        }

    def _build_skill_output(self, selected_action, decision, state, **kwargs):
        output = super()._build_skill_output(
            selected_action, decision, state, **kwargs
        )
        if selected_action == "no_pagination":
            output["query_frame_delta"] = {
                "limit": None,
                "offset": None,
                "slimit": None,
                "soffset": None,
            }
        return output


class PaginationInstantiationSkill(BaseActionSkill):
    """P9A4: generate concrete LIMIT/OFFSET/SLIMIT/SOFFSET values."""

    skill_id = "PaginationInstantiationSkill"
    action_id = "P9A4"
    decision_name = "pagination_instantiation"

    candidate_descriptions = {
        "instantiate_pagination": {
            "meaning": "Generate concrete pagination parameters.",
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "limit",
                "artifact_type": "pagination_spec",
                "generation_instruction": (
                    "Generate LIMIT, OFFSET, SLIMIT, and SOFFSET values for "
                    "selected_pagination_policy. Use compact, realistic page "
                    "sizes. Avoid offsets likely to make the result empty. "
                    "For column pagination, obey result_shape's minimum value "
                    "column count. Follow output_instruction exactly."
                ),
            },
        }
    }

    def run(self, state, context, action_space):
        expected = "instantiate_pagination"
        if self._normalize_action_space(action_space) != [expected]:
            raise ValueError(f"{self.skill_id} requires the single action {expected!r}")
        policy = tool.selected_pagination_policy(state) if tool else None
        if policy in {None, "no_pagination"}:
            raise ValueError(f"{self.skill_id} requires an enabled pagination policy")
        output = self._build_skill_output(
            expected,
            {
                "selected_action": expected,
                "confidence": 1.0,
                "reason": "P9A4 directly instantiates pagination parameters.",
            },
            state,
            semantic_decision_skipped=True,
        )
        output["selected_pagination_policy"] = policy
        output["result_shape"] = tool.result_shape_summary(state)
        output["generation_output_instruction"] = (
            build_pagination_output_instruction(policy, output["result_shape"])
        )
        return output


SKILL_REGISTRY = {
    "OrderingModeSkill": OrderingModeSkill,
    "OrderingInstantiationSkill": OrderingInstantiationSkill,
    "PaginationPolicySkill": PaginationPolicySkill,
    "PaginationInstantiationSkill": PaginationInstantiationSkill,
}
