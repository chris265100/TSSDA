"""Semantic decision agent used by action-level skills."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from .agent import DataAgent, create_default_agent
from .guidance import user_goal_from_input_request
from .token_accounting import estimate_message_tokens


@dataclass
class SkillDecision:
    """Formatted decision returned by the semantic decision agent."""

    selected_action: str
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class SemanticDecisionAgent:
    """Choose one semantic action from an action space."""

    base_agent: DataAgent = field(default_factory=create_default_agent)
    fallback_to_first: bool = False
    last_decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    retry_instruction: str | None = None

    def decide(self, skill_input: dict[str, Any]) -> dict[str, Any]:
        """Return a SkillDecision dict for one action-level skill input."""

        action_space = list(skill_input.get("action_space", []))
        if not action_space:
            raise ValueError("SemanticDecisionAgent received an empty action_space")

        try:
            messages = self._build_messages(skill_input)
            decision = self._call_json_once(messages)
            selected_action = decision.get("selected_action")
            if selected_action not in action_space:
                raise ValueError(
                    f"selected_action must be one of {action_space}, got {selected_action!r}"
                )
            result = SkillDecision(
                selected_action=selected_action,
                confidence=self._coerce_confidence(decision.get("confidence", 0.0)),
                reason=str(decision.get("reason", "")),
            ).to_dict()
        except Exception:
            if not self.fallback_to_first:
                raise
            result = SkillDecision(
                selected_action=action_space[0],
                confidence=0.0,
                reason="Fallback decision: selected the first masked candidate.",
            ).to_dict()

        action_id = skill_input.get("action_id")
        if action_id:
            self.last_decisions[action_id] = result
        return result

    def decide_structured(
        self,
        skill_input: dict[str, Any],
        decision_field: str = "decision",
    ) -> dict[str, Any]:
        """Return a skill-specific structured decision without dropping fields."""

        normalized_input = self._normalize_skill_input_for_model(skill_input)
        action_id = normalized_input.get("action_id")

        action_space = list(normalized_input.get("action_space", []))
        if not action_space:
            raise ValueError("SemanticDecisionAgent received an empty action_space")

        try:
            raw_result = self._call_json_once(
                self._build_messages(normalized_input)
            )
            result = dict(raw_result)
            selected_action = result.get(decision_field)
            if isinstance(selected_action, dict):
                selected_action = selected_action.get("selected_action")
            selected_action = selected_action or result.get("selected_action")
            if selected_action not in action_space:
                raise ValueError(
                    f"{decision_field} must be one of {action_space}, "
                    f"got {selected_action!r}"
                )
            result[decision_field] = selected_action
            result["selected_action"] = selected_action
            result["confidence"] = self._coerce_confidence(
                result.get("confidence", 0.0)
            )
            result.setdefault("reason", "")
        except Exception:
            if action_id in {"P4A2", "P4A3"}:
                raise
            if not self.fallback_to_first:
                raise
            selected_action = action_space[0]
            result = {
                decision_field: selected_action,
                "selected_action": selected_action,
                "selected_expressions": [],
                "confidence": 0.0,
                "reason": "Fallback decision: selected the first masked candidate.",
            }

        if action_id:
            self.last_decisions[action_id] = dict(result)
        return result
    def generate_expression_list(
        self,
        skill_input: dict[str, Any],
    ) -> list[Any]:
        """Generate a skill-specific top-level JSON expression list."""

        normalized_input = self._normalize_skill_input_for_model(skill_input)

        try:
            messages = self._build_messages(normalized_input)
            if hasattr(self.base_agent, "call_json_value"):
                raw_result = self._call_json_value_once(messages)
            else:
                raw_result = self.base_agent.call_json(messages)
            if isinstance(raw_result, dict):
                raw_result = raw_result.get("selected_expressions")
            if not isinstance(raw_result, list):
                raise TypeError("Expression generation must return a JSON list")
            result = raw_result[:2]
        except Exception:
            if not self.fallback_to_first:
                raise
            series_expr = normalized_input.get("SeriesExpr", [])
            result = []
            for item in series_expr[:2]:
                if isinstance(item, dict) and item.get("name"):
                    result.append(f"AGG({item['name']})")

        action_id = normalized_input.get("action_id")
        if action_id:
            self.last_decisions[action_id] = {
                "selected_action": "make_agg_atom",
                "generated_expressions": list(result),
                "confidence": 0.0,
                "reason": "Single-action expression selection.",
            }
        return result

    def select_action_and_rewrite_expression(
        self,
        skill_input: dict[str, Any],
        expression: str,
    ) -> dict[str, Any]:
        """Select an action whose name is returned as an expression wrapper."""

        normalized_input = self._normalize_skill_input_for_model(skill_input)
        action_space = list(normalized_input.get("action_space", []))
        if not action_space:
            raise ValueError("SemanticDecisionAgent received an empty action_space")

        try:
            system_prompt = (
                "You are the semantic decision model for a TSSDA data "
                "generation agent. Follow output_instruction exactly. "
                "Return only the rewritten expression as plain text."
            )
            system_prompt = self._append_action_prior_instruction(
                system_prompt,
                normalized_input,
            )
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        normalized_input,
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ]
            rewritten_expression = self.base_agent.call(messages).strip()
            selected_action = self._action_from_expression(
                rewritten_expression,
                action_space,
            )
            result = {
                "selected_action": selected_action,
                "rewritten_expression": rewritten_expression,
                "confidence": 0.0,
                "reason": "Function family selected through expression rewrite.",
            }
        except Exception:
            if not self.fallback_to_first:
                raise
            selected_action = action_space[0]
            rewritten_expression = self._rewrite_expression_with_action(
                expression,
                selected_action,
            )
            result = {
                "selected_action": selected_action,
                "rewritten_expression": rewritten_expression,
                "confidence": 0.0,
                "reason": "Fallback decision: selected the first masked candidate.",
            }

        action_id = normalized_input.get("action_id")
        if action_id:
            self.last_decisions[action_id] = dict(result)
        return result

    def select_operator_and_rewrite_expression(
        self,
        skill_input: dict[str, Any],
        expected_rewrites: dict[str, str],
    ) -> dict[str, Any]:
        """Select an operator by returning its fully rewritten expression."""

        normalized_input = self._normalize_skill_input_for_model(skill_input)
        action_space = list(normalized_input.get("action_space", []))
        if not action_space:
            raise ValueError("SemanticDecisionAgent received an empty action_space")
        if set(action_space) != set(expected_rewrites):
            raise ValueError(
                "Operator rewrite candidates must match the action space"
            )

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the semantic decision model for a TSSDA data "
                        "generation agent. Follow output_instruction exactly. "
                        "Return only the rewritten expression as plain text."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        normalized_input,
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ]
            rewritten_expression = self.base_agent.call(messages).strip()
            matching_actions = [
                action
                for action, expected_expression in expected_rewrites.items()
                if rewritten_expression == expected_expression
            ]
            if len(matching_actions) != 1:
                raise ValueError(
                    "Operator expression must match exactly one candidate rewrite"
                )
            selected_action = matching_actions[0]
            result = {
                "selected_action": selected_action,
                "rewritten_expression": rewritten_expression,
                "confidence": 0.0,
                "reason": "Arithmetic operator selected through expression rewrite.",
            }
        except Exception:
            if not self.fallback_to_first:
                raise
            selected_action = action_space[0]
            result = {
                "selected_action": selected_action,
                "rewritten_expression": expected_rewrites[selected_action],
                "confidence": 0.0,
                "reason": "Fallback decision: selected the first masked candidate.",
            }

        action_id = normalized_input.get("action_id")
        if action_id:
            self.last_decisions[action_id] = dict(result)
        return result

    @staticmethod
    def _action_from_expression(
        expression: str,
        action_space: list[str],
    ) -> str:
        for action in sorted(action_space, key=len, reverse=True):
            if expression.startswith(f"{action}(") and expression.endswith(")"):
                return action
        raise ValueError(
            "Rewritten expression must start with one candidate action name; "
            f"candidates={action_space!r}, got={expression!r}"
        )

    @staticmethod
    def _rewrite_expression_with_action(expression: str, action: str) -> str:
        opening_parenthesis = expression.find("(")
        if opening_parenthesis <= 0 or not expression.endswith(")"):
            raise ValueError(
                "Function-family rewriting expects a placeholder expression"
            )
        return f"{action}{expression[opening_parenthesis:]}"
    def select_action(
        self,
        skill_id: str,
        action_id: str,
        decision_name: str,
        state: dict[str, Any] | None = None,
        context: str | None = None,
        action_space: list[str] | None = None,
        candidate_descriptions: dict[str, Any] | None = None,
        schema_view: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
        skill_input: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        """Compatibility API used by current skill classes."""

        if skill_input is None:
            skill_input = {
                "phase_id": action_id.split("A", 1)[0],
                "skill_id": skill_id,
                "action_id": action_id,
                "decision_name": decision_name,
                "context": context or "",
                "schema_view": schema_view or (state or {}).get("schema", {}),
                "query_frame": query_frame
                or (state or {}).get("query_frame_flat", {})
                or {},
                "state": state or {},
                "action_space": list(action_space or []),
                "candidate_descriptions": candidate_descriptions or {},
            }
        else:
            skill_input = dict(skill_input)
            skill_input.setdefault("phase_id", action_id.split("A", 1)[0])
            skill_input.setdefault("skill_id", skill_id)
            skill_input.setdefault("action_id", action_id)
            skill_input.setdefault("decision_name", decision_name)
            state_view = skill_input.get("state", {}) if isinstance(skill_input.get("state"), dict) else {}
            skill_input.setdefault(
                "query_frame",
                query_frame or state_view.get("query_frame_flat", {}) or {},
            )

        skill_input = self._normalize_skill_input_for_model(skill_input)

        decision = self.decide(skill_input)
        return decision["selected_action"]

    def get_last_decision(self, action_id: str) -> dict[str, Any]:
        """Return the latest decision dict for an action id."""

        return dict(self.last_decisions.get(action_id, {}))

    def estimate_input_tokens(self, skill_input: dict[str, Any]) -> int:
        """Estimate the exact semantic-decision prompt that Memory may skip."""

        return estimate_message_tokens(self._build_messages(skill_input))

    def _call_json_once(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Make one JSON attempt; SkillRunner owns validation retries."""

        if isinstance(self.base_agent, DataAgent):
            return self.base_agent.call_json(
                messages,
                retry_invalid_json=False,
            )
        return self.base_agent.call_json(messages)

    def _call_json_value_once(self, messages: list[dict[str, str]]) -> Any:
        """Make one JSON-value attempt; SkillRunner owns validation retries."""

        if isinstance(self.base_agent, DataAgent):
            return self.base_agent.call_json_value(
                messages,
                retry_invalid_json=False,
            )
        return self.base_agent.call_json_value(messages)

    def _build_messages(self, skill_input: dict[str, Any]) -> list[dict[str, str]]:
        skill_input = self._normalize_skill_input_for_model(skill_input)
        if skill_input.get("output_instruction"):
            system_prompt = (
                "You are the semantic decision model for a TSSDA data generation agent. "
                "Follow output_instruction exactly and return JSON only."
            )
        else:
            system_prompt = (
                "You are the semantic decision model for a TSSDA data generation agent. "
                "Choose exactly one selected_action from action_space. "
                "Return JSON only with selected_action, confidence, and reason."
            )
        user_goal = str(skill_input.get("user_generation_goal") or "").strip()
        if user_goal:
            system_prompt += (
                " Treat user_generation_goal as a global objective for every "
                "decision in this trajectory. Treat it as a preference rather "
                "than a hard grammar constraint. Among valid and semantically "
                "sensible candidates, prefer choices that advance the requested "
                "cross-trajectory diversity goal."
            )
        system_prompt = self._append_action_prior_instruction(
            system_prompt,
            skill_input,
        )
        if skill_input.get("skill_routing_instruction"):
            system_prompt += " " + skill_input["skill_routing_instruction"]
        user_prompt = json.dumps(skill_input, ensure_ascii=False, indent=2)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _append_action_prior_instruction(
        system_prompt: str,
        skill_input: dict[str, Any],
    ) -> str:
        action_prior_instruction = str(
            skill_input.get("action_prior_instruction") or ""
        ).strip()
        if skill_input.get("action_prior") and action_prior_instruction:
            return f"{system_prompt} {action_prior_instruction}"
        return system_prompt

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        """Normalize model confidence values into the 0.0-1.0 range."""

        if isinstance(value, Real) and not isinstance(value, bool):
            confidence = float(value)
        elif isinstance(value, str):
            normalized_value = value.strip().lower()
            named_confidences = {
                "very high": 0.95,
                "high": 0.85,
                "medium high": 0.75,
                "medium": 0.60,
                "moderate": 0.60,
                "medium low": 0.45,
                "low": 0.35,
                "very low": 0.15,
                "none": 0.0,
                "unknown": 0.0,
            }
            if normalized_value in named_confidences:
                confidence = named_confidences[normalized_value]
            else:
                numeric_match = re.search(r"-?\d+(?:\.\d+)?\s*%?", normalized_value)
                if numeric_match:
                    numeric_text = numeric_match.group(0).strip()
                    if numeric_text.endswith("%"):
                        confidence = float(numeric_text[:-1].strip()) / 100.0
                    else:
                        confidence = float(numeric_text)
                elif "high" in normalized_value:
                    confidence = 0.85
                elif "medium" in normalized_value or "moderate" in normalized_value:
                    confidence = 0.60
                elif "low" in normalized_value:
                    confidence = 0.35
                else:
                    return 0.0
        else:
            return 0.0

        if confidence != confidence:
            return 0.0
        return max(0.0, min(1.0, confidence))

    def _compact_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Keep only state fields needed for semantic decisions."""

        return {
            "run_id": state.get("run_id"),
            "input_request": state.get("input_request", {}),
            "current_phase": state.get("current_phase"),
            "phase_progress": state.get("phase_progress", []),
            "spo": state.get("spo", {}),
            "natural_language_semantics": state.get("natural_language_semantics", {}),
        }

    def _normalize_skill_input_for_model(
        self,
        skill_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Remove duplicated global context before a model-facing skill call."""

        normalized_input = dict(skill_input)
        normalized_input.setdefault(
            "phase_id",
            str(normalized_input.get("action_id", "")).split("A", 1)[0],
        )
        state_view = normalized_input.get("state", {})
        if not isinstance(state_view, dict):
            state_view = {}
        compact_state = self._compact_state(state_view)
        user_goal = user_goal_from_input_request(
            compact_state.get("input_request", {})
        )
        normalized_input.setdefault(
            "user_generation_goal",
            user_goal,
        )

        # Semantic notes are a first-class input field. Keep one copy outside
        # state so expression-specific skills can refer to it directly.
        top_level_semantics = normalized_input.get("natural_language_semantics")
        state_semantics = compact_state.get("natural_language_semantics", {})
        if top_level_semantics is None or (
            not top_level_semantics and state_semantics
        ):
            top_level_semantics = state_semantics
            normalized_input["natural_language_semantics"] = top_level_semantics
        compact_state.pop("natural_language_semantics", None)
        normalized_input["state"] = compact_state
        normalized_input["schema_view"] = self._select_schema_view(
            normalized_input.get("schema_view", {})
        )
        query_frame = normalized_input.get("query_frame")
        if not isinstance(query_frame, dict):
            normalized_input["query_frame"] = {}
        if self.retry_instruction:
            normalized_input["retry_instruction"] = self.retry_instruction
            output_instruction = str(
                normalized_input.get("output_instruction", "")
            ).strip()
            if self.retry_instruction not in output_instruction:
                normalized_input["output_instruction"] = (
                    f"{output_instruction}\n\n{self.retry_instruction}".strip()
                )
        return normalized_input

    def _select_schema_view(self, schema_context: dict[str, Any]) -> dict[str, Any]:
        """Choose working schema when active, otherwise compact meta schema."""

        if not isinstance(schema_context, dict):
            return {}

        working_schema = schema_context.get("working_schema")
        if working_schema:
            return working_schema

        meta_schema = schema_context.get("meta_schema")
        if meta_schema:
            return meta_schema

        return schema_context


