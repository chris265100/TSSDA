"""Action-level skills for the P3 Group Semantics skill package."""

from __future__ import annotations

try:
    from . import tool
except ImportError:
    tool = None

try:
    from . import outputs
except ImportError:
    outputs = None


class BaseActionSkill:
    """Base interface for a local, constrained group decision."""

    skill_id: str = "BaseActionSkill"
    action_id: str = ""
    decision_name: str = ""
    candidate_descriptions: dict[str, dict] = {}

    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        action_space = self._normalize_action_space(action_space)
        self._validate_action_space(action_space)

        skill_input = self.build_skill_input(
            state=state,
            context=context,
            action_space=action_space,
        )
        raw_decision = self.decide(
            state=state,
            context=context,
            action_space=action_space,
            skill_input=skill_input,
        )
        decision = self.parse_decision(raw_decision)
        selected_action = decision["selected_action"]

        if selected_action not in action_space:
            raise ValueError(
                f"{self.skill_id} selected invalid action: {selected_action}. "
                f"Valid candidates: {action_space}"
            )

        spo_delta, flag_delta = self._build_output_delta(
            selected_action=selected_action,
            decision=decision,
            state=state,
        )
        candidate_info = self.candidate_descriptions.get(selected_action, {})

        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "selected_action": selected_action,
            "action_meaning": candidate_info.get("meaning", ""),
            "requires_sql_generation": candidate_info.get(
                "requires_sql_generation", False
            ),
            "generation_target": candidate_info.get("generation_target"),
            "structured_decision": decision,
            "action_effect": self.build_action_effect(
                selected_action=selected_action,
                decision=decision,
                state=state,
            ),
            "spo_delta": spo_delta,
            "flag_delta": flag_delta,
            "query_frame_delta": {},
            "trace": f"{self.skill_id} selected {selected_action}.",
            "confidence": decision.get("confidence", 0.0),
        }

    def build_skill_input(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> dict:
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "state": state,
            "context": context,
            "schema_view": state.get("schema", {}),
            "query_frame": state.get("query_frame_flat", {}),
            "natural_language_semantics": state.get(
                "natural_language_semantics", {}
            ),
            "action_space": action_space,
            "candidate_descriptions": {
                action: self.candidate_descriptions[action]
                for action in action_space
                if action in self.candidate_descriptions
            },
        }

    def decide(
        self,
        state: dict,
        context: str,
        action_space: list[str],
        skill_input: dict,
    ):
        if self.llm is not None and hasattr(self.llm, "select_action"):
            return self.llm.select_action(
                skill_id=self.skill_id,
                action_id=self.action_id,
                decision_name=self.decision_name,
                skill_input=skill_input,
            )
        return action_space[0]

    def parse_decision(self, raw_decision) -> dict:
        if isinstance(raw_decision, str):
            return {"selected_action": raw_decision}
        if isinstance(raw_decision, dict):
            if "selected_action" not in raw_decision:
                raise ValueError(
                    f"{self.skill_id} decision dict must contain selected_action"
                )
            return raw_decision
        raise TypeError(
            f"{self.skill_id} received unsupported decision type: "
            f"{type(raw_decision)}"
        )

    def build_action_effect(
        self,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> dict:
        return {}

    def _build_output_delta(
        self,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> tuple[dict, dict]:
        spo_delta = {}
        flag_delta = {}
        if outputs is None:
            return spo_delta, flag_delta

        if hasattr(outputs, "has_spo_update") and outputs.has_spo_update(self.action_id):
            spo_delta = outputs.build_spo_delta(
                action_id=self.action_id,
                selected_action=selected_action,
            )
        if hasattr(outputs, "has_flag_update") and outputs.has_flag_update(self.action_id):
            flag_delta = outputs.build_flag_delta(
                action_id=self.action_id,
                selected_action=selected_action,
            )
        return spo_delta, flag_delta

    def _normalize_action_space(self, action_space: list[str]) -> list[str]:
        if tool is not None and hasattr(tool, "normalize_action_space"):
            return tool.normalize_action_space(action_space)
        return list(action_space)

    def _validate_action_space(self, action_space: list[str]) -> None:
        if not action_space:
            raise ValueError(f"{self.skill_id} received an empty action space")


class GroupParadigmSkill(BaseActionSkill):
    """Action P3A1: decide whether and how GROUP BY should be enabled."""

    skill_id = "GroupParadigmSkill"
    action_id = "P3A1"
    decision_name = "group_paradigm"
    memory_enabled = True
    memory_required_spo = ("P2.temporal_mode",)
    memory_profile_fields = (
        "query.clause_mask",
        "schema.device_count",
        "schema.timeseries_count",
        "schema.measurement_sharing",
        "schema.data_type_family",
        "temporal.intent_pattern",
        "temporal.data_span",
        "temporal.coverage_overlap",
        "temporal.query_scope.data_overlap",
        "temporal.window_shape.window_count",
        "temporal.window_shape.step_relation",
    )

    candidate_descriptions = {
        "none": {
            "meaning": (
                "Do not enable GROUP BY. The query keeps raw rows or directly "
                "computes/selects results without temporal segmentation or series "
                "grouping. Suitable for retrieval, simple computation, and latest "
                "point queries."
            ),
            "requires_sql_generation": False,
        },
        "segmentation_only": {
            "meaning": (
                "Use only time-axis or event-axis segmentation. The query divides "
                "time series by time windows, variation, condition, session, or "
                "count, and aggregates within each segment. Suitable for temporal "
                "interval or event-segment analysis."
            ),
            "requires_sql_generation": False,
        },
        "grouping_only": {
            "meaning": (
                "Use only cross-series grouping. The query groups multiple series "
                "by path level or tag key without explicit temporal segmentation. "
                "Suitable for summarizing devices, measurements, regions, or tagged "
                "entities."
            ),
            "requires_sql_generation": False,
        },
        "hybrid_grouping": {
            "meaning": (
                "Use both segmentation and cross-series grouping. The query organizes "
                "data by temporal/event segments and also groups series by path level "
                "or tags. Suitable for multi-dimensional aggregation."
            ),
            "requires_sql_generation": False,
        },
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        action_space = self._normalize_action_space(action_space)
        self._validate_action_space(action_space)
        if action_space == ["none"] and self._flag_is_true(state, "latest_time"):
            return self._build_latest_time_output(state)
        return super().run(state=state, context=context, action_space=action_space)

    def _build_latest_time_output(self, state: dict) -> dict:
        selected_action = "none"
        decision = {
            "selected_action": selected_action,
            "confidence": 1.0,
            "reason": (
                "Rule-selected none because P2 enabled latest-point temporal "
                "semantics."
            ),
        }
        spo_delta, flag_delta = self._build_output_delta(
            selected_action=selected_action,
            decision=decision,
            state=state,
        )
        flag_delta = {
            "where_value": False,
            "having_flag": False,
            "group_by_enabled": False,
        }
        candidate_info = self.candidate_descriptions[selected_action]
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "selected_action": selected_action,
            "action_meaning": candidate_info.get("meaning", ""),
            "requires_sql_generation": False,
            "generation_target": candidate_info.get("generation_target"),
            "structured_decision": decision,
            "action_effect": self.build_action_effect(
                selected_action=selected_action,
                decision=decision,
                state=state,
            ),
            "spo_delta": spo_delta,
            "flag_delta": flag_delta,
            "query_frame_delta": {},
            "trace": f"{self.skill_id} rule-selected {selected_action}.",
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "semantic_decision_skipped": True,
        }

    @staticmethod
    def _flag_is_true(state: dict, flag_name: str) -> bool:
        flags = state.get("flags", {}) if isinstance(state, dict) else {}
        if not isinstance(flags, dict):
            return False
        for phase_flags in flags.values():
            if isinstance(phase_flags, dict) and flag_name in phase_flags:
                return phase_flags[flag_name] is True
        return flags.get(flag_name) is True

    def build_action_effect(
        self,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> dict:
        route_table = {
            "segmentation_only": ["SegmentAxisSkill"],
            "grouping_only": ["GroupAxisSkill"],
            "hybrid_grouping": ["SegmentAxisSkill", "GroupAxisSkill"],
        }
        return {
            "activate_next_skills": route_table.get(selected_action, []),
            "group_route": selected_action,
        }


class SegmentAxisSkill(BaseActionSkill):
    """Action P3A2: decide the segmentation axis for GROUP BY."""

    skill_id = "SegmentAxisSkill"
    action_id = "P3A2"
    decision_name = "segment_axis"
    memory_enabled = True
    memory_required_spo = (
        "P2.temporal_mode",
        "P3.group_paradigm",
    )
    memory_profile_fields = (
        "query.clause_mask",
        "schema.timeseries_count",
        "schema.data_type_family",
        "temporal.intent_pattern",
        "temporal.data_span",
        "temporal.coverage_overlap",
        "temporal.query_scope.data_overlap",
        "temporal.query_scope.span_ratio",
        "temporal.window_shape.window_count",
        "temporal.window_shape.step_relation",
    )

    candidate_descriptions = {
        "time_interval": {
            "meaning": (
                "Divide the time axis by a display time range, a fixed interval, "
                "and an optional sliding step. The interval defines the window "
                "length, and the slidingStep controls how the window start time "
                "moves forward."
            ),
            "requires_sql_generation": False,
            "generation_target": {
                "query_frame_key": "group_by",
                "artifact_type": "segment_axis",
                "generation_instruction": (
                    "Generate a time-interval GROUP BY segment, equivalent to "
                    "GROUP BY ([startTime, endTime), interval [, slidingStep]). "
                    "Use the existing P2 time artifact, including startTime, "
                    "endTime, interval, and optional slidingStep. The display "
                    "time range defines the visible aggregation range, the "
                    "interval divides the time axis into windows, and the "
                    "slidingStep defines how each next window starts. Generate "
                    "a semantically coherent and syntactically correct time "
                    "GROUP BY clause under the `group_by` key."
                ),
            },
        },
        "variation": {
            "meaning": (
                "Divide the sequence by value variation of a control expression. "
                "For each segment, the first point is used as the base point; "
                "following points stay in the same segment if their difference "
                "from the base point is within delta, otherwise a new segment "
                "starts."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "group_by",
                "artifact_type": "segment_axis",
                "generation_instruction": (
                    "Generate a VARIATION-based GROUP BY segment, equivalent to "
                    "GROUP BY VARIATION(controlExpression [, delta] "
                    "[, ignoreNull=true/false]). Select a valid controlExpression "
                    "from the current measurement scope or a valid expression over "
                    "measurements. Generate a reasonable delta when needed; delta "
                    "is the maximum allowed difference from the segment's first "
                    "point. Set ignoreNull only when it is semantically necessary. "
                    "Generate a semantically coherent and syntactically correct "
                    "VARIATION function under the `group_by` key."
                ),
                "output_format": {
                    "group_by": "GROUP BY VARIATION(controlExpression [, delta] [, ignoreNull=true/false])"
                },
            },
        },
        "condition": {
            "meaning": (
                "Divide the sequence by continuous rows satisfying a boolean "
                "condition. Rows that do not satisfy the condition are ignored, "
                "and only continuous groups whose length satisfies the keep "
                "constraint are output."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "group_by",
                "artifact_type": "segment_axis",
                "generation_instruction": (
                    "Generate a CONDITION-based GROUP BY segment, equivalent to "
                    "GROUP BY CONDITION(predicate, keepCondition "
                    "[, ignoreNull=true/false]). The predicate must be a legal "
                    "boolean expression over the current measurement scope. The "
                    "keepCondition specifies how many continuous rows satisfying "
                    "the predicate are required to form an output group. Generate "
                    "a suitable predicate, keep condition, and optional ignoreNull "
                    "setting. Generate a semantically coherent and syntactically "
                    "correct CONDITION function under the `group_by` key."
                ),
                "output_format": {
                    "group_by": "GROUP BY CONDITION(predicate, keepCondition [, ignoreNull=true/false])"
                },
            },
        },
        "session": {
            "meaning": (
                "Divide the sequence by time gaps between adjacent data points. "
                "Adjacent points whose time gap is less than or equal to the "
                "given timeInterval belong to the same session; a larger gap "
                "starts a new session."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "group_by",
                "artifact_type": "segment_axis",
                "generation_instruction": (
                    "Generate a SESSION-based GROUP BY segment, equivalent to "
                    "GROUP BY SESSION(timeInterval). Choose a reasonable "
                    "timeInterval according to the current time range and data "
                    "granularity. The timeInterval is the maximum allowed time "
                    "gap between adjacent points in the same session; a larger "
                    "gap starts a new segment. Generate a semantically coherent "
                    "and syntactically correct SESSION function under the "
                    "`group_by` key."
                ),
                "output_format": {
                    "group_by": "GROUP BY SESSION(timeInterval)"
                },
            },
        },
        "count": {
            "meaning": (
                "Divide the sequence by a fixed number of continuous data points. "
                "Each segment contains size counted points according to the "
                "control expression, and incomplete final segments smaller than "
                "size are not output."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "group_by",
                "artifact_type": "segment_axis",
                "generation_instruction": (
                    "Generate a COUNT-based GROUP BY segment, equivalent to "
                    "GROUP BY COUNT(controlExpression, size "
                    "[, ignoreNull=true/false]). Select a valid controlExpression "
                    "from the current measurement scope or a valid expression over "
                    "measurements. Generate a reasonable size, which represents "
                    "the number of counted continuous points in each segment. Set "
                    "ignoreNull only when null handling should affect the counting "
                    "process. Generate a semantically coherent and syntactically "
                    "correct COUNT function under the `group_by` key."
                ),
                "output_format": {
                    "group_by": "GROUP BY COUNT(controlExpression, size [, ignoreNull=true/false])"
                },
            },
        },
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        action_space = self._normalize_action_space(action_space)
        self._validate_action_space(action_space)

        if self._should_force_time_interval(state=state, action_space=action_space):
            return self._build_time_interval_output(state=state)

        return super().run(state=state, context=context, action_space=action_space)

    def _should_force_time_interval(
        self,
        state: dict,
        action_space: list[str],
    ) -> bool:
        return (
            action_space == ["time_interval"]
            and (
                self._flag_is_true(state, "time_group")
                or self._p3_group_paradigm(state) == "hybrid_grouping"
            )
        )

    def _build_time_interval_output(self, state: dict) -> dict:
        selected_action = "time_interval"
        decision = {
            "selected_action": selected_action,
            "confidence": 1.0,
            "reason": (
                "Rule-selected time_interval from upstream time_group or "
                "hybrid_grouping."
            ),
        }
        spo_delta, flag_delta = self._build_output_delta(
            selected_action=selected_action,
            decision=decision,
            state=state,
        )
        candidate_info = self.candidate_descriptions[selected_action]
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "selected_action": selected_action,
            "action_meaning": candidate_info.get("meaning", ""),
            "requires_sql_generation": False,
            "generation_target": candidate_info.get("generation_target"),
            "structured_decision": decision,
            "action_effect": {},
            "spo_delta": spo_delta,
            "flag_delta": flag_delta,
            "query_frame_delta": {
                "group_by": [self._build_time_interval_group_by(state)]
            },
            "trace": f"{self.skill_id} rule-selected {selected_action}.",
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "semantic_decision_skipped": True,
        }

    def _build_time_interval_group_by(self, state: dict) -> str:
        time_value = self._read_query_frame_time(state)
        time_window = self._extract_time_window(time_value)
        return f"GROUP BY{time_window}"

    def _read_query_frame_time(self, state: dict):
        if not isinstance(state, dict):
            return ""
        query_frame = state.get("query_frame_flat", {})
        if isinstance(query_frame, dict):
            return query_frame.get("time", "")
        return ""

    def _extract_time_window(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            for key in ("group_by_time", "window", "time", "content"):
                if value.get(key):
                    return self._extract_time_window(value[key])
            return str(value)

        text = str(value).strip()
        if text.upper().startswith("GROUP BY "):
            return text[9:]
        if text.upper().startswith("GROUP BY"):
            return text[8:]
        return text

    def _flag_is_true(self, state: dict, flag_name: str) -> bool:
        flags = state.get("flags", {}) if isinstance(state, dict) else {}
        if not isinstance(flags, dict):
            return False
        for phase_flags in flags.values():
            if isinstance(phase_flags, dict) and flag_name in phase_flags:
                return phase_flags[flag_name] is True
        return flags.get(flag_name) is True

    def _p3_group_paradigm(self, state: dict) -> str | None:
        spo = state.get("spo", {}) if isinstance(state, dict) else {}
        phase_spo = spo.get("P3", {}) if isinstance(spo, dict) else {}
        if isinstance(phase_spo, dict):
            return phase_spo.get("group_paradigm")
        return None


class GroupAxisSkill(BaseActionSkill):
    """Action P3A3: decide the cross-series grouping axis."""

    skill_id = "GroupAxisSkill"
    action_id = "P3A3"
    decision_name = "group_axis"
    memory_enabled = True
    memory_required_spo = (
        "P1.measurement_scope",
        "P1.tag_context",
        "P3.group_paradigm",
    )
    memory_profile_fields = (
        "query.clause_mask",
        "schema.device_count",
        "schema.measurement_name_count",
        "schema.timeseries_count",
        "schema.measurement_sharing",
        "schema.data_type_family",
        "temporal.intent_pattern",
        "temporal.coverage_overlap",
    )

    candidate_descriptions = {
        "level": {
            "meaning": (
                "Group same-name time series by selected path levels in the "
                "tree schema. GROUP BY LEVEL = levelNum merges series whose "
                "paths are identical after keeping the specified levels."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "group_by",
                "artifact_type": "group_axis",
                "generation_instruction": (
                    "Generate a LEVEL-based GROUP BY axis, equivalent to "
                    "GROUP BY LEVEL = levelNum or GROUP BY LEVEL = levelNum, ... . "
                    "Choose valid level numbers according to the current working "
                    "paths, where ROOT is level 0. The selected levels should "
                    "produce meaningful cross-series groups over same-name "
                    "measurements. Generate a semantically coherent and "
                    "syntactically correct LEVEL grouping clause under the "
                    "`group_by` key."
                ),
            },
        },
        "tags": {
            "meaning": (
                "Group time series by tag key values. GROUP BY TAGS(tagKey, ...) "
                "aggregates series with the same tag values and outputs the "
                "selected tag keys as grouping columns."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "group_by",
                "artifact_type": "group_axis",
                "generation_instruction": (
                    "Generate a TAGS-based GROUP BY axis, equivalent to "
                    "GROUP BY TAGS(tagKey, ...) . Select valid tag keys from "
                    "the current tag schema. A single tag key forms one grouping "
                    "dimension, while multiple tag keys form ordered multi-tag "
                    "groups. Generate a semantically coherent and syntactically "
                    "correct TAGS grouping clause under the `group_by` key."
                ),
            },
        },
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        action_space = self._normalize_action_space(action_space)
        self._validate_action_space(action_space)

        forced_action = self._forced_action(state=state, action_space=action_space)
        if forced_action:
            return self._build_forced_output(
                state=state,
                selected_action=forced_action,
            )

        return super().run(state=state, context=context, action_space=action_space)

    def _forced_action(
        self,
        state: dict,
        action_space: list[str],
    ) -> str | None:
        tag_disabled = self._flag_is_false(state, "tag_agg")
        level_disabled = self._flag_is_false(state, "level_agg")

        if tag_disabled and not level_disabled and "level" in action_space:
            return "level"
        if level_disabled and not tag_disabled and "tags" in action_space:
            return "tags"
        return None

    def _build_forced_output(self, state: dict, selected_action: str) -> dict:
        decision = {
            "selected_action": selected_action,
            "confidence": 1.0,
            "reason": self._forced_reason(selected_action),
        }
        spo_delta, flag_delta = self._build_output_delta(
            selected_action=selected_action,
            decision=decision,
            state=state,
        )
        candidate_info = self.candidate_descriptions[selected_action]
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "selected_action": selected_action,
            "action_meaning": candidate_info.get("meaning", ""),
            "requires_sql_generation": True,
            "generation_target": candidate_info.get("generation_target"),
            "structured_decision": decision,
            "action_effect": {},
            "spo_delta": spo_delta,
            "flag_delta": flag_delta,
            "query_frame_delta": {},
            "trace": f"{self.skill_id} rule-selected {selected_action}.",
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "semantic_decision_skipped": True,
        }

    def _forced_reason(self, selected_action: str) -> str:
        if selected_action == "level":
            return "Rule-selected level because tag_agg is false."
        return "Rule-selected tags because level_agg is false."

    def _flag_is_false(self, state: dict, flag_name: str) -> bool:
        flags = state.get("flags", {}) if isinstance(state, dict) else {}
        if not isinstance(flags, dict):
            return False

        for phase_flags in flags.values():
            if isinstance(phase_flags, dict) and flag_name in phase_flags:
                return phase_flags[flag_name] is False
        return flags.get(flag_name) is False


SKILL_REGISTRY = {
    "GroupParadigmSkill": GroupParadigmSkill,
    "SegmentAxisSkill": SegmentAxisSkill,
    "GroupAxisSkill": GroupAxisSkill,
}
