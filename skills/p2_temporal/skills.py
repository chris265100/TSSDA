"""Action-level skills for the P2 Temporal Semantics skill package."""

try:
    from . import tool
except ImportError:
    tool = None

try:
    from . import outputs
except ImportError:
    outputs = None


class BaseActionSkill:
    """Base interface for a local, constrained temporal decision."""

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


class TemporalModeSkill(BaseActionSkill):
    """Action P2A1: decide the SQL temporal access mode."""

    skill_id = "TemporalModeSkill"
    action_id = "P2A1"
    decision_name = "temporal_mode"
    memory_enabled = True
    memory_required_spo = (
        "P1.query_object",
        "P1.device_scope",
        "P1.measurement_scope",
    )
    memory_profile_fields = (
        "schema.device_count",
        "schema.timeseries_count",
        "schema.measurement_sharing",
        "schema.data_type_family",
        "temporal.intent_pattern",
        "temporal.data_span",
        "temporal.coverage_overlap",
    )

    candidate_descriptions = {
        "no_time": {
            "meaning": "No explicit temporal semantics.",
            "requires_sql_generation": False,
        },
        "point_time": {
            "meaning": "Access a single time point or near-point.",
            "requires_sql_generation": False,
        },
        "range_time": {
            "meaning": "Access data within a bounded time range.",
            "requires_sql_generation": False,
        },
        "window_time": {
            "meaning": "Organize data by a temporal window for downstream grouping.",
            "requires_sql_generation": False,
        },
        "latest_time": {
            "meaning": "Use latest-point temporal semantics.",
            "requires_sql_generation": False,
        },
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        result = super().run(state=state, context=context, action_space=action_space)
        selected_action = result.get("selected_action")
        result["query_frame_delta"] = {
            "sql_query_mode": (
                "select_last" if selected_action == "latest_time" else "select"
            )
        }
        if selected_action in {"no_time", "point_time", "latest_time"}:
            result["query_frame_delta"]["time"] = {
                "temporal_mode": selected_action,
            }
        return result

    def build_action_effect(
        self,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> dict:
        route_table = {
            "range_time": ["TimeScopeSkill"],
            "window_time": ["WindowSizeSkill", "WindowStepSkill"],
        }
        return {
            "activate_next_skills": route_table.get(selected_action, []),
            "temporal_route": selected_action,
        }


class TimeScopeSkill(BaseActionSkill):
    """Action P2A2: bind the concrete temporal scope."""

    skill_id = "TimeScopeSkill"
    action_id = "P2A2"
    decision_name = "time_scope"
    memory_enabled = True
    memory_required_spo = (
        "P1.device_scope",
        "P1.measurement_scope",
        "P2.temporal_mode",
    )
    memory_profile_fields = (
        "schema.device_count",
        "schema.timeseries_count",
        "schema.data_type_family",
        "temporal.intent_pattern",
        "temporal.data_span",
        "temporal.coverage_overlap",
    )

    candidate_descriptions = {
        "bounded_range": {
            "meaning": (
                "Use explicit start and end time boundaries, such as "
                "time > t1 AND time < t2 or inclusive variants."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "time",
                "artifact_type": "time_scope",
                "generation_instruction": (
                    "Generate a bounded time range with explicit start and end "
                    "boundaries, equivalent to time > or >= t1 AND time < or <= t2. "
                    "The range can be randomly sampled, or calendar-aligned, such as "
                    "within a specific year or month. Return the result under the "
                    "`time` key."
                ),
            },
        },
        "single_sided_boundary": {
            "meaning": (
                "Use only one time boundary: start or end, such as time > t1, "
                "time >= t1, time < t1, or time <= t1."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "time",
                "artifact_type": "time_scope",
                "generation_instruction": (
                    "Generate a single-sided time boundary with only one start or "
                    "end condition, equivalent to time >, >=, <, or <= t1. The "
                    "boundary can be randomly sampled, or calendar-based, such as "
                    "before or after a specific year or month. Return the result "
                    "under the `time` key."
                ),
            },
        },
        "relative_range": {
            "meaning": (
                "Use a range relative to now(), such as time > now() - t1, "
                "time >= now() - t1, time < now() - t1, or time <= now() - t1."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "time",
                "artifact_type": "time_scope",
                "generation_instruction": (
                    "Generate a relative time range based on now(), equivalent to "
                    "time >, >=, <, or <= now() - t1. The relative duration can be "
                    "a short or long calendar duration, such as one day, one week, "
                    "one month, or one year before now(). Return the result under "
                    "the `time` key."
                ),
            },
        },
        "point_with_tolerance": {
            "meaning": (
                "Use a tolerance interval around a reference time point, such as "
                "from t1-2h to t1+2h, from t1-2h to t1, or from t1 to t1+2h."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "time",
                "artifact_type": "time_scope",
                "generation_instruction": (
                    "Generate a tolerance-based time range around a reference time "
                    "point. The range may cover both sides of the point, only before "
                    "the point, or only after the point. Return the result under "
                    "the `time` key."
                ),
            },
        },
    }


class WindowSizeSkill(BaseActionSkill):
    """Action P2A3: decide the temporal window size."""

    skill_id = "WindowSizeSkill"
    action_id = "P2A3"
    decision_name = "window_size"
    memory_enabled = True
    memory_required_spo = (
        "P1.device_scope",
        "P1.measurement_scope",
        "P2.temporal_mode",
    )
    memory_profile_fields = (
        "schema.timeseries_count",
        "schema.data_type_family",
        "temporal.intent_pattern",
        "temporal.data_span",
        "temporal.coverage_overlap",
    )

    candidate_descriptions = {
        "second_level_window": {
            "meaning": (
                "Use a second-level fixed time window, where the interval is "
                "measured in seconds."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "time",
                "artifact_type": "window_spec",
                "generation_instruction": (
                    "Generate a fixed window artifact equivalent to GROUP BY "
                    "([startTime, endTime), interval). Use standard timestamp "
                    "strings in YYYY-MM-DDTHH:MM:SS format, not epoch "
                    "milliseconds. Choose startTime and endTime within the "
                    "valid data time range and use a second-level interval. "
                    "Return `time` in this format, for example: "
                    "group by ([2017-11-01T00:00:00, "
                    "2017-11-07T23:00:00),1s). This example is only a format "
                    "reference; do not copy its concrete dates or numbers. "
                    "Also return a rich user-style window meaning under "
                    "`temporal_semantic`."
                ),
            },
        },
        "minute_level_window": {
            "meaning": (
                "Use a minute-level fixed time window, where the interval is "
                "measured in minutes."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "time",
                "artifact_type": "window_spec",
                "generation_instruction": (
                    "Generate a fixed window artifact equivalent to GROUP BY "
                    "([startTime, endTime), interval). Use standard timestamp "
                    "strings in YYYY-MM-DDTHH:MM:SS format, not epoch "
                    "milliseconds. Choose startTime and endTime within the "
                    "valid data time range and use a minute-level interval. "
                    "Return `time` in this format, for example: "
                    "group by ([2017-11-01T00:00:00, "
                    "2017-11-07T23:00:00),1m). This example is only a format "
                    "reference; do not copy its concrete dates or numbers. "
                    "Also return a rich user-style window meaning under "
                    "`temporal_semantic`."
                ),
            },
        },
        "hour_level_window": {
            "meaning": (
                "Use an hour-level fixed time window, where the interval is "
                "measured in hours."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "time",
                "artifact_type": "window_spec",
                "generation_instruction": (
                    "Generate a fixed window artifact equivalent to GROUP BY "
                    "([startTime, endTime), interval). Use standard timestamp "
                    "strings in YYYY-MM-DDTHH:MM:SS format, not epoch "
                    "milliseconds. Choose startTime and endTime within the "
                    "valid data time range and use an hour-level interval. "
                    "Return `time` in this format, for example: "
                    "group by ([2017-11-01T00:00:00, "
                    "2017-11-07T23:00:00),1h). This example is only a format "
                    "reference; do not copy its concrete dates or numbers. "
                    "Also return a rich user-style window meaning under "
                    "`temporal_semantic`."
                ),
            },
        },
        "calendar_level_window": {
            "meaning": (
                "Use a calendar-level fixed time window, where the interval "
                "follows natural calendar units such as day, week, month, or year."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "time",
                "artifact_type": "window_spec",
                "generation_instruction": (
                    "Generate a fixed window artifact equivalent to GROUP BY "
                    "([startTime, endTime), interval). Use standard timestamp "
                    "strings in YYYY-MM-DDTHH:MM:SS format, not epoch "
                    "milliseconds. Choose startTime and endTime within the "
                    "valid data time range and use a calendar-level interval. "
                    "Return `time` in this format, for example: "
                    "group by ([2017-11-01T00:00:00, "
                    "2017-11-07T23:00:00),1d) or "
                    "group by ([2017-11-01T00:00:00, "
                    "2017-11-07T23:00:00),1mo). These examples are only "
                    "format references; do not copy their concrete dates or "
                    "numbers. Also return a rich user-style window meaning "
                    "under `temporal_semantic`."
                ),
            },
        },
    }


class WindowStepSkill(BaseActionSkill):
    """Action P2A4: decide the sliding step size for a generated time window."""

    skill_id = "WindowStepSkill"
    action_id = "P2A4"
    decision_name = "window_step"
    memory_enabled = True
    memory_required_spo = (
        "P2.temporal_mode",
        "P2.window_size",
    )
    memory_profile_fields = (
        "query.clause_mask",
        "schema.timeseries_count",
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
        "no_sliding_step": {
            "meaning": (
                "Do not use an explicit sliding step. The query uses a fixed "
                "non-overlapping window, equivalent to GROUP BY "
                "([startTime, endTime), interval)."
            ),
            "requires_sql_generation": False,
        },
        "second_minute_hour_level_step": {
            "meaning": (
                "Use a second-, minute-, or hour-level sliding step, where "
                "the step controls how far the window start time moves forward "
                "each time."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "time",
                "artifact_type": "window_step",
                "generation_instruction": (
                    "Generate a sliding step for the existing time window, "
                    "equivalent to GROUP BY ([startTime, endTime), interval, "
                    "slidingStep). Use standard timestamp strings in "
                    "YYYY-MM-DDTHH:MM:SS format, not epoch milliseconds. "
                    "The step advances each window start, e.g. 1:00, 1:10, "
                    "1:20 for a 10-minute step. Choose a reasonable second-, "
                    "minute-, or hour-level slidingStep from the current "
                    "startTime, endTime, and interval. The interval and step "
                    "values may use seconds `s`, minutes `m`, or hours `h`. "
                    "Return `time` in this format, for example: "
                    "group by([2017-11-01T00:00:00, "
                    "2019-11-07T23:00:00), 1s, 2s); This example is only a "
                    "format reference; do not copy its concrete dates or "
                    "numbers. Also return a rich user-style sliding-window "
                    "meaning under `temporal_semantic`."
                ),
            },
        },
        "calendar_level_step": {
            "meaning": (
                "Use a calendar-level sliding step, where the step follows "
                "natural calendar units such as day, month, or year."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "time",
                "artifact_type": "window_step",
                "generation_instruction": (
                    "Generate a calendar-level sliding step for the existing "
                    "time window, equivalent to GROUP BY ([startTime, endTime), "
                    "interval, slidingStep). Use standard timestamp strings in "
                    "YYYY-MM-DDTHH:MM:SS format, not epoch milliseconds. The "
                    "step advances each window start from the previous start "
                    "position. Choose a reasonable day- or month-level "
                    "slidingStep from the current startTime, endTime, and "
                    "interval. The interval and step values may use days `d` "
                    "or months `mo`. Return `time` in this format, for example: "
                    "group by([2017-11-01T00:00:00, "
                    "2019-11-07T23:00:00), 1d, 2d); This example is only a "
                    "format reference; do not copy its concrete dates or "
                    "numbers. Also return a rich user-style sliding-window "
                    "meaning under `temporal_semantic`."
                ),
            },
        },
    }


SKILL_REGISTRY = {
    "TemporalModeSkill": TemporalModeSkill,
    "TimeScopeSkill": TimeScopeSkill,
    "WindowSizeSkill": WindowSizeSkill,
    "WindowStepSkill": WindowStepSkill,
}
