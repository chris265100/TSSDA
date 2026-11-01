"""Action-level skills for the P1 Scope skill package.

P1 makes schema-scope decisions over the hierarchical time-series schema.
Each action-level skill performs one local semantic decision from its masked
candidate space and returns structured SPO/Flag deltas.
"""

try:
    from . import tool
except ImportError:
    tool = None

try:
    from . import outputs
except ImportError:
    outputs = None


class BaseActionSkill:
    """Base interface for a local, constrained semantic decision."""

    skill_id: str = "BaseActionSkill"
    action_id: str = ""
    decision_name: str = ""

    candidate_descriptions: dict[str, dict] = {}

    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        """Execute one action-level semantic decision."""
        action_space = self._normalize_action_space(action_space)
        action_space = self.filter_action_space(
            state=state,
            context=context,
            action_space=action_space,
        )
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
            "structured_decision": decision,
            "action_effect": self.build_action_effect(
                selected_action=selected_action,
                decision=decision,
                state=state,
            ),
            "spo_delta": spo_delta,
            "flag_delta": flag_delta,
            "trace": f"{self.skill_id} selected {selected_action}.",
            "confidence": decision.get("confidence", 0.0),
        }

    def build_skill_input(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> dict:
        """Build structured input for the current skill."""
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "state": state,
            "context": context,
            "schema_view": state.get("schema", {}),
            "query_frame": state.get("query_frame_flat", {}),
            "action_space": action_space,
            "candidate_descriptions": self.candidate_descriptions,
        }

    def decide(
        self,
        state: dict,
        context: str,
        action_space: list[str],
        skill_input: dict,
    ):
        """Select one candidate from the masked action space.

        Expected LLM output:
            Either a string action value, or a dict with field:
            {
                "selected_action": "...",
                ...
            }
        """
        if self.llm is not None and hasattr(self.llm, "select_action"):
            return self.llm.select_action(
                skill_id=self.skill_id,
                action_id=self.action_id,
                decision_name=self.decision_name,
                skill_input=skill_input,
            )

        return action_space[0]

    def parse_decision(self, raw_decision) -> dict:
        """Normalize LLM/rule output into a structured decision dict."""
        if isinstance(raw_decision, str):
            return {
                "selected_action": raw_decision
            }

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
        """Return additional execution effect of the selected action."""
        return {}

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        """Filter candidates before sending them to the semantic agent."""

        return list(action_space)

    def _build_output_delta(
        self,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> tuple[dict, dict]:
        """Build SPO and Flag deltas according to outputs.py."""
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

        spo_delta = self.postprocess_spo_delta(
            spo_delta=spo_delta,
            selected_action=selected_action,
            decision=decision,
            state=state,
        )

        return spo_delta, flag_delta

    def postprocess_spo_delta(
        self,
        spo_delta: dict,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> dict:
        """Optionally modify SPO delta after the default output mapping."""
        return spo_delta

    def _normalize_action_space(self, action_space: list[str]) -> list[str]:
        """Normalize candidate values before decision."""
        if tool is not None and hasattr(tool, "normalize_action_space"):
            return tool.normalize_action_space(action_space)

        return list(action_space)

    def _validate_action_space(self, action_space: list[str]) -> None:
        """Ensure the current skill receives a non-empty masked action space."""
        if not action_space:
            raise ValueError(f"{self.skill_id} received an empty action space")


class ScopePatternSkill(BaseActionSkill):
    """Action P1A1: decide the query scope pattern."""

    skill_id = "ScopePatternSkill"
    action_id = "P1A1"
    decision_name = "query_object"

    route_table = {
        "series_oriented": ["DeviceScopeSkill", "MeasurementScopeSkill"],
        "device_oriented": ["DeviceScopeSkill"],
        "tag_aware": [
            "DeviceScopeSkill",
            "MeasurementScopeSkill",
            "TagScopeSkill",
        ],
    }

    candidate_routing_notes = {
        "series_oriented": (
            "Continue with device scope and measurement scope decisions."
        ),
        "device_oriented": "Continue with device scope only.",
        "tag_aware": (
            "Continue with device scope, measurement scope, and tag context "
            "decisions."
        ),
    }

    candidate_descriptions = {
        "series_oriented": {
            "meaning": "Target concrete time series or measurements.",
            "requires_sql_generation": False,
        },
        "device_oriented": {
            "meaning": "Target device-level objects and their measurements.",
            "requires_sql_generation": False,
        },
        "tag_aware": {
            "meaning": "Target objects organized or filtered by tag context.",
            "requires_sql_generation": False,
        },
    }

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        if "tag_aware" not in action_space:
            return list(action_space)
        if tool is not None and hasattr(tool, "get_tag_schema"):
            if tool.get_tag_schema(state=state):
                return list(action_space)
        return [action for action in action_space if action != "tag_aware"]

    def build_skill_input(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> dict:
        tag_schema = (
            tool.get_tag_schema(state=state)
            if tool is not None and hasattr(tool, "get_tag_schema")
            else {}
        )

        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "state": state,
            "context": context,
            "schema_view": state.get("schema", {}),
            "query_frame": state.get("query_frame_flat", {}),
            "action_space": action_space,
            "candidate_descriptions": {
                action: self.candidate_descriptions[action]
                for action in action_space
                if action in self.candidate_descriptions
            },
            "candidate_routing": {
                action: {
                    "activate_next_skills": self.route_table.get(action, []),
                    "routing_note": self.candidate_routing_notes.get(action, ""),
                }
                for action in action_space
            },
            "tag_schema": tag_schema,
        }

    def build_action_effect(
        self,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> dict:
        if selected_action not in self.route_table:
            raise ValueError(f"Unsupported scope pattern: {selected_action}")

        return {
            "activate_next_skills": self.route_table[selected_action],
            "scope_route": selected_action,
            "routing_note": self.candidate_routing_notes.get(selected_action, ""),
        }

    def postprocess_spo_delta(
        self,
        spo_delta: dict,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> dict:
        return spo_delta


class DeviceScopeSkill(BaseActionSkill):
    """Action P1A2: decide the device-level scope."""

    skill_id = "DeviceScopeSkill"
    action_id = "P1A2"
    decision_name = "device_scope"

    candidate_descriptions = {
        "no_device_binding": {
            "meaning": "Do not explicitly bind devices.",
            "requires_sql_generation": False,
        },
        "single_device": {
            "meaning": "Select one concrete device.",
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "from_paths",
                "artifact_type": "device_paths",
                "generation_instruction": (
                    "Select exactly one target device from schema_view.devices. "
                    "The target must be a concrete device in the schema, and the "
                    "generated query-frame value must be that device's full path."
                ),
            },
        },
        "device_set": {
            "meaning": "Select multiple concrete devices.",
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "from_paths",
                "artifact_type": "device_paths",
                "generation_instruction": (
                    "Select multiple target devices from schema_view.devices. "
                    "Every target must be a concrete device in the schema, and the "
                    "generated query-frame value must contain the selected devices' "
                    "full paths."
                ),
            },
        },
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
            "action_space": action_space,
            "candidate_descriptions": self.candidate_descriptions,
            "expected_output_format": {
                "selected_action": "one candidate from action_space",
                "target_devices": "optional list of selected devices",
                "device_paths": "optional list of generated device-level paths",
                "working_schema": "optional updated working schema",
            },
        }

    def postprocess_spo_delta(
        self,
        spo_delta: dict,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> dict:
        return spo_delta


class MeasurementScopeSkill(BaseActionSkill):
    """Action P1A3: decide the measurement-level scope."""

    skill_id = "MeasurementScopeSkill"
    action_id = "P1A3"
    decision_name = "measurement_scope"

    candidate_descriptions = {
        "no_measurement_binding": {
            "meaning": "Do not explicitly specify measurements.",
            "requires_sql_generation": False,
        },
        "single_measurement": {
            "meaning": "Select one concrete measurement.",
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "measurement",
                "artifact_type": "measurement_paths",
                "generation_instruction": (
                    "Select exactly one target measurement from each device in "
                    "working_schema.devices. The target measurements must exist in "
                    "the schema, and the output must return the full measurement "
                    "path for every device."
                ),
            },
        },
        "measurement_set": {
            "meaning": "Select multiple concrete measurements.",
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "measurement",
                "artifact_type": "measurement_paths",
                "generation_instruction": (
                    "Select multiple target measurements from working_schema.devices. "
                    "The target measurements must exist in the schema, and the "
                    "output must return full measurement paths with at least one "
                    "selected measurement for every device."
                ),
            },
        },
        "same_measurement_across_devices": {
            "meaning": "Select the same measurement across multiple devices.",
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "measurement",
                "artifact_type": "measurement_paths",
                "generation_instruction": (
                    "Identify measurement names shared by different devices in "
                    "working_schema.devices. Select multiple of these shared "
                    "measurements and return all full measurement paths for the "
                    "selected shared measurement names across the matching devices."
                ),
            },
        },
    }

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        if (
            "same_measurement_across_devices" not in action_space
            or tool is None
            or not hasattr(tool, "has_same_measurement_across_devices")
        ):
            return list(action_space)

        if tool.has_same_measurement_across_devices(state=state):
            return list(action_space)

        return [
            action
            for action in action_space
            if action != "same_measurement_across_devices"
        ]

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
            "action_space": action_space,
            "candidate_descriptions": {
                action: self.candidate_descriptions[action]
                for action in action_space
                if action in self.candidate_descriptions
            },
            "expected_output_format": {
                "selected_action": "one candidate from action_space",
                "target_measurements": "optional list of selected measurements",
                "series_paths": "optional list of generated series-level paths",
                "working_schema": "optional updated working schema",
            },
        }

    def postprocess_spo_delta(
        self,
        spo_delta: dict,
        selected_action: str,
        decision: dict,
        state: dict,
    ) -> dict:
        return spo_delta


class TagScopeSkill(BaseActionSkill):
    """Action P1A4: decide the tag-related scope context."""

    skill_id = "TagScopeSkill"
    action_id = "P1A4"
    decision_name = "tag_context"

    candidate_descriptions = {
        "no_tag_context": {
            "meaning": "Do not use tag context.",
            "requires_sql_generation": False,
        },
        "single_tag": {
            "meaning": "Select one tag explicitly.",
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "tag",
                "artifact_type": "tagged_measurement_paths",
                "generation_instruction": (
                    "Select exactly one tagged measurement from working_tag. "
                    "The target must be a timeseries listed in working_tag.timeseries. "
                    "Return only the selected measurement path under the `tag` key; "
                    "do not return tag metadata."
                ),
                "output_format": {
                    "tag": ["root.database.device.measurement"]
                },
            },
        },
        "tag_set": {
            "meaning": "Select multiple tags explicitly.",
            "requires_sql_generation": True,
            "generation_target": {
                "query_frame_key": "tag",
                "artifact_type": "tagged_measurement_paths",
                "generation_instruction": (
                    "Select multiple tagged measurements from working_tag. "
                    "Every target must be a timeseries listed in working_tag.timeseries. "
                    "Return only the selected measurement paths under the `tag` key; "
                    "do not return tag metadata."
                ),
                "output_format": {
                    "tag": [
                        "root.database.device.measurement",
                        "root.database.device.other_measurement",
                    ]
                },
            },
        },
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        working_tag = (
            tool.get_working_tag_schema(state=state)
            if tool is not None and hasattr(tool, "get_working_tag_schema")
            else {}
        )
        if not working_tag:
            selected_action = "no_tag_context"
            decision = {
                "selected_action": selected_action,
                "confidence": 1.0,
                "reason": "No tag metadata exists inside the current working schema.",
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
                "structured_decision": decision,
                "action_effect": {},
                "spo_delta": spo_delta,
                "flag_delta": flag_delta,
                "query_frame_delta": {},
                "working_tag": {},
                "trace": f"{self.skill_id} selected {selected_action} without agent.",
                "confidence": decision["confidence"],
                "reason": decision["reason"],
                "semantic_decision_skipped": True,
            }

        result = super().run(state=state, context=context, action_space=action_space)
        result["working_tag"] = working_tag
        return result

    def build_skill_input(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> dict:
        working_tag = (
            tool.get_working_tag_schema(state=state)
            if tool is not None and hasattr(tool, "get_working_tag_schema")
            else {}
        )

        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "state": state,
            "context": context,
            "schema_view": state.get("schema", {}),
            "query_frame": state.get("query_frame_flat", {}),
            "action_space": action_space,
            "candidate_descriptions": {
                action: self.candidate_descriptions[action]
                for action in action_space
                if action in self.candidate_descriptions
            },
            "working_tag": working_tag,
        }


SKILL_REGISTRY = {
    "ScopePatternSkill": ScopePatternSkill,
    "DeviceScopeSkill": DeviceScopeSkill,
    "MeasurementScopeSkill": MeasurementScopeSkill,
    "TagScopeSkill": TagScopeSkill,
}
