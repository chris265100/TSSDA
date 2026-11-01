"""Output definitions for the P3 Group Semantics skill package."""

from __future__ import annotations


PHASE_ID = "P3"


SPO_SCHEMA = {
    "group_paradigm": {
        "description": "Whether GROUP BY is enabled and which grouping paradigm is used.",
        "source_actions": ["P3A1"],
    },
    "segment_axis": {
        "description": "The selected time-axis or event-axis segmentation mechanism.",
        "source_actions": ["P3A2"],
    },
    "group_axis": {
        "description": "The selected cross-series grouping mechanism.",
        "source_actions": ["P3A3"],
    },
}


ACTION_TO_SPO_FIELD = {
    "P3A1": "group_paradigm",
    "P3A2": "segment_axis",
    "P3A3": "group_axis",
}


FLAG_SCHEMA = {
    "where_value": {
        "description": "Whether later value predicates may be expressed in WHERE.",
        "source_actions": ["P3A1"],
    },
    "having_flag": {
        "description": "Whether later aggregate predicates may be expressed in HAVING.",
        "source_actions": ["P3A1", "P3A3"],
    },
    "group_by_enabled": {
        "description": "Whether P3 enables any GROUP BY aggregation semantics.",
        "source_actions": ["P3A1"],
    },
}


def has_spo_update(action_id: str) -> bool:
    """Return whether the action should update P3 SPO."""

    return action_id in ACTION_TO_SPO_FIELD


def build_spo_delta(action_id: str, selected_action: str) -> dict:
    """Build the SPO delta for one completed P3 action."""

    spo_field = ACTION_TO_SPO_FIELD.get(action_id)
    if not spo_field:
        return {}
    return {
        PHASE_ID: {
            spo_field: selected_action,
        }
    }


def has_flag_update(action_id: str) -> bool:
    """Return whether the action should update P3 flags."""

    return action_id in {"P3A1", "P3A3"}


def build_flag_delta(action_id: str, selected_action: str) -> dict:
    """Build phase-local flags from P3 semantic decisions."""

    if action_id == "P3A1":
        return {
            "where_value": selected_action
            not in {"grouping_only", "hybrid_grouping"},
            "having_flag": selected_action != "none",
            "group_by_enabled": selected_action != "none",
        }

    if action_id == "P3A3":
        return {
            "having_flag": selected_action != "tags",
        }

    return {}


def build_phase_spo_delta(phase_spo: dict, skill_outputs: list[dict]) -> dict:
    """Ensure all P3 decision SPO fields exist after phase execution."""

    return {
        PHASE_ID: {
            field: phase_spo.get(field)
            for field in ACTION_TO_SPO_FIELD.values()
        }
    }
