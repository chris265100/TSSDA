"""Output definitions for the P2 Temporal Semantics skill package."""

PHASE_ID = "P2"


SPO_SCHEMA = {
    "temporal_mode": {
        "description": "The high-level temporal access mode.",
        "source_actions": ["P2A1"],
    },
    "time_scope": {
        "description": "The concrete temporal scope strategy.",
        "source_actions": ["P2A2"],
    },
    "window_size": {
        "description": "The temporal window size strategy.",
        "source_actions": ["P2A3"],
    },
    "window_step": {
        "description": "The temporal window sliding-step strategy.",
        "source_actions": ["P2A4"],
    },
}


ACTION_TO_SPO_FIELD = {
    "P2A1": "temporal_mode",
    "P2A2": "time_scope",
    "P2A3": "window_size",
    "P2A4": "window_step",
}


FLAG_SCHEMA = {
    "time_group": {
        "description": "Whether the query should use time-window grouping.",
        "source_actions": ["P2A1"],
    },
    "where_time": {
        "description": "Whether the query should use an explicit time condition in WHERE.",
        "source_actions": ["P2A1"],
    },
    "latest_time": {
        "description": "Whether the query uses latest-point temporal semantics.",
        "source_actions": ["P2A1"],
    },
}


def has_spo_update(action_id: str) -> bool:
    """Return whether the action should update P2 SPO."""

    return action_id in ACTION_TO_SPO_FIELD


def build_spo_delta(action_id: str, selected_action: str) -> dict:
    """Build the SPO delta for one completed P2 action."""

    spo_field = ACTION_TO_SPO_FIELD.get(action_id)
    if not spo_field:
        return {}
    return {
        PHASE_ID: {
            spo_field: selected_action,
        }
    }


def has_flag_update(action_id: str) -> bool:
    """Return whether the action should update P2 flags."""

    return action_id == "P2A1"


def build_flag_delta(action_id: str, selected_action: str) -> dict:
    """Build phase-local flags from the temporal mode decision."""

    if action_id != "P2A1":
        return {}

    return {
        "time_group": selected_action == "window_time",
        "where_time": selected_action in {"point_time", "range_time"},
        "latest_time": selected_action == "latest_time",
    }


def build_phase_spo_delta(phase_spo: dict, skill_outputs: list[dict]) -> dict:
    """Ensure all P2 decision SPO fields exist after phase execution."""

    return {
        PHASE_ID: {
            field: phase_spo.get(field)
            for field in ACTION_TO_SPO_FIELD.values()
        }
    }
