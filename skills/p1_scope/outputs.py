"""Output definitions for the P1 Scope skill package.

This file defines the SPO and Flag schema for P1.
It does not execute skills and does not mutate the global state directly.
"""


PHASE_ID = "P1"


# ---------------------------------------------------------------------------
# SPO definition
# ---------------------------------------------------------------------------

SPO_SCHEMA = {
    "query_object": {
        "description": "The main object type targeted by the current query.",
        "source_actions": ["P1A1"],
    },
    "device_scope": {
        "description": "The device-level scope involved in the current query.",
        "source_actions": ["P1A2"],
    },
    "measurement_scope": {
        "description": "The measurement-level scope involved in the current query.",
        "source_actions": ["P1A3"],
    },
    "tag_context": {
        "description": "The tag-related context involved in the current query.",
        "source_actions": ["P1A4"],
    },
}


ACTION_TO_SPO_FIELD = {
    "P1A1": "query_object",
    "P1A2": "device_scope",
    "P1A3": "measurement_scope",
    "P1A4": "tag_context",
}


def has_spo_update(action_id: str) -> bool:
    """Return whether the action should update P1 SPO."""
    return action_id in ACTION_TO_SPO_FIELD


def build_spo_delta(action_id: str, selected_action: str) -> dict:
    """Build the SPO delta for one completed action.

    The selected action value is stored into its corresponding P1 SPO field.
    """
    if not has_spo_update(action_id):
        return {}

    phase_delta = {}

    if action_id in ACTION_TO_SPO_FIELD:
        spo_field = ACTION_TO_SPO_FIELD[action_id]
        phase_delta[spo_field] = selected_action

    return {
        PHASE_ID: phase_delta
    }


# ---------------------------------------------------------------------------
# Flag definition
# ---------------------------------------------------------------------------

FLAG_SCHEMA = {
    "group_agg": {
        "description": "Whether the current working schema supports grouping or aggregation.",
        "source_actions": ["P1 phase finalization"],
    },
    "tag_agg": {
        "description": "Whether the current tag context supports tag-based aggregation.",
        "source_actions": ["P1 phase finalization"],
    },
    "level_agg": {
        "description": "Whether same-name measurements across devices support level-based aggregation.",
        "source_actions": ["P1A3"],
    },
    "measurement_count": {
        "description": "The number of measurements in the final working schema.",
        "source_actions": ["P1 phase finalization"],
    },
}


def has_flag_update(action_id: str) -> bool:
    """Return whether the action should update P1 flags."""
    return action_id == "P1A3"


def build_flag_delta(action_id: str, selected_action: str) -> dict:
    """Build the flag delta for one completed action."""

    if action_id != "P1A3":
        return {}

    return {
        "level_agg": selected_action == "same_measurement_across_devices",
    }


def build_phase_spo_delta(phase_spo: dict, skill_outputs: list[dict]) -> dict:
    """Ensure all P1 decision SPO fields exist after phase execution."""

    phase_delta = {}
    for field in ACTION_TO_SPO_FIELD.values():
        phase_delta[field] = phase_spo.get(field)
    return {
        PHASE_ID: phase_delta
    }


def build_phase_flag_delta(
    working_schema: dict | None,
    query_frame: dict,
) -> dict:
    """Build P1 phase-level flags from final runtime state."""

    measurement_count = _measurement_count(working_schema or {})
    flags = {
        "group_agg": True,
        "tag_agg": True,
        "measurement_count": measurement_count,
    }
    if measurement_count <= 1:
        flags["group_agg"] = False
    if not query_frame.get("tag"):
        flags["tag_agg"] = False
    return {
        PHASE_ID: flags
    }


def _measurement_count(working_schema: dict) -> int:
    devices = working_schema.get("devices", [])
    if isinstance(devices, list):
        count = 0
        for device in devices:
            if isinstance(device, dict) and isinstance(device.get("measurements"), list):
                count += len(device["measurements"])
        if count:
            return count

    measurements = working_schema.get("measurements", [])
    if isinstance(measurements, list) and measurements:
        return len(measurements)

    timeseries_count = working_schema.get("timeseries_count")
    if isinstance(timeseries_count, int):
        return timeseries_count

    return 0
