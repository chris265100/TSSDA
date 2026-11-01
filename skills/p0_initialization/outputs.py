"""Output definitions for the P0 Initialization skill package.

This file defines the SPO and Flag schema for P0.
It does not execute skills and does not mutate the global state directly.
"""


PHASE_ID = "P0"


# ---------------------------------------------------------------------------
# SPO definition
# ---------------------------------------------------------------------------

SPO_SCHEMA = {
    "task_family": {
        "description": "The selected main task family for the current SQL generation trajectory.",
        "source_actions": ["P0A1"],
        "source_skills": ["TaskFamilySkill"],
    },
    "task_subtype": {
        "description": "The selected fine-grained task subtype under the selected task family.",
        "source_actions": ["P0A2-1", "P0A2-2", "P0A2-3", "P0A2-4"],
        "source_skills": [
            "RetrievalSubtypeSkill",
            "ComputationSubtypeSkill",
            "AggregationSubtypeSkill",
            "MaterializationSubtypeSkill",
        ],
    },
}


ACTION_OUTPUT_SCHEMA = {
    "P0A1": {
        "skill_id": "TaskFamilySkill",
        "spo_field": "task_family",
        "flag_fields": ["writeback_required"],
    },
    "P0A2-1": {
        "skill_id": "RetrievalSubtypeSkill",
        "spo_field": "task_subtype",
        "flag_fields": [],
    },
    "P0A2-2": {
        "skill_id": "ComputationSubtypeSkill",
        "spo_field": "task_subtype",
        "flag_fields": [],
    },
    "P0A2-3": {
        "skill_id": "AggregationSubtypeSkill",
        "spo_field": "task_subtype",
        "flag_fields": [],
    },
    "P0A2-4": {
        "skill_id": "MaterializationSubtypeSkill",
        "spo_field": "task_subtype",
        "flag_fields": [],
    },
}


def get_spo_schema() -> dict:
    """Return the SPO schema of P0."""
    return SPO_SCHEMA


def get_action_output_schema() -> dict:
    """Return the action-to-output mapping of P0."""
    return ACTION_OUTPUT_SCHEMA


def get_action_output_spec(action_id: str) -> dict:
    """Return the output specification of one action."""
    return ACTION_OUTPUT_SCHEMA.get(action_id, {})


def has_spo_update(action_id: str) -> bool:
    """Return whether the action should update P0 SPO."""
    spec = get_action_output_spec(action_id)
    return bool(spec.get("spo_field"))


def build_spo_delta(action_id: str, selected_action: str) -> dict:
    """Build the SPO delta for one completed action.

    The selected action value is directly stored into the corresponding
    P0 SPO field.
    """
    spec = get_action_output_spec(action_id)

    if not spec or not spec.get("spo_field"):
        return {}

    spo_field = spec["spo_field"]

    return {
        PHASE_ID: {
            spo_field: selected_action
        }
    }


# ---------------------------------------------------------------------------
# Flag definition
# ---------------------------------------------------------------------------

FLAG_SCHEMA = {
    "writeback_required": {
        "description": "Whether P0 selected the materialization task family.",
        "source_actions": ["P0A1"],
    }
}


def get_flag_schema() -> dict:
    """Return the P0 Flag schema."""
    return FLAG_SCHEMA


def has_flag_update(action_id: str) -> bool:
    """Return whether the action should update P0 flags."""
    spec = get_action_output_spec(action_id)
    return bool(spec.get("flag_fields"))


def build_flag_delta(action_id: str, selected_action: str) -> dict:
    """Build the Flag delta for one completed action.

    P0 exposes the materialization intent for the P8 INTO phase.
    """
    if action_id != "P0A1":
        return {}
    return {
        "writeback_required": selected_action == "materialization",
    }


# ---------------------------------------------------------------------------
# Unified output interface
# ---------------------------------------------------------------------------

def build_output_delta(action_id: str, selected_action: str) -> dict:
    """Build both SPO and Flag deltas for one completed action.

    This is the unified interface used by the runner or BaseActionSkill.
    """
    return {
        "spo_delta": build_spo_delta(
            action_id=action_id,
            selected_action=selected_action,
        ),
        "flag_delta": build_flag_delta(
            action_id=action_id,
            selected_action=selected_action,
        ),
    }


def build_phase_query_frame_delta(phase_spo: dict) -> dict:
    """Build P0 phase-level QueryFrame updates after all P0 skills finish."""

    task_family = phase_spo.get("task_family")
    task_subtype = phase_spo.get("task_subtype")
    if not task_family or not task_subtype:
        return {}

    return {
        "query_type": f"本组查询意图：{task_family}，子任务类型：{task_subtype}。"
    }
