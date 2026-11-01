"""Output definitions for P5 Select Semantics."""

PHASE_ID = "P5"

SPO_SCHEMA = {
    "alias_mode": {
        "description": "Whether the SELECT expressions use aliases.",
        "source_actions": ["P5A2"],
    }
}

ACTION_TO_SPO_FIELD = {"P5A2": "alias_mode"}
FLAG_SCHEMA = {}


def has_spo_update(action_id: str) -> bool:
    return action_id in ACTION_TO_SPO_FIELD


def build_spo_delta(action_id: str, selected_action: str | None) -> dict:
    field = ACTION_TO_SPO_FIELD.get(action_id)
    return {PHASE_ID: {field: selected_action}} if field else {}


def has_flag_update(action_id: str) -> bool:
    return False


def build_flag_delta(action_id: str, selected_action: str | None) -> dict:
    return {}


def build_phase_spo_delta(phase_spo: dict, skill_outputs: list[dict]) -> dict:
    return {
        PHASE_ID: {
            field: phase_spo.get(field)
            for field in ACTION_TO_SPO_FIELD.values()
        }
    }
