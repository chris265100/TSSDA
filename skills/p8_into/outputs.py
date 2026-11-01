"""SPO and Flag definitions for P8 INTO semantics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PHASE_ID = "P8"

SPO_SCHEMA = {
    "result_alignment": {
        "description": "The selected time- or device-aligned result layout.",
        "source_actions": ["P8A1"],
    },
    "writeback_mode": {
        "description": "Whether INTO is disabled, non-aligned, or aligned.",
        "source_actions": ["P8A2"],
    },
    "target_mapping": {
        "description": "The validated INTO target mapping generated for SELECT groups.",
        "source_actions": ["P8A3"],
    },
}

ACTION_TO_SPO_FIELD = {
    "P8A1": "result_alignment",
    "P8A2": "writeback_mode",
}

FLAG_SCHEMA = {
    "align_by_device": {
        "description": "Whether the result uses ALIGN BY DEVICE.",
        "source_actions": ["P8A1"],
    },
    "has_into": {
        "description": "Whether the query uses an INTO clause.",
        "source_actions": ["P8A2"],
    },
    "into_clause_ready": {
        "description": "Whether validated INTO mappings are stored in QueryFrame.",
        "source_actions": ["P8A2", "P8A3"],
    },
}


def has_spo_update(action_id: str) -> bool:
    return action_id in ACTION_TO_SPO_FIELD


def build_spo_delta(action_id: str, selected_action: str | None) -> dict:
    field = ACTION_TO_SPO_FIELD.get(action_id)
    return {PHASE_ID: {field: selected_action}} if field else {}


def has_flag_update(action_id: str) -> bool:
    return action_id in {"P8A1", "P8A2", "P8A3"}


def build_flag_delta(action_id: str, selected_action: str | None) -> dict:
    if action_id == "P8A1":
        return {"align_by_device": selected_action == "device_alignment"}
    if action_id == "P8A2":
        enabled = selected_action in {
            "non_aligned_writeback",
            "aligned_writeback",
        }
        return {"has_into": enabled, "into_clause_ready": not enabled}
    if action_id == "P8A3":
        return {"has_into": True, "into_clause_ready": True}
    return {}


def build_phase_spo_delta(
    phase_spo: dict[str, Any],
    skill_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    target_mapping = phase_spo.get("target_mapping")
    if target_mapping is None:
        for output in reversed(skill_outputs):
            if output.get("action_id") != "P8A3":
                continue
            into_value = (output.get("query_frame_delta") or {}).get("into")
            if into_value:
                target_mapping = deepcopy(into_value)
                break
    return {
        PHASE_ID: {
            "result_alignment": phase_spo.get("result_alignment"),
            "writeback_mode": phase_spo.get("writeback_mode"),
            "target_mapping": target_mapping,
        }
    }
