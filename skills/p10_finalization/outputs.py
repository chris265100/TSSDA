"""Output definitions for P10 final NL2SQL assembly."""

from __future__ import annotations


PHASE_ID = "P10"

SPO_SCHEMA = {
    "final_sql": {
        "description": "The complete generated IoTDB SQL statement.",
        "source_actions": ["P10A1"],
    },
    "natural_language_query": {
        "description": "The user-style natural-language request paired with SQL.",
        "source_actions": ["P10A1"],
    },
    "final_samples": {
        "description": "The complete generated NL2SQL sample list.",
        "source_actions": ["P10A1"],
    },
    "nl2sql_data": {
        "description": "The final structured NL2SQL data payload.",
        "source_actions": ["P10A1"],
    },
}

FLAG_SCHEMA = {}


def has_spo_update(action_id: str) -> bool:
    return False


def build_spo_delta(action_id: str, selected_action: str | None) -> dict:
    return {}


def has_flag_update(action_id: str) -> bool:
    return False


def build_flag_delta(action_id: str, selected_action: str | None) -> dict:
    return {}


def build_phase_spo_delta(phase_spo: dict, skill_outputs: list[dict]) -> dict:
    return {
        PHASE_ID: {
            "final_sql": phase_spo.get("final_sql"),
            "natural_language_query": phase_spo.get("natural_language_query"),
            "final_samples": phase_spo.get("final_samples"),
            "nl2sql_data": phase_spo.get("nl2sql_data"),
        }
    }
