"""Output definitions for P6 Predicate Semantics."""

from __future__ import annotations


PHASE_ID = "P6"


SPO_SCHEMA = {
    "predicate_mode": {
        "description": "Whether predicates are absent, WHERE-only, HAVING-only, or both.",
        "source_actions": ["P6A1"],
    },
    "where_clause_plan": {
        "description": "The value-side WHERE predicate generation action.",
        "source_actions": ["P6A2"],
    },
    "having_clause_plan": {
        "description": "The HAVING predicate generation action.",
        "source_actions": ["P6A3"],
    },
}


ACTION_TO_SPO_FIELD = {
    "P6A1": "predicate_mode",
    "P6A2": "where_clause_plan",
    "P6A3": "having_clause_plan",
}


FLAG_SCHEMA = {
    "has_predicate": {
        "description": "Whether the query uses any predicate clause.",
        "source_actions": ["P6A1"],
    },
    "has_where": {
        "description": "Whether P6 should generate a WHERE clause.",
        "source_actions": ["P6A1"],
    },
    "has_having": {
        "description": "Whether P6 should generate a HAVING clause.",
        "source_actions": ["P6A1"],
    },
}


def has_spo_update(action_id: str) -> bool:
    return action_id in ACTION_TO_SPO_FIELD


def build_spo_delta(action_id: str, selected_action: str | None) -> dict:
    field = ACTION_TO_SPO_FIELD.get(action_id)
    return {PHASE_ID: {field: selected_action}} if field else {}


def has_flag_update(action_id: str) -> bool:
    return action_id == "P6A1"


def build_flag_delta(action_id: str, selected_action: str | None) -> dict:
    if action_id != "P6A1":
        return {}
    return {
        "has_predicate": selected_action != "none",
        "has_where": selected_action in {"where_only", "both"},
        "has_having": selected_action in {"having_only", "both"},
    }


def build_phase_spo_delta(phase_spo: dict, skill_outputs: list[dict]) -> dict:
    return {
        PHASE_ID: {
            field: phase_spo.get(field)
            for field in ACTION_TO_SPO_FIELD.values()
        }
    }
