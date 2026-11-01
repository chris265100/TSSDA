"""Output definitions for the P4 Operator Semantics skill package."""

from __future__ import annotations


PHASE_ID = "P4"


SPO_SCHEMA = {
    "operator_regime": {
        "description": "The main operator mode selected for the query.",
        "source_actions": ["P4A1"],
    },
}


ACTION_TO_SPO_FIELD = {
    "P4A1": "operator_regime",
}


FLAG_SCHEMA = {
    "having_flag2": {
        "description": (
            "Whether P4 produced at least one level-1 aggregate expression "
            "that may support later HAVING-style predicate generation."
        ),
        "source_actions": ["P4 phase finalization"],
    },
}


def has_spo_update(action_id: str) -> bool:
    """Return whether the action should update P4 SPO."""

    return action_id in ACTION_TO_SPO_FIELD


def build_spo_delta(action_id: str, selected_action: str | None) -> dict:
    """Build the SPO delta for one completed P4 action."""

    spo_field = ACTION_TO_SPO_FIELD.get(action_id)
    if not spo_field:
        return {}
    return {
        PHASE_ID: {
            spo_field: selected_action,
        }
    }


def has_flag_update(action_id: str) -> bool:
    """Return whether the action should update P4 flags."""

    return False


def build_flag_delta(action_id: str, selected_action: str | None) -> dict:
    """P4 flag design is deferred."""

    return {}


def build_phase_spo_delta(phase_spo: dict, skill_outputs: list[dict]) -> dict:
    """Ensure all P4 decision SPO fields exist after phase execution."""

    return {
        PHASE_ID: {
            field: phase_spo.get(field)
            for field in ACTION_TO_SPO_FIELD.values()
        }
    }


def build_phase_flag_delta(
    working_schema: dict | None,
    query_frame: dict,
    state=None,
) -> dict:
    """Build P4 phase-level flags from the completed operator store."""

    return {
        PHASE_ID: {
            "having_flag2": _has_level_one_agg_expr(state),
        }
    }


def _has_level_one_agg_expr(state) -> bool:
    operator_store = _operator_store_from_state(state)
    agg_expr = operator_store.get("AggExpr", [])
    if not isinstance(agg_expr, list):
        return False

    return any(_expression_level(expression) == 1 for expression in agg_expr)


def _operator_store_from_state(state) -> dict:
    if state is None:
        return {}
    if hasattr(state, "get_operator_store"):
        return state.get_operator_store()
    if isinstance(state, dict):
        operator_store = state.get("operator_store", {})
        if isinstance(operator_store, dict):
            return operator_store
    return {}


def _expression_level(expression) -> int | None:
    if not isinstance(expression, dict):
        return None

    raw_level = expression.get("op_level", expression.get("level"))
    try:
        return int(raw_level)
    except (TypeError, ValueError):
        return None

