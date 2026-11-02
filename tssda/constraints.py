"""Manually curated constraints and dependency-DAG ordering.

RT/RD/RA classify temporal semantics, documented SQL restrictions and runtime
Analyzer compatibility checks. This is not an automatic rule extractor.
"""

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Callable


@dataclass(frozen=True)
class ConstraintRule:
    rule_id: str
    source: str
    action_id: str
    description: str
    allows: Callable[[dict, str], bool]


def _spo(state, phase, key):
    return state.get("spo", {}).get(phase, {}).get(key)


def _select_pool_available(state, _):
    regime = _spo(state, "P4", "operator_regime")
    if regime is None:
        return True
    kind = "AggExpr" if regime in {"aggregate_query", "mixed_nested_query"} else "SeriesExpr"
    return bool(state.get("operator_store", {}).get(kind))


def _alias_allowed(state, action):
    if action != "use_alias":
        return True
    schema = state.get("schema", {}).get("working_schema") or {}
    count = schema.get("device_count", len(schema.get("devices", [])))
    frame = state.get("query_frame_flat", state.get("query_frame", {}))
    return bool(frame.get("select")) and count <= 1


RULES = (
    ConstraintRule("T01", "R_T", "P2A2", "Range scope follows range-time mode.",
                   lambda s, a: _spo(s, "P2", "temporal_mode") in {None, "range_time"}),
    ConstraintRule("T02", "R_T", "P2A3", "Window size follows window-time mode.",
                   lambda s, a: _spo(s, "P2", "temporal_mode") in {None, "window_time"}),
    ConstraintRule("T03", "R_T", "P2A4", "Window step requires window-time mode and a resolved size.",
                   lambda s, a: _spo(s, "P2", "temporal_mode") == "window_time" and bool(_spo(s, "P2", "window_size"))),
    ConstraintRule("D01", "R_D", "P2A1", "SELECT LAST is a raw retrieval route, not an aggregate/computation route.",
                   lambda s, a: a != "latest_time" or _spo(s, "P0", "task_family") not in {"aggregation", "computation"}),
    ConstraintRule("A02", "R_A", "P5A2", "Aliases require selected expressions and a single-device scope.", _alias_allowed),
    ConstraintRule("A01", "R_A", "P5A1", "SELECT requires expressions of the current operator regime.", _select_pool_available),
)
R_T = tuple(rule for rule in RULES if rule.source == "R_T")
R_D = tuple(rule for rule in RULES if rule.source == "R_D")
R_A = tuple(rule for rule in RULES if rule.source == "R_A")
R_c = R_T + R_D + R_A


def mask_actions(action_id, state, actions):
    rules = [rule for rule in R_c if rule.action_id == action_id]
    return [action for action in actions if all(rule.allows(state, action) for rule in rules)]


def topological_phase_order(packages):
    """Build edges from required phases, SPOs and flags, then TopoSort(G)."""
    graph = {}
    available = {package.phase_id for package in packages}
    for package in packages:
        deps = package.package_config.get("dependencies", {})
        predecessors = set(deps.get("required_phases", []) or [])
        for key in ("required_spo", "required_flags"):
            predecessors.update(ref.split(".", 1)[0] for ref in deps.get(key, []) or [])
        graph[package.phase_id] = predecessors & available
    try:
        return list(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        raise ValueError(f"Cyclic phase dependencies: {exc.args[1]}") from exc
