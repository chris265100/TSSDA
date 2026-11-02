"""Temporal retrieval, incremental routing and proactive experience checks."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import json

import pytest

from iotdb_tools.temporal_sampling import read_timestamp_sample
from skills.p2_temporal.mask import build_mask as temporal_mask
from skills.p5_select.mask import build_mask as select_mask
from tssda.constraints import R_c, topological_phase_order
from tssda.execution_knowledge import ExecutionKnowledgeBase, query_signature
from tssda.memory.expressions import ExpressionMemoryController, ExpressionMemoryStore, build_expression_record
from tssda.memory.matcher import MemoryMatchConfig, StateMemoryMatcher
from tssda.memory.models import MemoryIndex, MemoryStatistics, StateMemoryRecord, ValidatedStateTransition
from tssda.memory.profiles import StateProfileBuilder
from tssda.memory.runtime import MemoryValidationResult
from tssda.memory.store import StateMemoryStore
from tssda.memory.temporal import sampling_features
from tssda.package_loader import SkillPackageLoader
from tssda.runner import TSSDARunner
from tssda.schema import DatabaseSchema
from tssda.skill_routing import next_skill_targets, routing_candidates, SkillRoutingAgentProxy
from tssda.sql_generation_agent import SQLGenerationAgent, SQLGenerationInput
from tssda.sql_validator import SQLValidationResult
from tssda.state import State
from test_framework import build_schema, FirstCandidateSemanticAgent, FrameworkGenerationAgent


def observations(asynchronous=False):
    start = 1700000000000
    return {"measurements": [
        {"device": "root.test.d1", "measurement": name,
         "start_time_ms": start, "end_time_ms": start + 64000,
         "timestamps_ms": [start + i * 1000 + shift for i in range(64)]}
        for name, shift in (("temperature", 0), ("humidity", 500 if asynchronous else 0))
    ]}


def memory_state(asynchronous=False):
    raw = deepcopy(build_schema().meta_schema)
    for device in raw["devices"]:
        device["measurements"][1].update(name="humidity", data_type="DOUBLE",
                                           path=device["path"] + ".humidity")
    schema = DatabaseSchema.from_raw(raw, schema_root="root.test")
    state = State.create(schema=schema)
    schema.update_working_schema(raw, phase_id="P1")
    state.update_spo("P0", {"task_family": "retrieval", "task_subtype": "raw_series"})
    state.update_spo("P2", {"temporal_mode": "no_time"})
    state.update_spo("P3", {"group_paradigm": "none"})
    state.update_spo("P4", {"operator_regime": "aggregate_query"})
    state.set_memory_temporal_observations(observations(asynchronous))
    return state


def expression_record(state, expression="AVG(temperature)"):
    record = build_expression_record(state, "AggExpr", {"name": expression, "data_type": "DOUBLE", "op_level": 2})
    assert record is not None
    record.statistics = MemoryStatistics(ns=2)
    return record


def test_timestamp_alignment_is_not_endpoint_overlap():
    left, right = memory_state(), memory_state(True)
    builder = StateProfileBuilder()
    a, b = [builder.build_profile(state=s, masked_action_space=["range_time"]) for s in (left, right)]
    assert a.temporal.coverage_overlap == b.temporal.coverage_overlap == "common"
    assert a.temporal.timestamp_alignment == "aligned"
    assert b.temporal.timestamp_alignment == "asynchronous"
    index = MemoryIndex("P2", "TemporalModeSkill", "retrieval")
    record = StateMemoryRecord(index, a, ValidatedStateTransition("range_time"), MemoryStatistics(ns=3))
    matcher = StateMemoryMatcher()
    assert matcher.retrieve(index=index, current_profile=a, candidates=[record]).hit
    rejected = matcher.retrieve(index=index, current_profile=b, candidates=[record])
    assert not rejected.hit
    assert "temporal_timestamp_alignment_incompatible" in rejected.rejections[record.record_key]


def test_sampling_profiles_use_observations_and_preserve_unknowns():
    value = sampling_features(observations())
    assert value["sampling_interval"] == "second"
    assert value["sampling_regularity"] == "regular"
    assert value["point_distribution"] == "uniform"
    assert set(sampling_features({}).values()) == {"UNKNOWN"}
    sparse = observations()
    for item in sparse["measurements"]:
        item["timestamps_ms"] = [0, 1000, 2000, 3000, 100000]
    value = sampling_features(sparse)
    assert value["sampling_regularity"] == "irregular"
    assert value["point_distribution"] == "uneven"
    assert set(sampling_features(observations(), (0, 1)).values()) == {"UNKNOWN"}


def test_timestamp_probe_is_bounded_and_closes_cursor():
    dataset = Mock()
    dataset.has_next.return_value = True
    dataset.next.side_effect = [SimpleNamespace(get_timestamp=lambda i=i: i) for i in range(8)]
    session = Mock(execute_query_statement=Mock(return_value=dataset))
    result = read_timestamp_sample(session, {"measurement": "temperature", "device": "root.test.d1"}, start=0, end=100, limit=8)
    assert result == list(range(8))
    assert "LIMIT 8" in session.execute_query_statement.call_args.args[0]
    assert dataset.next.call_count == 8
    dataset.close_operation_handle.assert_called_once()


def test_p2_collects_sampling_evidence_with_a_fixed_query_budget():
    from skills.p2_temporal.tool import read_working_schema_time_ranges

    start = 1700000000000
    cursors = []

    def execute(sql):
        times = ([start + 1000 * i for i in range(64)] if "LIMIT 64" in sql
                 else [start + 63000] if "DESC" in sql else [start])
        cursor = Mock()
        cursor.has_next.side_effect = [True] * len(times) + [False]
        cursor.next.side_effect = [SimpleNamespace(get_timestamp=lambda t=t: t) for t in times]
        cursors.append(cursor)
        return cursor

    session = Mock(execute_query_statement=Mock(side_effect=execute))
    schema = {"devices": [{"path": "root.test.d1", "measurements": [
        {"name": f"s{i}", "data_type": "DOUBLE"} for i in range(10)]}]}
    observed = read_working_schema_time_ranges(working_schema=schema, session=session)
    assert observed["scope_sampled"] and observed["total_measurement_count"] == 10
    assert len(observed["measurements"]) == 8
    assert all(len(item["timestamps_ms"]) == 64 for item in observed["measurements"])
    assert session.execute_query_statement.call_count == 24
    assert sampling_features(observed)["timestamp_alignment"] == "aligned"
    for cursor in cursors:
        cursor.close_operation_handle.assert_called_once()
    session.close.assert_not_called()


def test_expression_retrieval_samples_one_and_settles_only_used_expression(tmp_path):
    source = memory_state()
    store = StateMemoryStore(storage_path=tmp_path / "memory.json")
    for expression in ("AVG(temperature)", "MAX(temperature)", "MIN(temperature)"):
        store.expressions.add(expression_record(source, expression))
    store.save()
    restored = StateMemoryStore.load(store.storage_path)
    controller = ExpressionMemoryController(restored.expressions, MemoryMatchConfig(random_seed=7))
    target = memory_state()
    trace = controller.populate(target, kind="AggExpr", max_level=10)
    assert trace["hit"] and len(trace["expressions"]) == 1
    assert len(trace["matches"]) == 3
    assert sum(item["selection_probability"] for item in trace["matches"]) == pytest.approx(1)
    assert all(item["profile_similarity"] == pytest.approx(1) for item in trace["matches"])
    expression = trace["expressions"][0]["name"]
    target.final_samples = [{"sql": f"SELECT {expression} FROM root.test.d1", "nl": "Aggregate."}]
    controller.finish_trajectory(MemoryValidationResult.passed_result())
    records = restored.expressions.to_list()
    assert sum(item["statistics"]["Nr"] for item in records) == 1
    assert sum(item["statistics"]["Ns"] for item in records) == 7


def test_expression_memory_rejects_asynchronous_inputs_and_unrelated_names():
    store = ExpressionMemoryStore([expression_record(memory_state(), "AVG(temperature - humidity)")])
    controller = ExpressionMemoryController(store, MemoryMatchConfig(random_seed=1))
    assert not controller.populate(memory_state(True), kind="AggExpr", max_level=10)["hit"]
    target = memory_state()
    raw = target.schema.get_active_working_schema()
    for device in raw["devices"]:
        for measurement in device["measurements"]:
            measurement["name"] = "unrelated_" + measurement["name"]
    target.schema.update_working_schema(raw, phase_id="P1")
    assert not controller.populate(target, kind="AggExpr", max_level=10)["hit"]


def test_legacy_profiles_load_without_bypassing_temporal_checks():
    state = memory_state()
    current = StateProfileBuilder().build_profile(state=state, masked_action_space=["range_time"])
    index = MemoryIndex("P2", "TemporalModeSkill", "retrieval")
    record = StateMemoryRecord(index, current, ValidatedStateTransition("range_time"), MemoryStatistics(ns=3))
    old = record.to_dict()
    old["profile"]["profile_version"] = 1
    for key in ("sampling_interval", "sampling_regularity", "timestamp_alignment", "point_distribution"):
        del old["profile"]["temporal"][key]
    legacy = StateMemoryRecord.from_dict(old)
    assert legacy.profile.temporal.timestamp_alignment == "UNKNOWN"
    assert not StateMemoryMatcher().retrieve(index=index, current_profile=current, candidates=[legacy]).hit
    expression = expression_record(state)
    expression.context.pop("profile")
    controller = ExpressionMemoryController(ExpressionMemoryStore([expression]), MemoryMatchConfig())
    assert not controller.populate(state, kind="AggExpr", max_level=10)["hit"]


def test_topological_sort_obeys_edges_and_rejects_cycles():
    def package(name, parents):
        return SimpleNamespace(phase_id=name, package_config={"dependencies": {"required_phases": parents}})
    order = topological_phase_order([package("P3", ["P1", "P2"]), package("P2", []), package("P1", [])])
    assert order[-1] == "P3"
    with pytest.raises(ValueError, match="Cyclic"):
        topological_phase_order([package("P1", ["P2"]), package("P2", ["P1"])])


def test_curated_masks_are_executable():
    assert {rule.source for rule in R_c} == {"R_T", "R_D", "R_A"}
    assert temporal_mask("P2A1", {"spo": {"P0": {"task_family": "aggregation"}}}, ["latest_time", "no_time"]) == ["no_time"]
    assert temporal_mask("P2A4", {"spo": {"P2": {"temporal_mode": "no_time"}}}, ["no_sliding_step"]) == []
    assert select_mask("P5A1", {"spo": {"P4": {"operator_regime": "aggregate_query"}}, "operator_store": {"AggExpr": []}}, ["select_from_operator_store"]) == []
    assert select_mask("P5A2", {"schema": {"working_schema": {"device_count": 2}}, "query_frame_flat": {"select": ["temperature"]}}, ["use_alias", "no_alias"]) == ["no_alias"]


def test_next_route_is_resolved_after_each_action():
    loader = SkillPackageLoader()
    p2 = loader.load("P2")
    assert next_skill_targets(p2, "TemporalModeSkill", "window_time", {}) == ["WindowSizeSkill"]
    assert next_skill_targets(p2, "WindowSizeSkill", "hour_level_window", {"spo": {"P2": {"temporal_mode": "window_time"}}}) == ["WindowStepSkill"]
    assert next_skill_targets(p2, "WindowSizeSkill", "hour_level_window", {"spo": {"P2": {"temporal_mode": "no_time"}}}) == []
    skill = p2.skill_registry["WindowSizeSkill"]()
    assert routing_candidates(p2, skill, {"spo": {"P2": {"temporal_mode": "window_time"}}})
    proxy = SkillRoutingAgentProxy(delegate=Mock(), phase="P2", routes={"window_time": ["WindowSizeSkill", "WindowStepSkill"]}, knowledge={})
    prepared = proxy.prepare_input({"state": {"package_config": p2.package_config}, "action_space": ["window_time"], "candidate_routing": {"old": "full plan"}})
    assert "package_config" not in prepared["state"]
    assert "candidate_routing" not in prepared
    assert prepared["candidate_descriptions"]["window_time"]["next_skills"] == ["WindowSizeSkill"]


def test_validated_experience_is_retrieved_before_generation_and_persists(tmp_path):
    kb = ExecutionKnowledgeBase(tmp_path / "knowledge.json")
    frame = {"select": ["AVG(temperature)"], "group_by": ["TIME(1h)"]}
    kb.settle_repair(retrieved={}, outcomes={"701": True}, rule="Keep valid aggregate grouping syntax.", run_id="learn", attempt=2, context=query_signature(frame))
    kb = ExecutionKnowledgeBase(kb.storage_path)
    assert len(kb.rules_for_generation(frame)) == 1
    assert not kb.rules_for_generation({"select": ["SIN(temperature)"]})
    base = Mock()
    base.call_json.return_value = {"select": ["AVG(temperature)"]}
    agent = SQLGenerationAgent(base_agent=base, execution_knowledge_base=kb)
    generation_input = SQLGenerationInput(context="", base_info={"action_id": "P5A1"}, generation_target={
        "generation_instruction": "Select a candidate.", "query_frame_key": "select", "artifact_type": "select_expression_list"},
        working_schema=None, query_frame=frame, state={}, schema_view={},
        extra_payload={"select_expression_candidates": [{"name": "AVG(temperature)", "data_type": "DOUBLE", "op_level": 2}]})
    agent.generate(generation_input)
    payload = json.loads(base.call_json.call_args.args[0][-1]["content"])
    assert payload["execution_knowledge"][0]["rule"] == "Keep valid aggregate grouping syntax."
    assert "execution_feedback" not in payload


class StopExplorationAgent(FirstCandidateSemanticAgent):
    def __init__(self):
        super().__init__({"P4A1": "aggregate_query", "P4A3": "stop_agg_expr"})
        self.structured_inputs = []

    def decide_structured(self, skill_input, decision_field="decision"):
        self.structured_inputs.append(deepcopy(skill_input))
        action = self.select_action(action_id=skill_input["action_id"], skill_input=skill_input)
        return {decision_field: action, "selected_action": action, "selected_expressions": []}


def test_p0_to_p10_memory_hit_keeps_p4_exploration_and_exports_sql(monkeypatch, tmp_path):
    from skills.p2_temporal import tool

    seed = memory_state()
    store = StateMemoryStore(storage_path=tmp_path / "memory.json")
    store.expressions.add(expression_record(seed))
    semantic = StopExplorationAgent()
    monkeypatch.setattr(tool, "read_working_schema_time_ranges", lambda **kwargs: observations())
    runner = TSSDARunner(schema=deepcopy(seed.schema), state_memory_store=store,
        semantic_decision_agent=semantic,
        sql_generation_agent=SQLGenerationAgent(base_agent=FrameworkGenerationAgent()),
        sql_validator=Mock(validate=Mock(return_value=SQLValidationResult(passed=True))))
    state = runner.run({"max_samples": 1})
    assert state.phase_progress == [f"P{i}" for i in range(11)]
    p4 = next(output for output in state.phase_outputs if output["phase_id"] == "P4")
    assert p4["expression_memory"]["hit"]
    assert any(item["skill_id"] == "AggregateExpressionSkill" for item in p4["skill_outputs"])
    assert semantic.structured_inputs
    assert "skill_routing_knowledge" in semantic.structured_inputs[0]
    assert state.sql_validation["executed_count"] == 1
    assert "AVG" in state.final_sql
    assert not state.skill_failures
    # The first validated trajectory activates distribution guidance. A later
    # memory hit must still pass that guidance into the exploration decision.
    runner.schema = deepcopy(seed.schema)
    second = runner.run({"max_samples": 1})
    assert second.sql_validation["executed_count"] == 1
    assert any("action_prior" in item for item in semantic.structured_inputs)
