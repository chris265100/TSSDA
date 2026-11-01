"""Check generation coverage, execution boundaries and learning persistence."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skills.p0_initialization.mask import build_mask as task_mask
from skills.p1_scope.mask import build_mask as scope_mask
from skills.p3_group.mask import build_mask as group_mask
from skills.p4_operator import skills as operators
from skills.p4_operator.function_tables import FUNCTION_TABLES_BY_CATEGORY
from skills.p5_select.skills import SelectExpressionSkill
from skills.p8_into.mask import build_mask as alignment_mask
from tssda.decision_learning import DecisionLearningConfig, DecisionLearningStore
from tssda.iotdb_knowledge import load_non_function_iotdb_knowledge
from tssda.package_loader import SkillPackageLoader
from tssda.runner import DEFAULT_IMPLEMENTED_PHASES, TSSDARunner
from tssda.sql_generation_agent import SQLGenerationAgent, SQLGenerationInput
from tssda.sql_validator import IoTDBSQLValidator
from test_framework import build_schema


def test_all_phases_and_bundled_knowledge_are_self_contained():
    loader = SkillPackageLoader()
    for phase in DEFAULT_IMPLEMENTED_PHASES:
        package = loader.load(phase)
        assert package.skill_registry
        assert package.package_context
    assert "GROUP BY TAGS" in load_non_function_iotdb_knowledge()


def test_tag_scope_and_grouping_candidates_are_available():
    assert scope_mask("P1A1", {}, ["tag_aware", "all_scope"]) == ["tag_aware", "all_scope"]
    state = {"flags": {"P1": {"group_agg": True, "tag_agg": True, "level_agg": True},
                       "P2": {"time_group": True}}}
    candidates = ["none", "segmentation_only", "grouping_only", "hybrid_grouping"]
    assert group_mask("P3A1", state, candidates) == candidates
    assert group_mask("P3A3", state, ["tags", "level"]) == ["tags", "level"]


def test_every_catalog_function_belongs_to_a_selectable_family():
    families = (operators.BUILTIN_AGGREGATE_FAMILIES, operators.UDF_AGGREGATE_FAMILIES,
                operators.BUILTIN_TIMESERIES_FAMILIES, operators.UDF_TIMESERIES_FAMILIES)
    reachable = {name for group in families for family in group.values() for name in family["functions"]}
    catalog = {name for definitions in FUNCTION_TABLES_BY_CATEGORY.values() for name in definitions}
    assert catalog == reachable
    assert "TIMESTAMPREPAIR" in reachable
    IoTDBSQLValidator._check_statement("SELECT TIMESTAMPREPAIR(temperature) FROM root.example.device")


def test_select_allows_one_expression_from_a_larger_pool_without_depth_ordering():
    candidates = [{"name": "temperature", "op_level": 1},
                  {"name": "SIN(temperature)", "op_level": 2}]
    ordered, _ = SelectExpressionSkill._select_expression_pool(
        {"operator_store": {"SeriesExpr": candidates, "AggExpr": []}})
    assert [item["name"] for item in ordered] == [item["name"] for item in candidates]
    generation_input = SQLGenerationInput(
        context="", base_info={"action_id": "P5A1"}, generation_target={},
        working_schema=None, query_frame={}, state={}, schema_view={},
        extra_payload={"select_expression_candidates": candidates},
    )
    output = SQLGenerationAgent()._normalize_select_expression_output(
        {"select": ["temperature"]}, generation_input)
    assert output.query_frame_delta["select"] == ["temperature"]


def test_writeback_is_blocked_without_hiding_device_alignment():
    assert task_mask("P0A1", {}, ["retrieval", "materialization"]) == ["retrieval"]
    state = {"schema": {"working_schema": {"devices": [
        {"path": "root.example.d1", "measurements": [{"name": "temperature", "data_type": "DOUBLE"}]},
        {"path": "root.example.d2", "measurements": [{"name": "temperature", "data_type": "DOUBLE"}]},
    ]}}}
    assert "device_alignment" in alignment_mask("P8A1", state, ["time_alignment", "device_alignment"])
    assert alignment_mask("P8A2", {}, ["no_writeback", "non_aligned", "aligned"]) == ["no_writeback"]


@pytest.mark.parametrize("sql", [
    "SELECT temperature INTO root.example.copy FROM root.example.device",
    "SELECT temperature /* write */ INTO root.example.copy FROM root.example.device",
    "DELETE FROM root.example.device",
    "SELECT temperature FROM root.example.device; DROP DATABASE root.example",
])
def test_execution_guard_rejects_writeback_and_other_statements_before_connecting(sql):
    factory = Mock()
    result = IoTDBSQLValidator(session_factory=factory).validate(sql)
    assert not result.passed
    factory.assert_not_called()


def test_execution_guard_preserves_quoted_into_and_single_select():
    IoTDBSQLValidator._check_statement("SELECT 'into', `into` FROM root.example.device;")


def test_memory_and_learning_start_enabled_and_persist(tmp_path):
    runner = TSSDARunner(schema=build_schema())
    assert runner.memory_controller.read_enabled
    assert runner.decision_learning_controller is not None
    assert runner.skill_routing_controller is not None
    assert not DecisionLearningConfig().bypass_state_memory
    store = runner.decision_learning_controller.store
    assert store.trajectory_count == 0
    store.register_actions("TemporalModeSkill", ["no_time", "point_time", "latest_time"])
    store.save()
    restored = DecisionLearningStore.load(store.storage_path)
    assert set(restored.action_catalog_for("TemporalModeSkill")) == {"no_time", "point_time", "latest_time"}


def test_cli_exports_only_executed_pairs_and_uses_full_schema(tmp_path, monkeypatch):
    import run_tssda as cli

    schema = build_schema()
    agent = SimpleNamespace(config=SimpleNamespace(api_key="test-placeholder", model="test-model"))
    monkeypatch.setattr(cli, "create_default_agent", lambda: agent)
    session = Mock()
    monkeypatch.setattr(cli, "open_default_session", lambda **kwargs: session)
    monkeypatch.setattr(cli.DatabaseSchema, "from_access", lambda **kwargs: schema)
    accepted = {"sql": "SELECT temperature FROM root.test.d1", "nl": "Read the temperature."}
    unsupported = {"sql": "SELECT AVG(temperature) FROM root.test.** GROUP BY TAGS(city)",
                   "nl": "Average temperature by city."}
    runners = []

    class FakeRunner:
        def __init__(self, **kwargs):
            self.schema = kwargs["schema"]
            self.kwargs = kwargs
            self.seen_schemas = []
            runners.append(self)

        def run(self, input_request):
            self.seen_schemas.append(self.schema)
            assert len(self.schema.meta_schema["devices"]) == 2
            return SimpleNamespace(sql_validation={"validated": True, "passed": True,
                                                   "accepted_samples": [accepted],
                                                   "unsupported_samples": [unsupported]})

    monkeypatch.setattr(cli, "TSSDARunner", FakeRunner)
    assert cli.main(["--database", "root.test", "--num-trajectories", "2", "--quiet"]) == 0
    import json
    assert json.loads((tmp_path / "runtime/nl2sql.json").read_text()) == [accepted]
    assert runners[0].seen_schemas[0] is not runners[0].seen_schemas[1]
    session.close.assert_called_once()
