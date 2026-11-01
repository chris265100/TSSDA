import json
import unittest
from unittest.mock import Mock

from skills.p3_group.skills import GroupParadigmSkill, SegmentAxisSkill
from tssda.runner import DEFAULT_IMPLEMENTED_PHASES, TSSDARunner
from tssda.renderer import TreeSQLRenderer
from tssda.schema import DatabaseSchema
from tssda.sql_generation_agent import SQLGenerationAgent
from tssda.sql_validator import SQLValidationResult


class FirstCandidateSemanticAgent:
    def __init__(self, selections=None):
        self.last_decisions = {}
        self.selections = selections or {}

    def select_action(self, **kwargs):
        skill_input = kwargs.get("skill_input") or {}
        action_space = skill_input.get("action_space") or kwargs.get(
            "action_space", []
        )
        selected_action = self.selections.get(kwargs["action_id"], action_space[0])
        if selected_action not in action_space:
            raise AssertionError(
                f"Test selected {selected_action!r} outside {action_space}"
            )
        decision = {
            "selected_action": selected_action,
            "confidence": 1.0,
            "reason": "Full-framework deterministic test decision.",
        }
        self.last_decisions[kwargs["action_id"]] = decision
        return selected_action

    def get_last_decision(self, action_id):
        return self.last_decisions.get(action_id, {})


class FrameworkGenerationAgent:
    def call_json(self, messages):
        payload = json.loads(messages[-1]["content"])
        action_id = payload["base_info"]["action_id"]
        if action_id == "P5A1":
            candidates = payload["select_expression_candidates"]
            expression = ", ".join(item.get("name") or item["expression"] for item in candidates[:2])
            return {"select": [expression]}
        if action_id == "P10A1":
            contract = payload["assembly_contract"]
            sql = self._render_contract(contract)
            result = {
                "samples": [
                    {
                        "sql": sql,
                        "nl": "Retrieve the selected measurement values from all devices.",
                    }
                ]
            }
            if "Execution-Aware Repair" in messages[0]["content"]:
                result["rule"] = "Use valid IoTDB syntax without changing the intended query operations."
            return result
        raise AssertionError(f"Unexpected generation action: {action_id}")

    @classmethod
    def _render_contract(cls, contract):
        variant = contract["select_variants"][0]
        keyword = (
            "SELECT LAST"
            if contract["sql_query_mode"] == "select_last"
            else "SELECT"
        )
        clauses = [f"{keyword} {variant['select']}"]
        if variant.get("into_clause"):
            clauses.append(variant["into_clause"])
        clauses.append(f"FROM {', '.join(contract['from_paths'])}")
        if contract["where_predicates"]:
            clauses.append(
                f"WHERE {' AND '.join(contract['where_predicates'])}"
            )
        if contract["group_by_items"]:
            clauses.append(
                "GROUP BY "
                + ", ".join(
                    cls._strip_keyword(value, "GROUP BY")
                    for value in contract["group_by_items"]
                )
            )
        if contract["having_predicates"]:
            clauses.append(
                f"HAVING {' AND '.join(contract['having_predicates'])}"
            )
        if contract["order_by_items"]:
            clauses.append(
                f"ORDER BY {', '.join(contract['order_by_items'])}"
            )
        if contract.get("fill_clause"):
            clauses.append(contract["fill_clause"])
        for keyword_name in ("slimit", "soffset", "limit", "offset"):
            value = contract["pagination"].get(keyword_name)
            if value is not None:
                clauses.append(f"{keyword_name.upper()} {value}")
        if contract["result_alignment"] == "device_alignment":
            clauses.append("ALIGN BY DEVICE")
        return " ".join(clauses)

    @staticmethod
    def _strip_keyword(value, keyword):
        text = str(value).strip()
        return text[len(keyword) :].strip() if text.upper().startswith(keyword) else text


def build_schema():
    devices = []
    for device_name in ("d1", "d2"):
        device_path = f"root.test.{device_name}"
        devices.append(
            {
                "name": device_name,
                "path": device_path,
                "measurements": [
                    {
                        "name": "temperature",
                        "path": f"{device_path}.temperature",
                        "data_type": "DOUBLE",
                    },
                    {
                        "name": "status",
                        "path": f"{device_path}.status",
                        "data_type": "BOOLEAN",
                    },
                ],
            }
        )
    return DatabaseSchema.from_raw(
        {
            "schema_type": "iotdb_tree",
            "database": "root.test",
            "device_count": 2,
            "timeseries_count": 4,
            "devices": devices,
        },
        schema_root="root.test",
    )


class FullFrameworkTest(unittest.TestCase):
    def test_p3_group_flags_are_not_overwritten_by_base_skill(self):
        semantic_agent = FirstCandidateSemanticAgent()
        group_output = GroupParadigmSkill(llm=semantic_agent).run(
            state={},
            context="",
            action_space=["segmentation_only"],
        )
        self.assertEqual(
            group_output["flag_delta"],
            {
                "where_value": True,
                "having_flag": True,
                "group_by_enabled": True,
            },
        )

        segment_output = SegmentAxisSkill(llm=semantic_agent).run(
            state={},
            context="",
            action_space=["variation"],
        )
        self.assertEqual(segment_output["flag_delta"], {})

    def test_p0_through_p10_complete_with_normalized_contract(self):
        schema = build_schema()
        runner = TSSDARunner(
            schema=schema,
            state_memory_enabled=False,
            sql_validator=Mock(validate=Mock(return_value=SQLValidationResult(passed=True))),
            semantic_decision_agent=FirstCandidateSemanticAgent(),
            sql_generation_agent=SQLGenerationAgent(
                base_agent=FrameworkGenerationAgent(),
                fallback_to_empty=False,
            ),
            enabled_phase_ids=list(DEFAULT_IMPLEMENTED_PHASES),
        )

        state = runner.run(
            input_request={
                "generation_request": "Generate a schema-grounded query.",
                "max_samples": 2,
            }
        )

        self.assertEqual(state.phase_progress, [f"P{i}" for i in range(11)])
        self.assertEqual(state.query_frame.from_paths, ["root.test.**"])
        self.assertEqual(state.query_frame.select, ["temperature, status"])
        self.assertEqual(
            state.final_samples,
            [
                {
                    "sql": "SELECT temperature, status FROM root.test.**",
                    "nl": "Retrieve the selected measurement values from all devices.",
                }
            ],
        )
        self.assertTrue(state.get_phase_flags("P9")["result_organization_ready"])
        self.assertEqual(state.get_spo_value("P10", "final_sql"), state.final_samples[0]["sql"])
        self.assertEqual(
            TreeSQLRenderer().render(state.query_frame),
            "SELECT temperature, status FROM root.test.**",
        )

    def test_latest_query_is_consistent_across_p3_p4_p8_p9_and_p10(self):
        schema = build_schema()
        runner = TSSDARunner(
            schema=schema,
            state_memory_enabled=False,
            sql_validator=Mock(validate=Mock(return_value=SQLValidationResult(passed=True))),
            semantic_decision_agent=FirstCandidateSemanticAgent(
                {"P2A1": "latest_time"}
            ),
            sql_generation_agent=SQLGenerationAgent(
                base_agent=FrameworkGenerationAgent(),
                fallback_to_empty=False,
            ),
        )

        state = runner.run(input_request={"max_samples": 1})

        self.assertEqual(state.get_phase_spo("P3")["group_paradigm"], "none")
        self.assertFalse(state.get_phase_flags("P3")["where_value"])
        self.assertEqual(state.get_phase_spo("P4")["operator_regime"], "latest_query")
        self.assertEqual(
            state.get_phase_spo("P8")["result_alignment"],
            "time_alignment",
        )
        self.assertEqual(state.get_phase_spo("P9")["pagination_policy"], "no_pagination")
        self.assertEqual(
            state.final_samples[0]["sql"],
            "SELECT LAST temperature, status FROM root.test.**",
        )
        self.assertEqual(state.query_frame.sql_query_mode, "select_last")
        self.assertEqual(
            TreeSQLRenderer().render(state.query_frame),
            "SELECT LAST temperature, status FROM root.test.**",
        )


if __name__ == "__main__":
    unittest.main()
