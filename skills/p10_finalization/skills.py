"""Final NL2SQL assembly skill for P10."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


FINALIZATION_OUTPUT_INSTRUCTION = """
Return JSON only with exactly one field named `samples`.

{
  "samples": [
    {
      "sql": "a complete IoTDB SQL statement",
      "nl": "the corresponding natural-language request"
    }
  ]
}

Generate between 1 and `max_sample_count` samples when `max_sample_count` is
provided. Prefer to use the full allowance (normally three samples) whenever
the input supports that many reasonable and legal queries. Each sample must
contain exactly `sql` and `nl` string fields.
If multiple samples are generated, every sample must have a different SQL
statement and a different natural-language request. Prefer reasonable,
executable IoTDB SQL over length or complexity. Use meaningfully different
supported SELECT variants and semantic combinations; do not create nominal
diversity through formatting changes or NL paraphrases of the same SQL.
Do not include explanations, markdown, comments, validation reports, or any
other fields.
"""


class NL2SQLAssemblySkill:
    """P10A1: assemble one complete IoTDB NL2SQL data item."""

    skill_id = "NL2SQLAssemblySkill"
    action_id = "P10A1"
    decision_name = "nl2sql_assembly"

    candidate_descriptions = {
        "assemble_nl2sql": {
            "meaning": (
                "Assemble the completed QueryFrame and working schema into "
                "multiple high-quality IoTDB SQL and natural-language pairs."
            ),
            "requires_sql_generation": True,
            "generation_target": {
                "artifact_type": "nl2sql_pair",
                "generation_instruction": (
                    "Use the complete `query_frame` as the authoritative SQL "
                    "construction plan, `assembly_contract` as its normalized "
                    "executable form, and `working_schema` as the available "
                    "IoTDB tree-model schema scope. Generate multiple complete, "
                    "syntactically correct, semantically coherent IoTDB SQL "
                    "statements and matching user-style natural-language "
                    "requests. Use `max_sample_count` as an upper bound, not "
                    "a fixed target count. Prefer to generate the maximum "
                    "allowed number (normally three) when the input supports "
                    "that many reasonable, legal, and meaningfully different "
                    "queries. Strictly follow the supplied `query_frame`: do "
                    "not add SELECT expressions, FROM paths, time conditions, "
                    "GROUP BY clauses, filters, aliases, limits, or other SQL "
                    "parts that are not present in `query_frame`. If "
                    "`query_frame.select` contains multiple groups, each SQL "
                    "must use exactly one of those groups as its SELECT list. "
                    "Use `assembly_contract.sql_query_mode` to choose SELECT or "
                    "SELECT LAST; `query_frame.query_type` is a semantic intent "
                    "description and is not the SQL keyword. Use one complete "
                    "entry from `assembly_contract.select_variants` per sample. "
                    "If `query_frame.fill` is an object with a `clause` field, "
                    "use that field as the complete FILL clause. "
                    "If `query_frame.into` contains a `mappings` list, match "
                    "the SELECT group by its zero-based select index and use "
                    "exactly that mapping's `clause`; never mix an INTO "
                    "mapping with another SELECT group. For "
                    "`time_alignment`, time alignment is the default and no "
                    "ALIGN BY clause is required. For `device_alignment`, "
                    "append `ALIGN BY DEVICE`. "
                    "Treat `query_frame.order_by` as the ordered list of "
                    "items following the ORDER BY keyword. Render each "
                    "non-null pagination field with its exact keyword: "
                    "LIMIT, OFFSET, SLIMIT, and SOFFSET. Do not omit, change, "
                    "or invent any ordering or pagination value. Preserve "
                    "IoTDB clause order: ORDER BY, FILL, SLIMIT, SOFFSET, "
                    "LIMIT, OFFSET, then ALIGN BY. "
                    "Each "
                    "natural-language request must accurately express every "
                    "important SQL constraint and operation in its paired SQL. "
                    "It should be meaningful and possible to translate "
                    "unambiguously back into the SQL. Prioritize SQL legality, "
                    "semantic coherence, schema grounding, and data quality "
                    "rather than query length or complexity. When multiple "
                    "samples are possible, use meaningfully different supported "
                    "SELECT variants and semantic combinations to improve "
                    "diversity while preserving SQL validity and natural-"
                    "language quality. "
                    "If multiple samples are generated, every SQL statement "
                    "must be different from the others, and every "
                    "natural-language request must also be different from the "
                    "others. Do not invent "
                    "paths or measurements outside the "
                    "working schema. In the final SELECT clause, expressions "
                    "must not contain full paths. If `query_frame.select` "
                    "contains path-qualified expressions such as "
                    "`root.database.device.measurement`, remove the path "
                    "prefix and keep only the measurement name inside SELECT "
                    "expressions. Keep path-qualified scopes in FROM as "
                    "needed. Follow `output_instruction` exactly."
                ),
            },
        }
    }

    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state: dict, context: str, action_space: list[str]) -> dict[str, Any]:
        expected_action = "assemble_nl2sql"
        normalized_actions = list(dict.fromkeys(action_space))
        if normalized_actions != [expected_action]:
            raise ValueError(
                f"{self.skill_id} requires the single action {expected_action!r}"
            )
        candidate = self.candidate_descriptions[expected_action]
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "selected_action": expected_action,
            "structured_decision": {
                "selected_action": expected_action,
                "confidence": 1.0,
                "reason": "P10 uses a deterministic final assembly action.",
            },
            "action_meaning": candidate["meaning"],
            "action_effect": {"activate_next_skills": []},
            "requires_sql_generation": True,
            "generation_target": deepcopy(candidate["generation_target"]),
            "generation_output_instruction": FINALIZATION_OUTPUT_INSTRUCTION,
            "spo_delta": {},
            "flag_delta": {},
            "query_frame_delta": {},
            "trace": "NL2SQLAssemblySkill prepared final SQL/NL generation.",
            "confidence": 1.0,
            "reason": "P10 uses a deterministic final assembly action.",
            "semantic_decision_skipped": True,
        }


SKILL_REGISTRY = {"NL2SQLAssemblySkill": NL2SQLAssemblySkill}
