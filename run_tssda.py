"""Run the complete TSSDA workflow against a user-configured IoTDB database."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from iotdb_tools.schema_access import open_default_session
from tssda.agent import create_default_agent, create_qwen_agent
from tssda.config import get_default_schema_root
from tssda.decision_learning import DecisionLearningConfig, DecisionLearningStore
from tssda.execution_knowledge import ExecutionKnowledgeBase
from tssda.memory import StateMemoryStore
from tssda.runner import TSSDARunner
from tssda.schema import DatabaseSchema
from tssda.skill_routing import SkillRoutingController
from tssda.sql_validator import IoTDBSQLValidator


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=get_default_schema_root(),
                        help="Tree-model database root, e.g. root.example")
    parser.add_argument("--provider", choices=("deepseek", "qwen"),
                        default=os.getenv("TSSDA_PROVIDER", "deepseek"))
    parser.add_argument("--request", default="Generate meaningful, schema-grounded IoTDB NL2SQL pairs.")
    parser.add_argument("--user-goal", default="", help="Optional domain-specific generation request")
    parser.add_argument("--num-trajectories", type=positive_int, default=1)
    parser.add_argument("--max-samples", type=positive_int, default=3,
                        help="Maximum final pairs per trajectory")
    parser.add_argument("--output", type=Path, default=Path("runtime/nl2sql.json"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"),
                        help="Directory for new memory, learning state and execution knowledge")
    parser.add_argument("--log", type=Path, help="Optional trajectory audit log (JSON Lines)")
    parser.add_argument("--query-timeout-ms", type=positive_int, default=60000)
    parser.add_argument("--validation-timeout-ms", type=positive_int, default=75000)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _write_samples(path: Path, samples: list[dict[str, str]]) -> int:
    """Append validated pairs without duplicating previously exported SQL."""
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not isinstance(existing, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("sql"), str)
        or not isinstance(item.get("nl"), str) for item in existing
    ):
        raise ValueError("Output must contain a JSON array of SQL/NL pairs")
    seen = {item["sql"].strip().rstrip(";").strip() for item in existing}
    added = 0
    for sample in samples:
        key = sample["sql"].strip().rstrip(";").strip()
        if key not in seen:
            existing.append({"sql": sample["sql"].strip(), "nl": sample["nl"].strip()})
            seen.add(key)
            added += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return added


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.database.startswith("root."):
        parser.error("--database must name a database under root")
    # Reject path collisions before opening a database or making model calls.
    runtime = args.runtime_dir
    memory_path = runtime / "knowledge/state_memory.json"
    decision_path = runtime / "knowledge/decision_learning.json"
    routing_path = runtime / "knowledge/skill_routing.json"
    execution_path = runtime / "knowledge/execution_errors.json"
    destinations = [args.output, memory_path, decision_path, routing_path, execution_path]
    if args.log is not None:
        destinations.append(args.log)
    if len({path.resolve() for path in destinations}) != len(destinations):
        parser.error("Output, log and knowledge files must have distinct paths")

    agent = create_qwen_agent() if args.provider == "qwen" else create_default_agent()
    if not agent.config.api_key or not agent.config.model:
        parser.error("Configure the selected provider's API key and model through environment variables")
    session = open_default_session(connection_timeout_ms=args.query_timeout_ms)
    try:
        schema = DatabaseSchema.from_access(schema_root=args.database, session=session)
    finally:
        session.close()
    if not schema.meta_schema.get("devices"):
        parser.error("No devices found under the selected database")

    runner = TSSDARunner(
        schema=schema,
        base_agent=agent,
        progress_callback=None if args.quiet else print,
        state_memory_store=StateMemoryStore.load_or_create(memory_path),
        decision_learning_store=DecisionLearningStore.load(decision_path),
        decision_learning_config=DecisionLearningConfig(),
        skill_routing_controller=SkillRoutingController(storage_path=routing_path),
        execution_knowledge_base=ExecutionKnowledgeBase(execution_path),
        sql_validator=IoTDBSQLValidator(
            session_factory=lambda: open_default_session(connection_timeout_ms=args.query_timeout_ms),
            timeout_ms=args.query_timeout_ms,
            wall_timeout_ms=args.validation_timeout_ms,
        ),
    )
    exported = 0
    for index in range(args.num_trajectories):
        # Each trajectory starts from an independent, full schema snapshot.
        runner.schema = deepcopy(schema)
        runner.schema.reset_working_schema()
        state = runner.run(input_request={
            "database": args.database,
            "generation_request": args.request,
            "user_generation_goal": args.user_goal,
            "max_samples": args.max_samples,
        })
        report = state.sql_validation
        samples = (report.get("accepted_samples", [])
                   if report.get("validated") and report.get("passed") else [])
        if args.log is not None:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with args.log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "run_id": state.run_id,
                    "trajectory": index + 1,
                    "spo": state.spo,
                    "flags": state.flags,
                    "validation": report,
                }, ensure_ascii=False) + "\n")
        if samples:
            exported += _write_samples(args.output, samples)
        if not args.quiet:
            print(f"[TSSDA] Trajectory {index + 1}/{args.num_trajectories}: "
                  f"{len(samples)} executable pair(s)")
    if not args.quiet:
        print(f"[TSSDA] Exported {exported} new pair(s) to {args.output}")
    return 0 if exported else 1


if __name__ == "__main__":
    sys.exit(main())
