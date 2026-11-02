"""P10 execution feedback loop, with one shared three-generation budget."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .execution_knowledge import ExecutionKnowledgeBase, query_signature
from .infrastructure import InsufficientBalanceError
from .sql_generation_agent import SQLGenerationAgent, SQLGenerationInput
from .sql_validator import IoTDBSQLValidator


MAX_FINAL_GENERATIONS = 3


class ValidatedSQLGeneration:
    def __init__(
        self, agent: SQLGenerationAgent, validator: IoTDBSQLValidator,
        knowledge_base: ExecutionKnowledgeBase | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.agent = agent
        self.validator = validator
        configured_limit = getattr(validator, "connection_failure_limit", 3)
        self.connection_failure_limit = (
            configured_limit
            if isinstance(configured_limit, int) and configured_limit >= 1
            else 3
        )
        self.knowledge_base = knowledge_base if knowledge_base is not None else ExecutionKnowledgeBase()
        self.agent.execution_knowledge_base = self.knowledge_base
        self.progress_callback = progress_callback

    def run(
        self, generation_input: SQLGenerationInput, *, run_id: str, database: str,
    ) -> dict[str, Any]:
        accepted: list[dict[str, str]] = []
        unsupported: list[dict[str, str]] = []
        errors: list[dict[str, Any]] = []
        repair_targets: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        executed: dict[str, dict[str, Any]] = {}
        connection_failure_count = 0
        connection_failure_limit_reached = False
        base_agent = getattr(self.agent, "base_agent", None)
        base_output_limit = getattr(
            getattr(base_agent, "config", None),
            "max_output_tokens",
            None,
        )
        if isinstance(base_output_limit, bool) or not isinstance(
            base_output_limit, int
        ):
            base_output_limit = None
        max_samples = int(generation_input.extra_payload.get("max_sample_count") or 3)
        context = query_signature(generation_input.query_frame, generation_input.state)
        for attempt in range(1, MAX_FINAL_GENERATIONS + 1):
            successful = [*accepted, *unsupported]
            current_input = deepcopy(generation_input)
            current_input.execution_validation_pending = True
            current_input.extra_payload["max_sample_count"] = max_samples - len(successful)
            retrieved: dict[str, list[dict[str, Any]]] = {}
            if errors:
                current_input.extra_payload["execution_feedback"] = deepcopy(errors)
                current_input.extra_payload["accepted_samples"] = deepcopy(successful)
                current_input.extra_payload["repair_targets"] = deepcopy(repair_targets)
                if repair_targets:
                    current_input.extra_payload["max_sample_count"] = len(repair_targets)
                retrieved = self._retrieve_knowledge(repair_targets)
                current_input.extra_payload["repair_knowledge"] = {
                    code: [{"rule_id": item["rule_id"], "rule": item["rule"]} for item in rules]
                    for code, rules in retrieved.items()
                }
                # Repair can change an invalid expression in the original plan.
                # The context override limits such changes to necessary repairs.
                current_input.extra_payload["strict_assembly_contract"] = False
            self._emit(f"[TSSDA][P10] Generate and validate: attempt {attempt}/3")
            attempt_log: dict[str, Any] = {
                "attempt": attempt, "samples": [],
                "repair_targets": deepcopy(repair_targets),
                "retrieved_knowledge": deepcopy(retrieved),
            }
            attempts.append(attempt_log)
            try:
                attempt_output_limit = (
                    base_output_limit * 2
                    if attempt > 1 and base_output_limit is not None
                    else base_output_limit
                )
                temporary_limit = getattr(
                    base_agent,
                    "temporary_max_output_tokens",
                    None,
                )
                if base_output_limit is not None and callable(temporary_limit):
                    with temporary_limit(attempt_output_limit):
                        generated = self.agent.generate(current_input)
                else:
                    generated = self.agent.generate(current_input)
                candidates = generated.get("final_samples")
                if not isinstance(candidates, list) or not candidates:
                    raise ValueError("Final generation returned no SQL/NL samples")
            except InsufficientBalanceError:
                # Billing exhaustion is terminal for the entire run. Do not
                # spend the two repair generations on a request that cannot
                # succeed, and do not settle execution knowledge from it.
                raise
            except Exception as exc:
                error = {
                    "attempt": attempt, "sql": None, "nl": None,
                    "stage": "generation", "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                errors.append(error)
                attempt_log["generation_error"] = deepcopy(error)
                attempt_log["knowledge_update"] = self._settle_knowledge(
                    repair_targets, retrieved, [], None, run_id, attempt, context,
                )
                self._emit(f"[TSSDA][P10] Generation failed: {exc}")
                continue

            failed_this_attempt = False
            next_targets: list[dict[str, Any]] = []
            target_results: list[bool | None] = []
            attempt_log["repair_rule"] = generated.get("repair_rule")
            for index, sample in enumerate(candidates[:max_samples - len(successful)]):
                sql, nl = sample["sql"], sample["nl"]
                # Reuse terminal validations. A true failure may be retried, but
                # successful writebacks and confirmed server limitations are not.
                reused_result = sql in executed
                contract_error = self._contract_error(sql, current_input, index)
                has_into = self.agent._find_top_level_keyword(sql, "into") is not None
                if has_into and contract_error:
                    # Reject an unauthorized/mismatched writeback mapping before
                    # executing SQL that could mutate the database.
                    result = contract_error
                else:
                    if reused_result:
                        result = deepcopy(executed[sql])
                    else:
                        result = self.validator.validate(sql).to_dict()
                        if result["passed"] or result.get("unsupported_but_valid"):
                            executed[sql] = deepcopy(result)
                    if (
                        result["passed"] or result.get("unsupported_but_valid")
                    ) and contract_error:
                        result = contract_error
                entry = {"sql": sql, "nl": nl, **result, "reused_result": reused_result}
                attempt_log["samples"].append(entry)
                sample_accepted = False
                deferred_unsupported = bool(result.get("unsupported_but_valid"))
                connection_failure = IoTDBSQLValidator.is_connection_failure(result)
                if connection_failure:
                    connection_failure_count += 1
                if result["passed"] or deferred_unsupported:
                    current_successes = [*accepted, *unsupported]
                    if any(
                        item["sql"] == sql or item["nl"] == nl
                        for item in current_successes
                    ):
                        failed_this_attempt = True
                        error = {
                            "attempt": attempt, "sql": sql, "nl": nl,
                            "stage": "diversity", "error_type": "DuplicateSample",
                            "error_message": "Both SQL and NL must differ from accepted samples.",
                        }
                        errors.append(error)
                        next_targets.append(error)
                    else:
                        destination = unsupported if deferred_unsupported else accepted
                        destination.append({"sql": sql, "nl": nl})
                        sample_accepted = True
                        entry["storage_class"] = (
                            "unsupported_nl2sql"
                            if deferred_unsupported
                            else "validated_nl2sql"
                        )
                        if deferred_unsupported:
                            self._emit(
                                "[TSSDA][P10] Accepted separately: IoTDB server "
                                f"limitation {result.get('unsupported_feature')}"
                            )
                else:
                    failed_this_attempt = True
                    error = {"attempt": attempt, "sql": sql, "nl": nl,
                             **{key: result.get(key) for key in (
                                 "stage", "error_type", "error_code", "error_message"
                             )}}
                    error["error_type"] = ExecutionKnowledgeBase.error_type_for_failure(error) or error["error_type"]
                    errors.append(error)
                    # Infrastructure failures cannot be repaired by changing
                    # SQL and must never train execution-error knowledge.
                    if not connection_failure:
                        next_targets.append(error)
                    self._emit(f"[TSSDA][P10] SQL validation failed: {result['error_message']}")
                entry["accepted"] = sample_accepted
                # A server-limited sample is terminal success for the dataset,
                # but cannot prove or disprove a repair rule for another error.
                target_results.append(
                    None
                    if connection_failure or (sample_accepted and deferred_unsupported)
                    else sample_accepted
                )
                if (
                    connection_failure_count
                    >= self.connection_failure_limit
                ):
                    connection_failure_limit_reached = True
                    attempt_log["connection_failure_limit_reached"] = True
                    attempt_log["connection_failure_count"] = (
                        connection_failure_count
                    )
                    self._emit(
                        "[TSSDA][P10] IoTDB connection failure limit "
                        f"reached ({connection_failure_count}/"
                        f"{self.connection_failure_limit}); "
                        "skipping remaining validation and repair generations"
                    )
                    break
            if connection_failure_limit_reached:
                # Do not settle a pending repair rule as failed merely because
                # its validation was interrupted by unavailable infrastructure.
                attempt_log["knowledge_update"] = {}
                repair_targets = []
                break
            # Missing replacements are not evidence that their error types were
            # repaired. Keep them pending and mark the matching advice as failed.
            if len(target_results) < len(repair_targets):
                failed_this_attempt = True
                next_targets.extend(deepcopy(repair_targets[len(target_results):]))
            attempt_log["knowledge_update"] = self._settle_knowledge(
                repair_targets, retrieved, target_results,
                generated.get("repair_rule"), run_id, attempt, context,
            )
            repair_targets = next_targets
            self._emit(
                "[TSSDA][P10] Accepted "
                f"{len(accepted)} executable and {len(unsupported)} "
                "server-limited sample(s) so far"
            )
            if not failed_this_attempt or len(accepted) + len(unsupported) >= max_samples:
                break

        successful = [*accepted, *unsupported]
        passed = bool(successful)
        if connection_failure_limit_reached and not successful:
            reason = (
                "IoTDB connection failure limit reached during validation; "
                "remaining SQL and repair generations were skipped."
            )
        elif accepted and unsupported:
            reason = (
                "At least one SQL/NL sample passed execution validation and at "
                "least one was confirmed valid behind a current IoTDB server "
                "limitation."
            )
        elif accepted:
            reason = "At least one final SQL/NL sample passed execution validation."
        elif unsupported:
            reason = (
                "At least one final SQL/NL sample was confirmed valid behind a "
                "current IoTDB server limitation."
            )
        else:
            reason = "No sample passed validation within three final generations."
        report = {
            "validated": True, "passed": passed, "validation_scope": "iotdb_execution",
            "reason": reason,
            "generation_count": len(attempts),
            "accepted_count": len(successful),
            "executed_count": len(accepted),
            "unsupported_count": len(unsupported),
            "database": database,
            "accepted_samples": deepcopy(accepted),
            "unsupported_samples": deepcopy(unsupported),
            "attempts": attempts,
            "connection_failure_count": connection_failure_count,
            "connection_failure_limit": self.connection_failure_limit,
            "connection_failure_limit_reached": connection_failure_limit_reached,
            "knowledge_base_path": (str(self.knowledge_base.storage_path.resolve())
                                    if self.knowledge_base is not None else None),
        }
        first = successful[0] if successful else {}
        return {
            "final_samples": successful, "final_sql": first.get("sql"),
            "natural_language_query": first.get("nl"),
            "query_frame_delta": {},
            "generated_artifact": {
                "type": "nl2sql_samples",
                "content": {"samples": successful},
            },
            "sql_validation": report,
        }

    def _retrieve_knowledge(
        self, targets: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        codes = {code for target in targets
                 if (code := ExecutionKnowledgeBase.error_type_for_failure(target)) is not None}
        if not codes:
            return {}
        if self.knowledge_base is None:
            self.knowledge_base = ExecutionKnowledgeBase()
        return {code: self.knowledge_base.top_rules(code) for code in sorted(codes)}

    def _settle_knowledge(
        self, targets: list[dict[str, Any]], retrieved: dict[str, list[dict[str, Any]]],
        results: list[bool | None], rule: str | None, run_id: str, attempt: int,
        context: dict | None = None,
    ) -> dict[str, Any]:
        grouped: dict[str, list[bool]] = {}
        for index, target in enumerate(targets):
            code = ExecutionKnowledgeBase.error_type_for_failure(target)
            if code is None:
                continue
            result = results[index] if index < len(results) else False
            if result is not None:
                grouped.setdefault(code, []).append(result)
        outcomes = {code: all(values) for code, values in grouped.items()}
        if not outcomes or self.knowledge_base is None:
            return {}
        # The model supplies one shared rule for the repair batch. Do not admit
        # that rule from a partially failed or incomplete repair batch.
        complete_success = (
            len(results) == len(targets)
            and all(result is True for result in results)
        )
        updates = self.knowledge_base.settle_repair(
            retrieved=retrieved, outcomes=outcomes,
            rule=rule if complete_success else None, run_id=run_id, attempt=attempt,
            context=context,
        )
        return {"outcomes_by_error_type": outcomes, **updates}

    def _contract_error(
        self, sql: str, generation_input: SQLGenerationInput, index: int,
    ) -> dict[str, Any] | None:
        try:
            self.agent._validate_final_sql_contract(
                sql, generation_input=generation_input, sample_index=index,
            )
        except (TypeError, ValueError) as exc:
            return {
                "passed": False, "stage": "assembly_contract",
                "error_type": type(exc).__name__, "error_message": str(exc),
                "error_code": None,
            }
        return None

    def _emit(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)
