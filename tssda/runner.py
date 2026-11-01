"""Top-level TSSDA runtime runner.

The runner no longer calls hard-coded phase classes. It loads skill packages
from `skills/`, executes them through the phase/skill runners, and updates the
global State, Schema, and QueryFrame objects.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

from .agent import DataAgent, create_default_agent
from .decision_learning import (
    DecisionLearningConfig,
    DecisionLearningController,
    DecisionLearningStore,
)
from .errors import TrajectoryFailureLimitExceeded
from .execution_knowledge import ExecutionKnowledgeBase
from .infrastructure import (
    InsufficientBalanceError,
    infrastructure_error_reason,
    state_infrastructure_failure_reason,
)
from .package_loader import SkillPackageLoader
from .memory import (
    MemoryActivationPolicy,
    MemoryMatchConfig,
    StateMemoryController,
    StateMemoryStore,
    MemoryValidationResult,
)
from .phase_runner import PhaseRunner
from .schema import DatabaseSchema
from .semantic_decision_agent import SemanticDecisionAgent
from .skill_runner import SkillRunner
from .skill_routing import SkillRoutingController
from .sql_generation_agent import SQLGenerationAgent
from .state import PHASE_IDS, State
from .sql_validator import IoTDBSQLValidator
from .validated_generation import ValidatedSQLGeneration


DEFAULT_IMPLEMENTED_PHASES = [
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "P9",
    "P10",
]


class TSSDARunner:
    """Run TSSDA skill packages in phase order."""

    phase_order = PHASE_IDS

    def __init__(
        self,
        schema: DatabaseSchema,
        *,
        semantic_decision_agent: SemanticDecisionAgent | None = None,
        sql_generation_agent: SQLGenerationAgent | None = None,
        base_agent: DataAgent | None = None,
        skills_root: str | Path | None = None,
        enabled_phase_ids: list[str] | None = None,
        progress_callback: Callable[[str], None] | None = None,
        sql_validator: IoTDBSQLValidator | None = None,
        execution_knowledge_base: ExecutionKnowledgeBase | None = None,
        state_memory_enabled: bool = True,
        state_memory_store: StateMemoryStore | None = None,
        memory_match_config: MemoryMatchConfig | None = None,
        memory_activation_policy: MemoryActivationPolicy | None = None,
        memory_read_enabled: bool = True,
        memory_autosave: bool = True,
        memory_validator: Callable[
            [State], bool | MemoryValidationResult
        ] | None = None,
        decision_learning_enabled: bool = True,
        decision_learning_store: DecisionLearningStore | None = None,
        decision_learning_config: DecisionLearningConfig | None = None,
        decision_learning_controller: DecisionLearningController | None = None,
        decision_learning_autosave: bool = True,
        decision_learning_validator: Callable[
            [State], bool | MemoryValidationResult
        ] | None = None,
        decision_learning_accept_clean_trajectory: bool = False,
        skill_routing_enabled: bool = True,
        skill_routing_controller: SkillRoutingController | None = None,
    ):
        base_agent = base_agent or create_default_agent()
        self.schema = schema
        self.progress_callback = progress_callback
        self.semantic_decision_agent = semantic_decision_agent or SemanticDecisionAgent(
            base_agent=base_agent
        )
        self.sql_generation_agent = sql_generation_agent or SQLGenerationAgent(
            base_agent=base_agent
        )
        self.final_generation = ValidatedSQLGeneration(
            self.sql_generation_agent,
            sql_validator if sql_validator is not None else IoTDBSQLValidator(),
            knowledge_base=execution_knowledge_base,
            progress_callback=progress_callback,
        )
        self.package_loader = SkillPackageLoader(skills_root=skills_root)
        self.memory_validator = memory_validator
        self.decision_learning_validator = decision_learning_validator
        self.decision_learning_accept_clean_trajectory = (
            decision_learning_accept_clean_trajectory
        )
        self.last_memory_validation_result = MemoryValidationResult.not_run(
            "No trajectory has been finalized yet."
        )
        resolved_memory_store = state_memory_store
        if state_memory_enabled and resolved_memory_store is None:
            resolved_memory_store = StateMemoryStore.load_or_create()
        self.memory_controller = (
            StateMemoryController(
                store=resolved_memory_store,
                match_config=memory_match_config,
                activation_policy=memory_activation_policy,
                read_enabled=memory_read_enabled,
                autosave=memory_autosave,
            )
            if state_memory_enabled
            else None
        )
        if decision_learning_controller is not None:
            self.decision_learning_controller = decision_learning_controller
        elif decision_learning_enabled:
            self.decision_learning_controller = DecisionLearningController(
                store=decision_learning_store or DecisionLearningStore.load(),
                config=decision_learning_config,
                autosave=decision_learning_autosave,
            )
        else:
            self.decision_learning_controller = None
        self.last_decision_learning_validation_result = (
            MemoryValidationResult.not_run(
                "Decision learning is disabled or no trajectory has run yet."
            )
        )
        self.skill_routing_controller = (
            (skill_routing_controller or SkillRoutingController())
            if skill_routing_enabled else None
        )
        self.skill_runner = SkillRunner(
            semantic_decision_agent=self.semantic_decision_agent,
            sql_generation_agent=self.sql_generation_agent,
            memory_controller=self.memory_controller,
            decision_learning_controller=self.decision_learning_controller,
            final_generation=self.final_generation,
            skill_routing_controller=self.skill_routing_controller,
        )
        self.phase_runner = PhaseRunner(
            skill_runner=self.skill_runner,
            progress_callback=progress_callback,
        )
        self.enabled_phase_ids = enabled_phase_ids or list(DEFAULT_IMPLEMENTED_PHASES)
        self.last_state: State | None = None

    def run(
        self,
        input_request: dict[str, Any] | None = None,
        *,
        enabled_phase_ids: list[str] | None = None,
    ) -> State:
        """Run enabled phases and return the final global state."""

        state = State.create(schema=self.schema, input_request=input_request or {})
        self.last_state = state
        if self.memory_controller is not None:
            self.memory_controller.begin_trajectory()
        if self.decision_learning_controller is not None:
            self.decision_learning_controller.begin_trajectory()
        try:
            for phase_id in self._resolve_phase_ids(enabled_phase_ids):
                package = self.package_loader.load(phase_id)
                self._emit(
                    "[TSSDA] Enter phase "
                    f"{phase_id}: {package.package_config.get('phase_name', phase_id)}"
                )
                try:
                    package_output = self.phase_runner.run(
                        package=package,
                        state=state,
                        schema=state.schema,
                        query_frame=state.query_frame,
                        # P10 is deliberately best-effort. Missing upstream fields
                        # are sent to its agent as warnings instead of preventing
                        # final SQL/NL generation.
                        validate_dependencies=phase_id != "P10",
                    )
                except (TrajectoryFailureLimitExceeded, InsufficientBalanceError):
                    raise
                except Exception as exc:
                    # A recoverable phase-level dependency or local processing
                    # error must not prevent P10 from using the state accumulated
                    # so far. Only the explicit three-failed-skill exception aborts
                    # a trajectory early.
                    if phase_id == "P10":
                        raise
                    state.start_phase(phase_id)
                    state.complete_phase(phase_id)
                    package_output = {
                        "phase_id": phase_id,
                        "phase_status": "failed",
                        "spo_delta": {},
                        "flag_delta": {},
                        "query_frame_delta": {},
                        "working_schema": {},
                        "natural_language_semantics": {},
                        "errors": [
                            {
                                "code": "phase_execution_failed",
                                "message": str(exc),
                                "recoverable": True,
                                "phase_id": phase_id,
                                "error_type": type(exc).__name__,
                                "traceback": "".join(
                                    traceback.format_exception(
                                        type(exc), exc, exc.__traceback__
                                    )
                                ),
                            }
                        ],
                        "skill_outputs": [],
                        "package_input": {},
                    }
                state.phase_outputs.append(package_output)
                self._emit(
                    "[TSSDA] Complete phase "
                    f"{phase_id}: {package_output.get('phase_status', 'completed')}"
                )
        except Exception as exc:
            infrastructure_reason = (
                state_infrastructure_failure_reason(state)
                or infrastructure_error_reason(
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            )
            if self.memory_controller is not None:
                if infrastructure_reason:
                    self.last_memory_validation_result = (
                        MemoryValidationResult.not_run(
                            "Infrastructure failure; Memory changes were discarded.",
                            infrastructure_failure=infrastructure_reason,
                        )
                    )
                    self.memory_controller.discard_trajectory(
                        validation_result=self.last_memory_validation_result
                    )
                else:
                    self.last_memory_validation_result = (
                        MemoryValidationResult.failed_result(
                            "Trajectory failed before final SQL validation."
                        )
                    )
                    self.memory_controller.finish_trajectory(
                        validation_result=self.last_memory_validation_result
                    )
            if self.decision_learning_controller is not None:
                if infrastructure_reason:
                    self.last_decision_learning_validation_result = (
                        MemoryValidationResult.not_run(
                            "Infrastructure failure; decision learning was discarded.",
                            infrastructure_failure=infrastructure_reason,
                        )
                    )
                    self.decision_learning_controller.discard_trajectory(
                        validation_result=(
                            self.last_decision_learning_validation_result
                        )
                    )
                else:
                    self.last_decision_learning_validation_result = (
                        MemoryValidationResult.failed_result(
                            "Trajectory failed before decision-learning acceptance."
                        )
                    )
                    self.decision_learning_controller.finish_trajectory(
                        validation_result=(
                        self.last_decision_learning_validation_result
                        ),
                        state=state,
                    )
            if self.skill_routing_controller is not None:
                self.skill_routing_controller.finish(state, self.semantic_decision_agent, successful=False)
            raise

        infrastructure_reason = state_infrastructure_failure_reason(state)
        if infrastructure_reason:
            self.last_memory_validation_result = MemoryValidationResult.not_run(
                "Infrastructure failure; Memory changes were discarded.",
                infrastructure_failure=infrastructure_reason,
            )
            self.last_decision_learning_validation_result = (
                MemoryValidationResult.not_run(
                    "Infrastructure failure; decision learning was discarded.",
                    infrastructure_failure=infrastructure_reason,
                )
            )
            if self.memory_controller is not None:
                self.memory_controller.discard_trajectory(
                    validation_result=self.last_memory_validation_result
                )
            if self.decision_learning_controller is not None:
                self.decision_learning_controller.discard_trajectory(
                    validation_result=(
                        self.last_decision_learning_validation_result
                    )
                )
        else:
            self.last_memory_validation_result = self._validate_memory_trajectory(state)
            if self.memory_controller is not None:
                self.memory_controller.finish_trajectory(
                    validation_result=self.last_memory_validation_result
                )
            if self.decision_learning_controller is not None:
                self.last_decision_learning_validation_result = (
                    self._validate_decision_learning_trajectory(state)
                )
                self.decision_learning_controller.finish_trajectory(
                    validation_result=self.last_decision_learning_validation_result,
                    state=state,
                )
        if self.skill_routing_controller is not None:
            successful = bool(
                not infrastructure_reason and state.final_samples
                and (state.sql_validation.get("passed") if state.sql_validation
                     else self._trajectory_reached_validation_gate(state))
            )
            if successful and state.skill_routing_decisions:
                self._emit("[TSSDA] Validate Skill route semantics (one LLM call)")
            self.skill_routing_controller.finish(state, self.semantic_decision_agent, successful=successful)
        return state

    def _validate_decision_learning_trajectory(
        self,
        state: State,
    ) -> MemoryValidationResult:
        """Apply the acceptance gate before crediting action distribution."""

        if state.sql_validation:
            return self._execution_validation_result(state)
        if not self._trajectory_reached_validation_gate(state):
            return MemoryValidationResult.failed_result(
                "Trajectory did not produce a clean, complete P10 result."
            )
        if self.decision_learning_validator is not None:
            return self._invoke_validator(
                self.decision_learning_validator,
                state,
                validator_name="Decision-learning SQL validator",
            )
        if self.memory_validator is not None:
            if self.memory_controller is not None:
                # Memory validation runs immediately before this method. Reuse
                # the result so an execution validator (especially SELECT
                # INTO) is never invoked twice for the same trajectory.
                return self.last_memory_validation_result
            return self._invoke_validator(
                self.memory_validator,
                state,
                validator_name="Shared final SQL validator",
            )
        if self.decision_learning_accept_clean_trajectory:
            return MemoryValidationResult.passed_result(
                "Trajectory was accepted by the internal TSSDA P10 contract gate.",
                validation_scope="internal_contract",
            )
        return MemoryValidationResult.not_run(
            "Decision-learning validator is not configured; action counts were "
            "not updated."
        )

    def _validate_memory_trajectory(self, state: State) -> MemoryValidationResult:
        """Run the explicit final-SQL validation gate used by memory admission."""

        if state.sql_validation:
            return self._execution_validation_result(state)
        if not self._trajectory_reached_validation_gate(state):
            return MemoryValidationResult.failed_result(
                "Trajectory did not produce a clean, complete P10 result."
            )
        if self.memory_validator is None:
            return MemoryValidationResult.not_run(
                "Final SQL validator is not configured; provisional decisions "
                "were not admitted."
            )
        return self._invoke_validator(
            self.memory_validator,
            state,
            validator_name="Final SQL validator",
        )

    @staticmethod
    def _execution_validation_result(state: State) -> MemoryValidationResult:
        """Settle once, using the final outcome, never intermediate repair failures."""

        report = state.sql_validation
        factory = (MemoryValidationResult.passed_result
                   if report.get("passed") and state.final_samples
                   else MemoryValidationResult.failed_result)
        return factory(
            report.get("reason", "Final IoTDB execution validation completed."),
            validation_scope="iotdb_execution",
            generation_count=report.get("generation_count", 0),
            accepted_count=report.get("accepted_count", 0),
        )

    @staticmethod
    def _invoke_validator(
        validator: Callable[[State], bool | MemoryValidationResult],
        state: State,
        *,
        validator_name: str,
    ) -> MemoryValidationResult:
        try:
            result = validator(state)
        except Exception as exc:
            return MemoryValidationResult.not_run(
                f"{validator_name} raised an exception.",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        if isinstance(result, MemoryValidationResult):
            return result
        if isinstance(result, bool):
            return (
                MemoryValidationResult.passed_result()
                if result
                else MemoryValidationResult.failed_result()
            )
        return MemoryValidationResult.not_run(
            f"{validator_name} returned an unsupported result.",
            result_type=type(result).__name__,
        )

    @staticmethod
    def _trajectory_reached_validation_gate(state: State) -> bool:
        """Check prerequisites for invoking the future final SQL validator."""

        return bool(
            state.final_samples
            and state.failed_skill_count == 0
            and state.phase_outputs
            and all(
                output.get("phase_status") == "completed"
                for output in state.phase_outputs
            )
        )

    def _resolve_phase_ids(self, enabled_phase_ids: list[str] | None = None) -> list[str]:
        requested = enabled_phase_ids or self.enabled_phase_ids
        unknown = [phase_id for phase_id in requested if phase_id not in self.phase_order]
        if unknown:
            raise ValueError(f"Unknown phase ids: {unknown}")
        return [phase_id for phase_id in self.phase_order if phase_id in requested]

    def _emit(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)


# Backward-compatible alias used by existing imports.
SequentialRunner = TSSDARunner
