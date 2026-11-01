"""Action-level skill execution for TSSDA packages."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import Any

from .mask_runner import MaskRunner
from .errors import (
    MAX_FAILED_SKILLS_PER_TRAJECTORY,
    MAX_MODEL_OUTPUT_RETRIES,
    TrajectoryFailureLimitExceeded,
    build_model_output_error,
    build_retry_instruction,
)
from .decision_learning import (
    DecisionLearningAgentProxy,
    DecisionLearningController,
)
from .package_loader import SkillPackage
from .memory import MemoryAwareDecisionProxy, StateMemoryController
from .schema import DatabaseSchema
from .semantic_decision_agent import SemanticDecisionAgent
from .sql_generation_agent import SQLGenerationAgent
from .state import QueryFrame, State
from .skill_routing import (
    SkillRoutingAgentProxy, SkillRoutingController, routing_candidates, semantic_facts,
)
from .sql_validator import IoTDBSQLValidator
from .validated_generation import ValidatedSQLGeneration


class SkillRunner:
    """Run one action-level skill with mask and model decision."""

    def __init__(
        self,
        semantic_decision_agent: SemanticDecisionAgent,
        sql_generation_agent: SQLGenerationAgent,
        mask_runner: MaskRunner | None = None,
        memory_controller: StateMemoryController | None = None,
        decision_learning_controller: DecisionLearningController | None = None,
        final_generation: ValidatedSQLGeneration | None = None,
        skill_routing_controller: SkillRoutingController | None = None,
    ):
        self.semantic_decision_agent = semantic_decision_agent
        self.sql_generation_agent = sql_generation_agent
        self.mask_runner = mask_runner or MaskRunner()
        self.memory_controller = memory_controller
        self.decision_learning_controller = decision_learning_controller
        self.skill_routing_controller = skill_routing_controller
        self.final_generation = final_generation or ValidatedSQLGeneration(
            sql_generation_agent, IoTDBSQLValidator()
        )

    def run(
        self,
        package: SkillPackage,
        skill_id: str,
        state: State,
        schema: DatabaseSchema,
        query_frame: QueryFrame,

    ) -> dict[str, Any]:
        """Run one skill, retrying model-output validation failures."""

        retry_errors: list[dict[str, Any]] = []
        # P10 owns its initial generation plus two repair attempts. Do not
        # multiply that budget by the generic model-output retry loop.
        total_attempts = 1 if package.phase_id == "P10" else MAX_MODEL_OUTPUT_RETRIES + 1
        for attempt in range(1, total_attempts + 1):
            retry_instruction = (
                build_retry_instruction(retry_errors[-1])
                if retry_errors
                else None
            )
            self.semantic_decision_agent.retry_instruction = retry_instruction
            self.sql_generation_agent.retry_instruction = retry_instruction
            try:
                with self._attempt_output_token_limit(attempt) as token_limit:
                    output = self._run_once(
                        package=package,
                        skill_id=skill_id,
                        state=state,
                        schema=schema,
                        query_frame=query_frame,
                    )
                self._stage_decision_learning(
                    skill_output=output,
                    state=state,
                    phase_id=package.phase_id,
                )
                return output
            except (TypeError, ValueError) as exc:
                error = build_model_output_error(exc, attempt=attempt)
                error["max_output_tokens"] = token_limit
                retry_errors.append(error)
                if attempt >= total_attempts:
                    break
            finally:
                self.semantic_decision_agent.retry_instruction = None
                self.sql_generation_agent.retry_instruction = None

        failure_output = self._build_failed_skill_output(
            package=package,
            skill_id=skill_id,
            retry_errors=retry_errors,
        )
        state.record_skill_failure(failure_output["pipeline_error"])
        if state.failed_skill_count >= MAX_FAILED_SKILLS_PER_TRAJECTORY:
            raise TrajectoryFailureLimitExceeded(
                state=state,
                failure=failure_output["pipeline_error"],
            )
        return self._normalize_skill_output(failure_output)

    @contextmanager
    def _attempt_output_token_limit(self, attempt: int):
        """Use the base limit once, then its doubled value for two retries."""

        bases = []
        seen = set()
        for owner in (
            self.semantic_decision_agent,
            self.sql_generation_agent,
        ):
            base = getattr(owner, "base_agent", None)
            if base is None or id(base) in seen:
                continue
            seen.add(id(base))
            bases.append(base)

        limits = []
        with ExitStack() as stack:
            for base in bases:
                configured = getattr(
                    getattr(base, "config", None),
                    "max_output_tokens",
                    None,
                )
                if isinstance(configured, bool) or not isinstance(
                    configured, int
                ):
                    configured = None
                limit = (
                    configured * 2
                    if attempt > 1 and configured is not None
                    else configured
                )
                limits.append(limit)
                temporary_limit = getattr(
                    base,
                    "temporary_max_output_tokens",
                    None,
                )
                if configured is not None and callable(temporary_limit):
                    stack.enter_context(temporary_limit(limit))
            yield next((limit for limit in limits if limit is not None), None)

    def _run_once(
        self,
        package: SkillPackage,
        skill_id: str,
        state: State,
        schema: DatabaseSchema,
        query_frame: QueryFrame,

    ) -> dict[str, Any]:
        """Run one skill attempt and return a normalized SkillOutput."""

        skill_cls = package.skill_registry.get(skill_id)
        if skill_cls is None:
            raise KeyError(f"Skill {skill_id} is not registered in {package.package_name}")

        skill = skill_cls(llm=self.semantic_decision_agent)
        routes = routing_candidates(package, skill) if self.skill_routing_controller is not None else {}
        raw_action_space = self._get_raw_action_space(skill)
        state_view = state.to_skill_state()
        state_view["package_config"] = deepcopy(package.package_config)
        if not raw_action_space and getattr(skill, "allow_empty_action_space", False):
            masked_action_space = []
        else:
            masked_action_space = self.mask_runner.apply(
                package=package,
                skill_id=skill.skill_id,
                action_id=skill.action_id,
                state_view=state_view,
                action_space=raw_action_space,
            )
        if hasattr(skill, "filter_action_space"):
            masked_action_space = skill.filter_action_space(
                state=state_view,
                context=package.package_context,
                action_space=masked_action_space,
            )
        learning_controller = self.decision_learning_controller
        learning_enabled = bool(
            learning_controller is not None and learning_controller.tracks(skill)
        )
        if learning_enabled and learning_controller is not None:
            learning_controller.observe_action_space(
                skill_id=skill.skill_id,
                actions=masked_action_space,
            )
        bypass_state_memory = bool(
            learning_enabled
            and learning_controller is not None
            and learning_controller.should_bypass_state_memory(skill)
        )
        # Routing stays an agent decision so online suitability can guide it.
        bypass_state_memory = bypass_state_memory or bool(routes)

        if not bypass_state_memory:
            self._prepare_memory_profile_context(
                package=package,
                skill=skill,
                state=state,
            )

        skill_input = self.build_skill_input(
            phase_id=package.phase_id,
            skill=skill,
            context=package.package_context,
            schema=schema,
            state=state,
            action_space=masked_action_space,
        )

        memory_proxy = None
        if (
            not bypass_state_memory
            and self.memory_controller is not None
            and self.memory_controller.supports(package.phase_id, skill)
        ):
            memory_proxy = MemoryAwareDecisionProxy(
                delegate=self.semantic_decision_agent,
                controller=self.memory_controller,
                state=state,
                phase_id=package.phase_id,
                skill=skill,
            )
            skill.llm = memory_proxy

        learning_proxy = None
        if learning_enabled and learning_controller is not None:
            learning_proxy = DecisionLearningAgentProxy(
                delegate=skill.llm,
                controller=learning_controller,
                skill_id=skill.skill_id,
            )
            skill.llm = learning_proxy

        routing_proxy = None
        if routes and self.skill_routing_controller is not None:
            facts = semantic_facts(state_view)
            knowledge = self.skill_routing_controller.guidance(package.phase_id, facts, masked_action_space)
            routing_proxy = SkillRoutingAgentProxy(
                delegate=skill.llm, phase=package.phase_id, routes=routes, knowledge=knowledge,
            )
            skill.llm = routing_proxy

        skill_output = skill.run(
            state=state_view,
            context=package.package_context,
            action_space=masked_action_space,
        )
        if memory_proxy is not None and memory_proxy.memory_hit:
            skill_output["semantic_decision_skipped"] = True
            decision = deepcopy(memory_proxy.memory_decision or {})
        else:
            decision = (
                {}
                if skill_output.get("semantic_decision_skipped")
                else self.semantic_decision_agent.get_last_decision(skill.action_id)
            )
        skill_output["decision_skill_invoked"] = bool(
            (memory_proxy is not None and memory_proxy.memory_hit)
            or not skill_output.get("semantic_decision_skipped")
        )
        skill_output["confidence"] = decision.get("confidence", skill_output.get("confidence", 0.0))
        skill_output["reason"] = decision.get("reason", skill_output.get("reason", ""))
        if routing_proxy is not None:
            record = self.skill_routing_controller.record(
                state=state, phase=package.phase_id, skill_id=skill.skill_id,
                action=skill_output["selected_action"], routes=routes, facts=facts,
                reason=skill_output["reason"], knowledge=knowledge,
            )
            skill_output["skill_routing"] = record
            skill_input = routing_proxy.prepare_input(skill_input)
        if learning_proxy is not None and learning_proxy.last_guidance is not None:
            guidance = learning_proxy.last_guidance
            skill_output["decision_learning"] = guidance.to_trace_dict()
            skill_input.update(guidance.to_model_fields())
        skill_output["skill_input"] = skill_input

        if not skill_output.get("spo_delta") and package.outputs_module is not None:
            skill_output.update(
                self._build_output_delta(
                    outputs_module=package.outputs_module,
                    action_id=skill.action_id,
                    selected_action=skill_output["selected_action"],
                )
            )

        # A fresh semantic decision enters the provisional buffer immediately.
        # SQL generation and later phases may still fail, so long-term admission
        # remains gated by the trajectory's final SQL validation result.
        self._stage_memory_decision(
            memory_proxy=memory_proxy,
            skill_output=skill_output,
            state=state,
        )

        generation_result = None
        query_frame_delta = {}
        if self._requires_generation(skill, skill_output):
            generation_target = self._get_generation_target(skill, skill_output)
            skill_output["generation_target"] = deepcopy(generation_target)
            generation_input = self.sql_generation_agent.build_input(
                phase_id=package.phase_id,
                phase_name=package.package_config.get("phase_name", package.phase_id),
                skill_output=skill_output,
                generation_target=generation_target,
                state=state,
                schema=schema,
                query_frame=query_frame,
                constraints=skill_output.get("generation_constraints")
                or skill_output.get("constraints"),
            )
            if self._is_tag_generation(skill_output) and skill_output.get("working_tag"):
                generation_input.extra_payload["working_tag"] = deepcopy(
                    skill_output["working_tag"]
                )
            if self._needs_temporal_generation_context(skill_output):
                self._attach_temporal_generation_context(
                    package=package,
                    state=state,
                    generation_input=generation_input,
                )
            if self._needs_window_step_generation_context(skill_output):
                self._attach_window_step_generation_context(
                    state=state,
                    generation_input=generation_input,
                )
            if self._needs_group_axis_generation_context(skill_output):
                self._attach_group_axis_generation_context(
                    state=state,
                    generation_input=generation_input,
                )
            if self._is_component_instantiation(skill_output):
                self._attach_component_instantiation_context(
                    skill_output=skill_output,
                    generation_input=generation_input,
                )
            if self._is_select_expression_generation(skill_output):
                self._attach_select_expression_generation_context(
                    skill_output=skill_output,
                    generation_input=generation_input,
                )
            if self._is_alias_generation(skill_output):
                self._attach_alias_generation_context(
                    skill_output=skill_output,
                    generation_input=generation_input,
                )
            if self._is_where_value_predicate_generation(skill_output):
                self._attach_where_value_predicate_generation_context(
                    skill_output=skill_output,
                    generation_input=generation_input,
                )
            if self._is_having_predicate_generation(skill_output):
                self._attach_having_predicate_generation_context(
                    skill_output=skill_output,
                    generation_input=generation_input,
                )
            if self._is_fill_instantiation_generation(skill_output):
                self._attach_fill_instantiation_generation_context(
                    skill_output=skill_output,
                    generation_input=generation_input,
                )
            if self._is_writeback_target_generation(skill_output):
                self._attach_writeback_target_generation_context(
                    skill_output=skill_output,
                    generation_input=generation_input,
                )
            if self._is_ordering_instantiation_generation(skill_output):
                self._attach_ordering_instantiation_generation_context(
                    skill_output=skill_output,
                    generation_input=generation_input,
                )
            if self._is_pagination_instantiation_generation(skill_output):
                self._attach_pagination_instantiation_generation_context(
                    skill_output=skill_output,
                    generation_input=generation_input,
                )
            if self._is_final_nl2sql_generation(skill_output):
                self._attach_final_nl2sql_generation_context(
                    skill_output=skill_output,
                    generation_input=generation_input,
                )
            if self._is_final_nl2sql_generation(skill_output):
                generation_result = self.final_generation.run(
                    generation_input, run_id=state.run_id, database=schema.schema_root,
                )
                state.sql_validation = deepcopy(generation_result["sql_validation"])
                skill_output["sql_validation"] = deepcopy(state.sql_validation)
            else:
                generation_result = self.sql_generation_agent.generate(generation_input)
            if self._is_where_value_predicate_generation(skill_output):
                generation_result = self._complete_where_value_predicate_generation(
                    generation_result=generation_result,
                    state=state,
                )
            if self._is_fill_instantiation_generation(skill_output):
                self._complete_fill_instantiation_generation(
                    skill_output=skill_output,
                    generation_result=generation_result,
                )
            if self._is_writeback_target_generation(skill_output):
                self._complete_writeback_target_generation(
                    skill_output=skill_output,
                    generation_result=generation_result,
                )
            if self._is_ordering_instantiation_generation(skill_output):
                self._complete_ordering_instantiation_generation(
                    skill_output=skill_output,
                    generation_result=generation_result,
                )
            if self._is_pagination_instantiation_generation(skill_output):
                self._complete_pagination_instantiation_generation(
                    skill_output=skill_output,
                    generation_result=generation_result,
                )
            if self._is_component_instantiation(skill_output):
                self._complete_component_instantiation(
                    skill_output=skill_output,
                    generation_result=generation_result,
                    state=state,
                )
            if self._is_tag_generation(skill_output):
                generation_result = self._complete_tag_generation_result(
                    generation_result=generation_result,
                    working_tag=skill_output.get("working_tag", {}),
                )
            if self._is_final_nl2sql_generation(skill_output):
                self._complete_final_nl2sql_generation(
                    skill_output=skill_output,
                    generation_result=generation_result,
                )
            natural_language_semantics = self._build_natural_language_semantics(
                skill_output=skill_output,
                generation_result=generation_result,
            )
            if natural_language_semantics:
                skill_output["natural_language_semantics"] = natural_language_semantics
            query_frame_delta = generation_result.get("query_frame_delta", {})
            if self._should_replace_group_axis_group_by(
                skill_output=skill_output,
                state=state,
                query_frame_delta=query_frame_delta,
            ):
                self._clear_group_by_before_group_axis_update(
                    state=state,
                    query_frame=query_frame,
                    phase_id=package.phase_id,
                )
            if generation_result.get("working_schema"):
                skill_output["working_schema"] = generation_result["working_schema"]
        else:
            working_schema = self._build_non_generation_working_schema(
                package=package,
                skill_output=skill_output,
                schema=schema,
            )
            if working_schema:
                skill_output["working_schema"] = working_schema
            query_frame_delta = self._build_non_generation_query_frame_delta(
                package=package,
                skill_output=skill_output,
                schema=schema,
            )

        if query_frame_delta:
            skill_output["query_frame_delta"] = query_frame_delta
        else:
            skill_output.setdefault("query_frame_delta", {})
        skill_output["generation_result"] = generation_result
        skill_output.setdefault("working_schema", {})
        return self._normalize_skill_output(skill_output)

    def _stage_decision_learning(
        self,
        *,
        skill_output: dict[str, Any],
        state: State,
        phase_id: str,
    ) -> None:
        """Buffer one valid high-level decision until trajectory acceptance."""

        controller = self.decision_learning_controller
        if controller is None or skill_output.get("pipeline_error"):
            return
        if (
            skill_output.get("skill_id") == "ComponentInstantiationSkill"
            and skill_output.get("source_component_skill")
            == "OperatorFamilySkill"
        ):
            # P4A8 only makes a semantic function decision after A5/A6. The
            # A7 path merely finalizes an operator that has already been chosen.
            return
        controller.stage_decision(
            skill_id=str(skill_output.get("skill_id") or ""),
            action_id=str(skill_output.get("action_id") or ""),
            selected_action=skill_output.get("selected_action"),
            phase_id=phase_id,
            run_id=state.run_id,
            guidance=skill_output.get("decision_learning", {}),
        )

    def _prepare_memory_profile_context(
        self,
        *,
        package: SkillPackage,
        skill: Any,
        state: State,
    ) -> None:
        if self.memory_controller is None or not self.memory_controller.supports(
            package.phase_id,
            skill,
        ):
            return
        if not self.memory_controller.should_prefetch_temporal_profile(
            state=state,
            phase_id=package.phase_id,
            skill=skill,
        ):
            return
        tool_module = package.tool_module
        if tool_module is None or not hasattr(
            tool_module,
            "read_working_schema_time_ranges",
        ):
            return
        try:
            observations = tool_module.read_working_schema_time_ranges(
                state=state.to_skill_state(),
                working_schema=state.schema.get_active_working_schema(),
            )
        except Exception:
            return
        state.set_memory_temporal_observations(observations)

    def _stage_memory_decision(
        self,
        *,
        memory_proxy: MemoryAwareDecisionProxy | None,
        skill_output: dict[str, Any],
        state: State,
    ) -> None:
        if (
            self.memory_controller is None
            or memory_proxy is None
            or not memory_proxy.invoked
            or memory_proxy.lookup_result is None
        ):
            return
        lookup = memory_proxy.lookup_result
        memory_trace = lookup.to_trace_dict()
        memory_trace["source"] = "memory" if lookup.hit else "agent"
        memory_trace["token_accounting"] = deepcopy(
            memory_proxy.token_accounting
        )
        skill_output["memory"] = memory_trace
        self.memory_controller.stage_decision(
            lookup=lookup,
            selected_action=str(skill_output["selected_action"]),
            state_delta={
                "spo_delta": deepcopy(skill_output.get("spo_delta", {})),
                "flag_delta": deepcopy(skill_output.get("flag_delta", {})),
                "action_effect": deepcopy(skill_output.get("action_effect", {})),
                "requires_sql_generation": bool(
                    skill_output.get("requires_sql_generation", False)
                ),
            },
            metadata={
                "run_id": state.run_id,
                "action_id": skill_output.get("action_id"),
                "decision_name": skill_output.get("decision_name"),
                "agent_confidence": skill_output.get("confidence", 0.0),
            },
        )

    def _build_failed_skill_output(
        self,
        *,
        package: SkillPackage,
        skill_id: str,
        retry_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a recoverable output that records and skips a failed skill."""

        skill_cls = package.skill_registry.get(skill_id)
        action_id = getattr(skill_cls, "action_id", None)
        decision_name = getattr(skill_cls, "decision_name", None)
        last_error = retry_errors[-1] if retry_errors else {}
        failure = {
            "code": "skill_model_output_failed",
            "message": (
                f"{skill_id} still failed after "
                f"{MAX_MODEL_OUTPUT_RETRIES} model-output retries: "
                f"{last_error.get('error_type', 'Error')}: "
                f"{last_error.get('message', '')}"
            ),
            "recoverable": True,
            "phase_id": package.phase_id,
            "skill_id": skill_id,
            "action_id": action_id,
            "attempt_count": len(retry_errors),
            "retry_count": max(0, len(retry_errors) - 1),
            "errors": deepcopy(retry_errors),
        }
        return {
            "skill_id": skill_id,
            "action_id": action_id,
            "decision_name": decision_name,
            "selected_action": None,
            "semantic_decision_skipped": True,
            "skill_skipped": True,
            "retry_count": failure["retry_count"],
            "retry_errors": deepcopy(retry_errors),
            "pipeline_error": failure,
            "trace": failure["message"],
        }

    def build_skill_input(
        self,
        phase_id: str,
        skill: Any,
        context: str,
        schema: DatabaseSchema,
        state: State,
        action_space: list[str],
    ) -> dict[str, Any]:
        """Build the unified SkillInput shape."""

        schema_context = schema.to_agent_context(include_detail=False)
        schema_view = (
            schema_context.get("working_schema")
            or schema_context.get("meta_schema")
            or schema_context
        )
        if hasattr(skill, "get_candidate_descriptions"):
            candidate_descriptions = skill.get_candidate_descriptions(
                state=state.to_skill_state(),
                action_space=action_space,
            )
        else:
            all_candidate_descriptions = getattr(
                skill,
                "candidate_descriptions",
                {},
            )
            candidate_descriptions = {
                action: deepcopy(all_candidate_descriptions[action])
                for action in action_space
                if action in all_candidate_descriptions
            }
        skill_input = {
            "phase_id": phase_id,
            "skill_id": skill.skill_id,
            "action_id": skill.action_id,
            "decision_name": skill.decision_name,
            "context": context,
            "schema_view": schema_view,
            "query_frame": state.query_frame.to_flat_dict(),
            "state": state.to_decision_state(),
            "natural_language_semantics": deepcopy(state.natural_language_semantics),
            "action_space": list(action_space),
            "candidate_descriptions": deepcopy(candidate_descriptions),
        }
        if hasattr(skill, "build_additional_skill_input"):
            additional_input = skill.build_additional_skill_input(
                state.to_skill_state()
            )
            if isinstance(additional_input, dict):
                skill_input.update(deepcopy(additional_input))
        return skill_input

    def _get_raw_action_space(self, skill: Any) -> list[str]:
        return list(getattr(skill, "candidate_descriptions", {}).keys())

    def _requires_generation(self, skill: Any, skill_output: dict[str, Any]) -> bool:
        if "requires_sql_generation" in skill_output:
            return bool(skill_output["requires_sql_generation"])
        candidate_info = self._get_candidate_info(skill, skill_output.get("selected_action"))
        if isinstance(candidate_info, dict):
            return bool(candidate_info.get("requires_sql_generation", False))
        return False

    def _get_generation_target(
        self,
        skill: Any,
        skill_output: dict[str, Any],
    ) -> dict[str, Any]:
        generation_target = skill_output.get("generation_target")
        if generation_target:
            return deepcopy(generation_target)

        candidate_info = self._get_candidate_info(skill, skill_output.get("selected_action"))
        if isinstance(candidate_info, dict) and candidate_info.get("generation_target"):
            return deepcopy(candidate_info["generation_target"])

        if hasattr(skill, "get_generation_target"):
            generation_target = skill.get_generation_target(
                selected_action=skill_output.get("selected_action"),
                skill_output=skill_output,
            )
            if generation_target:
                return deepcopy(generation_target)

        raise ValueError(
            f"{skill_output.get('skill_id')} requires SQL generation but did not "
            "provide generation_target"
        )

    def _get_candidate_info(self, skill: Any, selected_action: str | None) -> Any:
        if selected_action is None:
            return {}
        return getattr(skill, "candidate_descriptions", {}).get(selected_action, {})

    def _is_tag_generation(self, skill_output: dict[str, Any]) -> bool:
        return (
            skill_output.get("action_id") == "P1A4"
            and skill_output.get("selected_action") in {"single_tag", "tag_set"}
        )

    def _needs_temporal_generation_context(self, skill_output: dict[str, Any]) -> bool:
        return skill_output.get("action_id") in {"P2A2", "P2A3"}

    def _needs_window_step_generation_context(self, skill_output: dict[str, Any]) -> bool:
        return skill_output.get("action_id") == "P2A4"

    def _needs_group_axis_generation_context(self, skill_output: dict[str, Any]) -> bool:
        return skill_output.get("action_id") == "P3A3"

    def _is_component_instantiation(
        self,
        skill_output: dict[str, Any],
    ) -> bool:
        return skill_output.get("action_id") == "P4A8"

    @staticmethod
    def _is_select_expression_generation(
        skill_output: dict[str, Any],
    ) -> bool:
        return skill_output.get("action_id") == "P5A1"

    @staticmethod
    def _is_alias_generation(skill_output: dict[str, Any]) -> bool:
        return (
            skill_output.get("action_id") == "P5A2"
            and skill_output.get("selected_action") == "use_alias"
        )

    @staticmethod
    def _is_final_nl2sql_generation(skill_output: dict[str, Any]) -> bool:
        return skill_output.get("action_id") == "P10A1"

    @staticmethod
    def _is_where_value_predicate_generation(
        skill_output: dict[str, Any],
    ) -> bool:
        return skill_output.get("action_id") == "P6A2"

    @staticmethod
    def _is_having_predicate_generation(skill_output: dict[str, Any]) -> bool:
        return skill_output.get("action_id") == "P6A3"

    @staticmethod
    def _is_fill_instantiation_generation(skill_output: dict[str, Any]) -> bool:
        return skill_output.get("action_id") == "P7A2"

    @staticmethod
    def _is_writeback_target_generation(skill_output: dict[str, Any]) -> bool:
        return skill_output.get("action_id") == "P8A3"

    @staticmethod
    def _is_ordering_instantiation_generation(
        skill_output: dict[str, Any],
    ) -> bool:
        return skill_output.get("action_id") == "P9A2"

    @staticmethod
    def _is_pagination_instantiation_generation(
        skill_output: dict[str, Any],
    ) -> bool:
        return skill_output.get("action_id") == "P9A4"

    @staticmethod
    def _attach_select_expression_generation_context(
        skill_output: dict[str, Any],
        generation_input: Any,
    ) -> None:
        candidates = skill_output.get("select_expression_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                "P5A1 requires a non-empty select_expression_candidates list"
            )
        generation_input.extra_payload.update(
            {
                "select_expression_candidates": deepcopy(candidates),
                "select_expression_source": skill_output.get(
                    "select_expression_source"
                ),
                "output_instruction": skill_output.get(
                    "generation_output_instruction"
                ),
            }
        )

    @staticmethod
    def _attach_alias_generation_context(
        skill_output: dict[str, Any],
        generation_input: Any,
    ) -> None:
        current_select = generation_input.query_frame.get("select")
        if isinstance(current_select, str):
            if not current_select.strip():
                raise ValueError("P5A2 requires a non-empty query_frame.select")
            normalized_current_select = current_select.strip()
        elif isinstance(current_select, list):
            normalized_current_select = [
                str(group).strip()
                for group in current_select
                if str(group).strip()
            ]
            if not normalized_current_select:
                raise ValueError("P5A2 requires a non-empty query_frame.select")
        else:
            raise ValueError(
                "P5A2 requires query_frame.select to be a string or list"
            )
        generation_input.extra_payload.update(
            {
                "current_select": normalized_current_select,
                "output_instruction": skill_output.get(
                    "generation_output_instruction"
                ),
            }
        )

    @staticmethod
    def _attach_where_value_predicate_generation_context(
        skill_output: dict[str, Any],
        generation_input: Any,
    ) -> None:
        generation_input.extra_payload.update(
            {
                "working_measurements": deepcopy(
                    skill_output.get("working_measurements", [])
                ),
                "output_instruction": skill_output.get(
                    "generation_output_instruction"
                ),
                "where_generation_note": (
                    "Generate only value-related WHERE predicates over "
                    "working_measurements. Do not include time predicates. If "
                    "query_frame.time contains a time predicate, the runner "
                    "will merge it with the generated value predicate."
                ),
            }
        )

    @staticmethod
    def _attach_having_predicate_generation_context(
        skill_output: dict[str, Any],
        generation_input: Any,
    ) -> None:
        generation_input.extra_payload.update(
            {
                "level_one_aggregate_expressions": deepcopy(
                    skill_output.get("level_one_aggregate_expressions", [])
                ),
                "output_instruction": skill_output.get(
                    "generation_output_instruction"
                ),
                "having_generation_note": (
                    "Generate HAVING predicates only over "
                    "level_one_aggregate_expressions. Do not include the "
                    "HAVING keyword."
                ),
            }
        )

    def _complete_where_value_predicate_generation(
        self,
        generation_result: dict[str, Any],
        state: State,
    ) -> dict[str, Any]:
        completed = deepcopy(generation_result)
        query_frame_delta = completed.get("query_frame_delta") or {}
        value_predicate = self._predicate_text(query_frame_delta.get("where"))
        if not value_predicate:
            return completed

        time_predicate = self._time_predicate_text(
            state.query_frame.to_flat_dict().get("time")
        )
        if time_predicate:
            merged_predicate = f"({value_predicate}) AND ({time_predicate})"
        else:
            merged_predicate = value_predicate

        completed["query_frame_delta"] = {"where": [merged_predicate]}
        generated_artifact = completed.get("generated_artifact")
        if isinstance(generated_artifact, dict):
            generated_artifact["content"] = merged_predicate
        else:
            completed["generated_artifact"] = {
                "type": "where_predicate",
                "content": merged_predicate,
            }
        return completed

    @staticmethod
    def _attach_fill_instantiation_generation_context(
        skill_output: dict[str, Any],
        generation_input: Any,
    ) -> None:
        generation_input.extra_payload.update(
            {
                "selected_fill_mode": skill_output.get("selected_fill_mode"),
                "selected_expression_type_summary": deepcopy(
                    skill_output.get("selected_expression_type_summary", {})
                ),
                "fill_mode_support": deepcopy(
                    skill_output.get("fill_mode_support", {})
                ),
                "output_instruction": skill_output.get(
                    "generation_output_instruction"
                ),
                "fill_generation_note": (
                    "Generate only the FILL clause object. Do not generate a "
                    "complete SQL query. The FILL mode must match "
                    "selected_fill_mode and be compatible with the selected "
                    "expression data types."
                ),
            }
        )

    @staticmethod
    def _complete_fill_instantiation_generation(
        skill_output: dict[str, Any],
        generation_result: dict[str, Any],
    ) -> None:
        fill_value = (generation_result.get("query_frame_delta") or {}).get("fill")
        if not isinstance(fill_value, dict):
            raise ValueError("P7A2 SQL generation did not return a fill object")
        fill_clause = fill_value.get("clause")
        if not isinstance(fill_clause, str) or not fill_clause.strip():
            raise ValueError("P7A2 SQL generation did not return fill.clause")
        phase_delta = skill_output.setdefault("spo_delta", {}).setdefault("P7", {})
        phase_delta["fill_clause"] = fill_clause.strip()
        parameter = fill_value.get("parameter") or fill_value.get("interval") or fill_value.get("value")
        phase_delta["fill_parameter"] = parameter if parameter is not None else None

    @staticmethod
    def _attach_writeback_target_generation_context(
        skill_output: dict[str, Any],
        generation_input: Any,
    ) -> None:
        plan = skill_output.get("writeback_cardinality_plan")
        if not isinstance(plan, dict) or not plan.get("select_groups"):
            raise ValueError("P8A3 requires a non-empty writeback_cardinality_plan")
        generation_input.extra_payload.update(
            {
                "selected_alignment": skill_output.get("selected_alignment"),
                "selected_writeback_mode": skill_output.get(
                    "selected_writeback_mode"
                ),
                "writeback_cardinality_plan": deepcopy(plan),
                "output_instruction": skill_output.get(
                    "generation_output_instruction"
                ),
                "writeback_generation_note": (
                    "Generate only INTO mappings. The runner validates every "
                    "target-device and target-measurement count against the "
                    "cardinality plan and reconstructs each INTO clause from "
                    "the returned items."
                ),
            }
        )

    @staticmethod
    def _complete_writeback_target_generation(
        skill_output: dict[str, Any],
        generation_result: dict[str, Any],
    ) -> None:
        into_value = (generation_result.get("query_frame_delta") or {}).get("into")
        if not isinstance(into_value, dict) or not into_value.get("mappings"):
            raise ValueError(
                "P8A3 SQL generation did not return validated INTO mappings"
            )
        skill_output.setdefault("spo_delta", {}).setdefault("P8", {})[
            "target_mapping"
        ] = deepcopy(into_value)

    @staticmethod
    def _attach_ordering_instantiation_generation_context(
        skill_output: dict[str, Any],
        generation_input: Any,
    ) -> None:
        generation_input.extra_payload.update(
            {
                "selected_ordering_mode": skill_output.get(
                    "selected_ordering_mode"
                ),
                "result_alignment": skill_output.get("result_alignment"),
                "sortable_expressions": deepcopy(
                    skill_output.get("sortable_expressions", [])
                ),
                "output_instruction": skill_output.get(
                    "generation_output_instruction"
                ),
            }
        )

    @staticmethod
    def _complete_ordering_instantiation_generation(
        skill_output: dict[str, Any],
        generation_result: dict[str, Any],
    ) -> None:
        order_by = (generation_result.get("query_frame_delta") or {}).get(
            "order_by"
        )
        if not isinstance(order_by, list) or not order_by:
            raise ValueError("P9A2 SQL generation did not return ORDER BY items")
        skill_output.setdefault("spo_delta", {}).setdefault("P9", {})[
            "order_by_items"
        ] = deepcopy(order_by)

    @staticmethod
    def _attach_pagination_instantiation_generation_context(
        skill_output: dict[str, Any],
        generation_input: Any,
    ) -> None:
        generation_input.extra_payload.update(
            {
                "selected_pagination_policy": skill_output.get(
                    "selected_pagination_policy"
                ),
                "result_shape": deepcopy(skill_output.get("result_shape", {})),
                "output_instruction": skill_output.get(
                    "generation_output_instruction"
                ),
            }
        )

    @staticmethod
    def _complete_pagination_instantiation_generation(
        skill_output: dict[str, Any],
        generation_result: dict[str, Any],
    ) -> None:
        query_frame_delta = generation_result.get("query_frame_delta") or {}
        required_fields = {"limit", "offset", "slimit", "soffset"}
        if set(query_frame_delta) != required_fields:
            raise ValueError(
                "P9A4 SQL generation did not return all pagination fields"
            )
        skill_output.setdefault("spo_delta", {}).setdefault("P9", {})[
            "pagination_values"
        ] = deepcopy(query_frame_delta)

    @classmethod
    def _predicate_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value.strip().rstrip(";")
            lowered = text.lower()
            if lowered.startswith("where "):
                text = text[6:].strip()
            if lowered.startswith("having "):
                text = text[7:].strip()
            return text
        if isinstance(value, list):
            parts = [cls._predicate_text(item) for item in value]
            parts = [part for part in parts if part]
            return " AND ".join(parts)
        if isinstance(value, dict):
            for key in ("where", "having", "predicate", "condition", "content", "expression", "time"):
                if key in value:
                    text = cls._predicate_text(value.get(key))
                    if text:
                        return text
            return ""
        return str(value).strip()

    @classmethod
    def _time_predicate_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value.strip().rstrip(";")
            if not text:
                return ""
            lowered = text.lower()
            if lowered.startswith("where "):
                text = text[6:].strip()
                lowered = text.lower()
            if lowered.startswith("group by") or "group by" in lowered:
                return ""
            if lowered in {"no_time", "point_time", "latest_time", "window_time"}:
                return ""
            if "time" not in lowered and "now()" not in lowered:
                return ""
            return text
        if isinstance(value, list):
            parts = [cls._time_predicate_text(item) for item in value]
            parts = [part for part in parts if part]
            return " AND ".join(parts)
        if isinstance(value, dict):
            temporal_mode = value.get("temporal_mode")
            if temporal_mode in {"no_time", "point_time", "latest_time", "window_time"}:
                remaining_values = [
                    item
                    for key, item in value.items()
                    if key != "temporal_mode"
                ]
                parts = [cls._time_predicate_text(item) for item in remaining_values]
                parts = [part for part in parts if part]
                return " AND ".join(parts)
            for key in ("where", "predicate", "condition", "content", "expression", "time"):
                if key in value:
                    text = cls._time_predicate_text(value.get(key))
                    if text:
                        return text
            parts = [cls._time_predicate_text(item) for item in value.values()]
            parts = [part for part in parts if part]
            return " AND ".join(parts)
        return ""

    @classmethod
    def _attach_final_nl2sql_generation_context(
        cls,
        skill_output: dict[str, Any],
        generation_input: Any,
    ) -> None:
        input_request = generation_input.state.get("input_request", {})
        requested_count = input_request.get("max_samples")
        if requested_count is None:
            requested_count = input_request.get("num_samples")
        try:
            max_sample_count = max(1, int(requested_count or 3))
        except (TypeError, ValueError):
            max_sample_count = 3
        assembly_contract = cls._build_final_assembly_contract(generation_input)
        assembly_warnings = cls._final_assembly_contract_errors(
            assembly_contract,
            generation_input.state or {},
        )
        best_effort_assembly = bool(assembly_warnings)
        generation_input.extra_payload.update(
            {
                "output_instruction": skill_output.get(
                    "generation_output_instruction"
                ),
                "max_sample_count": max_sample_count,
                "assembly_contract": assembly_contract,
                "assembly_warnings": assembly_warnings,
                "best_effort_assembly": best_effort_assembly,
                "strict_assembly_contract": not best_effort_assembly,
                "input_description": (
                    "query_frame is the complete SQL construction plan; "
                    "assembly_contract is its normalized executable view when "
                    "strict_assembly_contract is true; working_schema is the "
                    "schema scope available to the SQL. When "
                    "best_effort_assembly is true, preserve usable current "
                    "information and complete missing essentials instead of "
                    "refusing final generation."
                ),
            }
        )

    @classmethod
    def _build_final_assembly_contract(cls, generation_input: Any) -> dict[str, Any]:
        """Normalize cross-phase QueryFrame data for deterministic P10 assembly."""

        query_frame = deepcopy(generation_input.query_frame or {})
        state = generation_input.state or {}
        spo = state.get("spo", {})
        flags = state.get("flags", {})
        latest_query = bool(
            query_frame.get("sql_query_mode") == "select_last"
            or flags.get("P2", {}).get("latest_time")
            or spo.get("P2", {}).get("temporal_mode") == "latest_time"
            or spo.get("P4", {}).get("operator_regime") == "latest_query"
        )

        raw_select = query_frame.get("select")
        if isinstance(raw_select, str):
            select_groups = [raw_select]
        elif isinstance(raw_select, list):
            select_groups = [
                str(value).strip()
                for value in raw_select
                if str(value).strip()
            ]
        else:
            select_groups = []

        into_value = query_frame.get("into")
        into_mappings: dict[int, str] = {}
        if isinstance(into_value, dict):
            mappings = into_value.get("mappings")
            if isinstance(mappings, list):
                for mapping in mappings:
                    if not isinstance(mapping, dict):
                        continue
                    select_index = mapping.get("select_index")
                    clause = mapping.get("clause")
                    if isinstance(select_index, int) and clause:
                        into_mappings[select_index] = str(clause).strip()
            elif into_value.get("clause"):
                into_mappings[0] = str(into_value["clause"]).strip()

        select_variants = [
            {
                "select_index": index,
                "select": select_group,
                "into_clause": into_mappings.get(index),
            }
            for index, select_group in enumerate(select_groups)
        ]

        where_items = cls._string_list(query_frame.get("where"))
        time_predicate = cls._time_predicate_text(query_frame.get("time"))
        if not where_items and time_predicate:
            where_items = [time_predicate]

        fill_value = query_frame.get("fill")
        fill_clause = None
        if isinstance(fill_value, dict):
            fill_clause = fill_value.get("clause")
        elif isinstance(fill_value, str):
            fill_clause = fill_value

        result_alignment = spo.get("P8", {}).get("result_alignment")
        if result_alignment not in {"time_alignment", "device_alignment"}:
            result_alignment = (
                "device_alignment"
                if str(query_frame.get("align_by") or "").lower() == "device"
                else "time_alignment"
            )

        return {
            "sql_query_mode": "select_last" if latest_query else "select",
            "query_intent": query_frame.get("query_type"),
            "select_variants": select_variants,
            "from_paths": cls._string_list(query_frame.get("from_paths")),
            "where_predicates": where_items,
            "group_by_items": cls._string_list(query_frame.get("group_by")),
            "having_predicates": cls._string_list(query_frame.get("having")),
            "fill_clause": str(fill_clause).strip() if fill_clause else None,
            "has_into": bool(into_value),
            "result_alignment": result_alignment,
            "order_by_items": cls._string_list(query_frame.get("order_by")),
            "pagination": {
                key: query_frame.get(key)
                for key in ("slimit", "soffset", "limit", "offset")
            },
            "clause_order": [
                "SELECT",
                "INTO",
                "FROM",
                "WHERE",
                "GROUP BY",
                "HAVING",
                "ORDER BY",
                "FILL",
                "SLIMIT",
                "SOFFSET",
                "LIMIT",
                "OFFSET",
                "ALIGN BY",
            ],
        }

    @staticmethod
    def _validate_final_assembly_contract(
        contract: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        """Reject incomplete cross-phase output before the final model call."""

        errors = SkillRunner._final_assembly_contract_errors(contract, state)
        if errors:
            raise ValueError("P10 assembly contract is incomplete: " + "; ".join(errors))

    @staticmethod
    def _final_assembly_contract_errors(
        contract: dict[str, Any],
        state: dict[str, Any],
    ) -> list[str]:
        """Return contract gaps so P10 can switch to best-effort generation."""

        errors = []
        variants = contract.get("select_variants") or []
        if not variants:
            errors.append("P5 did not provide any SELECT expression group")
        if not contract.get("from_paths"):
            errors.append("P1 did not provide any FROM path")
        if contract.get("has_into") and any(
            not variant.get("into_clause")
            for variant in variants
            if isinstance(variant, dict)
        ):
            errors.append("P8 INTO mappings do not cover every SELECT group")

        spo = state.get("spo", {})
        flags = state.get("flags", {})
        for phase_id, flag_name in (
            ("P7", "fill_clause_ready"),
            ("P8", "into_clause_ready"),
            ("P9", "result_organization_ready"),
        ):
            if flags.get(phase_id, {}).get(flag_name) is not True:
                errors.append(f"{phase_id}.{flag_name} is not ready")

        fill_mode = spo.get("P7", {}).get("fill_mode")
        if (fill_mode == "none") != (contract.get("fill_clause") is None):
            errors.append("P7 fill mode and QueryFrame.fill disagree")

        writeback_mode = spo.get("P8", {}).get("writeback_mode")
        writeback_enabled = writeback_mode in {
            "non_aligned_writeback",
            "aligned_writeback",
        }
        if writeback_enabled != bool(contract.get("has_into")):
            errors.append("P8 writeback mode and QueryFrame.into disagree")

        ordering_mode = spo.get("P9", {}).get("ordering_mode")
        if (ordering_mode == "no_order") != (
            not contract.get("order_by_items")
        ):
            errors.append("P9 ordering mode and QueryFrame.order_by disagree")

        pagination_policy = spo.get("P9", {}).get("pagination_policy")
        pagination_enabled = any(
            value is not None
            for value in (contract.get("pagination") or {}).values()
        )
        if (pagination_policy != "no_pagination") != pagination_enabled:
            errors.append("P9 pagination policy and QueryFrame pagination disagree")

        if (
            contract.get("sql_query_mode") == "select_last"
            and contract.get("result_alignment") == "device_alignment"
        ):
            errors.append("SELECT LAST cannot use device alignment")
        return errors

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        return [str(item).strip() for item in values if str(item).strip()]

    @staticmethod
    def _complete_final_nl2sql_generation(
        skill_output: dict[str, Any],
        generation_result: dict[str, Any],
    ) -> None:
        final_samples = generation_result.get("final_samples") or []
        if not final_samples and generation_result.get("final_sql"):
            final_samples = [
                {
                    "sql": generation_result.get("final_sql"),
                    "nl": generation_result.get("natural_language_query"),
                }
            ]
        validation = generation_result.get("sql_validation", {})
        if not final_samples and validation.get("validated") and not validation.get("passed"):
            skill_output["pipeline_error"] = {
                "code": "sql_validation_failed", "message": validation["reason"],
                "phase_id": "P10", "skill_id": skill_output["skill_id"],
                "recoverable": False,
            }
            skill_output["spo_delta"] = {"P10": {
                "final_sql": None, "natural_language_query": None,
                "final_samples": [], "nl2sql_data": {"samples": []},
            }}
            return
        if not isinstance(final_samples, list) or not final_samples:
            raise ValueError("P10A1 did not produce any complete SQL/NL samples")

        first_sample = final_samples[0]
        final_sql = first_sample.get("sql")
        natural_language_query = first_sample.get("nl")
        if not final_sql or not natural_language_query:
            raise ValueError("P10A1 did not produce a complete first SQL/NL pair")
        nl2sql_data = {"samples": deepcopy(final_samples)}
        skill_output["final_sql"] = final_sql
        skill_output["natural_language_query"] = natural_language_query
        skill_output["final_samples"] = deepcopy(final_samples)
        skill_output["nl2sql_data"] = nl2sql_data
        skill_output["spo_delta"] = {
            "P10": {
                "final_sql": final_sql,
                "natural_language_query": natural_language_query,
                "final_samples": deepcopy(final_samples),
                "nl2sql_data": nl2sql_data,
            }
        }

    def _attach_component_instantiation_context(
        self,
        skill_output: dict[str, Any],
        generation_input: Any,
    ) -> None:
        prepared_expression = skill_output.get(
            "prepared_constructing_expression"
        )
        if not prepared_expression:
            raise ValueError(
                "P4A8 requires a prepared constructing expression"
            )
        generation_input.extra_payload.update(
            {
                "constructing_expression": prepared_expression,
                "output_instruction": skill_output.get(
                    "generation_output_instruction"
                ),
                "source_component_skill": skill_output.get(
                    "source_component_skill"
                ),
            }
        )
        function_definition = skill_output.get(
            "selected_function_definition"
        )
        if function_definition:
            generation_input.extra_payload[
                "selected_function_definition"
            ] = deepcopy(function_definition)

    def _complete_component_instantiation(
        self,
        skill_output: dict[str, Any],
        generation_result: dict[str, Any],
        state: State,
    ) -> None:
        generated_expression = generation_result.get("generated_expression")
        if not isinstance(generated_expression, str) or not generated_expression.strip():
            raise ValueError(
                "P4A8 SQL generation did not return a complete expression"
            )

        current_expression = (
            state.operator_store.current_constructing_expression
        )
        if not isinstance(current_expression, dict):
            raise ValueError(
                "P4A8 requires the current constructing expression descriptor"
            )
        completed_expression = deepcopy(current_expression)
        expression_key = (
            "expression"
            if "expression" in completed_expression
            and "name" not in completed_expression
            else "name"
        )
        completed_expression[expression_key] = generated_expression.strip()
        function_definition = skill_output.get(
            "selected_function_definition"
        )
        if function_definition:
            resolved_output_type = self._resolve_component_output_type(
                function_definition=function_definition,
                input_data_type=current_expression.get("data_type"),
                generated_expression=generated_expression.strip(),
            )
            if resolved_output_type is None:
                raise ValueError(
                    "P4A8 could not resolve output data type for function "
                    f"{function_definition.get('function_name')!r} from "
                    f"expression {generated_expression.strip()!r}"
                )
            completed_expression["data_type"] = resolved_output_type
            skill_output["resolved_output_type"] = deepcopy(
                resolved_output_type
            )
        skill_output["constructing_expression_update"] = completed_expression
        skill_output["generated_expression"] = generated_expression.strip()
        skill_output["trace"] = (
            f"{skill_output.get('skill_id')} generated "
            f"{generated_expression.strip()}."
        )

    @staticmethod
    def _resolve_component_output_type(
        function_definition: dict[str, Any],
        input_data_type: Any,
        generated_expression: str,
    ) -> Any:
        from skills.p4_operator.function_tables import (
            resolve_function_output_type,
        )

        return resolve_function_output_type(
            function_definition=function_definition,
            input_data_type=input_data_type,
            generated_expression=generated_expression,
        )

    def _attach_temporal_generation_context(
        self,
        package: SkillPackage,
        state: State,
        generation_input: Any,
    ) -> None:
        tool_module = package.tool_module
        if tool_module is None or not hasattr(tool_module, "read_working_schema_time_ranges"):
            return

        time_range = tool_module.read_working_schema_time_ranges(
            state=state.to_skill_state(),
            working_schema=state.schema.get_active_working_schema(),
        )
        state.set_memory_temporal_observations(time_range)
        generation_input.extra_payload["data_time_range"] = deepcopy(time_range)
        if generation_input.base_info.get("action_id") == "P2A3":
            generation_input.extra_payload["data_time_range_instruction"] = (
                "This is the valid data time range after excluding implausible "
                "timestamps outside the configured calendar-year bounds. The start_time and "
                "end_time values use YYYY-MM-DDTHH:MM:SS format; *_ms fields "
                "are raw epoch milliseconds for reference only. When choosing "
                "the window range, use YYYY-MM-DDTHH:MM:SS timestamps and keep "
                "startTime and endTime within this range as much as possible. "
                "Measurements with null bounds have no points in the valid range; "
                "never invent or use 1970-era timestamps for them. "
                "Do not output epoch-millisecond timestamps in the time artifact."
            )
            generation_input.extra_payload["time_generation_output_format"] = {
                "time": (
                    "The generated GROUP BY time-window expression to write into "
                    "QueryFrame.time, using YYYY-MM-DDTHH:MM:SS timestamps."
                ),
                "temporal_semantic": (
                    "The natural-language temporal meaning of the generated window."
                ),
            }
            return

        generation_input.extra_payload["data_time_range_instruction"] = (
            "This is the valid data time range after excluding implausible "
            "timestamps outside the configured calendar-year bounds. The start_time and end_time "
            "values use YYYY-MM-DDTHH:MM:SS format; *_ms fields are raw epoch "
            "milliseconds for reference only. Generate a time artifact that "
            "uses YYYY-MM-DDTHH:MM:SS timestamps and overlaps this range so the "
            "query can return non-empty results; range filters do not need to "
            "be fully contained in it. Measurements with null bounds have no "
            "points in the valid range; never invent or use 1970-era timestamps "
            "for them. Do not output epoch-millisecond "
            "timestamps in the time artifact."
        )
        generation_input.extra_payload["time_generation_output_format"] = {
            "time": "The generated time scope to write into QueryFrame.time."
        }

    def _attach_window_step_generation_context(
        self,
        state: State,
        generation_input: Any,
    ) -> None:
        natural_language_semantics = deepcopy(state.natural_language_semantics)
        generation_input.extra_payload["natural_language_semantics"] = (
            natural_language_semantics
        )
        generation_input.extra_payload["window_step_context_instruction"] = (
            "natural_language_semantics.temporal_semantic and query_frame.time "
            "are the already designed window meaning and expression. Design the "
            "sliding step from this existing window. Keep timestamps in "
            "YYYY-MM-DDTHH:MM:SS format and do not output epoch-millisecond "
            "timestamps."
        )
        generation_input.extra_payload["time_generation_output_format"] = {
            "time": (
                "Return the updated GROUP BY window expression in the form "
                "group by([YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM:SS), "
                "interval, slidingStep)."
            ),
            "temporal_semantic": (
                "The updated natural-language temporal meaning of the sliding window."
            ),
        }

    def _attach_group_axis_generation_context(
        self,
        state: State,
        generation_input: Any,
    ) -> None:
        group_paradigm = state.get_spo_value("P3", "group_paradigm")
        is_hybrid_grouping = group_paradigm == "hybrid_grouping"
        selected_action = generation_input.base_info.get("selected_action")
        axis_format = self._group_axis_output_axis_format(selected_action)
        existing_group_by = deepcopy(state.query_frame.to_flat_dict().get("group_by", []))

        generation_input.extra_payload["group_axis_context"] = {
            "group_paradigm": group_paradigm,
            "is_hybrid_grouping": is_hybrid_grouping,
            "existing_group_by": existing_group_by,
        }
        if is_hybrid_grouping:
            generation_input.extra_payload["group_axis_generation_instruction"] = (
                "query_frame.group_by already contains the time/window GROUP BY. "
                "Combine that existing window aggregation with the selected group "
                "axis and return one final group_by value, not a second independent "
                "GROUP BY clause."
            )
            generation_input.extra_payload["group_axis_output_format"] = {
                "group_by": (
                    "GROUP BY([startTime, endTime), interval [, slidingStep]), "
                    f"{axis_format}"
                )
            }
            return

        generation_input.extra_payload["group_axis_generation_instruction"] = (
            "query_frame.group_by is expected to be empty. Generate only the "
            "selected cross-series grouping clause."
        )
        generation_input.extra_payload["group_axis_output_format"] = {
            "group_by": f"GROUP BY {axis_format}"
        }

    def _group_axis_output_axis_format(self, selected_action: Any) -> str:
        if selected_action == "tags":
            return "TAGS(tagKey, ...)"
        return "LEVEL = levelNum"

    def _should_replace_group_axis_group_by(
        self,
        skill_output: dict[str, Any],
        state: State,
        query_frame_delta: dict[str, Any],
    ) -> bool:
        return (
            skill_output.get("action_id") == "P3A3"
            and state.get_spo_value("P3", "group_paradigm") == "hybrid_grouping"
            and "group_by" in query_frame_delta
        )

    def _clear_group_by_before_group_axis_update(
        self,
        state: State,
        query_frame: QueryFrame,
        phase_id: str,
    ) -> None:
        self._clear_group_by(query_frame=query_frame, phase_id=phase_id)
        if query_frame is not state.query_frame:
            self._clear_group_by(query_frame=state.query_frame, phase_id=phase_id)

    def _clear_group_by(self, query_frame: QueryFrame, phase_id: str) -> None:
        query_frame.phase_frames.setdefault(phase_id, {})["group_by"] = []
        query_frame.group_by = []

    def _build_natural_language_semantics(
        self,
        skill_output: dict[str, Any],
        generation_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if skill_output.get("action_id") not in {"P2A3", "P2A4"} or not generation_result:
            return {}

        raw_output = generation_result.get("raw_output", {})
        temporal_semantic = self._extract_temporal_semantic(raw_output)
        if temporal_semantic is None:
            temporal_semantic = self._extract_temporal_semantic(
                generation_result.get("generated_artifact", {})
            )
        if temporal_semantic is None:
            return {}
        return {"temporal_semantic": temporal_semantic}

    def _extract_temporal_semantic(self, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return None

        for key in (
            "temporal_semantic",
            "temporal semantic",
            "temporalSemantic",
            "temporal_meaning",
            "natural_language_description",
        ):
            if payload.get(key):
                return payload[key]

        content = payload.get("content")
        if isinstance(content, dict):
            return self._extract_temporal_semantic(content)
        return None

    def _complete_tag_generation_result(
        self,
        generation_result: dict[str, Any],
        working_tag: dict[str, Any],
    ) -> dict[str, Any]:
        if not generation_result or not working_tag:
            return generation_result

        query_frame_delta = generation_result.get("query_frame_delta", {})
        selected_paths = self._as_list(query_frame_delta.get("tag"))
        if not selected_paths:
            return generation_result

        completed_tag = self._build_selected_working_tag(
            working_tag=working_tag,
            selected_paths=selected_paths,
        )
        generation_result = deepcopy(generation_result)
        generation_result["query_frame_delta"] = {"tag": completed_tag}
        generation_result["selected_tag_paths"] = selected_paths
        return generation_result

    def _build_selected_working_tag(
        self,
        working_tag: dict[str, Any],
        selected_paths: list[Any],
    ) -> dict[str, Any]:
        selected_path_set = {str(path) for path in selected_paths if path}
        selected_timeseries = [
            deepcopy(series)
            for series in working_tag.get("timeseries", [])
            if isinstance(series, dict) and str(series.get("path")) in selected_path_set
        ]
        if not selected_timeseries:
            return {}

        tag_keys = sorted(
            {
                tag_key
                for series in selected_timeseries
                for tag_key in (
                    series.get("tags", {}).keys()
                    if isinstance(series.get("tags"), dict)
                    else []
                )
            }
        )
        return {
            "schema_type": working_tag.get("schema_type", "iotdb_tree_tags"),
            "database": working_tag.get("database"),
            "timeseries_count": len(selected_timeseries),
            "tag_keys": tag_keys,
            "timeseries": selected_timeseries,
        }

    def _as_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _build_non_generation_working_schema(
        self,
        package: SkillPackage,
        skill_output: dict[str, Any],
        schema: DatabaseSchema,
    ) -> dict[str, Any]:
        """Build working-schema patches for non-generation actions."""

        if (
            package.phase_id == "P1"
            and skill_output.get("action_id") == "P1A2"
            and skill_output.get("selected_action") == "no_device_binding"
        ):
            return deepcopy(
                schema.to_agent_context(include_detail=False).get("meta_schema", {})
            )

        return {}

    def _build_non_generation_query_frame_delta(
        self,
        package: SkillPackage,
        skill_output: dict[str, Any],
        schema: DatabaseSchema,
    ) -> dict[str, Any]:
        """Build QueryFrame defaults for non-generating scope decisions."""

        if (
            package.phase_id == "P1"
            and skill_output.get("action_id") == "P1A2"
            and skill_output.get("selected_action") == "no_device_binding"
        ):
            meta_schema = schema.to_agent_context(include_detail=False).get(
                "meta_schema", {}
            )
            database = meta_schema.get("database") or schema.schema_root
            if database:
                return {"from_paths": [f"{str(database).rstrip('.')}.**"]}
        return {}

    def _build_output_delta(
        self,
        outputs_module: Any,
        action_id: str,
        selected_action: str,
    ) -> dict[str, Any]:
        if hasattr(outputs_module, "build_output_delta"):
            return outputs_module.build_output_delta(action_id, selected_action)
        return {
            "spo_delta": (
                outputs_module.build_spo_delta(action_id, selected_action)
                if hasattr(outputs_module, "build_spo_delta")
                else {}
            ),
            "flag_delta": (
                outputs_module.build_flag_delta(action_id, selected_action)
                if hasattr(outputs_module, "build_flag_delta")
                else {}
            ),
        }

    def _normalize_skill_output(self, skill_output: dict[str, Any]) -> dict[str, Any]:
        return {
            "skill_id": skill_output.get("skill_id"),
            "action_id": skill_output.get("action_id"),
            "decision_name": skill_output.get("decision_name"),
            "selected_action": skill_output.get("selected_action"),
            "structured_decision": skill_output.get("structured_decision", {}),
            "selected_expressions": skill_output.get("selected_expressions", []),
            "constructing_expressions": skill_output.get("constructing_expressions", []),
            "constructing_expression_update": skill_output.get(
                "constructing_expression_update"
            ),
            "prepared_constructing_expression": skill_output.get(
                "prepared_constructing_expression"
            ),
            "generated_expression": skill_output.get("generated_expression"),
            "selected_function_definition": skill_output.get(
                "selected_function_definition"
            ),
            "resolved_output_type": skill_output.get("resolved_output_type"),
            "source_component_skill": skill_output.get(
                "source_component_skill"
            ),
            "completed_constructing_expression": skill_output.get(
                "completed_constructing_expression"
            ),
            "input_constructing_expression": skill_output.get(
                "input_constructing_expression"
            ),
            "current_constructing_expression": skill_output.get(
                "current_constructing_expression"
            ),
            "current_constructing_expression_index": skill_output.get(
                "current_constructing_expression_index"
            ),
            "operator_store_delta": skill_output.get("operator_store_delta", {}),
            "action_meaning": skill_output.get("action_meaning", ""),
            "action_effect": skill_output.get("action_effect", {}),
            "spo_delta": skill_output.get("spo_delta", {}),
            "flag_delta": skill_output.get("flag_delta", {}),
            "query_frame_delta": skill_output.get("query_frame_delta", {}),
            "generation_result": skill_output.get("generation_result"),
            "requires_sql_generation": skill_output.get("requires_sql_generation", False),
            "generation_target": skill_output.get("generation_target"),
            "working_schema": skill_output.get("working_schema", {}),
            "trace": skill_output.get("trace", ""),
            "confidence": skill_output.get("confidence", 0.0),
            "reason": skill_output.get("reason", ""),
            "skill_input": skill_output.get("skill_input", {}),
            "working_tag": skill_output.get("working_tag", {}),
            "natural_language_semantics": skill_output.get(
                "natural_language_semantics", {}
            ),
            "semantic_decision_skipped": skill_output.get("semantic_decision_skipped", False),
            "decision_skill_invoked": skill_output.get(
                "decision_skill_invoked", False
            ),
            "memory": deepcopy(skill_output.get("memory", {})),
            "decision_learning": deepcopy(
                skill_output.get("decision_learning", {})
            ),
            "skill_routing": skill_output.get("skill_routing", {}),
            "skill_skipped": skill_output.get("skill_skipped", False),
            "retry_count": skill_output.get("retry_count", 0),
            "retry_errors": skill_output.get("retry_errors", []),
            "op_level_limit_reached": skill_output.get(
                "op_level_limit_reached", False
            ),
            "pipeline_error": skill_output.get("pipeline_error"),
            "function_source_compatibility": skill_output.get(
                "function_source_compatibility", {}
            ),
            "select_expression_candidates": skill_output.get(
                "select_expression_candidates", []
            ),
            "select_expression_source": skill_output.get(
                "select_expression_source"
            ),
            "selected_alignment": skill_output.get("selected_alignment"),
            "selected_writeback_mode": skill_output.get(
                "selected_writeback_mode"
            ),
            "writeback_cardinality_plan": skill_output.get(
                "writeback_cardinality_plan", {}
            ),
            "selected_ordering_mode": skill_output.get(
                "selected_ordering_mode"
            ),
            "result_alignment": skill_output.get("result_alignment"),
            "sortable_expressions": skill_output.get(
                "sortable_expressions", []
            ),
            "selected_pagination_policy": skill_output.get(
                "selected_pagination_policy"
            ),
            "result_shape": skill_output.get("result_shape", {}),
            "final_sql": skill_output.get("final_sql"),
            "natural_language_query": skill_output.get(
                "natural_language_query"
            ),
            "final_samples": skill_output.get("final_samples", []),
            "nl2sql_data": skill_output.get("nl2sql_data"),
            "sql_validation": skill_output.get("sql_validation"),
        }
