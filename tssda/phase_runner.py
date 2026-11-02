"""Phase-level package runner for TSSDA."""

from __future__ import annotations

from copy import deepcopy
from inspect import Parameter, signature
from typing import Any, Callable

from .package_loader import SkillPackage
from .schema import DatabaseSchema
from .skill_runner import SkillRunner
from .state import QueryFrame, State
from .memory.expressions import KINDS_BY_REGIME
from .skill_routing import next_skill_targets


P4_COMPONENT_SKILLS = frozenset(
    {
        "FunctionSourceSkill",
        "AggregateFunctionFamilySkill",
        "TimeSeriesFunctionFamilySkill",
        "OperatorFamilySkill",
        "ComponentInstantiationSkill",
    }
)


class PhaseRunner:
    """Run one loaded skill package according to package.yaml."""

    def __init__(
        self,
        skill_runner: SkillRunner,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.skill_runner = skill_runner
        self.progress_callback = progress_callback

    def run(
        self,
        package: SkillPackage,
        state: State,
        schema: DatabaseSchema,
        query_frame: QueryFrame,
        *,
        validate_dependencies: bool = True,
    ) -> dict[str, Any]:
        """Run a phase package and return PackageOutput."""

        if validate_dependencies:
            self._validate_dependencies(package, state)
        package_input = self.build_package_input(package, state, schema, query_frame)
        execution_policy = package.package_config.get("execution_policy", {})
        entry_skill = execution_policy.get("entry_skill")
        if not entry_skill:
            raise ValueError(f"{package.package_name} has no execution_policy.entry_skill")

        state.start_phase(package.phase_id)
        skill_queue = [entry_skill]
        executed_skills = set()
        skill_execution_counts: dict[str, int] = {}
        skill_outputs = []
        package_output = {
            "phase_id": package.phase_id,
            "phase_status": "running",
            "spo_delta": {},
            "flag_delta": {},
            "query_frame_delta": {},
            "working_schema": {},
            "natural_language_semantics": {},
            "errors": [],
            "skill_outputs": skill_outputs,
            "package_input": package_input,
        }

        while skill_queue:
            skill_id = skill_queue.pop(0)
            repeatable = self._is_repeatable_skill(package, skill_id)
            if skill_id in executed_skills and not repeatable:
                continue
            skill_execution_counts[skill_id] = (
                skill_execution_counts.get(skill_id, 0) + 1
            )
            current_iteration = skill_execution_counts[skill_id]

            skill_output = self.skill_runner.run(
                package=package,
                skill_id=skill_id,
                state=state,
                schema=schema,
                query_frame=query_frame,
            )
            if not repeatable:
                executed_skills.add(skill_id)
            skill_outputs.append(skill_output)
            self._merge_package_output(package_output, skill_output)
            self._apply_skill_output(state, package.phase_id, skill_output)

            next_skills = self._resolve_next_skills(
                execution_policy=execution_policy,
                skill_output=skill_output,
            )
            next_skills = next_skill_targets(
                package, skill_id, skill_output.get("selected_action"),
                state.to_skill_state(), declared=next_skills,
            )
            if package.phase_id == "P4" and skill_id == "OperatorRegimeSkill":
                trace = self._reuse_p4_expressions(package, state)
                if trace is not None:
                    package_output["expression_memory"] = trace
                    if trace["hit"]:
                        # Reuse seeds the pool; the normal action prior still
                        # chooses whether to extend it or stop exploration.
                        if trace["kind"] == "AggExpr" and state.get_spo_value("P4", "operator_regime") == "aggregate_query":
                            next_skills = ["AggregateExpressionSkill"]
                        self._emit(f"[TSSDA][P4] Expression memory reused {len(trace['expressions'])} {trace['kind']} expression(s).")
            skill_output["next_skills"] = list(next_skills[:1])
            continuation_skills = self._run_constructing_expression_pipeline(
                package=package,
                state=state,
                schema=schema,
                query_frame=query_frame,
                parent_output=skill_output,
                next_skills=next_skills,
                package_output=package_output,
                skill_outputs=skill_outputs,
                parent_iteration=current_iteration,
            )
            if continuation_skills is None:
                self._emit_p4_loop_stop_progress(
                    package=package,
                    skill_output=skill_output,
                    iteration=current_iteration,
                )
            skill_queue.extend(
                next_skills
                if continuation_skills is None
                else continuation_skills
            )

        self._apply_phase_query_frame_delta(package, state, package_output)
        self._apply_phase_spo_delta(package, state, package_output, skill_outputs)
        self._apply_phase_flag_delta(package, state, query_frame, package_output)
        if package.phase_id == "P1":
            # P4 must only see measurements inside the final P1 query scope.
            # State creation initially uses the sampled meta-schema because P1
            # has not selected a working schema yet; replace that provisional
            # pool as soon as P1 finishes.
            state.initialize_operator_store_from_working_schema()
        state.complete_phase(package.phase_id)
        package_output["phase_status"] = (
            "completed_with_errors"
            if package_output.get("errors")
            else "completed"
        )
        if package.phase_id == "P10" and state.sql_validation and not state.sql_validation.get("passed"):
            package_output["phase_status"] = "failed"
        return package_output

    def _reuse_p4_expressions(self, package: SkillPackage, state: State) -> dict[str, Any] | None:
        controller = getattr(self.skill_runner, "memory_controller", None)
        kind = KINDS_BY_REGIME.get(state.get_spo_value("P4", "operator_regime"))
        if controller is None or kind is None:
            return None
        # Use exactly the same request/package depth policy as fresh P4 skills.
        skill = package.skill_registry["SeriesExpressionSkill"]()
        state_view = state.to_skill_state()
        state_view["package_config"] = deepcopy(package.package_config)
        limit_key, default = ("series_expr", 8) if kind == "SeriesExpr" else ("agg_expr", 10)
        limit = skill._op_level_limit(state_view, limit_key, default)
        if state.get_spo_value("P4", "operator_regime") == "mixed_nested_query":
            # Total depth minus its aggregate wrapper upper-bounds the nested
            # series depth. Respect a tighter series cap conservatively.
            limit = min(limit, skill._op_level_limit(state_view, "series_expr", 8) + 1)
        return controller.expression_memory.populate(state, kind=kind, max_level=limit)

    @staticmethod
    def _validate_dependencies(package: SkillPackage, state: State) -> None:
        """Require every declared upstream phase field before phase execution."""

        dependencies = package.package_config.get("dependencies", {}) or {}
        missing: list[str] = []
        for phase_id in dependencies.get("required_phases", []) or []:
            if (
                phase_id not in state.phase_progress
                and phase_id not in state.spo
                and phase_id not in state.flags
            ):
                missing.append(f"phase:{phase_id}")

        for reference in dependencies.get("required_spo", []) or []:
            phase_id, key = PhaseRunner._split_dependency_reference(reference)
            if key not in state.spo.get(phase_id, {}):
                missing.append(f"spo:{reference}")

        for reference in dependencies.get("required_flags", []) or []:
            phase_id, key = PhaseRunner._split_dependency_reference(reference)
            if key not in state.flags.get(phase_id, {}):
                missing.append(f"flag:{reference}")

        if missing:
            raise RuntimeError(
                f"{package.phase_id} dependency validation failed: "
                f"{', '.join(missing)}"
            )

    @staticmethod
    def _split_dependency_reference(reference: Any) -> tuple[str, str]:
        value = str(reference)
        if "." not in value:
            raise ValueError(
                f"Dependency reference must use PHASE.field format: {value!r}"
            )
        return tuple(value.split(".", 1))

    def _run_constructing_expression_pipeline(
        self,
        package: SkillPackage,
        state: State,
        schema: DatabaseSchema,
        query_frame: QueryFrame,
        parent_output: dict[str, Any],
        next_skills: list[str],
        package_output: dict[str, Any],
        skill_outputs: list[dict[str, Any]],
        parent_iteration: int,
    ) -> list[str] | None:
        """Run required P4 component skills once for each pending expression."""

        if package.phase_id != "P4":
            return None

        if "constructing_expressions" not in parent_output:
            return None
        expressions = parent_output.get("constructing_expressions")
        if not isinstance(expressions, list):
            return None

        component_skills = [
            skill_id for skill_id in next_skills if skill_id in P4_COMPONENT_SKILLS
        ]
        if not component_skills:
            return None

        continuation_skills = [
            skill_id for skill_id in next_skills if skill_id not in P4_COMPONENT_SKILLS
        ]
        return_to_skill = parent_output.get("action_effect", {}).get(
            "return_to_skill"
        )
        if return_to_skill and return_to_skill not in continuation_skills:
            continuation_skills.append(return_to_skill)

        if not expressions:
            self._emit_p4_pipeline_progress(
                parent_output=parent_output,
                iteration=parent_iteration,
                generated_count=0,
                aborted=False,
            )
            return continuation_skills

        completed_expressions: list[dict[str, Any]] = []
        pipeline_aborted = False
        state.update_operator_store(
            {"clear_processed_constructing_expressions": True}
        )
        pipeline_completed = False
        try:
            for index, expression in enumerate(expressions[:2]):
                current_expression = deepcopy(expression)
                expression_completed = False
                state.update_operator_store(
                    {
                        "current_constructing_expression": current_expression,
                        "current_constructing_expression_index": index,
                    }
                )
                for component_skill_id in component_skills:
                    component_output = self.skill_runner.run(
                        package=package,
                        skill_id=component_skill_id,
                        state=state,
                        schema=schema,
                        query_frame=query_frame,
                    )
                    component_output["input_constructing_expression"] = deepcopy(
                        current_expression
                    )
                    component_output["current_constructing_expression_index"] = index
                    expression_update = component_output.get(
                        "constructing_expression_update"
                    )
                    if isinstance(expression_update, dict):
                        current_expression = deepcopy(expression_update)
                        state.update_operator_store(
                            {
                                "current_constructing_expression": (
                                    current_expression
                                ),
                                "current_constructing_expression_index": index,
                            }
                        )

                    if component_skill_id == "ComponentInstantiationSkill":
                        component_output[
                            "completed_constructing_expression"
                        ] = deepcopy(current_expression)
                        completed_expressions.append(deepcopy(current_expression))
                        state.update_operator_store(
                            {
                                "append_processed_constructing_expressions": [
                                    current_expression
                                ]
                            }
                        )
                        expression_completed = True

                    skill_outputs.append(component_output)
                    self._merge_package_output(package_output, component_output)
                    self._apply_skill_output(
                        state,
                        package.phase_id,
                        component_output,
                    )

                    if component_output.get("pipeline_error"):
                        pipeline_aborted = True
                        expression_completed = True
                        break

                if not expression_completed:
                    raise RuntimeError(
                        "P4 expression component pipeline did not complete "
                        f"ComponentInstantiationSkill for expression {index}"
                    )
                if pipeline_aborted:
                    break
            pipeline_completed = True
        finally:
            if pipeline_aborted:
                state.update_operator_store(
                    {
                        "clear_constructing_expressions": True,
                        "clear_processed_constructing_expressions": True,
                    }
                )
            elif pipeline_completed and completed_expressions:
                state.update_operator_store(
                    {"clear_constructing_expressions": True}
                )
                source_action_id = parent_output.get("action_id")
                if source_action_id == "P4A2":
                    state.update_operator_store(
                        {
                            "append_SeriesExpr": completed_expressions,
                        }
                    )
                elif source_action_id in {"P4A3", "P4A9"}:
                    state.update_operator_store(
                        {
                            "append_AggExpr": completed_expressions,
                        }
                    )
                else:
                    raise RuntimeError(
                        "P4 component pipeline cannot commit expressions for "
                        f"source action {source_action_id!r}"
                    )
                controller = getattr(self.skill_runner, "memory_controller", None)
                if controller is not None:
                    kind = "SeriesExpr" if source_action_id == "P4A2" else "AggExpr"
                    controller.expression_memory.stage(state, kind, completed_expressions)
                self._emit_p4_pipeline_progress(
                    parent_output=parent_output,
                    iteration=parent_iteration,
                    generated_count=len(completed_expressions),
                    aborted=False,
                )
            state.update_operator_store(
                {"clear_current_constructing_expression": True}
            )

        if pipeline_aborted:
            self._emit_p4_pipeline_progress(
                parent_output=parent_output,
                iteration=parent_iteration,
                generated_count=len(completed_expressions),
                aborted=True,
            )

        return [] if pipeline_aborted else continuation_skills

    def _emit_p4_loop_stop_progress(
        self,
        package: SkillPackage,
        skill_output: dict[str, Any],
        iteration: int,
    ) -> None:
        if package.phase_id != "P4":
            return
        action_id = skill_output.get("action_id")
        selected_action = skill_output.get("selected_action")
        if action_id == "P4A2" and selected_action == "stop_series_expr":
            self._emit(
                "[TSSDA][P4] P4A2 SeriesExpressionSkill iteration "
                f"{iteration}: stop_series_expr; generated 0 new SeriesExpr "
                "expression(s); series loop stopped."
            )
        elif action_id == "P4A3" and selected_action == "stop_agg_expr":
            self._emit(
                "[TSSDA][P4] P4A3 AggregateExpressionSkill iteration "
                f"{iteration}: stop_agg_expr; generated 0 new AggExpr "
                "expression(s); aggregate loop stopped."
            )

    def _emit_p4_pipeline_progress(
        self,
        parent_output: dict[str, Any],
        iteration: int,
        generated_count: int,
        aborted: bool,
    ) -> None:
        action_id = parent_output.get("action_id")
        selected_action = parent_output.get("selected_action")
        status = "aborted" if aborted else "completed"
        if action_id == "P4A2":
            self._emit(
                "[TSSDA][P4] P4A2 SeriesExpressionSkill iteration "
                f"{iteration}: {selected_action}; generated "
                f"{generated_count} new SeriesExpr expression(s); "
                f"component pipeline {status}."
            )
        elif action_id == "P4A3":
            self._emit(
                "[TSSDA][P4] P4A3 AggregateExpressionSkill iteration "
                f"{iteration}: {selected_action}; generated "
                f"{generated_count} new AggExpr expression(s); "
                f"component pipeline {status}."
            )
        elif action_id == "P4A9":
            self._emit(
                f"[TSSDA][P4] P4A9 AggregateAtomSkill: {selected_action}; "
                f"generated {generated_count} new AggExpr seed expression(s); "
                f"component pipeline {status}."
            )

    def _emit(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def _is_repeatable_skill(
        self,
        package: SkillPackage,
        skill_id: str,
    ) -> bool:
        """Return whether the package allows repeated execution of one skill."""

        skill_cls = package.skill_registry.get(skill_id)
        return bool(skill_cls and getattr(skill_cls, "repeatable", False))

    def build_package_input(
        self,
        package: SkillPackage,
        state: State,
        schema: DatabaseSchema,
        query_frame: QueryFrame,
    ) -> dict[str, Any]:
        """Build the unified PackageInput shape."""

        return {
            "phase_id": package.phase_id,
            "state": state.to_skill_state(),
            "schema": schema.to_agent_context(include_detail=False),
            "query_frame": query_frame.to_phase_dict(),
            "package_context": package.package_context,
            "package_config": deepcopy(package.package_config),
        }

    def _resolve_next_skills(
        self,
        execution_policy: dict[str, Any],
        skill_output: dict[str, Any],
    ) -> list[str]:
        action_effect = skill_output.get("action_effect", {})
        if "activate_next_skills" in action_effect:
            next_skills = action_effect.get("activate_next_skills") or []
            if isinstance(next_skills, str):
                return [next_skills]
            return list(next_skills)

        selected_action = skill_output.get("selected_action")
        conditional = execution_policy.get("conditional_skills", {})
        if isinstance(conditional, list):
            return list(conditional)
        configured = conditional.get(selected_action, [])
        if isinstance(configured, str):
            return [configured]
        return list(configured)

    def _merge_package_output(
        self,
        package_output: dict[str, Any],
        skill_output: dict[str, Any],
    ) -> None:
        for phase_id, values in skill_output.get("spo_delta", {}).items():
            package_output["spo_delta"].setdefault(phase_id, {}).update(deepcopy(values))

        flag_delta = skill_output.get("flag_delta", {})
        if flag_delta and all(not isinstance(value, dict) for value in flag_delta.values()):
            package_output["flag_delta"].setdefault(package_output["phase_id"], {}).update(
                deepcopy(flag_delta)
            )
        else:
            for phase_id, values in flag_delta.items():
                if isinstance(values, dict):
                    package_output["flag_delta"].setdefault(phase_id, {}).update(
                        deepcopy(values)
                    )

        query_frame_delta = skill_output.get("query_frame_delta", {})
        if query_frame_delta:
            package_output["query_frame_delta"].update(deepcopy(query_frame_delta))

        working_schema = skill_output.get("working_schema", {})
        if working_schema:
            package_output["working_schema"].update(deepcopy(working_schema))

        natural_language_semantics = skill_output.get("natural_language_semantics", {})
        if natural_language_semantics:
            package_output["natural_language_semantics"].update(
                deepcopy(natural_language_semantics)
            )

        pipeline_error = skill_output.get("pipeline_error")
        if pipeline_error:
            package_output.setdefault("errors", []).append(
                deepcopy(pipeline_error)
            )

    def _apply_skill_output(
        self,
        state: State,
        phase_id: str,
        skill_output: dict[str, Any],
    ) -> None:
        state.apply_skill_result(skill_output, default_phase_id=phase_id)

    def _apply_phase_query_frame_delta(
        self,
        package: SkillPackage,
        state: State,
        package_output: dict[str, Any],
    ) -> None:
        outputs_module = package.outputs_module
        if outputs_module is None or not hasattr(outputs_module, "build_phase_query_frame_delta"):
            return

        query_frame_delta = outputs_module.build_phase_query_frame_delta(
            state.get_phase_spo(package.phase_id)
        )
        if not query_frame_delta:
            return

        state.update_query_frame(package.phase_id, query_frame_delta)
        package_output["query_frame_delta"].update(deepcopy(query_frame_delta))

    def _apply_phase_spo_delta(
        self,
        package: SkillPackage,
        state: State,
        package_output: dict[str, Any],
        skill_outputs: list[dict[str, Any]],
    ) -> None:
        outputs_module = package.outputs_module
        if outputs_module is None or not hasattr(outputs_module, "build_phase_spo_delta"):
            return

        spo_delta = outputs_module.build_phase_spo_delta(
            state.get_phase_spo(package.phase_id),
            skill_outputs,
        )
        if not spo_delta:
            return

        for phase_id, values in spo_delta.items():
            if isinstance(values, dict):
                state.update_spo(phase_id, values)
                package_output["spo_delta"].setdefault(phase_id, {}).update(
                    deepcopy(values)
                )

    def _apply_phase_flag_delta(
        self,
        package: SkillPackage,
        state: State,
        query_frame: QueryFrame,
        package_output: dict[str, Any],
    ) -> None:
        outputs_module = package.outputs_module
        if outputs_module is None or not hasattr(outputs_module, "build_phase_flag_delta"):
            return

        flag_delta = self._call_phase_flag_builder(
            outputs_module.build_phase_flag_delta,
            state=state,
            query_frame=query_frame,
        )
        if not flag_delta:
            return

        for phase_id, values in flag_delta.items():
            if isinstance(values, dict):
                state.update_flags(phase_id, values)
                package_output["flag_delta"].setdefault(phase_id, {}).update(
                    deepcopy(values)
                )

    @staticmethod
    def _call_phase_flag_builder(
        builder: Callable[..., dict[str, Any]],
        *,
        state: State,
        query_frame: QueryFrame,
    ) -> dict[str, Any]:
        """Call phase flag builders with backward-compatible arguments."""

        working_schema = state.schema.get_active_working_schema()
        flat_query_frame = query_frame.to_flat_dict()
        builder_signature = signature(builder)
        parameters = builder_signature.parameters
        supports_varargs = any(
            parameter.kind == Parameter.VAR_POSITIONAL
            for parameter in parameters.values()
        )
        positional_parameters = [
            parameter
            for parameter in parameters.values()
            if parameter.kind
            in {
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        if "state" in parameters:
            return builder(working_schema, flat_query_frame, state=state)
        if supports_varargs or len(positional_parameters) >= 3:
            return builder(working_schema, flat_query_frame, state)
        return builder(working_schema, flat_query_frame)
