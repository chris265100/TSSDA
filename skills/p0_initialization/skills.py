"""Action-level skills for the P0 Initialization skill package.

P0 makes high-level semantic decisions for the current SQL generation trajectory.
Each action-level skill performs one local semantic decision from its masked
candidate space and returns a structured decision result.
"""

try:
    from . import tool
except ImportError:
    tool = None

try:
    from . import outputs
except ImportError:
    outputs = None


class BaseActionSkill:
    """Base interface for a local, constrained semantic decision.

    This class provides the external interface used by the runner.
    The runner only needs to call `run(...)` and consume the structured result.
    """

    skill_id: str = "BaseActionSkill"
    action_id: str = ""
    decision_name: str = ""

    candidate_descriptions: dict[str, dict] = {}

    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        """Execute one action-level semantic decision.

        The skill selects one action from the masked candidate space and
        immediately builds the corresponding SPO/Flag deltas according to
        outputs.py.
        """
        action_space = self._normalize_action_space(action_space)
        self._validate_action_space(action_space)

        selected_action = self.decide(
            state=state,
            context=context,
            action_space=action_space,
        )

        if selected_action not in action_space:
            raise ValueError(
                f"{self.skill_id} selected invalid action: {selected_action}. "
                f"Valid candidates: {action_space}"
            )

        spo_delta, flag_delta = self._build_output_delta(selected_action)
        candidate_info = self.candidate_descriptions.get(selected_action, {})

        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "selected_action": selected_action,
            "action_meaning": candidate_info.get("meaning", ""),
            "requires_sql_generation": candidate_info.get(
                "requires_sql_generation", False
            ),
            "action_effect": self.build_action_effect(selected_action, state),
            "spo_delta": spo_delta,
            "flag_delta": flag_delta,
            "trace": f"{self.skill_id} selected {selected_action}.",
            "confidence": 0.0,
        }

    def decide(self, state: dict, context: str, action_space: list[str]) -> str:
        """Select one candidate from the masked action space.

        The runner provides state, shared context, and masked action space.
        If an LLM policy is provided, the skill delegates the decision to it.
        Otherwise, it returns the first valid candidate for MVP execution.
        """
        if self.llm is not None and hasattr(self.llm, "select_action"):
            return self.llm.select_action(
                skill_id=self.skill_id,
                action_id=self.action_id,
                decision_name=self.decision_name,
                state=state,
                context=context,
                action_space=action_space,
                candidate_descriptions=self.candidate_descriptions,
            )

        return action_space[0]

    def build_action_effect(self, selected_action: str, state: dict) -> dict:
        """Return additional execution effect of the selected action.

        By default, an action has no additional effect.
        Subclasses can override this method to provide routing information.
        """
        return {}

    def _build_output_delta(self, selected_action: str) -> tuple[dict, dict]:
        """Build SPO and Flag deltas according to outputs.py."""
        spo_delta = {}
        flag_delta = {}

        if outputs is None:
            return spo_delta, flag_delta

        if hasattr(outputs, "has_spo_update") and outputs.has_spo_update(self.action_id):
            spo_delta = outputs.build_spo_delta(
                action_id=self.action_id,
                selected_action=selected_action,
            )

        if hasattr(outputs, "has_flag_update") and outputs.has_flag_update(self.action_id):
            flag_delta = outputs.build_flag_delta(
                action_id=self.action_id,
                selected_action=selected_action,
            )

        return spo_delta, flag_delta

    def _normalize_action_space(self, action_space: list[str]) -> list[str]:
        """Normalize candidate values before decision."""
        if tool is not None and hasattr(tool, "normalize_action_space"):
            return tool.normalize_action_space(action_space)

        return list(action_space)

    def _validate_action_space(self, action_space: list[str]) -> None:
        """Ensure the current skill receives a non-empty masked action space."""
        if not action_space:
            raise ValueError(f"{self.skill_id} received an empty action space")


class TaskFamilySkill(BaseActionSkill):
    """Action P0A1: decide the main task family."""

    skill_id = "TaskFamilySkill"
    action_id = "P0A1"
    decision_name = "task_family"

    candidate_descriptions = {
        "retrieval": {
            "meaning": (
                "Read raw or near-raw time-series records. "
                "The main focus is data access."
            ),
            "requires_sql_generation": False,
        },
        "computation": {
            "meaning": (
                "Apply expressions, functions, operators, or UDFs. "
                "The main focus is deriving new values."
            ),
            "requires_sql_generation": False,
        },
        "aggregation": {
            "meaning": (
                "Perform statistical aggregation, temporal segmentation, "
                "or cross-series grouping analysis."
            ),
            "requires_sql_generation": False,
        },
        "materialization": {
            "meaning": (
                "Write query results back into target time-series structures."
            ),
            "requires_sql_generation": False,
        },
    }

    def build_action_effect(self, selected_action: str, state: dict) -> dict:
        """Activate the subtype skill corresponding to the selected task family."""
        route_table = {
            "retrieval": "RetrievalSubtypeSkill",
            "computation": "ComputationSubtypeSkill",
            "aggregation": "AggregationSubtypeSkill",
            "materialization": "MaterializationSubtypeSkill",
        }

        if selected_action not in route_table:
            raise ValueError(f"Unsupported task family: {selected_action}")

        return {
            "activate_next_skills": [route_table[selected_action]]
        }


class RetrievalSubtypeSkill(BaseActionSkill):
    """Action P0A2-1: decide the retrieval-oriented task subtype."""

    skill_id = "RetrievalSubtypeSkill"
    action_id = "P0A2-1"
    decision_name = "retrieval_subtype"

    candidate_descriptions = {
        "raw_series": {
            "meaning": (
                "Read one or more raw time series without emphasizing complex "
                "filtering or computation."
            ),
            "requires_sql_generation": False,
        },
        "time_range": {
            "meaning": (
                "Read raw time-series records within a specified temporal range."
            ),
            "requires_sql_generation": False,
        },
        "condition_filter": {
            "meaning": (
                "Filter raw records by time, value, or status conditions."
            ),
            "requires_sql_generation": False,
        },
        "latest_point": {
            "meaning": (
                "Query the latest data point of one or more time series."
            ),
            "requires_sql_generation": False,
        },
        "multi_target": {
            "meaning": (
                "Read multiple measurements, devices, or time series together."
            ),
            "requires_sql_generation": False,
        },
        "wildcard": {
            "meaning": (
                "Use prefix or wildcard paths to read a batch of time series."
            ),
            "requires_sql_generation": False,
        },
    }


class ComputationSubtypeSkill(BaseActionSkill):
    """Action P0A2-2: decide the computation-oriented task subtype."""

    skill_id = "ComputationSubtypeSkill"
    action_id = "P0A2-2"
    decision_name = "computation_subtype"

    candidate_descriptions = {
        "basic_expression": {
            "meaning": (
                "Apply basic arithmetic, comparison, or logical expressions "
                "to time-series values."
            ),
            "requires_sql_generation": False,
        },
        "trend_change": {
            "meaning": (
                "Model temporal variation, difference, rate of change, "
                "or trend direction."
            ),
            "requires_sql_generation": False,
        },
        "interval_state": {
            "meaning": (
                "Identify state intervals, durations, occurrences, "
                "or continuous state segments."
            ),
            "requires_sql_generation": False,
        },
        "data_quality": {
            "meaning": (
                "Evaluate or process missing values, anomalies, completeness, "
                "or data repair."
            ),
            "requires_sql_generation": False,
        },
        "sampling_reconstruction": {
            "meaning": (
                "Perform sampling, downsampling, reconstruction, "
                "or compact representation of time-series data."
            ),
            "requires_sql_generation": False,
        },
        "nested_expression": {
            "meaning": (
                "Compose time series, constants, functions, and operators "
                "into nested expressions."
            ),
            "requires_sql_generation": False,
        },
    }


class AggregationSubtypeSkill(BaseActionSkill):
    """Action P0A2-3: decide the aggregation-oriented task subtype."""

    skill_id = "AggregationSubtypeSkill"
    action_id = "P0A2-3"
    decision_name = "aggregation_subtype"

    candidate_descriptions = {
        "summary_statistics": {
            "meaning": (
                "Compute global statistical summaries over selected "
                "time-series data."
            ),
            "requires_sql_generation": False,
        },
        "temporal_analysis": {
            "meaning": (
                "Analyze aggregated results along the temporal dimension."
            ),
            "requires_sql_generation": False,
        },
        "entity_comparison": {
            "meaning": (
                "Compare aggregated results across devices, series, regions, "
                "or other entities."
            ),
            "requires_sql_generation": False,
        },
        "conditional_aggregation": {
            "meaning": (
                "Aggregate under specific conditions or filter aggregation results."
            ),
            "requires_sql_generation": False,
        },
        "multi_dimensional_analysis": {
            "meaning": (
                "Combine temporal, entity, and conditional dimensions "
                "in aggregation."
            ),
            "requires_sql_generation": False,
        },
    }


class MaterializationSubtypeSkill(BaseActionSkill):
    """Action P0A2-4: decide the materialization-oriented task subtype."""

    skill_id = "MaterializationSubtypeSkill"
    action_id = "P0A2-4"
    decision_name = "materialization_subtype"

    candidate_descriptions = {
        "raw_backup": {
            "meaning": (
                "Copy or save raw query results into a target time-series structure."
            ),
            "requires_sql_generation": False,
        },
        "computed_result_save": {
            "meaning": (
                "Save computed expression, function, or UDF results as new series."
            ),
            "requires_sql_generation": False,
        },
        "aggregated_result_save": {
            "meaning": (
                "Persist aggregation or windowed analysis results into target series."
            ),
            "requires_sql_generation": False,
        },
        "reorganized_writeback": {
            "meaning": (
                "Rewrite results into a reorganized device or measurement structure."
            ),
            "requires_sql_generation": False,
        },
        "analysis_artifact_save": {
            "meaning": (
                "Save intermediate features or analysis artifacts for later reuse."
            ),
            "requires_sql_generation": False,
        },
    }


SKILL_REGISTRY = {
    "TaskFamilySkill": TaskFamilySkill,
    "RetrievalSubtypeSkill": RetrievalSubtypeSkill,
    "ComputationSubtypeSkill": ComputationSubtypeSkill,
    "AggregationSubtypeSkill": AggregationSubtypeSkill,
    "MaterializationSubtypeSkill": MaterializationSubtypeSkill,
}