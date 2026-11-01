"""Action-level skills for the P4 Operator Planning skill package.

P4 plans and incrementally instantiates SELECT-side SeriesExpr and AggExpr
expressions through reusable component pipelines.
"""

from __future__ import annotations

import random
import re
from copy import deepcopy
from typing import Any

try:
    from . import tool
except ImportError:
    tool = None

try:
    from . import outputs
except ImportError:
    outputs = None

try:
    from .function_tables import (
        BUILTIN_AGGREGATE,
        BUILTIN_TIMESERIES,
        UDF_AGGREGATE,
        UDF_TIMESERIES,
        get_function_definition,
        get_primary_allowed_input_types,
        list_operator_definitions,
        normalize_data_types,
    )
except ImportError:
    BUILTIN_AGGREGATE = "builtin_aggregate"
    BUILTIN_TIMESERIES = "builtin_timeseries"
    UDF_AGGREGATE = "udf_aggregate"
    UDF_TIMESERIES = "udf_timeseries"
    get_function_definition = None
    get_primary_allowed_input_types = None
    list_operator_definitions = None
    normalize_data_types = None


def _ordered_expression_candidates(
    candidates: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    return [
        (name, candidate)
        for name, candidate in candidates.items()
        if name and isinstance(candidate, dict)
    ]


def _replace_index_placeholders(
    expression: str,
    candidates: dict[str, dict[str, Any]],
    placeholder_prefix: str,
) -> str:
    ordered = _ordered_expression_candidates(candidates)
    if not ordered:
        return expression

    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(placeholder_prefix)}([1-9]\d*)"
        r"(?![A-Za-z0-9_])"
    )

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if 0 <= index < len(ordered):
            return ordered[index][0]
        return match.group(0)

    return pattern.sub(replace, expression)


def _find_expression_sources(
    expression: str,
    candidates: dict[str, dict[str, Any]],
    placeholder_prefix: str,
) -> list[dict[str, Any]]:
    if expression in candidates:
        return [candidates[expression]]

    ordered = _ordered_expression_candidates(candidates)
    placeholder_pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(placeholder_prefix)}([1-9]\d*)"
        r"(?![A-Za-z0-9_])"
    )
    placeholder_sources: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for match in placeholder_pattern.finditer(expression):
        index = int(match.group(1)) - 1
        if not 0 <= index < len(ordered):
            continue
        source = ordered[index][1]
        source_id = id(source)
        if source_id in seen_ids:
            continue
        seen_ids.add(source_id)
        placeholder_sources.append(source)
    if placeholder_sources:
        return placeholder_sources

    matches = []
    for name in sorted(candidates, key=len, reverse=True):
        if name in expression:
            matches.append(candidates[name])
    return matches


SERIES_EXPR_INSTRUCTION = """SeriesExpr represents a time-series expression. It always corresponds to a time-series output and is constructed from raw series paths, time-series generating functions, unary operators, and binary operators:

SeriesExpr :=
    SeriesPath
  | TSGF(SeriesExpr, attrs)
  | UnaryOp(SeriesExpr)
  | BinaryOp(SeriesExpr, SeriesExpr | Constant)"""


AGG_ATOM_INSTRUCTION = """AggAtom describes the activation of an aggregate function over a SeriesExpr. It converts a valid time-series expression into an initial aggregate expression.

AggAtom :=
    AGG(SeriesExpr)

Here, SeriesExpr can be either a raw time series or a time-series expression already transformed by time-series generating functions and operators. The aggregate function summarizes this SeriesExpr into an aggregated scalar value, such as AVG(s1), SUM(ABS(s1)), or COUNT(TSGF(s1, attrs))."""


AGG_EXPR_INSTRUCTION = """AggExpr describes the iterative construction of aggregate expressions after an AggAtom has been created. It further extends an existing aggregate expression using scalar-compatible time-series generating functions and operators.

AggExpr :=
    AggAtom
  | ScalarCompatibleTSGF(AggExpr, attrs)
  | UnaryOp(AggExpr)
  | BinaryOp(AggExpr, AggExpr | Constant)

Only scalar-compatible time-series generating functions can wrap an AggExpr after aggregation. Operators can further transform or combine aggregate expressions into more complex multi-level expressions."""

AGG_ATOM_OUTPUT_INSTRUCTION = """
This action has only one candidate action: make_agg_atom. No decision is required.

Select at most 2 suitable SeriesExpr candidates for aggregation. The selected
SeriesExpr should be valid inputs for aggregate functions, including raw series
or series expressions transformed by time-series generating functions and
operators.

For each selected SeriesExpr, assemble it into the abstract form:

AGG(SeriesExpr)

Do not instantiate the concrete aggregate function name, such as AVG, SUM, or
COUNT. Keep AGG as an abstract placeholder.

Return only the generated expressions. Do not include any extra text,
explanation, markdown, comments, or decision fields.

Select expressions according to coherent query semantics. Each generated
AggAtom must have a reasonable aggregation meaning. If no current SeriesExpr
is suitable for aggregation, return an empty JSON list instead of forcing an
invalid or meaningless expression. `s1`, `s2`, ... refer to the first, second,
and following entries in the provided SeriesExpr list.


Example output:
[
  "AGG(s1)",
  "AGG(ABS(s2))"
]
"""

SERIES_EXPR_OUTPUT_INSTRUCTION = """
Return only a structured JSON output with two fields: decision and selected_expressions.
Do not include any extra text, explanation, markdown, or comments outside the JSON.

1. decision:
   The selected action for this SeriesExpr construction step.

2. selected_expressions:
   Select at most 2 suitable SeriesExpr candidates and assemble them into the
   corresponding abstract expression forms.

Assembly rules:
- wrap_tsgf:
  TSGF(s1, attrs), TSGF(s2, attrs), ...

- wrap_unary_op:
  UnaryOp(s1), UnaryOp(s2), ...

- extend_binary_const:
  BinaryOp(s1, Constant), BinaryOp(s2, Constant), ...

- extend_binary_series:
  BinaryOp(s1, s2), BinaryOp(s3, s4), ...

- stop_series_expr:
  s1, s2, ...

Do not instantiate concrete function names, operator symbols, constants, or
attributes in this skill.

Make the decision and expression selection semantically coherent. Generated
expressions must have a reasonable query meaning. If the current SeriesExpr
candidates are not suitable for further nesting, select stop_series_expr.
`s1`, `s2`, ... refer to the first, second, and following entries in the
provided SeriesExpr list.


Example output:
{
  "decision": "wrap_tsgf",
  "selected_expressions": [
    "TSGF(s1, attrs)",
    "TSGF(s2, attrs)"
  ]
}
"""

AGG_EXPR_OUTPUT_INSTRUCTION = """
Return only a structured JSON output with two fields: decision and selected_expressions.
Do not include any extra text, explanation, markdown, or comments outside the JSON.

1. decision:
   The selected action for this AggExpr construction step.

2. selected_expressions:
   Select at most 2 suitable AggExpr candidates and assemble them into the
   corresponding abstract aggregate expression forms.

Assembly rules:
- wrap_scalar_tsgf_post:
  ScalarCompatibleTSGF(a1, attrs), ScalarCompatibleTSGF(a2, attrs), ...

- wrap_unary_agg:
  UnaryOp(a1), UnaryOp(a2), ...

- extend_binary_agg_const:
  BinaryOp(a1, Constant), BinaryOp(a2, Constant), ...

- extend_binary_agg_expr:
  BinaryOp(a1, a2), BinaryOp(a3, a4), ...

- stop_agg_expr:
  a1, a2, ...

Do not instantiate concrete function names, operator symbols, constants, or
attributes in this skill. Only select the action, choose the input AggExpr
candidates, and assemble the abstract expression forms.

Make the decision and expression selection semantically coherent. Generated
expressions must have a reasonable query meaning. If the current AggExpr
candidates are not suitable for further nesting, select stop_agg_expr. `a1`,
`a2`, ... refer to the first, second, and following entries in the provided
AggExpr list.


Example output:
{
  "decision": "wrap_scalar_tsgf_post",
  "selected_expressions": [
    "ScalarCompatibleTSGF(a1, attrs)",
    "ScalarCompatibleTSGF(a2, attrs)"
  ]
}
"""

AGGREGATE_FAMILY_OUTPUT_INSTRUCTION = """
Choose the aggregate function family that gives the current abstract aggregate
expression a coherent query meaning. Then replace the outer AGG placeholder
with the selected family name while preserving the complete inner expression.

Return only the rewritten expression as plain text. Do not return JSON,
markdown, comments, explanations, confidence, or reasoning.

Example:
- current expression: AGG(s1)
- selected family: basic_stat_agg
- output: basic_stat_agg(s1)
"""

TIMESERIES_FAMILY_OUTPUT_INSTRUCTION = """
Choose the time-series generating function family that gives the current
abstract expression a coherent query meaning. Then replace the outer TSGF or
ScalarCompatibleTSGF placeholder with the selected family name while preserving
the complete inner expression and attributes.

Return only the rewritten expression as plain text. Do not return JSON,
markdown, comments, explanations, confidence, or reasoning.

Examples:
- current expression: TSGF(s1, attrs)
- selected family: math_transform_ts
- output: math_transform_ts(s1, attrs)

- current expression: ScalarCompatibleTSGF(a1, attrs)
- selected family: math_transform_ts
- output: math_transform_ts(a1, attrs)
"""

OPERATOR_OUTPUT_INSTRUCTION = """
Choose one arithmetic operator from the candidate actions and replace the
abstract UnaryOp or BinaryOp placeholder with that operator's symbol.

Formatting rules:
- UnaryOp(expr) becomes symbol(expr).
- BinaryOp(left, right) becomes (left) symbol (right).

Return only the rewritten expression as plain text. Do not return JSON,
markdown, comments, explanations, confidence, reasoning, or the action name.

Examples:
- current expression: UnaryOp(s1)
- selected action: unary_minus
- output: -(s1)

- current expression: BinaryOp(s1, s2)
- selected action: multiply
- output: (s1) * (s2)
"""

FUNCTION_INSTANTIATION_INSTRUCTION = (
    "Complete the current partial expression as a concrete function expression. "
    "Organize the selected function, input expressions, and attributes according "
    "to the function definition, using a form such as func(Expr, attr) or "
    "func(Expr). If required constant parameters are missing, choose reasonable "
    "values and add them to construct a valid function call. Produce one "
    "semantically meaningful complete expression and return only that expression."
)

OPERATOR_INSTANTIATION_INSTRUCTION = (
    "Complete the current partial operator expression. The current operator uses "
    "parentheses around one or two operand expressions; remove parentheses where "
    "normal expression style and precedence make that safe. If the expression "
    "contains CONST, replace it with a reasonable constant value. Produce one "
    "semantically meaningful complete expression and return only that expression."
)

COMPONENT_DECISION_OUTPUT_INSTRUCTION = (
    "Return JSON only with one field: selected_action. The value must be one "
    "candidate from action_space. Do not return explanations or extra fields."
)

COMPONENT_GENERATION_OUTPUT_INSTRUCTION = (
    "Return JSON only in the exact form {\"expression\": \"...\"}. The JSON "
    "object must contain no other fields, and expression must be the single "
    "complete expression generated for this component."
)

P4_EXPRESSION_INSTRUCTIONS = {
    "SeriesExpr": SERIES_EXPR_INSTRUCTION,
    "AggAtom": AGG_ATOM_INSTRUCTION,
    "AggExpr": AGG_EXPR_INSTRUCTION,
}


BUILTIN_AGGREGATE_FAMILIES = {
    "basic_stat_agg": {
        "meaning": (
            "Basic statistical aggregation over a group of points, including "
            "summation, counting, and average value calculation."
        ),
        "functions": ["SUM", "COUNT", "AVG"],
    },
    "variance_std_agg": {
        "meaning": (
            "Dispersion aggregation that measures the fluctuation or spread of "
            "values, including standard deviation and variance."
        ),
        "functions": [
            "STDDEV",
            "STDDEV_POP",
            "STDDEV_SAMP",
            "VARIANCE",
            "VAR_POP",
            "VAR_SAMP",
        ],
    },
    "extreme_value_agg": {
        "meaning": (
            "Extreme-value aggregation that returns maximum, minimum, or the "
            "value with the largest absolute magnitude."
        ),
        "functions": ["MAX_VALUE", "MIN_VALUE", "EXTREME"],
    },
    "first_last_value_agg": {
        "meaning": (
            "Time-order value aggregation that returns the first or last value "
            "according to timestamp order."
        ),
        "functions": ["FIRST_VALUE", "LAST_VALUE"],
    },
    "time_position_agg": {
        "meaning": (
            "Timestamp-position aggregation that returns the earliest or latest "
            "timestamp among selected points."
        ),
        "functions": ["MIN_TIME", "MAX_TIME"],
    },
    "conditional_extreme_locator_agg": {
        "meaning": (
            "Conditional extreme-location aggregation that returns the value "
            "associated with the maximum or minimum value of another variable."
        ),
        "functions": ["MAX_BY", "MIN_BY"],
    },
}


UDF_AGGREGATE_FAMILIES = {
    "profile_integral_time_weighted_agg": {
        "meaning": (
            "Data profiling aggregation for integral and time-weighted average "
            "statistics over a time series."
        ),
        "functions": ["INTEGRAL", "INTEGRALAVG"],
    },
    "profile_robust_stat_agg": {
        "meaning": (
            "Robust statistical aggregation that reduces sensitivity to outliers, "
            "including median, median absolute deviation, and quantile statistics."
        ),
        "functions": ["MAD", "MEDIAN", "PERCENTILE", "QUANTILE"],
    },
    "profile_period_distribution_agg": {
        "meaning": (
            "Data profiling aggregation for period, distribution shape, skewness, "
            "and value spread."
        ),
        "functions": ["PERIOD", "SKEW", "SPREAD"],
    },
    "matching_similarity_correlation_agg": {
        "meaning": (
            "Whole-series matching aggregation that measures similarity, distance, "
            "covariance, or correlation between two time series."
        ),
        "functions": ["COV", "DTW", "PEARSON"],
    },
}


BUILTIN_TIMESERIES_FAMILIES = {
    "math_transform_ts": {
        "meaning": (
            "Point-wise mathematical transformation over numeric time-series "
            "values."
        ),
        "functions": [
            "SIN",
            "COS",
            "TAN",
            "ASIN",
            "ACOS",
            "ATAN",
            "SINH",
            "COSH",
            "TANH",
            "DEGREES",
            "RADIANS",
            "ABS",
            "SIGN",
            "CEIL",
            "FLOOR",
            "ROUND",
            "EXP",
            "LN",
            "LOG10",
            "SQRT",
        ],
    },
    "comparison_state_ts": {
        "meaning": (
            "State-conversion functions that transform numeric series into "
            "boolean or state-like series."
        ),
        "functions": ["ON_OFF", "IN_RANGE"],
    },
    "string_processing_ts": {
        "meaning": (
            "String-processing functions for text measurements, including "
            "matching, locating, concatenating, slicing, replacing, and case "
            "conversion."
        ),
        "functions": [
            "STRING_CONTAINS",
            "STRING_MATCHES",
            "LENGTH",
            "LOCATE",
            "STARTSWITH",
            "ENDSWITH",
            "CONCAT",
            "SUBSTRING",
            "REPLACE",
            "UPPER",
            "LOWER",
            "TRIM",
            "STRCMP",
        ],
    },
    "type_cast_ts": {
        "meaning": (
            "Type-conversion functions that cast measurement values into target "
            "output types."
        ),
        "functions": ["CAST"],
    },
    "constant_series_ts": {
        "meaning": (
            "Constant-series generation functions that produce fixed values or "
            "mathematical constants as time-series outputs."
        ),
        "functions": ["CONST", "PI", "E"],
    },
    "selection_top_bottom_ts": {
        "meaning": (
            "Selection functions that return top or bottom data points according "
            "to value order."
        ),
        "functions": ["TOP_K", "BOTTOM_K"],
    },
    "interval_query_ts": {
        "meaning": (
            "Interval query functions that identify zero or non-zero durations "
            "and counts in a time series."
        ),
        "functions": [
            "ZERO_DURATION",
            "NON_ZERO_DURATION",
            "ZERO_COUNT",
            "NON_ZERO_COUNT",
        ],
    },
    "trend_calculation_ts": {
        "meaning": (
            "Trend calculation functions for time difference, value difference, "
            "derivative, and non-negative change rate."
        ),
        "functions": [
            "TIME_DIFFERENCE",
            "DIFFERENCE",
            "NON_NEGATIVE_DIFFERENCE",
            "DERIVATIVE",
            "NON_NEGATIVE_DERIVATIVE",
            "DIFF",
        ],
    },
    "sampling_downsampling_ts": {
        "meaning": (
            "Sampling and downsampling functions that reduce or summarize dense "
            "time-series data."
        ),
        "functions": [
            "EQUAL_SIZE_BUCKET_RANDOM_SAMPLE",
            "EQUAL_SIZE_BUCKET_AGG_SAMPLE",
            "EQUAL_SIZE_BUCKET_M4_SAMPLE",
            "EQUAL_SIZE_BUCKET_OUTLIER_SAMPLE",
            "M4",
        ],
    },
    "change_point_ts": {
        "meaning": (
            "Time-series processing functions that detect change points or "
            "continuous identical-value patterns."
        ),
        "functions": ["CHANGE_POINTS"],
    },
}


UDF_TIMESERIES_FAMILIES = {
    "data_quality_ts": {
        "meaning": (
            "Data quality assessment functions for completeness, consistency, "
            "timeliness, and validity of time-series data."
        ),
        "functions": ["COMPLETENESS", "CONSISTENCY", "TIMELINESS", "VALIDITY"],
    },
    "profile_autocorr_stat_ts": {
        "meaning": (
            "Data profiling functions that generate autocorrelation, partial "
            "autocorrelation, histogram, distinct-value, or stationarity outputs."
        ),
        "functions": ["ACF", "PACF", "HISTOGRAM", "DISTINCT", "QLB"],
    },
    "profile_normalize_smooth_sample_segment_ts": {
        "meaning": (
            "Data profiling and preprocessing functions for normalization, moving "
            "average, resampling, sampling, segmentation, and spline interpolation."
        ),
        "functions": [
            "MINMAX",
            "ZSCORE",
            "MVAVG",
            "RESAMPLE",
            "SAMPLE",
            "SEGMENT",
            "SPLINE",
        ],
    },
    "anomaly_detection_ts": {
        "meaning": (
            "Anomaly detection functions for distribution outliers, K-sigma "
            "outliers, local outliers, missing anomalies, range anomalies, "
            "two-sided filtering, and distance-based outliers."
        ),
        "functions": [
            "IQR",
            "KSIGMA",
            "LOF",
            "MISSDETECT",
            "RANGE",
            "TWOSIDEDFILTER",
            "OUTLIER",
        ],
    },
    "frequency_domain_ts": {
        "meaning": (
            "Frequency-domain functions for convolution, deconvolution, wavelet "
            "transform, Fourier transform, filtering, and envelope analysis."
        ),
        "functions": [
            "CONV",
            "DECONV",
            "DWT",
            "IDWT",
            "FFT",
            "IFFT",
            "HIGHPASS",
            "LOWPASS",
            "ENVELOPE",
        ],
    },
    "matching_local_pattern_corr_ts": {
        "meaning": (
            "Local pattern and cross-correlation functions for symmetric pattern "
            "matching and cross-correlation sequence generation."
        ),
        "functions": ["PTNSYM", "XCORR"],
    },
    "data_repair_ts": {
        "meaning": (
            "Data repair functions for timestamp repair, missing-value filling, "
            "and value repair."
        ),
        "functions": ["TIMESTAMPREPAIR", "VALUEFILL", "VALUEREPAIR"],
    },
    "sequence_discovery_ts": {
        "meaning": (
            "Sequence discovery functions that identify consecutive sequences "
            "or consecutive windows in time-series data."
        ),
        "functions": ["CONSECUTIVESEQUENCES", "CONSECUTIVEWINDOWS"],
    },
    "machine_learning_ts": {
        "meaning": "Machine learning function for autoregressive modeling.",
        "functions": ["AR"],
    },
}


_ANY_FAMILY_INPUT_TYPE = "ANY_TYPE"
_ANY_FUNCTION_INPUT_TYPES = {"ANY_TYPE", "ALL_TYPES"}


def _build_family_input_type_indexes(
    families: dict[str, dict],
    category: str,
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, tuple[str, ...]]]]:
    """Precompute family-level and function-level input-type indexes."""

    if get_function_definition is None or get_primary_allowed_input_types is None:
        raise RuntimeError("Function type-index helpers are unavailable")

    family_input_types = {}
    family_function_input_types = {}
    for family_name, family in families.items():
        supported_types = []
        supports_any_type = False
        function_input_types = {}

        for function_name in family.get("functions", []):
            definition = get_function_definition(function_name, category)
            if definition is None:
                raise ValueError(
                    f"Missing function definition for {function_name} in "
                    f"family {family_name}"
                )
            allowed_types = tuple(get_primary_allowed_input_types(definition))
            if not allowed_types:
                raise ValueError(
                    f"Function {function_name} in family {family_name} has no "
                    "allowed input types"
                )
            normalized_allowed_types = (
                (_ANY_FAMILY_INPUT_TYPE,)
                if set(allowed_types) & _ANY_FUNCTION_INPUT_TYPES
                else allowed_types
            )
            function_input_types[function_name] = normalized_allowed_types

            if _ANY_FAMILY_INPUT_TYPE in normalized_allowed_types:
                supports_any_type = True
                continue
            for data_type in normalized_allowed_types:
                if data_type not in supported_types:
                    supported_types.append(data_type)

        family_input_types[family_name] = (
            (_ANY_FAMILY_INPUT_TYPE,)
            if supports_any_type
            else tuple(supported_types)
        )
        family_function_input_types[family_name] = function_input_types

    return family_input_types, family_function_input_types


(
    BUILTIN_AGGREGATE_FAMILY_INPUT_TYPES,
    _BUILTIN_AGGREGATE_FAMILY_FUNCTION_INPUT_TYPES,
) = _build_family_input_type_indexes(
    BUILTIN_AGGREGATE_FAMILIES,
    BUILTIN_AGGREGATE,
)
(
    UDF_AGGREGATE_FAMILY_INPUT_TYPES,
    _UDF_AGGREGATE_FAMILY_FUNCTION_INPUT_TYPES,
) = _build_family_input_type_indexes(
    UDF_AGGREGATE_FAMILIES,
    UDF_AGGREGATE,
)
(
    BUILTIN_TIMESERIES_FAMILY_INPUT_TYPES,
    _BUILTIN_TIMESERIES_FAMILY_FUNCTION_INPUT_TYPES,
) = _build_family_input_type_indexes(
    BUILTIN_TIMESERIES_FAMILIES,
    BUILTIN_TIMESERIES,
)
(
    UDF_TIMESERIES_FAMILY_INPUT_TYPES,
    _UDF_TIMESERIES_FAMILY_FUNCTION_INPUT_TYPES,
) = _build_family_input_type_indexes(
    UDF_TIMESERIES_FAMILIES,
    UDF_TIMESERIES,
)


def _collect_category_supported_input_types(
    family_input_types: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Combine all supported input types for one function category."""

    supported_types = []
    for allowed_types in family_input_types.values():
        if _ANY_FAMILY_INPUT_TYPE in allowed_types:
            return (_ANY_FAMILY_INPUT_TYPE,)
        for data_type in allowed_types:
            if data_type not in supported_types:
                supported_types.append(data_type)
    return tuple(supported_types)


BUILTIN_AGGREGATE_SUPPORTED_INPUT_TYPES = (
    _collect_category_supported_input_types(
        BUILTIN_AGGREGATE_FAMILY_INPUT_TYPES
    )
)
UDF_AGGREGATE_SUPPORTED_INPUT_TYPES = (
    _collect_category_supported_input_types(
        UDF_AGGREGATE_FAMILY_INPUT_TYPES
    )
)
BUILTIN_TIMESERIES_SUPPORTED_INPUT_TYPES = (
    _collect_category_supported_input_types(
        BUILTIN_TIMESERIES_FAMILY_INPUT_TYPES
    )
)
UDF_TIMESERIES_SUPPORTED_INPUT_TYPES = (
    _collect_category_supported_input_types(
        UDF_TIMESERIES_FAMILY_INPUT_TYPES
    )
)

FUNCTION_CATEGORY_SUPPORTED_INPUT_TYPES = {
    "builtin_aggregate": BUILTIN_AGGREGATE_SUPPORTED_INPUT_TYPES,
    "udf_aggregate": UDF_AGGREGATE_SUPPORTED_INPUT_TYPES,
    "builtin_timeseries": BUILTIN_TIMESERIES_SUPPORTED_INPUT_TYPES,
    "udf_timeseries": UDF_TIMESERIES_SUPPORTED_INPUT_TYPES,
}


def _build_operator_candidates() -> dict[str, dict[str, Any]]:
    """Build concrete arithmetic operator candidates for P4A7."""

    if list_operator_definitions is None:
        raise RuntimeError("IoTDB operator lookup helpers are unavailable")

    candidates = {}
    for category in ("arithmetic_operators",):
        for definition in list_operator_definitions(category):
            operator_name = definition["operator_name"]
            candidates[operator_name] = {
                "meaning": definition["meaning"],
                "symbol": definition["symbol"],
                "operator_arity": definition["operator_arity"],
                "operator_category": category,
                "requires_sql_generation": False,
            }
    return candidates


OPERATOR_CANDIDATES = _build_operator_candidates()


FUNCTION_FAMILY_COMPONENT_INDEX = {
    **{
        family_name: {
            "functions": tuple(family["functions"]),
            "function_input_types": (
                _BUILTIN_AGGREGATE_FAMILY_FUNCTION_INPUT_TYPES[family_name]
            ),
            "family_kind": "aggregate",
        }
        for family_name, family in BUILTIN_AGGREGATE_FAMILIES.items()
    },
    **{
        family_name: {
            "functions": tuple(family["functions"]),
            "function_input_types": (
                _UDF_AGGREGATE_FAMILY_FUNCTION_INPUT_TYPES[family_name]
            ),
            "family_kind": "aggregate",
        }
        for family_name, family in UDF_AGGREGATE_FAMILIES.items()
    },
    **{
        family_name: {
            "functions": tuple(family["functions"]),
            "function_input_types": (
                _BUILTIN_TIMESERIES_FAMILY_FUNCTION_INPUT_TYPES[family_name]
            ),
            "family_kind": "timeseries",
        }
        for family_name, family in BUILTIN_TIMESERIES_FAMILIES.items()
    },
    **{
        family_name: {
            "functions": tuple(family["functions"]),
            "function_input_types": (
                _UDF_TIMESERIES_FAMILY_FUNCTION_INPUT_TYPES[family_name]
            ),
            "family_kind": "timeseries",
        }
        for family_name, family in UDF_TIMESERIES_FAMILIES.items()
    },
}


def _build_component_instantiation_candidates() -> dict[str, dict[str, Any]]:
    """Build all concrete function/operator actions that A8 can expose."""

    candidates = {}
    function_categories = (
        BUILTIN_AGGREGATE,
        UDF_AGGREGATE,
        BUILTIN_TIMESERIES,
        UDF_TIMESERIES,
    )
    for family_name, family_info in FUNCTION_FAMILY_COMPONENT_INDEX.items():
        for function_name in family_info["functions"]:
            definition = None
            for category in function_categories:
                definition = get_function_definition(function_name, category)
                if definition:
                    break
            if definition is None:
                raise ValueError(
                    f"Missing component definition for function {function_name}"
                )
            candidates[function_name] = {
                "meaning": definition["description"],
                "function_family": family_name,
                "requires_sql_generation": True,
                "generation_target": {
                    "generation_instruction": FUNCTION_INSTANTIATION_INSTRUCTION,
                },
            }

    for operator_name, operator_info in OPERATOR_CANDIDATES.items():
        candidates[operator_name] = {
            "meaning": operator_info["meaning"],
            "symbol": operator_info["symbol"],
            "operator_arity": operator_info["operator_arity"],
            "requires_sql_generation": True,
            "generation_target": {
                "generation_instruction": OPERATOR_INSTANTIATION_INSTRUCTION,
            },
        }
    return candidates


def _get_component_function_definition(
    function_name: str,
) -> dict[str, Any] | None:
    """Return one concrete function definition from the four P4 tables."""

    for category in (
        BUILTIN_AGGREGATE,
        UDF_AGGREGATE,
        BUILTIN_TIMESERIES,
        UDF_TIMESERIES,
    ):
        definition = get_function_definition(function_name, category)
        if definition:
            return definition
    return None


COMPONENT_INSTANTIATION_CANDIDATES = (
    _build_component_instantiation_candidates()
)


SERIES_COMPONENT_ACTIONS = {
    "wrap_tsgf",
    "wrap_unary_op",
    "extend_binary_const",
    "extend_binary_series",
}


AGG_COMPONENT_ACTIONS = {
    "make_agg_atom",
    "wrap_scalar_tsgf_post",
    "wrap_unary_agg",
    "extend_binary_agg_const",
    "extend_binary_agg_expr",
}


def _generation_target(artifact_type: str, instruction: str) -> dict[str, Any]:
    return {
        "query_frame_key": "select",
        "artifact_type": artifact_type,
        "generation_instruction": instruction,
    }


def _family_candidates(*families: dict[str, dict]) -> dict[str, dict]:
    candidates = {}
    for family_group in families:
        for family_name, family_info in family_group.items():
            candidates[family_name] = {
                "meaning": family_info["meaning"],
                "functions": list(family_info["functions"]),
                "requires_sql_generation": False,
            }
    return candidates


def _timeseries_family_candidates() -> dict[str, dict]:
    candidates = _family_candidates(
        BUILTIN_TIMESERIES_FAMILIES,
        UDF_TIMESERIES_FAMILIES,
    )
    for family_name, candidate in candidates.items():
        scalar_functions = [
            function_name
            for function_name in candidate["functions"]
            if _is_scalar_compatible_timeseries_function(function_name)
        ]
        candidate["scalar_compatible_functions"] = scalar_functions
        candidate["has_scalar_compatible_function"] = bool(scalar_functions)
    return candidates


def _is_scalar_compatible_timeseries_function(function_name: str) -> bool:
    if get_function_definition is None:
        return False
    for category in (BUILTIN_TIMESERIES, UDF_TIMESERIES):
        definition = get_function_definition(function_name, category)
        if definition and definition.get("scalar_compatible"):
            return True
    return False


class BaseActionSkill:
    """Base interface for local P4 operator planning decisions."""

    skill_id: str = "BaseActionSkill"
    action_id: str = ""
    decision_name: str = ""
    candidate_descriptions: dict[str, dict] = {}
    allow_empty_action_space: bool = False
    repeatable: bool = False

    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        action_space = self._normalize_action_space(action_space)
        action_space = self.filter_action_space(
            state=state,
            context=context,
            action_space=action_space,
        )
        self._validate_action_space(action_space)

        skill_input = self.build_skill_input(
            state=state,
            context=context,
            action_space=action_space,
        )
        raw_decision = self.decide(
            state=state,
            context=context,
            action_space=action_space,
            skill_input=skill_input,
        )
        decision = self.parse_decision(raw_decision)
        selected_action = decision["selected_action"]

        if selected_action not in action_space:
            raise ValueError(
                f"{self.skill_id} selected invalid action: {selected_action}. "
                f"Valid candidates: {action_space}"
            )

        return self._build_skill_output(
            selected_action=selected_action,
            decision=decision,
            state=state,
        )

    def build_skill_input(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> dict:
        skill_input = {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "decision_name": self.decision_name,
            "state": state,
            "context": context,
            "schema_view": state.get("schema", {}),
            "query_frame": state.get("query_frame_flat", {}),
            "natural_language_semantics": state.get(
                "natural_language_semantics", {}
            ),
            "action_space": action_space,
            "candidate_descriptions": self.get_candidate_descriptions(
                state=state,
                action_space=action_space,
            ),
        }
        skill_input.update(self.build_additional_skill_input(state))
        return skill_input

    def get_candidate_descriptions(
        self,
        state: dict,
        action_space: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Return candidate metadata visible to the semantic decision agent."""

        return {
            action: deepcopy(self.candidate_descriptions[action])
            for action in action_space
            if action in self.candidate_descriptions
        }

    def build_additional_skill_input(self, state: dict) -> dict[str, Any]:
        """Return expression context only to the P4 skill that needs it."""

        operator_store = state.get("operator_store", {})
        if self.action_id == "P4A2":
            return {
                "SeriesExpr": deepcopy(
                    operator_store.get("SeriesExpr", [])
                ),
                "series_expr_instruction": SERIES_EXPR_INSTRUCTION,
                "output_instruction": SERIES_EXPR_OUTPUT_INSTRUCTION,
            }
        if self.action_id == "P4A9":
            return {
                "SeriesExpr": deepcopy(
                    operator_store.get("SeriesExpr", [])
                ),
                "agg_atom_instruction": AGG_ATOM_INSTRUCTION,
                "output_instruction": AGG_ATOM_OUTPUT_INSTRUCTION,
            }
        if self.action_id == "P4A3":
            return {
                "AggExpr": deepcopy(
                    operator_store.get("AggExpr", [])
                ),
                "agg_expr_instruction": AGG_EXPR_INSTRUCTION,
                "output_instruction": AGG_EXPR_OUTPUT_INSTRUCTION,
            }
        if self.action_id in {"P4A4", "P4A5", "P4A6", "P4A7", "P4A8"}:
            additional_input = {
                "constructing_expression": deepcopy(
                    operator_store.get("current_constructing_expression")
                ),
                "constructing_expression_index": operator_store.get(
                    "current_constructing_expression_index"
                ),
            }
            if self.action_id == "P4A5":
                additional_input["output_instruction"] = (
                    AGGREGATE_FAMILY_OUTPUT_INSTRUCTION
                )
            elif self.action_id == "P4A6":
                additional_input["output_instruction"] = (
                    TIMESERIES_FAMILY_OUTPUT_INSTRUCTION
                )
            elif self.action_id == "P4A7":
                additional_input["output_instruction"] = (
                    OPERATOR_OUTPUT_INSTRUCTION
                )
            elif self.action_id == "P4A8":
                additional_input["output_instruction"] = (
                    COMPONENT_DECISION_OUTPUT_INSTRUCTION
                )
            return additional_input
        return {}

    def decide(
        self,
        state: dict,
        context: str,
        action_space: list[str],
        skill_input: dict,
    ):
        if self.llm is not None and hasattr(self.llm, "select_action"):
            return self.llm.select_action(
                skill_id=self.skill_id,
                action_id=self.action_id,
                decision_name=self.decision_name,
                skill_input=skill_input,
            )
        return action_space[0]

    def parse_decision(self, raw_decision) -> dict:
        if isinstance(raw_decision, str):
            return {"selected_action": raw_decision}
        if isinstance(raw_decision, dict):
            if "selected_action" not in raw_decision:
                raise ValueError(
                    f"{self.skill_id} decision dict must contain selected_action"
                )
            return raw_decision
        raise TypeError(
            f"{self.skill_id} received unsupported decision type: "
            f"{type(raw_decision)}"
        )

    def build_action_effect(
        self,
        selected_action: str | None,
        decision: dict,
        state: dict,
    ) -> dict:
        candidate_info = self.candidate_descriptions.get(selected_action, {})
        next_skills = candidate_info.get("required_next_skills", [])
        effect = {"activate_next_skills": list(next_skills)}
        if candidate_info.get("return_to_skill"):
            effect["return_to_skill"] = candidate_info["return_to_skill"]
        if candidate_info.get("component_role"):
            effect["component_role"] = candidate_info["component_role"]
        return effect

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        return list(action_space)

    def _build_skill_output(
        self,
        selected_action: str | None,
        decision: dict,
        state: dict,
        semantic_decision_skipped: bool = False,
    ) -> dict:
        spo_delta, flag_delta = self._build_output_delta(
            selected_action=selected_action,
            decision=decision,
            state=state,
        )
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
            "generation_target": candidate_info.get("generation_target"),
            "required_next_skills": candidate_info.get("required_next_skills", []),
            "structured_decision": decision,
            "action_effect": self.build_action_effect(
                selected_action=selected_action,
                decision=decision,
                state=state,
            ),
            "spo_delta": spo_delta,
            "flag_delta": flag_delta,
            "query_frame_delta": {},
            "trace": f"{self.skill_id} selected {selected_action}.",
            "confidence": decision.get("confidence", 0.0),
            "reason": decision.get("reason", ""),
            "semantic_decision_skipped": semantic_decision_skipped,
        }

    def _build_output_delta(
        self,
        selected_action: str | None,
        decision: dict,
        state: dict,
    ) -> tuple[dict, dict]:
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
        if tool is not None and hasattr(tool, "normalize_action_space"):
            return tool.normalize_action_space(action_space)
        return list(action_space)

    def _validate_action_space(self, action_space: list[str]) -> None:
        if not action_space and not self.allow_empty_action_space:
            raise ValueError(f"{self.skill_id} received an empty action space")

    def _p4_operator_regime(self, state: dict) -> str | None:
        return self._p4_spo_value(state=state, key="operator_regime")

    def _p4_function_source(self, state: dict) -> str | None:
        operator_store = state.get("operator_store", {})
        current_expression = operator_store.get(
            "current_constructing_expression"
        )
        if isinstance(current_expression, dict):
            function_source = current_expression.get("function_source")
            if function_source in {
                "builtin_function",
                "udf_function",
                "mixed_function",
            }:
                return function_source
        return None

    def _p4_spo_value(self, state: dict, key: str) -> str | None:
        spo = state.get("spo", {}) if isinstance(state, dict) else {}
        phase_spo = spo.get("P4", {}) if isinstance(spo, dict) else {}
        if isinstance(phase_spo, dict):
            return phase_spo.get(key)
        return None

    @staticmethod
    def _op_level_limit(
        state: dict,
        limit_key: str,
        default: int,
    ) -> int:
        input_request = state.get("input_request", {})
        specific_input_keys = {
            "series_expr": "max_series_expression_level",
            "agg_expr": "max_aggregate_expression_level",
        }
        specific_limit = None
        global_limit = None
        if isinstance(input_request, dict):
            specific_limit = input_request.get(specific_input_keys.get(limit_key))
            global_limit = input_request.get("max_operator_level")
        package_config = state.get("package_config", {})
        execution_policy = (
            package_config.get("execution_policy", {})
            if isinstance(package_config, dict)
            else {}
        )
        limits = execution_policy.get("op_level_limits", {})
        configured_limit = (
            specific_limit
            if specific_limit is not None
            else (
                global_limit
                if global_limit is not None
                else (
                    limits.get(limit_key, default)
                    if isinstance(limits, dict)
                    else default
                )
            )
        )
        try:
            limit = int(configured_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid P4 op-level limit {limit_key}={configured_limit!r}"
            ) from exc
        if limit < 1:
            raise ValueError(
                f"P4 op-level limit {limit_key} must be positive"
            )
        return limit

    @staticmethod
    def _highest_level_expression(
        state: dict,
        store_key: str,
    ) -> dict[str, Any] | None:
        operator_store = state.get("operator_store", {})
        expressions = operator_store.get(store_key, [])
        if not isinstance(expressions, list):
            return None

        highest_expression = None
        highest_level = -1
        for expression in expressions:
            if not isinstance(expression, dict):
                continue
            try:
                op_level = int(expression.get("op_level", 1))
            except (TypeError, ValueError):
                op_level = 1
            if op_level > highest_level:
                highest_expression = expression
                highest_level = op_level
        return highest_expression

    @classmethod
    def _op_level_limit_reached(
        cls,
        state: dict,
        store_key: str,
        limit_key: str,
        default: int,
    ) -> bool:
        expression = cls._highest_level_expression(state, store_key)
        if expression is None:
            return False
        try:
            op_level = int(expression.get("op_level", 1))
        except (TypeError, ValueError):
            op_level = 1
        return op_level >= cls._op_level_limit(state, limit_key, default)

    def _latest_component_action(self, state: dict) -> str | None:
        operator_store = state.get("operator_store", {})
        current_expression = operator_store.get(
            "current_constructing_expression"
        )
        if isinstance(current_expression, dict):
            source_action = current_expression.get("source_expression_action")
            if source_action in SERIES_COMPONENT_ACTIONS | AGG_COMPONENT_ACTIONS:
                return source_action

        return None


class OperatorRegimeSkill(BaseActionSkill):
    """Action P4A1: decide the main operator regime."""

    skill_id = "OperatorRegimeSkill"
    action_id = "P4A1"
    decision_name = "operator_regime"

    candidate_descriptions = {
        "no_function": {
            "meaning": (
                "Do not use functions in the current operator phase. The query "
                "directly selects raw measurements, constants, or simple raw targets "
                "without function wrapping or nested operator-chain construction."
            ),
            "requires_sql_generation": False,
        },
        "timeseries_function_query": {
            "meaning": (
                "Generate an expression using time-series generating functions, which "
                "process input series and produce derived series for temporal "
                "transformation, trend analysis, sampling, repair, anomaly detection, "
                "and other time-series analysis tasks."
            ),
            "requires_sql_generation": False,
        },
        "aggregate_query": {
            "meaning": (
                "Generate an aggregate expression using aggregate functions only. "
                "The aggregate function summarizes raw measurements or simple "
                "series expressions into aggregate values."
            ),
            "requires_sql_generation": False,
        },
        "mixed_nested_query": {
            "meaning": (
                "Generate a complex nested operator chain that may combine "
                "time-series generating functions, aggregate functions, and "
                "operators into a multi-layer expression, such as "
                "ROUND(AVG(ABS(s1)))."
            ),
            "requires_sql_generation": False,
        },
        "latest_query": {
            "meaning": (
                "Use latest-point query semantics. This mode does not enter the "
                "nested function/operator generation loop."
            ),
            "requires_sql_generation": False,
        },
    }

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        """Remove regimes whose first expression would exceed the level cap."""

        candidates = list(action_space)
        series_limit = self._op_level_limit(state, "series_expr", 8)
        agg_limit = self._op_level_limit(state, "agg_expr", 10)

        if series_limit < 2:
            candidates = [
                action
                for action in candidates
                if action not in {
                    "timeseries_function_query",
                    "mixed_nested_query",
                }
            ]
        if agg_limit < 2:
            candidates = [
                action
                for action in candidates
                if action not in {"aggregate_query", "mixed_nested_query"}
            ]
        return candidates

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        action_space = self._normalize_action_space(action_space)
        action_space = self.filter_action_space(
            state=state,
            context=context,
            action_space=action_space,
        )
        self._validate_action_space(action_space)
        if action_space == ["latest_query"] and self._flag_is_true(
            state,
            phase_id="P2",
            flag_name="latest_time",
        ):
            selected_action = "latest_query"
            decision = {
                "selected_action": selected_action,
                "confidence": 1.0,
                "reason": (
                    "Rule-selected latest_query because P2 enabled latest-point "
                    "temporal semantics."
                ),
            }
            output = self._build_skill_output(
                selected_action=selected_action,
                decision=decision,
                state=state,
                semantic_decision_skipped=True,
            )
            output["trace"] = f"{self.skill_id} rule-selected {selected_action}."
            return output
        if action_space == ["no_function"]:
            selected_action = "no_function"
            level_limit = self._op_level_limit(state, "series_expr", 8)
            decision = {
                "selected_action": selected_action,
                "confidence": 1.0,
                "reason": (
                    "Rule-selected no_function because the operator-expression "
                    f"level cap {level_limit} permits no extension beyond raw "
                    "level-1 SeriesExpr inputs."
                ),
            }
            output = self._build_skill_output(
                selected_action=selected_action,
                decision=decision,
                state=state,
                semantic_decision_skipped=True,
            )
            output["trace"] = f"{self.skill_id} rule-selected {selected_action}."
            return output
        return super().run(state=state, context=context, action_space=action_space)

    @staticmethod
    def _flag_is_true(
        state: dict,
        phase_id: str,
        flag_name: str,
    ) -> bool:
        flags = state.get("flags", {}) if isinstance(state, dict) else {}
        if not isinstance(flags, dict):
            return False
        phase_flags = flags.get(phase_id, {})
        return (
            isinstance(phase_flags, dict)
            and phase_flags.get(flag_name) is True
        )

    def build_action_effect(
        self,
        selected_action: str | None,
        decision: dict,
        state: dict,
    ) -> dict:
        route_table = {
            "timeseries_function_query": ["SeriesExpressionSkill"],
            "aggregate_query": ["AggregateAtomSkill"],
            "mixed_nested_query": ["SeriesExpressionSkill"],
        }
        effect = {
            "activate_next_skills": route_table.get(selected_action, []),
            "operator_route": selected_action,
            "operator_regime": selected_action,
        }
        if selected_action in {"no_function", "latest_query"}:
            effect["operator_unit_kind"] = "raw_candidate_unit"
            effect["p4_finished"] = True
        if selected_action == "timeseries_function_query":
            effect["p4_active_loop"] = "series_expr_loop"
            effect["requires_timeseries_function"] = True
        if selected_action == "aggregate_query":
            effect["p4_active_loop"] = "agg_expr_loop"
            effect["requires_aggregate_function"] = True
            effect["allow_timeseries_function"] = False
        if selected_action == "mixed_nested_query":
            effect["p4_active_loop"] = "series_expr_loop"
            effect["requires_aggregate_function"] = True
            effect["requires_timeseries_function"] = True
            effect["allow_timeseries_function"] = True
        return effect


class SeriesExpressionSkill(BaseActionSkill):
    """Action P4A2: iteratively construct a SeriesExpr."""

    skill_id = "SeriesExpressionSkill"
    action_id = "P4A2"
    decision_name = "series_expression_action"
    repeatable = True

    candidate_descriptions = {
        "wrap_tsgf": {
            "meaning": (
                "Generate TSGF(SeriesExpr, attrs) by wrapping the current "
                "SeriesExpr with a time-series generating function."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "FunctionSourceSkill",
                "TimeSeriesFunctionFamilySkill",
                "ComponentInstantiationSkill",
            ],
            "component_role": "timeseries_function",
            "return_to_skill": "SeriesExpressionSkill",
        },

        "wrap_unary_op": {
            "meaning": (
                "Generate UnaryOp(SeriesExpr) by applying a unary operator "
                "to the current SeriesExpr."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "OperatorFamilySkill",
                "ComponentInstantiationSkill",
            ],
            "component_role": "unary_operator",
            "return_to_skill": "SeriesExpressionSkill",
        },

        "extend_binary_const": {
            "meaning": (
                "Generate BinaryOp(SeriesExpr, Constant) by combining the "
                "current SeriesExpr with a constant using a binary operator."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "OperatorFamilySkill",
                "ComponentInstantiationSkill",
            ],
            "component_role": "binary_operator_const",
            "return_to_skill": "SeriesExpressionSkill",
        },

        "extend_binary_series": {
            "meaning": (
                "Generate BinaryOp(SeriesExpr, SeriesExpr) by combining the "
                "current SeriesExpr with another SeriesExpr using a binary operator."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "OperatorFamilySkill",
                "ComponentInstantiationSkill",
            ],
            "component_role": "binary_operator_series",
            "return_to_skill": "SeriesExpressionSkill",
        },

        "stop_series_expr": {
            "meaning": (
                "Stop constructing the current SeriesExpr and commit it as the "
                "generated time-series expression."
            ),
            "requires_sql_generation": False,
        },
    }

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        if self._op_level_limit_reached(
            state=state,
            store_key="SeriesExpr",
            limit_key="series_expr",
            default=8,
        ):
            return (
                ["stop_series_expr"]
                if "stop_series_expr" in action_space
                else []
            )
        return list(action_space)

    def decide(
        self,
        state: dict,
        context: str,
        action_space: list[str],
        skill_input: dict,
    ):
        if action_space == ["stop_series_expr"]:
            expression = self._highest_level_expression(state, "SeriesExpr")
            if expression is None:
                raise ValueError(
                    "SeriesExpressionSkill cannot stop without a SeriesExpr"
                )
            limit = self._op_level_limit(state, "series_expr", 8)
            op_level_limit_reached = self._op_level_limit_reached(
                state=state,
                store_key="SeriesExpr",
                limit_key="series_expr",
                default=8,
            )
            return {
                "decision": "stop_series_expr",
                "selected_action": "stop_series_expr",
                "selected_expressions": [expression["name"]],
                "confidence": 1.0,
                "reason": f"Stopped at the SeriesExpr depth limit {limit}.",
                "op_level_limit_reached": op_level_limit_reached,
            }
        if self.llm is not None and hasattr(self.llm, "decide_structured"):
            return self.llm.decide_structured(
                skill_input=skill_input,
                decision_field="decision",
            )
        return super().decide(
            state=state,
            context=context,
            action_space=action_space,
            skill_input=skill_input,
        )

    def parse_decision(self, raw_decision) -> dict:
        if not isinstance(raw_decision, dict) or "decision" not in raw_decision:
            decision = super().parse_decision(raw_decision)
            decision.setdefault("selected_expressions", [])
            return decision

        selected_action = raw_decision.get("decision")
        if isinstance(selected_action, dict):
            selected_action = selected_action.get("selected_action")
        selected_expressions = raw_decision.get("selected_expressions", [])
        if not isinstance(selected_expressions, list):
            raise TypeError("selected_expressions must be a list")

        decision = dict(raw_decision)
        decision["decision"] = selected_action
        decision["selected_action"] = selected_action
        decision["selected_expressions"] = selected_expressions[:2]
        return decision

    def _build_skill_output(
        self,
        selected_action: str | None,
        decision: dict,
        state: dict,
        semantic_decision_skipped: bool = False,
    ) -> dict:
        output = super()._build_skill_output(
            selected_action=selected_action,
            decision=decision,
            state=state,
            semantic_decision_skipped=semantic_decision_skipped,
        )
        selected_expressions = self._build_constructing_expression_descriptors(
            selected_action=selected_action,
            selected_expressions=decision.get("selected_expressions", []),
            series_expr=state.get("operator_store", {}).get("SeriesExpr", []),
        )
        constructing_expressions = (
            []
            if selected_action == "stop_series_expr"
            else selected_expressions
        )
        output["selected_expressions"] = selected_expressions
        output["constructing_expressions"] = constructing_expressions
        output["operator_store_delta"] = {
            "constructing_expressions": constructing_expressions,
        }
        if decision.get("op_level_limit_reached"):
            output["semantic_decision_skipped"] = True
            output["op_level_limit_reached"] = bool(
                decision.get("op_level_limit_reached")
            )
        return output

    def _build_constructing_expression_descriptors(
        self,
        selected_action: str | None,
        selected_expressions: list[Any],
        series_expr: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates = {
            str(item.get("name")): item
            for item in series_expr
            if isinstance(item, dict) and item.get("name")
        }
        descriptors = []

        for selected in selected_expressions[:2]:
            expression = self._selected_expression_text(selected)
            if not expression:
                continue
            expression = _replace_index_placeholders(
                expression=expression,
                candidates=candidates,
                placeholder_prefix="s",
            )
            sources = self._find_source_expressions(expression, candidates)
            if not sources:
                raise ValueError(
                    f"{self.skill_id} could not resolve a SeriesExpr source for "
                    f"{expression!r}"
                )

            data_types = []
            for source in sources:
                data_type = source.get("data_type")
                if data_type not in data_types:
                    data_types.append(data_type)
            resolved_data_type: Any = data_types[0] if len(data_types) == 1 else data_types

            source_levels = []
            for source in sources:
                try:
                    source_levels.append(int(source.get("op_level", 1)))
                except (TypeError, ValueError):
                    source_levels.append(1)
            op_level = max(source_levels or [1])
            if selected_action != "stop_series_expr":
                op_level += 1

            descriptors.append(
                {
                    "name": expression,
                    "data_type": resolved_data_type,
                    "op_level": op_level,
                    "source_expression_action": selected_action,
                }
            )
        return descriptors

    @staticmethod
    def _selected_expression_text(selected: Any) -> str:
        if isinstance(selected, str):
            return selected.strip()
        if isinstance(selected, dict):
            value = selected.get("name") or selected.get("expression")
            return str(value).strip() if value is not None else ""
        return ""

    @staticmethod
    def _find_source_expressions(
        expression: str,
        candidates: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return _find_expression_sources(
            expression=expression,
            candidates=candidates,
            placeholder_prefix="s",
        )
    def build_action_effect(
        self,
        selected_action: str | None,
        decision: dict,
        state: dict,
    ) -> dict:
        if selected_action == "stop_series_expr":
            next_skills = []
            if self._p4_operator_regime(state) == "mixed_nested_query":
                next_skills = ["AggregateAtomSkill"]
            return {
                "activate_next_skills": next_skills,
                "series_expr_committed": True,
                "p4_finished": not next_skills,
            }

        effect = super().build_action_effect(selected_action, decision, state)
        effect["p4_active_loop"] = "series_expr_loop"
        effect["series_expression_action"] = selected_action
        return effect


class AggregateAtomSkill(BaseActionSkill):
    """Action P4A9: construct the initial AggAtom from a SeriesExpr."""

    skill_id = "AggregateAtomSkill"
    action_id = "P4A9"
    decision_name = "aggregate_atom_action"

    candidate_descriptions = {
        "make_agg_atom": {
            "meaning": (
                "Generate AGG(SeriesExpr) by wrapping a selected SeriesExpr "
                "with an aggregate function. This creates the initial AggAtom "
                "before subsequent AggExpr construction begins."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "FunctionSourceSkill",
                "AggregateFunctionFamilySkill",
                "ComponentInstantiationSkill",
                "AggregateExpressionSkill",
            ],
            "component_role": "aggregate_function",
            "return_to_skill": "AggregateExpressionSkill",
        },
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        action_space = self._normalize_action_space(action_space)
        action_space = self.filter_action_space(
            state=state,
            context=context,
            action_space=action_space,
        )
        self._validate_action_space(action_space)
        selected_action = "make_agg_atom"
        if selected_action not in action_space:
            raise ValueError(
                f"{self.skill_id} requires make_agg_atom in action_space"
            )

        skill_input = self.build_skill_input(
            state=state,
            context=context,
            action_space=action_space,
        )
        agg_limit = self._op_level_limit(state, "agg_expr", 10)
        eligible_series_expr = []
        for expression in skill_input.get("SeriesExpr", []):
            if not isinstance(expression, dict):
                continue
            try:
                source_level = int(expression.get("op_level", 1))
            except (TypeError, ValueError):
                source_level = 1
            if source_level + 1 <= agg_limit:
                eligible_series_expr.append(expression)
        if not eligible_series_expr:
            raise RuntimeError(
                "AggregateAtomSkill has no SeriesExpr that can be wrapped "
                f"without exceeding the AggExpr op-level limit {agg_limit}"
            )
        skill_input["SeriesExpr"] = eligible_series_expr
        if self.llm is not None and hasattr(self.llm, "generate_expression_list"):
            raw_expressions = self.llm.generate_expression_list(skill_input)
        else:
            raw_expressions = [
                f"AGG({item['name']})"
                for item in skill_input.get("SeriesExpr", [])[:2]
                if isinstance(item, dict) and item.get("name")
            ]

        constructing_expressions = self._build_agg_atom_descriptors(
            selected_expressions=raw_expressions,
            series_expr=skill_input.get("SeriesExpr", []),
        )
        output = self._build_skill_output(
            selected_action=selected_action,
            decision={"selected_action": selected_action},
            state=state,
            semantic_decision_skipped=True,
        )
        output["constructing_expressions"] = constructing_expressions
        output["operator_store_delta"] = {
            "constructing_expressions": constructing_expressions,
        }
        output["trace"] = (
            f"{self.skill_id} selected {len(constructing_expressions)} SeriesExpr "
            "candidates for aggregate activation."
        )
        return output

    def _build_agg_atom_descriptors(
        self,
        selected_expressions: list[Any],
        series_expr: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates = {
            str(item.get("name")): item
            for item in series_expr
            if isinstance(item, dict) and item.get("name")
        }
        descriptors = []

        for selected in selected_expressions[:2]:
            expression = self._selected_expression_text(selected)
            if not expression:
                continue
            if not expression.startswith("AGG(") or not expression.endswith(")"):
                raise ValueError(
                    f"{self.skill_id} expected AGG(SeriesExpr), got {expression!r}"
                )

            inner_expression = expression[4:-1].strip()
            inner_expression = _replace_index_placeholders(
                expression=inner_expression,
                candidates=candidates,
                placeholder_prefix="s",
            )
            expression = f"AGG({inner_expression})"
            sources = self._find_series_sources(inner_expression, candidates)
            if not sources:
                raise ValueError(
                    f"{self.skill_id} could not resolve SeriesExpr source for "
                    f"{expression!r}"
                )

            data_types = []
            source_levels = []
            for source in sources:
                data_type = source.get("data_type")
                if data_type not in data_types:
                    data_types.append(data_type)
                try:
                    source_levels.append(int(source.get("op_level", 1)))
                except (TypeError, ValueError):
                    source_levels.append(1)

            descriptors.append(
                {
                    "name": expression,
                    "data_type": (
                        data_types[0] if len(data_types) == 1 else data_types
                    ),
                    "op_level": max(source_levels or [1]) + 1,
                    "source_expression_action": "make_agg_atom",
                }
            )
        return descriptors

    @staticmethod
    def _selected_expression_text(selected: Any) -> str:
        if isinstance(selected, str):
            return selected.strip()
        if isinstance(selected, dict):
            value = selected.get("name") or selected.get("expression")
            return str(value).strip() if value is not None else ""
        return ""

    @staticmethod
    def _find_series_sources(
        inner_expression: str,
        candidates: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return _find_expression_sources(
            expression=inner_expression,
            candidates=candidates,
            placeholder_prefix="s",
        )

class AggregateExpressionSkill(BaseActionSkill):
    """Action P4A3: iteratively construct an AggExpr."""

    skill_id = "AggregateExpressionSkill"
    action_id = "P4A3"
    decision_name = "aggregate_expression_action"
    repeatable = True

    candidate_descriptions = {
        "wrap_scalar_tsgf_post": {
            "meaning": (
                "Generate ScalarCompatibleTSGF(AggExpr, attrs) by wrapping "
                "the current AggExpr with a scalar-compatible time-series "
                "generating function."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "FunctionSourceSkill",
                "TimeSeriesFunctionFamilySkill",
                "ComponentInstantiationSkill",
            ],
            "component_role": "scalar_timeseries_function",
            "return_to_skill": "AggregateExpressionSkill",
        },

        "wrap_unary_agg": {
            "meaning": (
                "Generate UnaryOp(AggExpr) by applying a unary operator "
                "to the current AggExpr."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "OperatorFamilySkill",
                "ComponentInstantiationSkill",
            ],
            "component_role": "unary_operator",
            "return_to_skill": "AggregateExpressionSkill",
        },

        "extend_binary_agg_const": {
            "meaning": (
                "Generate BinaryOp(AggExpr, Constant) by combining the "
                "current AggExpr with a constant using a binary operator."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "OperatorFamilySkill",
                "ComponentInstantiationSkill",
            ],
            "component_role": "binary_operator_const",
            "return_to_skill": "AggregateExpressionSkill",
        },

        "extend_binary_agg_expr": {
            "meaning": (
                "Generate BinaryOp(AggExpr, AggExpr) by combining the "
                "current AggExpr with another valid AggExpr using a binary "
                "operator."
            ),
            "requires_sql_generation": False,
            "required_next_skills": [
                "OperatorFamilySkill",
                "ComponentInstantiationSkill",
            ],
            "component_role": "binary_operator_agg",
            "return_to_skill": "AggregateExpressionSkill",
        },

        "stop_agg_expr": {
            "meaning": (
                "Stop constructing the current AggExpr and commit it as the "
                "generated aggregate expression."
            ),
            "requires_sql_generation": False,
        },
    }

    def decide(
        self,
        state: dict,
        context: str,
        action_space: list[str],
        skill_input: dict,
    ):
        if action_space == ["stop_agg_expr"]:
            expression = self._highest_level_expression(state, "AggExpr")
            if expression is None:
                raise ValueError(
                    "AggregateExpressionSkill cannot stop without an AggExpr"
                )
            limit = self._op_level_limit(state, "agg_expr", 10)
            op_level_limit_reached = self._op_level_limit_reached(
                state=state,
                store_key="AggExpr",
                limit_key="agg_expr",
                default=10,
            )
            return {
                "decision": "stop_agg_expr",
                "selected_action": "stop_agg_expr",
                "selected_expressions": [expression["name"]],
                "confidence": 1.0,
                "reason": f"Stopped at the AggExpr depth limit {limit}.",
                "op_level_limit_reached": op_level_limit_reached,
            }
        if self.llm is not None and hasattr(self.llm, "decide_structured"):
            return self.llm.decide_structured(
                skill_input=skill_input,
                decision_field="decision",
            )
        return super().decide(
            state=state,
            context=context,
            action_space=action_space,
            skill_input=skill_input,
        )

    def parse_decision(self, raw_decision) -> dict:
        if not isinstance(raw_decision, dict) or "decision" not in raw_decision:
            decision = super().parse_decision(raw_decision)
            decision.setdefault("selected_expressions", [])
            return decision

        selected_action = raw_decision.get("decision")
        if isinstance(selected_action, dict):
            selected_action = selected_action.get("selected_action")
        selected_expressions = raw_decision.get("selected_expressions", [])
        if not isinstance(selected_expressions, list):
            raise TypeError("selected_expressions must be a list")

        decision = dict(raw_decision)
        decision["decision"] = selected_action
        decision["selected_action"] = selected_action
        decision["selected_expressions"] = selected_expressions[:2]
        return decision

    def _build_skill_output(
        self,
        selected_action: str | None,
        decision: dict,
        state: dict,
        semantic_decision_skipped: bool = False,
    ) -> dict:
        output = super()._build_skill_output(
            selected_action=selected_action,
            decision=decision,
            state=state,
            semantic_decision_skipped=semantic_decision_skipped,
        )
        selected_expressions = self._build_constructing_agg_expr_descriptors(
            selected_action=selected_action,
            selected_expressions=decision.get("selected_expressions", []),
            agg_expr=state.get("operator_store", {}).get("AggExpr", []),
        )
        constructing_expressions = (
            []
            if selected_action == "stop_agg_expr"
            else selected_expressions
        )
        output["selected_expressions"] = selected_expressions
        output["constructing_expressions"] = constructing_expressions
        output["operator_store_delta"] = {
            "constructing_expressions": constructing_expressions,
        }
        if decision.get("op_level_limit_reached"):
            output["semantic_decision_skipped"] = True
            output["op_level_limit_reached"] = bool(
                decision.get("op_level_limit_reached")
            )
        return output

    def _build_constructing_agg_expr_descriptors(
        self,
        selected_action: str | None,
        selected_expressions: list[Any],
        agg_expr: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates = {
            str(item.get("name")): item
            for item in agg_expr
            if isinstance(item, dict) and item.get("name")
        }
        descriptors = []

        for selected in selected_expressions[:2]:
            expression = self._selected_expression_text(selected)
            if not expression:
                continue
            expression = _replace_index_placeholders(
                expression=expression,
                candidates=candidates,
                placeholder_prefix="a",
            )
            sources = self._find_agg_expr_sources(expression, candidates)
            if not sources:
                raise ValueError(
                    f"{self.skill_id} could not resolve an AggExpr source for "
                    f"{expression!r}"
                )

            data_types = []
            source_levels = []
            for source in sources:
                data_type = source.get("data_type")
                if data_type not in data_types:
                    data_types.append(data_type)
                try:
                    source_levels.append(int(source.get("op_level", 1)))
                except (TypeError, ValueError):
                    source_levels.append(1)

            op_level = max(source_levels or [1])
            if selected_action != "stop_agg_expr":
                op_level += 1

            descriptors.append(
                {
                    "name": expression,
                    "data_type": (
                        data_types[0] if len(data_types) == 1 else data_types
                    ),
                    "op_level": op_level,
                    "source_expression_action": selected_action,
                }
            )
        return descriptors

    @staticmethod
    def _selected_expression_text(selected: Any) -> str:
        if isinstance(selected, str):
            return selected.strip()
        if isinstance(selected, dict):
            value = selected.get("name") or selected.get("expression")
            return str(value).strip() if value is not None else ""
        return ""

    @staticmethod
    def _find_agg_expr_sources(
        expression: str,
        candidates: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return _find_expression_sources(
            expression=expression,
            candidates=candidates,
            placeholder_prefix="a",
        )

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        if self._op_level_limit_reached(
            state=state,
            store_key="AggExpr",
            limit_key="agg_expr",
            default=10,
        ):
            return ["stop_agg_expr"] if "stop_agg_expr" in action_space else []

        operator_regime = self._p4_operator_regime(state)
        if operator_regime == "aggregate_query":
            action_space = [
                action
                for action in action_space
                if action != "wrap_scalar_tsgf_post"
            ]
        return action_space

    def build_action_effect(
        self,
        selected_action: str | None,
        decision: dict,
        state: dict,
    ) -> dict:
        if selected_action == "stop_agg_expr":
            return {
                "activate_next_skills": [],
                "agg_expr_committed": True,
                "p4_finished": True,
            }

        effect = super().build_action_effect(selected_action, decision, state)
        effect["p4_active_loop"] = "agg_expr_loop"
        effect["aggregate_expression_action"] = selected_action
        return effect


class FunctionSourceSkill(BaseActionSkill):
    """Action P4A4: randomly select the function source without an LLM call."""

    skill_id = "FunctionSourceSkill"
    action_id = "P4A4"
    decision_name = "function_source"

    candidate_descriptions = {
        "builtin_function": {
            "meaning": (
                "Use functions provided by the database engine, including built-in "
                "aggregate functions and built-in time-series generating functions."
            ),
            "requires_sql_generation": False,
        },
        "udf_function": {
            "meaning": (
                "Use functions from the UDF library, including user-defined "
                "aggregate functions and user-defined time-series processing "
                "functions."
            ),
            "requires_sql_generation": False,
        },
        "mixed_function": {
            "meaning": (
                "Use both built-in functions and UDF functions in the current "
                "operator plan."
            ),
            "requires_sql_generation": False,
        },
    }

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        """Select one available function source randomly and record the result."""

        action_space = self._normalize_action_space(action_space)
        action_space = self.filter_action_space(
            state=state,
            context=context,
            action_space=action_space,
        )
        if not action_space:
            return self._build_no_compatible_source_output(state)
        self._validate_action_space(action_space)

        selected_action = random.choice(action_space)
        output = self._build_skill_output(
            selected_action=selected_action,
            decision={
                "selected_action": selected_action,
                "confidence": 0.0,
                "reason": (
                    "Randomly selected from function sources compatible with "
                    "the current expression data type."
                ),
            },
            state=state,
            semantic_decision_skipped=True,
        )
        output["function_source_compatibility"] = (
            self._function_source_compatibility(state)
        )
        current_expression = deepcopy(self._current_expression(state))
        current_expression["function_source"] = selected_action
        output["constructing_expression_update"] = current_expression
        return output

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        compatibility = self._function_source_compatibility(state)
        builtin_supported = compatibility["builtin_supported"]
        udf_supported = compatibility["udf_supported"]

        if self._phase_flag_is_true(state, "P3", "group_by_enabled"):
            allowed_sources = {"builtin_function"} if builtin_supported else set()
        elif builtin_supported and udf_supported:
            allowed_sources = {
                "builtin_function",
                "udf_function",
                "mixed_function",
            }
        elif builtin_supported:
            allowed_sources = {"builtin_function"}
        elif udf_supported:
            allowed_sources = {"udf_function"}
        else:
            allowed_sources = set()
        return [action for action in action_space if action in allowed_sources]

    @staticmethod
    def _phase_flag_is_true(state: dict, phase_id: str, flag_name: str) -> bool:
        flags = state.get("flags", {}) if isinstance(state, dict) else {}
        if not isinstance(flags, dict):
            return False
        phase_flags = flags.get(phase_id, {})
        if isinstance(phase_flags, dict):
            return phase_flags.get(flag_name) is True
        return False

    def _function_source_compatibility(self, state: dict) -> dict[str, Any]:
        expression = self._current_expression(state)
        expression_text = self._expression_text(expression)
        data_type = expression.get("data_type")

        if expression_text.startswith("AGG("):
            target_skill = "AggregateFunctionFamilySkill"
            function_kind = "aggregate"
            builtin_families = BUILTIN_AGGREGATE_FAMILY_INPUT_TYPES
            udf_families = UDF_AGGREGATE_FAMILY_INPUT_TYPES
        elif expression_text.startswith(("TSGF(", "ScalarCompatibleTSGF(")):
            target_skill = "TimeSeriesFunctionFamilySkill"
            function_kind = "timeseries"
            builtin_families = BUILTIN_TIMESERIES_FAMILY_INPUT_TYPES
            udf_families = UDF_TIMESERIES_FAMILY_INPUT_TYPES
        else:
            raise ValueError(
                "FunctionSourceSkill expects AGG(...), TSGF(...), or "
                "ScalarCompatibleTSGF(...) as the current expression"
            )

        return {
            "target_skill": target_skill,
            "function_kind": function_kind,
            "data_type": deepcopy(data_type),
            "builtin_supported": self._family_group_accepts_input_type(
                builtin_families,
                data_type,
            ),
            "udf_supported": self._family_group_accepts_input_type(
                udf_families,
                data_type,
            ),
        }

    @staticmethod
    def _family_group_accepts_input_type(
        family_input_types: dict[str, tuple[str, ...]],
        data_type: Any,
    ) -> bool:
        input_types = normalize_data_types(data_type)
        if not input_types:
            return False
        for allowed_types in family_input_types.values():
            if _ANY_FAMILY_INPUT_TYPE in allowed_types:
                return True
            allowed_type_set = set(allowed_types)
            if all(input_type in allowed_type_set for input_type in input_types):
                return True
        return False

    @staticmethod
    def _current_expression(state: dict) -> dict[str, Any]:
        operator_store = state.get("operator_store", {})
        expression = operator_store.get("current_constructing_expression")
        if not isinstance(expression, dict):
            raise ValueError(
                "FunctionSourceSkill requires a current expression descriptor"
            )
        return expression

    @staticmethod
    def _expression_text(expression: dict[str, Any]) -> str:
        value = expression.get("name") or expression.get("expression")
        return str(value).strip() if value is not None else ""

    def _build_no_compatible_source_output(self, state: dict) -> dict[str, Any]:
        compatibility = self._function_source_compatibility(state)
        message = (
            "No compatible built-in or UDF "
            f"{compatibility['function_kind']} function source supports data "
            f"type {compatibility['data_type']!r}. The current P4 expression "
            "construction loop was exited without interrupting the run."
        )
        output = self._build_skill_output(
            selected_action=None,
            decision={
                "selected_action": None,
                "confidence": 1.0,
                "reason": message,
            },
            state=state,
            semantic_decision_skipped=True,
        )
        output["function_source_compatibility"] = compatibility
        output["pipeline_error"] = {
            "code": "no_compatible_function_source",
            "message": message,
            "recoverable": True,
        }
        output["trace"] = message
        return output


class _FunctionFamilySkill(BaseActionSkill):
    """Shared source-filtering behavior for function-family decisions."""

    builtin_families: dict[str, dict] = {}
    udf_families: dict[str, dict] = {}
    builtin_category: str = ""
    udf_category: str = ""
    builtin_family_input_types: dict[str, tuple[str, ...]] = {}
    udf_family_input_types: dict[str, tuple[str, ...]] = {}
    builtin_family_function_input_types: dict[
        str,
        dict[str, tuple[str, ...]],
    ] = {}
    udf_family_function_input_types: dict[
        str,
        dict[str, tuple[str, ...]],
    ] = {}

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        function_source = self._require_function_source(state)
        filtered_actions = self._filter_by_function_source(
            action_space=action_space,
            function_source=function_source,
            builtin_families=self.builtin_families,
            udf_families=self.udf_families,
        )
        filtered_actions = self._filter_families_by_input_type(
            action_space=filtered_actions,
            state=state,
        )
        return list(filtered_actions)

    def get_candidate_descriptions(
        self,
        state: dict,
        action_space: list[str],
    ) -> dict[str, dict[str, Any]]:
        descriptions = super().get_candidate_descriptions(
            state=state,
            action_space=action_space,
        )
        data_type = self._current_expression_data_type(state)
        for action, description in descriptions.items():
            description["functions"] = self._compatible_family_functions(
                family_name=action,
                data_type=data_type,
            )
        return descriptions

    def _require_function_source(self, state: dict) -> str:
        function_source = self._p4_function_source(state)
        if function_source not in {
            "builtin_function",
            "udf_function",
            "mixed_function",
        }:
            raise ValueError(
                f"{self.skill_id} requires a function_source selected by "
                "FunctionSourceSkill"
            )
        return function_source


    def _filter_families_by_input_type(
        self,
        action_space: list[str],
        state: dict,
    ) -> list[str]:
        data_type = self._current_expression_data_type(state)
        return [
            action
            for action in action_space
            if self._family_accepts_input_type(action, data_type)
        ]

    def _family_accepts_input_type(
        self,
        family_name: str,
        data_type: Any,
    ) -> bool:
        allowed_types = self._family_allowed_input_types(family_name)
        return self._allowed_types_accept_input(allowed_types, data_type)

    def _family_allowed_input_types(
        self,
        family_name: str,
    ) -> tuple[str, ...]:
        if family_name in self.builtin_families:
            return self.builtin_family_input_types.get(family_name, ())
        if family_name in self.udf_families:
            return self.udf_family_input_types.get(family_name, ())
        return ()

    def _compatible_family_functions(
        self,
        family_name: str,
        data_type: Any,
    ) -> list[str]:
        if family_name in self.builtin_families:
            family = self.builtin_families[family_name]
            function_input_types = (
                self.builtin_family_function_input_types.get(family_name, {})
            )
        elif family_name in self.udf_families:
            family = self.udf_families[family_name]
            function_input_types = self.udf_family_function_input_types.get(
                family_name,
                {},
            )
        else:
            return []

        return [
            function_name
            for function_name in family.get("functions", [])
            if self._allowed_types_accept_input(
                function_input_types.get(function_name, ()),
                data_type,
            )
        ]

    @staticmethod
    def _allowed_types_accept_input(
        allowed_types: tuple[str, ...],
        data_type: Any,
    ) -> bool:
        if normalize_data_types is None:
            raise RuntimeError("Data-type normalization helper is unavailable")
        input_types = normalize_data_types(data_type)
        if not input_types or not allowed_types:
            return False
        if _ANY_FAMILY_INPUT_TYPE in allowed_types:
            return True
        allowed_type_set = set(allowed_types)
        return all(input_type in allowed_type_set for input_type in input_types)

    @staticmethod
    def _current_expression_data_type(state: dict) -> Any:
        operator_store = state.get("operator_store", {})
        current_expression = operator_store.get(
            "current_constructing_expression"
        )
        if not isinstance(current_expression, dict):
            raise ValueError(
                "Function-family selection requires a current expression descriptor"
            )
        data_type = current_expression.get("data_type")
        if data_type is None or data_type == []:
            raise ValueError(
                "Function-family selection requires current expression data_type"
            )
        return data_type

    def _filter_by_function_source(
        self,
        action_space: list[str],
        function_source: str | None,
        builtin_families: dict[str, dict],
        udf_families: dict[str, dict],
    ) -> list[str]:
        if function_source == "builtin_function":
            allowed = set(builtin_families)
        elif function_source == "udf_function":
            allowed = set(udf_families)
        else:
            allowed = set(builtin_families) | set(udf_families)
        return [action for action in action_space if action in allowed]

    def _build_no_compatible_family_output(self, state: dict) -> dict[str, Any]:
        function_source = self._p4_function_source(state)
        data_type = self._current_expression_data_type(state)
        function_kind = (
            "aggregate" if self.action_id == "P4A5" else "time-series"
        )
        message = (
            f"No compatible {function_kind} function family is available for "
            f"source {function_source!r} and data type {data_type!r}. The "
            "current P4 expression construction loop was exited without "
            "interrupting the run."
        )
        output = self._build_skill_output(
            selected_action=None,
            decision={
                "selected_action": None,
                "confidence": 1.0,
                "reason": message,
            },
            state=state,
            semantic_decision_skipped=True,
        )
        output["pipeline_error"] = {
            "code": "no_compatible_function_family",
            "message": message,
            "recoverable": True,
        }
        output["trace"] = message
        return output


class AggregateFunctionFamilySkill(_FunctionFamilySkill):
    """Action P4A5: choose an aggregate function family."""

    skill_id = "AggregateFunctionFamilySkill"
    action_id = "P4A5"
    decision_name = "aggregate_function_family"
    builtin_families = BUILTIN_AGGREGATE_FAMILIES
    udf_families = UDF_AGGREGATE_FAMILIES
    builtin_category = BUILTIN_AGGREGATE
    udf_category = UDF_AGGREGATE
    builtin_family_input_types = BUILTIN_AGGREGATE_FAMILY_INPUT_TYPES
    udf_family_input_types = UDF_AGGREGATE_FAMILY_INPUT_TYPES
    builtin_family_function_input_types = (
        _BUILTIN_AGGREGATE_FAMILY_FUNCTION_INPUT_TYPES
    )
    udf_family_function_input_types = _UDF_AGGREGATE_FAMILY_FUNCTION_INPUT_TYPES

    candidate_descriptions = _family_candidates(
        BUILTIN_AGGREGATE_FAMILIES,
        UDF_AGGREGATE_FAMILIES,
    )

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        """Choose one aggregate family and rewrite the current AggAtom."""

        action_space = self._normalize_action_space(action_space)
        action_space = self.filter_action_space(
            state=state,
            context=context,
            action_space=action_space,
        )
        if not action_space:
            return self._build_no_compatible_family_output(state)
        self._validate_action_space(action_space)

        skill_input = self.build_skill_input(
            state=state,
            context=context,
            action_space=action_space,
        )
        current_expression = skill_input.get("constructing_expression")
        expression_text = self._expression_text(current_expression)
        if not expression_text:
            raise ValueError(
                "AggregateFunctionFamilySkill requires a current "
                "constructing expression"
            )

        if self.llm is not None and hasattr(
            self.llm,
            "select_action_and_rewrite_expression",
        ):
            decision = self.llm.select_action_and_rewrite_expression(
                skill_input=skill_input,
                expression=expression_text,
            )
        else:
            selected_action = action_space[0]
            decision = {
                "selected_action": selected_action,
                "rewritten_expression": self._fallback_rewrite(
                    expression_text,
                    selected_action,
                ),
                "confidence": 0.0,
                "reason": "Fallback aggregate-family selection.",
            }

        selected_action = decision.get("selected_action")
        if selected_action not in action_space:
            raise ValueError(
                f"{self.skill_id} selected invalid action: {selected_action}. "
                f"Valid candidates: {action_space}"
            )
        rewritten_expression = str(
            decision.get("rewritten_expression", "")
        ).strip()
        if not rewritten_expression:
            raise ValueError(
                "AggregateFunctionFamilySkill returned an empty expression"
            )
        expected_expression = self._fallback_rewrite(
            expression_text,
            selected_action,
        )
        if rewritten_expression != expected_expression:
            raise ValueError(
                "AggregateFunctionFamilySkill must only replace the outer "
                f"AGG placeholder; expected {expected_expression!r}, got "
                f"{rewritten_expression!r}"
            )

        processed_expression = self._build_processed_expression(
            current_expression=current_expression,
            rewritten_expression=rewritten_expression,
        )
        output = self._build_skill_output(
            selected_action=selected_action,
            decision=decision,
            state=state,
        )
        output["constructing_expression_update"] = processed_expression
        output["trace"] = (
            f"{self.skill_id} selected {selected_action} and rewrote "
            f"{expression_text} as {rewritten_expression}."
        )
        return output

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        function_source = self._require_function_source(state)
        filtered_actions = self._filter_by_function_source(
            action_space=action_space,
            function_source=function_source,
            builtin_families=self.builtin_families,
            udf_families=self.udf_families,
        )
        filtered_actions = self._filter_families_by_input_type(
            action_space=filtered_actions,
            state=state,
        )
        return list(filtered_actions)

    @staticmethod
    def _expression_text(expression: Any) -> str:
        if isinstance(expression, str):
            return expression.strip()
        if isinstance(expression, dict):
            value = expression.get("name") or expression.get("expression")
            return str(value).strip() if value is not None else ""
        return ""

    @staticmethod
    def _fallback_rewrite(expression: str, selected_action: str) -> str:
        if expression.startswith("AGG("):
            return f"{selected_action}{expression[3:]}"
        raise ValueError(
            "AggregateFunctionFamilySkill expects an AGG(SeriesExpr) input"
        )

    @staticmethod
    def _build_processed_expression(
        current_expression: Any,
        rewritten_expression: str,
    ) -> dict[str, Any]:
        if isinstance(current_expression, dict):
            processed = deepcopy(current_expression)
        else:
            processed = {
                "data_type": None,
                "op_level": 1,
            }
        if "expression" in processed and "name" not in processed:
            processed["expression"] = rewritten_expression
        else:
            processed["name"] = rewritten_expression
        return processed


class TimeSeriesFunctionFamilySkill(_FunctionFamilySkill):
    """Action P4A6: choose a time-series generating function family."""

    skill_id = "TimeSeriesFunctionFamilySkill"
    action_id = "P4A6"
    decision_name = "timeseries_function_family"
    builtin_families = BUILTIN_TIMESERIES_FAMILIES
    udf_families = UDF_TIMESERIES_FAMILIES
    builtin_category = BUILTIN_TIMESERIES
    udf_category = UDF_TIMESERIES
    builtin_family_input_types = BUILTIN_TIMESERIES_FAMILY_INPUT_TYPES
    udf_family_input_types = UDF_TIMESERIES_FAMILY_INPUT_TYPES
    builtin_family_function_input_types = (
        _BUILTIN_TIMESERIES_FAMILY_FUNCTION_INPUT_TYPES
    )
    udf_family_function_input_types = _UDF_TIMESERIES_FAMILY_FUNCTION_INPUT_TYPES

    candidate_descriptions = _timeseries_family_candidates()

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        """Choose one time-series family and rewrite the current placeholder."""

        action_space = self._normalize_action_space(action_space)
        action_space = self.filter_action_space(
            state=state,
            context=context,
            action_space=action_space,
        )
        if not action_space:
            return self._build_no_compatible_family_output(state)
        self._validate_action_space(action_space)

        skill_input = self.build_skill_input(
            state=state,
            context=context,
            action_space=action_space,
        )
        current_expression = skill_input.get("constructing_expression")
        expression_text = self._expression_text(current_expression)
        expression_mode = self._expression_mode(state)

        if self.llm is not None and hasattr(
            self.llm,
            "select_action_and_rewrite_expression",
        ):
            decision = self.llm.select_action_and_rewrite_expression(
                skill_input=skill_input,
                expression=expression_text,
            )
        else:
            selected_action = action_space[0]
            decision = {
                "selected_action": selected_action,
                "rewritten_expression": self._fallback_rewrite(
                    expression=expression_text,
                    selected_action=selected_action,
                    expression_mode=expression_mode,
                ),
                "confidence": 0.0,
                "reason": "Fallback time-series-family selection.",
            }

        selected_action = decision.get("selected_action")
        if selected_action not in action_space:
            raise ValueError(
                f"{self.skill_id} selected invalid action: {selected_action}. "
                f"Valid candidates: {action_space}"
            )

        rewritten_expression = str(
            decision.get("rewritten_expression", "")
        ).strip()
        if not rewritten_expression:
            raise ValueError(
                "TimeSeriesFunctionFamilySkill returned an empty expression"
            )
        expected_expression = self._fallback_rewrite(
            expression=expression_text,
            selected_action=selected_action,
            expression_mode=expression_mode,
        )
        if rewritten_expression != expected_expression:
            raise ValueError(
                "TimeSeriesFunctionFamilySkill must only replace the outer "
                "time-series function placeholder; expected "
                f"{expected_expression!r}, got {rewritten_expression!r}"
            )

        processed_expression = self._build_processed_expression(
            current_expression=current_expression,
            rewritten_expression=rewritten_expression,
        )
        output = self._build_skill_output(
            selected_action=selected_action,
            decision=decision,
            state=state,
        )
        output["constructing_expression_update"] = processed_expression
        output["trace"] = (
            f"{self.skill_id} selected {selected_action} and rewrote "
            f"{expression_text} as {rewritten_expression}."
        )
        return output

    def get_candidate_descriptions(
        self,
        state: dict,
        action_space: list[str],
    ) -> dict[str, dict[str, Any]]:
        descriptions = super().get_candidate_descriptions(
            state=state,
            action_space=action_space,
        )
        if self._expression_mode(state) != "scalar_compatible_tsgf":
            return descriptions

        for candidate in descriptions.values():
            compatible_functions = set(candidate.get("functions", []))
            scalar_functions = [
                function_name
                for function_name in candidate.get(
                    "scalar_compatible_functions", []
                )
                if function_name in compatible_functions
            ]
            candidate["functions"] = scalar_functions
            candidate["scalar_compatible_only"] = True
        return descriptions

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        function_source = self._require_function_source(state)
        action_space = self._filter_by_function_source(
            action_space=action_space,
            function_source=function_source,
            builtin_families=self.builtin_families,
            udf_families=self.udf_families,
        )
        action_space = self._filter_families_by_input_type(
            action_space=action_space,
            state=state,
        )
        if self._expression_mode(state) == "scalar_compatible_tsgf":
            action_space = [
                action
                for action in action_space
                if any(
                    function_name
                    in self.candidate_descriptions[action].get(
                        "scalar_compatible_functions", []
                    )
                    for function_name in self._compatible_family_functions(
                        family_name=action,
                        data_type=self._current_expression_data_type(state),
                    )
                )
            ]
        return list(action_space)

    def _expression_mode(self, state: dict) -> str:
        operator_store = state.get("operator_store", {})
        current_expression = operator_store.get(
            "current_constructing_expression"
        )
        expression_text = self._expression_text(current_expression)
        if expression_text.startswith("ScalarCompatibleTSGF("):
            return "scalar_compatible_tsgf"
        if expression_text.startswith("TSGF("):
            return "tsgf"
        raise ValueError(
            "TimeSeriesFunctionFamilySkill expects the current expression to "
            "start with TSGF( or ScalarCompatibleTSGF("
        )

    @staticmethod
    def _fallback_rewrite(
        expression: str,
        selected_action: str,
        expression_mode: str,
    ) -> str:
        prefix = (
            "ScalarCompatibleTSGF"
            if expression_mode == "scalar_compatible_tsgf"
            else "TSGF"
        )
        expected_prefix = f"{prefix}("
        if expression.startswith(expected_prefix):
            return f"{selected_action}{expression[len(prefix):]}"
        raise ValueError(
            "TimeSeriesFunctionFamilySkill received an expression that does "
            f"not match {expected_prefix}"
        )

    @staticmethod
    def _build_processed_expression(
        current_expression: Any,
        rewritten_expression: str,
    ) -> dict[str, Any]:
        if isinstance(current_expression, dict):
            processed = deepcopy(current_expression)
        else:
            processed = {
                "data_type": None,
                "op_level": 1,
            }
        if "expression" in processed and "name" not in processed:
            processed["expression"] = rewritten_expression
        else:
            processed["name"] = rewritten_expression
        return processed

    @staticmethod
    def _expression_text(expression: Any) -> str:
        if isinstance(expression, str):
            return expression.strip()
        if isinstance(expression, dict):
            value = expression.get("name") or expression.get("expression")
            return str(value).strip() if value is not None else ""
        return ""


class OperatorFamilySkill(BaseActionSkill):
    """Action P4A7: choose one concrete arithmetic operator."""

    skill_id = "OperatorFamilySkill"
    action_id = "P4A7"
    decision_name = "operator_family"
    candidate_descriptions = OPERATOR_CANDIDATES

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        """Choose one concrete operator and rewrite the abstract expression."""

        action_space = self._normalize_action_space(action_space)
        action_space = self.filter_action_space(
            state=state,
            context=context,
            action_space=action_space,
        )
        self._validate_action_space(action_space)

        skill_input = self.build_skill_input(
            state=state,
            context=context,
            action_space=action_space,
        )
        current_expression = skill_input.get("constructing_expression")
        expression_text = self._expression_text(current_expression)
        expected_rewrites = {
            action: self._rewrite_operator_expression(
                expression=expression_text,
                symbol=self.candidate_descriptions[action]["symbol"],
                operator_arity=self.candidate_descriptions[action][
                    "operator_arity"
                ],
            )
            for action in action_space
        }

        if self.llm is not None and hasattr(
            self.llm,
            "select_operator_and_rewrite_expression",
        ):
            decision = self.llm.select_operator_and_rewrite_expression(
                skill_input=skill_input,
                expected_rewrites=expected_rewrites,
            )
        else:
            selected_action = action_space[0]
            decision = {
                "selected_action": selected_action,
                "rewritten_expression": expected_rewrites[selected_action],
                "confidence": 0.0,
                "reason": "Fallback arithmetic-operator selection.",
            }

        selected_action = decision.get("selected_action")
        if selected_action not in action_space:
            raise ValueError(
                f"{self.skill_id} selected invalid action: {selected_action}. "
                f"Valid candidates: {action_space}"
            )
        rewritten_expression = str(
            decision.get("rewritten_expression", "")
        ).strip()
        expected_expression = expected_rewrites[selected_action]
        if rewritten_expression != expected_expression:
            raise ValueError(
                "OperatorFamilySkill must insert only the selected operator; "
                f"expected {expected_expression!r}, got "
                f"{rewritten_expression!r}"
            )

        processed_expression = self._build_processed_expression(
            current_expression=current_expression,
            rewritten_expression=rewritten_expression,
        )
        processed_expression["operator_family"] = selected_action
        output = self._build_skill_output(
            selected_action=selected_action,
            decision=decision,
            state=state,
        )
        output["constructing_expression_update"] = processed_expression
        output["trace"] = (
            f"{self.skill_id} selected {selected_action} and rewrote "
            f"{expression_text} as {rewritten_expression}."
        )
        return output

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        required_arity = self._required_operator_arity(state)
        return [
            action
            for action in action_space
            if self.candidate_descriptions.get(action, {}).get(
                "operator_arity"
            )
            == required_arity
        ]

    def _required_operator_arity(self, state: dict) -> str:
        operator_store = state.get("operator_store", {})
        current_expression = operator_store.get(
            "current_constructing_expression"
        )
        expression_text = self._expression_text(current_expression)
        if expression_text.startswith("UnaryOp("):
            return "unary"
        if expression_text.startswith("BinaryOp("):
            return "binary"
        raise ValueError(
            "OperatorFamilySkill expects the current expression to start "
            "with UnaryOp( or BinaryOp("
        )

    @classmethod
    def _rewrite_operator_expression(
        cls,
        expression: str,
        symbol: str,
        operator_arity: str,
    ) -> str:
        if operator_arity == "unary":
            inner_expression = cls._unwrap_expression(expression, "UnaryOp")
            return f"{symbol}({inner_expression})"
        if operator_arity == "binary":
            inner_expression = cls._unwrap_expression(expression, "BinaryOp")
            arguments = cls._split_top_level_arguments(inner_expression)
            if len(arguments) != 2:
                raise ValueError(
                    "BinaryOp expression must contain exactly two arguments"
                )
            return f"({arguments[0]}) {symbol} ({arguments[1]})"
        raise ValueError(f"Unsupported operator arity: {operator_arity}")

    @staticmethod
    def _unwrap_expression(expression: str, placeholder: str) -> str:
        prefix = f"{placeholder}("
        if not expression.startswith(prefix) or not expression.endswith(")"):
            raise ValueError(
                f"Expected {placeholder}(...) expression, got {expression!r}"
            )
        return expression[len(prefix):-1].strip()

    @staticmethod
    def _split_top_level_arguments(expression: str) -> list[str]:
        arguments = []
        start = 0
        depth = 0
        for index, character in enumerate(expression):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("Unbalanced parentheses in BinaryOp")
            elif character == "," and depth == 0:
                arguments.append(expression[start:index].strip())
                start = index + 1
        if depth != 0:
            raise ValueError("Unbalanced parentheses in BinaryOp")
        arguments.append(expression[start:].strip())
        if any(not argument for argument in arguments):
            raise ValueError("BinaryOp arguments cannot be empty")
        return arguments

    @staticmethod
    def _build_processed_expression(
        current_expression: Any,
        rewritten_expression: str,
    ) -> dict[str, Any]:
        if isinstance(current_expression, dict):
            processed = deepcopy(current_expression)
        else:
            processed = {
                "data_type": None,
                "op_level": 1,
            }
        if "expression" in processed and "name" not in processed:
            processed["expression"] = rewritten_expression
        else:
            processed["name"] = rewritten_expression
        return processed

    @staticmethod
    def _expression_text(expression: Any) -> str:
        if isinstance(expression, str):
            return expression.strip()
        if isinstance(expression, dict):
            value = expression.get("name") or expression.get("expression")
            return str(value).strip() if value is not None else ""
        return ""


class ComponentInstantiationSkill(BaseActionSkill):
    """Action P4A8: select a concrete function or finalize an operator."""

    skill_id = "ComponentInstantiationSkill"
    action_id = "P4A8"
    decision_name = "component_instantiation"
    candidate_descriptions = COMPONENT_INSTANTIATION_CANDIDATES

    def run(self, state: dict, context: str, action_space: list[str]) -> dict:
        """Skip semantic decision only when A7 already chose the operator."""

        action_space = self._normalize_action_space(action_space)
        action_space = self.filter_action_space(
            state=state,
            context=context,
            action_space=action_space,
        )
        self._validate_action_space(action_space)

        source_kind, _ = self._resolve_component_source(state)
        if source_kind != "operator":
            output = super().run(
                state=state,
                context=context,
                action_space=action_space,
            )
            return self._prepare_generation_output(
                output=output,
                state=state,
                source_kind=source_kind,
            )

        selected_action = action_space[0]
        output = self._build_skill_output(
            selected_action=selected_action,
            decision={
                "selected_action": selected_action,
                "confidence": 1.0,
                "reason": "Operator already selected by OperatorFamilySkill.",
            },
            state=state,
            semantic_decision_skipped=True,
        )
        return self._prepare_generation_output(
            output=output,
            state=state,
            source_kind=source_kind,
        )

    def filter_action_space(
        self,
        state: dict,
        context: str,
        action_space: list[str],
    ) -> list[str]:
        source_kind, source_name = self._resolve_component_source(state)
        if source_kind == "operator":
            return [source_name] if source_name in action_space else []

        family_info = FUNCTION_FAMILY_COMPONENT_INDEX[source_name]
        data_type = self._current_expression_data_type(state)
        compatible_functions = [
            function_name
            for function_name in family_info["functions"]
            if self._allowed_types_accept_input(
                family_info["function_input_types"].get(function_name, ()),
                data_type,
            )
        ]
        if self._is_scalar_compatible_source(state, family_info):
            compatible_functions = [
                function_name
                for function_name in compatible_functions
                if _is_scalar_compatible_timeseries_function(function_name)
            ]
        return [
            function_name
            for function_name in action_space
            if function_name in compatible_functions
        ]

    def _resolve_component_source(self, state: dict) -> tuple[str, str]:
        current_expression = self._current_expression(state)
        expression_text = self._expression_text(current_expression)
        outer_name = expression_text.split("(", 1)[0].strip()
        if outer_name in FUNCTION_FAMILY_COMPONENT_INDEX:
            return "function_family", outer_name

        selected_operator = current_expression.get("operator_family")
        if selected_operator in OPERATOR_CANDIDATES:
            return "operator", selected_operator
        raise ValueError(
            "ComponentInstantiationSkill could not resolve an A5, A6, or A7 "
            "source from the current constructing expression"
        )

    @staticmethod
    def _current_expression(state: dict) -> dict[str, Any]:
        operator_store = state.get("operator_store", {})
        current_expression = operator_store.get(
            "current_constructing_expression"
        )
        if not isinstance(current_expression, dict):
            raise ValueError(
                "ComponentInstantiationSkill requires a current expression descriptor"
            )
        return current_expression

    @classmethod
    def _current_expression_data_type(cls, state: dict) -> Any:
        data_type = cls._current_expression(state).get("data_type")
        if data_type is None or data_type == []:
            raise ValueError(
                "ComponentInstantiationSkill requires current expression data_type"
            )
        return data_type

    @staticmethod
    def _allowed_types_accept_input(
        allowed_types: tuple[str, ...],
        data_type: Any,
    ) -> bool:
        if normalize_data_types is None:
            raise RuntimeError("Data-type normalization helper is unavailable")
        input_types = normalize_data_types(data_type)
        if not input_types or not allowed_types:
            return False
        if _ANY_FAMILY_INPUT_TYPE in allowed_types:
            return True
        allowed_type_set = set(allowed_types)
        return all(input_type in allowed_type_set for input_type in input_types)

    def _is_scalar_compatible_source(
        self,
        state: dict,
        family_info: dict[str, Any],
    ) -> bool:
        return (
            family_info.get("family_kind") == "timeseries"
            and self._current_expression(state).get(
                "source_expression_action"
            )
            == "wrap_scalar_tsgf_post"
        )

    def _prepare_generation_output(
        self,
        output: dict[str, Any],
        state: dict,
        source_kind: str,
    ) -> dict[str, Any]:
        current_expression = self._current_expression(state)
        expression_text = self._expression_text(current_expression)
        selected_action = output.get("selected_action")

        if source_kind == "function_family":
            prepared_expression = self._replace_outer_function_family(
                expression=expression_text,
                function_name=selected_action,
            )
            function_definition = _get_component_function_definition(
                selected_action
            )
            if function_definition is None:
                raise ValueError(
                    "ComponentInstantiationSkill could not load the selected "
                    f"function definition: {selected_action}"
                )
            output["selected_function_definition"] = function_definition
            output["source_component_skill"] = (
                "AggregateFunctionFamilySkill"
                if FUNCTION_FAMILY_COMPONENT_INDEX[
                    expression_text.split("(", 1)[0].strip()
                ]["family_kind"]
                == "aggregate"
                else "TimeSeriesFunctionFamilySkill"
            )
        else:
            prepared_expression = expression_text
            output["source_component_skill"] = "OperatorFamilySkill"

        output["input_constructing_expression"] = deepcopy(current_expression)
        output["prepared_constructing_expression"] = prepared_expression
        output["generation_output_instruction"] = (
            COMPONENT_GENERATION_OUTPUT_INSTRUCTION
        )
        return output

    @staticmethod
    def _replace_outer_function_family(
        expression: str,
        function_name: str,
    ) -> str:
        opening_parenthesis = expression.find("(")
        if opening_parenthesis <= 0 or not expression.endswith(")"):
            raise ValueError(
                "ComponentInstantiationSkill expects a function-family "
                f"expression, got {expression!r}"
            )
        return f"{function_name}{expression[opening_parenthesis:]}"

    @staticmethod
    def _expression_text(expression: Any) -> str:
        value = expression.get("name") or expression.get("expression")
        return str(value).strip() if value is not None else ""

    def build_action_effect(
        self,
        selected_action: str | None,
        decision: dict,
        state: dict,
    ) -> dict:
        latest_action = self._latest_component_action(state)
        return_to_skill = None
        if latest_action in SERIES_COMPONENT_ACTIONS:
            return_to_skill = "SeriesExpressionSkill"
        elif latest_action in AGG_COMPONENT_ACTIONS:
            return_to_skill = "AggregateExpressionSkill"

        return {
            "activate_next_skills": [],
            "component_instantiated": selected_action,
            "source_expression_action": latest_action,
            "return_to_skill": return_to_skill,
        }


SKILL_REGISTRY = {
    "OperatorRegimeSkill": OperatorRegimeSkill,
    "SeriesExpressionSkill": SeriesExpressionSkill,
    "AggregateAtomSkill": AggregateAtomSkill,
    "AggregateExpressionSkill": AggregateExpressionSkill,
    "FunctionSourceSkill": FunctionSourceSkill,
    "AggregateFunctionFamilySkill": AggregateFunctionFamilySkill,
    "TimeSeriesFunctionFamilySkill": TimeSeriesFunctionFamilySkill,
    "OperatorFamilySkill": OperatorFamilySkill,
    "ComponentInstantiationSkill": ComponentInstantiationSkill,
}
