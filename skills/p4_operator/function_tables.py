"""Function definition tables and lookup indexes for P4 operators."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


BUILTIN_AGGREGATE = "builtin_aggregate"
UDF_AGGREGATE = "udf_aggregate"
BUILTIN_TIMESERIES = "builtin_timeseries"
UDF_TIMESERIES = "udf_timeseries"


FUNCTION_CATEGORIES = (
    BUILTIN_AGGREGATE,
    UDF_AGGREGATE,
    BUILTIN_TIMESERIES,
    UDF_TIMESERIES,
)


IOTDB_OPERATOR_CATEGORIES = (
    "arithmetic_operators",
    "comparison_operators",
    "logical_operators",
)


IOTDB_OPERATORS = {
    "arithmetic_operators": {
        "unary_plus": {
            "symbol": "+",
            "meaning": "Unary positive sign.",
            "operator_arity": "unary",
        },
        "unary_minus": {
            "symbol": "-",
            "meaning": "Unary negative sign.",
            "operator_arity": "unary",
        },
        "multiply": {
            "symbol": "*",
            "meaning": "Multiplication.",
            "operator_arity": "binary",
        },
        "divide": {
            "symbol": "/",
            "meaning": "Division.",
            "operator_arity": "binary",
        },
        "modulo": {
            "symbol": "%",
            "meaning": "Modulo / remainder.",
            "operator_arity": "binary",
        },
        "add": {
            "symbol": "+",
            "meaning": "Addition.",
            "operator_arity": "binary",
        },
        "subtract": {
            "symbol": "-",
            "meaning": "Subtraction.",
            "operator_arity": "binary",
        },
    },
    "comparison_operators": {
        "basic_comparison_operators": {
            "greater_than": {
                "symbol": ">",
                "meaning": "Greater than.",
                "operator_arity": "binary",
            },
            "greater_equal": {
                "symbol": ">=",
                "meaning": "Greater than or equal to.",
                "operator_arity": "binary",
            },
            "less_than": {
                "symbol": "<",
                "meaning": "Less than.",
                "operator_arity": "binary",
            },
            "less_equal": {
                "symbol": "<=",
                "meaning": "Less than or equal to.",
                "operator_arity": "binary",
            },
            "equal": {
                "symbol": "==",
                "meaning": "Equal to.",
                "operator_arity": "binary",
            },
            "not_equal_1": {
                "symbol": "!=",
                "meaning": "Not equal to.",
                "operator_arity": "binary",
            },
            "not_equal_2": {
                "symbol": "<>",
                "meaning": "Not equal to.",
                "operator_arity": "binary",
            },
        },
        "advanced_comparison_operators": {
            "between": {
                "symbol": "BETWEEN ... AND ...",
                "meaning": "Within the specified range.",
                "operator_arity": "binary",
            },
            "not_between": {
                "symbol": "NOT BETWEEN ... AND ...",
                "meaning": "Not within the specified range.",
                "operator_arity": "binary",
            },
            "like": {
                "symbol": "LIKE",
                "meaning": "Matches a simple pattern.",
                "operator_arity": "binary",
            },
            "not_like": {
                "symbol": "NOT LIKE",
                "meaning": "Does not match a simple pattern.",
                "operator_arity": "binary",
            },
            "regexp": {
                "symbol": "REGEXP",
                "meaning": "Matches a regular expression.",
                "operator_arity": "binary",
            },
            "not_regexp": {
                "symbol": "NOT REGEXP",
                "meaning": "Does not match a regular expression.",
                "operator_arity": "binary",
            },
            "is_null": {
                "symbol": "IS NULL",
                "meaning": "Is null.",
                "operator_arity": "unary",
            },
            "is_not_null": {
                "symbol": "IS NOT NULL",
                "meaning": "Is not null.",
                "operator_arity": "unary",
            },
            "in": {
                "symbol": "IN",
                "meaning": "Is one of the values in the specified list.",
                "operator_arity": "binary",
            },
            "contains": {
                "symbol": "CONTAINS",
                "meaning": "Is one of the values in the specified list.",
                "operator_arity": "binary",
            },
            "not_in": {
                "symbol": "NOT IN",
                "meaning": "Is not one of the values in the specified list.",
                "operator_arity": "binary",
            },
            "not_contains": {
                "symbol": "NOT CONTAINS",
                "meaning": "Is not one of the values in the specified list.",
                "operator_arity": "binary",
            },
        },
    },
    "logical_operators": {
        "not_1": {
            "symbol": "NOT",
            "meaning": "Logical negation.",
            "operator_arity": "unary",
        },
        "not_2": {
            "symbol": "!",
            "meaning": "Logical negation.",
            "operator_arity": "unary",
        },
        "and_1": {
            "symbol": "AND",
            "meaning": "Logical conjunction.",
            "operator_arity": "binary",
        },
        "and_2": {
            "symbol": "&",
            "meaning": "Logical conjunction.",
            "operator_arity": "binary",
        },
        "and_3": {
            "symbol": "&&",
            "meaning": "Logical conjunction.",
            "operator_arity": "binary",
        },
        "or_1": {
            "symbol": "OR",
            "meaning": "Logical disjunction.",
            "operator_arity": "binary",
        },
        "or_2": {
            "symbol": "|",
            "meaning": "Logical disjunction.",
            "operator_arity": "binary",
        },
        "or_3": {
            "symbol": "||",
            "meaning": "Logical disjunction.",
            "operator_arity": "binary",
        },
    },
}


def _flatten_iotdb_operator_definitions(
    operator_tree: dict[str, Any],
    category_path: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    flattened: dict[str, dict[str, Any]] = {}
    for key, value in operator_tree.items():
        if not isinstance(value, dict):
            continue
        if "symbol" in value and "meaning" in value:
            flattened[key] = {
                "operator_name": key,
                "category_path": list(category_path),
                "symbol": value["symbol"],
                "meaning": value["meaning"],
                **(
                    {"operator_arity": value["operator_arity"]}
                    if "operator_arity" in value
                    else {}
                ),
            }
            continue
        flattened.update(
            _flatten_iotdb_operator_definitions(value, category_path + (key,))
        )
    return flattened


IOTDB_OPERATOR_NAME_INDEX = _flatten_iotdb_operator_definitions(IOTDB_OPERATORS)


IOTDB_OPERATOR_SYMBOL_INDEX = {
    symbol: tuple(
        operator_name
        for operator_name, definition in IOTDB_OPERATOR_NAME_INDEX.items()
        if definition["symbol"] == symbol
    )
    for symbol in {
        definition["symbol"] for definition in IOTDB_OPERATOR_NAME_INDEX.values()
    }
}


BUILTIN_AGGREGATE_FUNCTION_DEFINITIONS = {
    "SUM": {
        "function_name": "SUM",
        "description": "Computes the sum of a group of numeric time-series values.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "COUNT": {
        "function_name": "COUNT",
        "description": "Counts the number of data points.",
        "parameters": ["expr"],
        "allowed_input_types": ["ALL_TYPES"],
        "required_attribute_parameters": None,
        "output_type": "INT",
    },
    "AVG": {
        "function_name": "AVG",
        "description": (
            "Computes the arithmetic average of a group of numeric time-series "
            "values."
        ),
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "STDDEV": {
        "function_name": "STDDEV",
        "description": "Alias of STDDEV_SAMP. Computes the sample standard deviation.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "STDDEV_POP": {
        "function_name": "STDDEV_POP",
        "description": "Computes the population standard deviation.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "STDDEV_SAMP": {
        "function_name": "STDDEV_SAMP",
        "description": "Computes the sample standard deviation.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "VARIANCE": {
        "function_name": "VARIANCE",
        "description": "Alias of VAR_SAMP. Computes the sample variance.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "VAR_POP": {
        "function_name": "VAR_POP",
        "description": "Computes the population variance.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "VAR_SAMP": {
        "function_name": "VAR_SAMP",
        "description": "Computes the sample variance.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "EXTREME": {
        "function_name": "EXTREME",
        "description": (
            "Returns the value with the largest absolute magnitude. If the largest "
            "positive and negative absolute values are equal, the positive value "
            "is returned."
        ),
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
    "MAX_VALUE": {
        "function_name": "MAX_VALUE",
        "description": "Returns the maximum value among selected data points.",
        "parameters": ["expr"],
        "allowed_input_types": [
            "INT32",
            "INT64",
            "FLOAT",
            "DOUBLE",
            "STRING",
            "TIMESTAMP",
            "DATE",
        ],
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
    "MIN_VALUE": {
        "function_name": "MIN_VALUE",
        "description": "Returns the minimum value among selected data points.",
        "parameters": ["expr"],
        "allowed_input_types": [
            "INT32",
            "INT64",
            "FLOAT",
            "DOUBLE",
            "STRING",
            "TIMESTAMP",
            "DATE",
        ],
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
    "FIRST_VALUE": {
        "function_name": "FIRST_VALUE",
        "description": "Returns the value with the smallest timestamp.",
        "parameters": ["expr"],
        "allowed_input_types": ["ALL_TYPES"],
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
    "LAST_VALUE": {
        "function_name": "LAST_VALUE",
        "description": "Returns the value with the largest timestamp.",
        "parameters": ["expr"],
        "allowed_input_types": ["ALL_TYPES"],
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
    "MAX_TIME": {
        "function_name": "MAX_TIME",
        "description": "Returns the largest timestamp among selected data points.",
        "parameters": ["expr"],
        "allowed_input_types": ["ALL_TYPES"],
        "required_attribute_parameters": None,
        "output_type": "TIMESTAMP",
    },
    "MIN_TIME": {
        "function_name": "MIN_TIME",
        "description": "Returns the smallest timestamp among selected data points.",
        "parameters": ["expr"],
        "allowed_input_types": ["ALL_TYPES"],
        "required_attribute_parameters": None,
        "output_type": "TIMESTAMP",
    },
    "MAX_BY": {
        "function_name": "MAX_BY",
        "description": (
            "MAX_BY(x, y) returns the value of x corresponding to the maximum "
            "value of y. MAX_BY(time, x) returns the timestamp corresponding to "
            "the maximum value of x."
        ),
        "parameters": ["x", "y"],
        "allowed_input_types": {
            "x": ["ANY_TYPE"],
            "y": [
                "INT32",
                "INT64",
                "FLOAT",
                "DOUBLE",
                "STRING",
                "TIMESTAMP",
                "DATE",
            ],
        },
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_FIRST_INPUT",
    },
    "MIN_BY": {
        "function_name": "MIN_BY",
        "description": (
            "MIN_BY(x, y) returns the value of x corresponding to the minimum "
            "value of y. MIN_BY(time, x) returns the timestamp corresponding to "
            "the minimum value of x."
        ),
        "parameters": ["x", "y"],
        "allowed_input_types": {
            "x": ["ANY_TYPE"],
            "y": [
                "INT32",
                "INT64",
                "FLOAT",
                "DOUBLE",
                "STRING",
                "TIMESTAMP",
                "DATE",
            ],
        },
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_FIRST_INPUT",
    },
}


UDF_AGGREGATE_FUNCTION_DEFINITIONS = {
    # ------------------------------------------------------------------
    # 1. Integral and time-weighted profiling aggregation
    # ------------------------------------------------------------------
    "INTEGRAL": {
        "function_name": "INTEGRAL",
        "description": (
            "Computes the numerical integral of a time series, i.e., the area "
            "under the piecewise-linear curve formed by timestamp-value points."
        ),
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "unit": (
                "Time-axis unit used for integration. Allowed values are "
                "'1S', '1s', '1m', '1H', and '1d'. Default is '1s'."
            )
        },
        "output_type": "DOUBLE",
    },
    "INTEGRALAVG": {
        "function_name": "INTEGRALAVG",
        "description": (
            "Computes the time-weighted average of a time series, defined as "
            "the numerical integral divided by the total time span."
        ),
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },

    # ------------------------------------------------------------------
    # 2. Robust statistical aggregation
    # ------------------------------------------------------------------
    "MAD": {
        "function_name": "MAD",
        "description": (
            "Computes the exact or approximate median absolute deviation of "
            "a numeric time series."
        ),
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "error": (
                "Value-based relative error for approximate MAD. Range: [0, 1). "
                "Default is 0, which means exact MAD."
            )
        },
        "output_type": "DOUBLE",
    },
    "MEDIAN": {
        "function_name": "MEDIAN",
        "description": (
            "Computes the exact or approximate median of a numeric time series."
        ),
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "error": (
                "Rank-based relative error for approximate median. Range: [0, 1). "
                "Default is 0, which means exact median."
            )
        },
        "output_type": "DOUBLE",
    },
    "PERCENTILE": {
        "function_name": "PERCENTILE",
        "description": (
            "Computes the exact or approximate percentile of a numeric time series."
        ),
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "rank": (
                "Target percentile rank in all data. Range: (0, 1]. "
                "Default is 0.5."
            ),
            "error": (
                "Rank-based relative error for approximate percentile. "
                "Range: [0, 1). Default is 0, which means exact percentile."
            ),
        },
        "output_type": "SAME_AS_INPUT",
    },
    "QUANTILE": {
        "function_name": "QUANTILE",
        "description": (
            "Computes an approximate quantile of a numeric time series based on "
            "the KLL sketch algorithm."
        ),
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "rank": (
                "Target quantile rank in all data. Range: (0, 1]. "
                "Default is 0.5."
            ),
            "K": (
                "Allowed KLL sketch size. Minimum value is 100. "
                "Default is 800."
            ),
        },
        "output_type": "SAME_AS_INPUT",
    },

    # ------------------------------------------------------------------
    # 3. Period, distribution shape, and spread aggregation
    # ------------------------------------------------------------------
    "PERIOD": {
        "function_name": "PERIOD",
        "description": "Computes the period of a single numeric time series.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "INT32",
    },
    "SKEW": {
        "function_name": "SKEW",
        "description": (
            "Computes the population skewness of a single numeric time series."
        ),
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "SPREAD": {
        "function_name": "SPREAD",
        "description": (
            "Computes the value spread of a time series, i.e., maximum value "
            "minus minimum value."
        ),
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },

    # ------------------------------------------------------------------
    # 4. Whole-series matching, similarity, and correlation aggregation
    # ------------------------------------------------------------------
    "COV": {
        "function_name": "COV",
        "description": (
            "Computes the population covariance between two numeric time series."
        ),
        "parameters": ["expr1", "expr2"],
        "allowed_input_types": {
            "expr1": ["INT32", "INT64", "FLOAT", "DOUBLE"],
            "expr2": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        },
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "DTW": {
        "function_name": "DTW",
        "description": (
            "Computes the Dynamic Time Warping distance between two numeric "
            "time series."
        ),
        "parameters": ["expr1", "expr2"],
        "allowed_input_types": {
            "expr1": ["INT32", "INT64", "FLOAT", "DOUBLE"],
            "expr2": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        },
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "PEARSON": {
        "function_name": "PEARSON",
        "description": (
            "Computes the Pearson correlation coefficient between two numeric "
            "time series."
        ),
        "parameters": ["expr1", "expr2"],
        "allowed_input_types": {
            "expr1": ["INT32", "INT64", "FLOAT", "DOUBLE"],
            "expr2": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        },
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
}


BUILTIN_TIMESERIES_FUNCTION_DEFINITIONS = {
    # ------------------------------------------------------------------
    # 1. Mathematical transformation functions
    # ------------------------------------------------------------------
    "SIN": {
        "function_name": "SIN",
        "description": "Computes the sine value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "COS": {
        "function_name": "COS",
        "description": "Computes the cosine value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "TAN": {
        "function_name": "TAN",
        "description": "Computes the tangent value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "ASIN": {
        "function_name": "ASIN",
        "description": "Computes the arc sine value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "ACOS": {
        "function_name": "ACOS",
        "description": "Computes the arc cosine value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "ATAN": {
        "function_name": "ATAN",
        "description": "Computes the arc tangent value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "SINH": {
        "function_name": "SINH",
        "description": "Computes the hyperbolic sine value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "COSH": {
        "function_name": "COSH",
        "description": "Computes the hyperbolic cosine value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "TANH": {
        "function_name": "TANH",
        "description": "Computes the hyperbolic tangent value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "DEGREES": {
        "function_name": "DEGREES",
        "description": "Converts each numeric input value from radians to degrees.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "RADIANS": {
        "function_name": "RADIANS",
        "description": "Converts each numeric input value from degrees to radians.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "ABS": {
        "function_name": "ABS",
        "description": "Computes the absolute value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
    "SIGN": {
        "function_name": "SIGN",
        "description": "Returns the sign of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "CEIL": {
        "function_name": "CEIL",
        "description": "Returns the smallest integer value greater than or equal to each input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "FLOOR": {
        "function_name": "FLOOR",
        "description": "Returns the largest integer value less than or equal to each input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "ROUND": {
        "function_name": "ROUND",
        "description": "Rounds each numeric input point according to the specified number of valid decimal or integer digits.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": {
            "places": "Number of valid digits for rounding. Positive values indicate decimal places; negative values indicate integer positions."
        },
        "output_type": "DOUBLE",
    },
    "EXP": {
        "function_name": "EXP",
        "description": "Computes the exponential value of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "LN": {
        "function_name": "LN",
        "description": "Computes the natural logarithm of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "LOG10": {
        "function_name": "LOG10",
        "description": "Computes the base-10 logarithm of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "SQRT": {
        "function_name": "SQRT",
        "description": "Computes the square root of each numeric input point.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },

    # ------------------------------------------------------------------
    # 2. Comparison / state-conversion functions
    # ------------------------------------------------------------------
    "ON_OFF": {
        "function_name": "ON_OFF",
        "description": "Converts a numeric series into a boolean series according to whether each value is greater than or equal to a threshold.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": {
            "threshold": "DOUBLE threshold. Output is true when ts_value >= threshold."
        },
        "output_type": "BOOLEAN",
    },
    "IN_RANGE": {
        "function_name": "IN_RANGE",
        "description": "Converts a numeric series into a boolean series according to whether each value falls within a closed interval.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": {
            "lower": "DOUBLE lower bound.",
            "upper": "DOUBLE upper bound.",
        },
        "output_type": "BOOLEAN",
    },

    # ------------------------------------------------------------------
    # 3. String-processing functions
    # ------------------------------------------------------------------
    "STRING_CONTAINS": {
        "function_name": "STRING_CONTAINS",
        "description": "Checks whether each string value contains the specified substring.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": {"s": "Substring to search for."},
        "output_type": "BOOLEAN",
    },
    "STRING_MATCHES": {
        "function_name": "STRING_MATCHES",
        "description": "Checks whether each string value matches the specified Java-style regular expression.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": {"regex": "Java-style regular expression."},
        "output_type": "BOOLEAN",
    },
    "LENGTH": {
        "function_name": "LENGTH",
        "description": "Returns the length of each string value.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": None,
        "output_type": "INT32",
    },
    "LOCATE": {
        "function_name": "LOCATE",
        "description": "Returns the first position of the target substring in each input string, or -1 if the target is not found.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": {"target": "Substring to locate."},
        "optional_attribute_parameters": {
            "reverse": "Whether to locate from right to left. Default is false."
        },
        "output_type": "INT32",
    },
    "STARTSWITH": {
        "function_name": "STARTSWITH",
        "description": "Checks whether each string value starts with the specified prefix.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": {"target": "Prefix to match."},
        "output_type": "BOOLEAN",
    },
    "ENDSWITH": {
        "function_name": "ENDSWITH",
        "description": "Checks whether each string value ends with the specified suffix.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": {"target": "Suffix to match."},
        "output_type": "BOOLEAN",
    },
    "CONCAT": {
        "function_name": "CONCAT",
        "description": "Concatenates each input string value with one or more target strings.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": {
            "targets": "A set of key-value attributes whose keys start with target and are not duplicated."
        },
        "optional_attribute_parameters": {
            "series_behind": "Whether the input series value should be placed after the target strings. Default is false."
        },
        "output_type": "TEXT",
    },
    "SUBSTRING": {
        "function_name": "SUBSTRING",
        "description": "Extracts a substring from each input string starting from the specified position and stopping after the specified length.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": {
            "from": "Start position of the substring. Index starts from 1.",
            "for": "Number of characters to extract.",
        },
        "output_type": "TEXT",
    },
    "REPLACE": {
        "function_name": "REPLACE",
        "description": "Replaces the target substring in each input string with another substring.",
        "parameters": ["expr", "target", "replacement"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": None,
        "output_type": "TEXT",
    },
    "UPPER": {
        "function_name": "UPPER",
        "description": "Converts each string value to uppercase.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": None,
        "output_type": "TEXT",
    },
    "LOWER": {
        "function_name": "LOWER",
        "description": "Converts each string value to lowercase.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": None,
        "output_type": "TEXT",
    },
    "TRIM": {
        "function_name": "TRIM",
        "description": "Removes leading and trailing spaces from each string value.",
        "parameters": ["expr"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": None,
        "output_type": "TEXT",
    },
    "STRCMP": {
        "function_name": "STRCMP",
        "description": "Compares two input string series. Returns 0 if equal, a negative value if the first value is smaller, and a positive value if the first value is larger.",
        "parameters": ["expr1", "expr2"],
        "allowed_input_types": ["TEXT", "STRING"],
        "required_attribute_parameters": None,
        "output_type": "TEXT",
    },

    # ------------------------------------------------------------------
    # 4. Type-conversion function
    # ------------------------------------------------------------------
    "CAST": {
        "function_name": "CAST",
        "description": "Casts input values into the target output type.",
        "parameters": ["expr"],
        "allowed_input_types": ["ANY_TYPE"],
        "required_attribute_parameters": {
            "type": "Target output type. Allowed values: INT32, INT64, FLOAT, DOUBLE, BOOLEAN, TEXT."
        },
        "output_type": "DETERMINED_BY_TYPE_PARAMETER",
    },

    # ------------------------------------------------------------------
    # 5. Constant-series generation functions
    # ------------------------------------------------------------------
    "CONST": {
        "function_name": "CONST",
        "description": (
            "Generates a constant time series according to the specified value "
            "and type. Input series are used as timestamp anchors for generating "
            "the output constant series."
        ),
        "parameters": ["expr", "..."],
        "allowed_input_types": ["ANY_TYPE"],
        "required_attribute_parameters": {
            "value": "Output data point value.",
            "type": "Output data point type. Allowed values: INT32, INT64, FLOAT, DOUBLE, BOOLEAN, TEXT.",
        },
        "output_type": "DETERMINED_BY_TYPE_PARAMETER",
    },
    "PI": {
        "function_name": "PI",
        "description": (
            "Generates a constant time series whose value is pi. Input series "
            "are used as timestamp anchors for generating the output constant "
            "series."
        ),
        "parameters": ["expr", "..."],
        "allowed_input_types": ["ANY_TYPE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "E": {
        "function_name": "E",
        "description": (
            "Generates a constant time series whose value is Euler's number e. "
            "Input series are used as timestamp anchors for generating the "
            "output constant series."
        ),
        "parameters": ["expr", "..."],
        "allowed_input_types": ["ANY_TYPE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },

    # ------------------------------------------------------------------
    # 6. Selection functions
    # ------------------------------------------------------------------
    "TOP_K": {
        "function_name": "TOP_K",
        "description": "Returns at most k data points with the largest values. If more than k points are tied, the points with the smallest timestamps are returned.",
        "parameters": ["expr"],
        "allowed_input_types": [
            "INT32",
            "INT64",
            "FLOAT",
            "DOUBLE",
            "TEXT",
            "STRING",
            "DATE",
            "TIMESTAMP",
        ],
        "required_attribute_parameters": {
            "k": "Maximum number of selected points. Must be greater than 0 and less than or equal to 1000."
        },
        "output_type": "SAME_AS_INPUT",
    },
    "BOTTOM_K": {
        "function_name": "BOTTOM_K",
        "description": "Returns at most k data points with the smallest values. If more than k points are tied, the points with the smallest timestamps are returned.",
        "parameters": ["expr"],
        "allowed_input_types": [
            "INT32",
            "INT64",
            "FLOAT",
            "DOUBLE",
            "TEXT",
            "STRING",
            "DATE",
            "TIMESTAMP",
        ],
        "required_attribute_parameters": {
            "k": "Maximum number of selected points. Must be greater than 0 and less than or equal to 1000."
        },
        "output_type": "SAME_AS_INPUT",
    },

    # ------------------------------------------------------------------
    # 7. Interval query functions
    # ------------------------------------------------------------------
    "ZERO_DURATION": {
        "function_name": "ZERO_DURATION",
        "description": "Returns the start time and duration of continuous zero or false intervals whose duration satisfies the specified range.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE", "BOOLEAN"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "min": "Minimum duration in milliseconds. Default is 0.",
            "max": "Maximum duration in milliseconds. Default is Long.MAX_VALUE.",
        },
        "output_type": "LONG",
    },
    "NON_ZERO_DURATION": {
        "function_name": "NON_ZERO_DURATION",
        "description": "Returns the start time and duration of continuous non-zero or true intervals whose duration satisfies the specified range.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE", "BOOLEAN"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "min": "Minimum duration in milliseconds. Default is 0.",
            "max": "Maximum duration in milliseconds. Default is Long.MAX_VALUE.",
        },
        "output_type": "LONG",
    },
    "ZERO_COUNT": {
        "function_name": "ZERO_COUNT",
        "description": "Returns the start time and number of consecutive zero or false points whose count satisfies the specified range.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE", "BOOLEAN"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "min": "Minimum number of consecutive points. Default is 1.",
            "max": "Maximum number of consecutive points. Default is Long.MAX_VALUE.",
        },
        "output_type": "LONG",
    },
    "NON_ZERO_COUNT": {
        "function_name": "NON_ZERO_COUNT",
        "description": "Returns the start time and number of consecutive non-zero or true points whose count satisfies the specified range.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE", "BOOLEAN"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "min": "Minimum number of consecutive points. Default is 1.",
            "max": "Maximum number of consecutive points. Default is Long.MAX_VALUE.",
        },
        "output_type": "LONG",
    },

    # ------------------------------------------------------------------
    # 8. Trend calculation functions
    # ------------------------------------------------------------------
    "TIME_DIFFERENCE": {
        "function_name": "TIME_DIFFERENCE",
        "description": "Computes the timestamp difference between each data point and its previous data point. The first point in the range has no output.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE", "BOOLEAN", "TEXT"],
        "required_attribute_parameters": None,
        "output_type": "INT64",
    },
    "DIFFERENCE": {
        "function_name": "DIFFERENCE",
        "description": "Computes the value difference between each numeric data point and its previous data point. The first point in the range has no output.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
    "NON_NEGATIVE_DIFFERENCE": {
        "function_name": "NON_NEGATIVE_DIFFERENCE",
        "description": "Computes the absolute value difference between each numeric data point and its previous data point. The first point in the range has no output.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
    "DERIVATIVE": {
        "function_name": "DERIVATIVE",
        "description": "Computes the rate of change of each numeric point relative to the previous point, equivalent to DIFFERENCE divided by TIME_DIFFERENCE.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "NON_NEGATIVE_DERIVATIVE": {
        "function_name": "NON_NEGATIVE_DERIVATIVE",
        "description": "Computes the absolute rate of change of each numeric point relative to the previous point, equivalent to NON_NEGATIVE_DIFFERENCE divided by TIME_DIFFERENCE.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "DIFF": {
        "function_name": "DIFF",
        "description": "Computes the difference between each numeric data point and the previous data point. The first point has no previous value and outputs null.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "ignoreNull": "Whether to ignore null previous values. Default is true."
        },
        "output_type": "DOUBLE",
    },

    # ------------------------------------------------------------------
    # 9. Sampling / downsampling functions
    # ------------------------------------------------------------------
    "EQUAL_SIZE_BUCKET_RANDOM_SAMPLE": {
        "function_name": "EQUAL_SIZE_BUCKET_RANDOM_SAMPLE",
        "description": "Performs equal-size bucket random sampling according to the specified sampling proportion.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "proportion": "Downsampling ratio in (0, 1]. Default is 0.1."
        },
        "output_type": "SAME_AS_INPUT",
    },
    "EQUAL_SIZE_BUCKET_AGG_SAMPLE": {
        "function_name": "EQUAL_SIZE_BUCKET_AGG_SAMPLE",
        "description": "Performs equal-size bucket aggregate sampling according to the specified sampling proportion and aggregation type.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "proportion": "Downsampling ratio in (0, 1]. Default is 0.1.",
            "type": "Aggregation type. Allowed values: avg, max, min, sum, extreme, variance. Default is avg.",
        },
        "output_type": "SAME_AS_INPUT",
    },
    "EQUAL_SIZE_BUCKET_M4_SAMPLE": {
        "function_name": "EQUAL_SIZE_BUCKET_M4_SAMPLE",
        "description": "Performs equal-size bucket M4 sampling according to the specified sampling proportion.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "proportion": "Downsampling ratio in (0, 1]. Default is 0.1."
        },
        "output_type": "SAME_AS_INPUT",
    },
    "EQUAL_SIZE_BUCKET_OUTLIER_SAMPLE": {
        "function_name": "EQUAL_SIZE_BUCKET_OUTLIER_SAMPLE",
        "description": "Performs equal-size bucket outlier sampling according to the specified sampling proportion, outlier type, and number of selected points per bucket.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "proportion": "Downsampling ratio in (0, 1]. Default is 0.1.",
            "type": "Outlier selection type. Allowed values include avg, stendis, cos, and prenextdis. Default is avg.",
            "number": "Number of selected outlier points in each bucket. Must be greater than 0. Default is 3.",
        },
        "output_type": "SAME_AS_INPUT",
    },
    "M4": {
        "function_name": "M4",
        "description": "Returns representative points in each window, including first, last, bottom, and top points. Points are sorted by timestamp and deduplicated before output.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": {
            "window_mode": "Either fixed-point window mode or sliding-time-window mode must be specified."
        },
        "attribute_parameter_options": {
            "fixed_point_window": {
                "windowSize": "Number of points in each fixed-size window.",
                "slidingStep": "Sliding step for the fixed-size window.",
            },
            "sliding_time_window": {
                "timeInterval": "Time length of each window.",
                "slidingStep": "Sliding step of the time window.",
                "displayWindowBegin": "Display window start time.",
                "displayWindowEnd": "Display window end time.",
            },
        },
        "output_type": "SAME_AS_INPUT",
    },

    # ------------------------------------------------------------------
    # 10. Time-series processing function
    # ------------------------------------------------------------------
    "CHANGE_POINTS": {
        "function_name": "CHANGE_POINTS",
        "description": "Removes continuous identical values from the input series and keeps change points.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
}


BUILTIN_TIMESERIES_SCALAR_COMPATIBLE_FLAGS = {
    "SIN": True,
    "COS": True,
    "TAN": True,
    "ASIN": True,
    "ACOS": True,
    "ATAN": True,
    "SINH": True,
    "COSH": True,
    "TANH": True,
    "DEGREES": True,
    "RADIANS": True,
    "ABS": True,
    "SIGN": True,
    "CEIL": True,
    "FLOOR": True,
    "ROUND": True,
    "EXP": True,
    "LN": True,
    "LOG10": True,
    "SQRT": True,
    "ON_OFF": True,
    "IN_RANGE": True,
    "STRING_CONTAINS": True,
    "STRING_MATCHES": True,
    "LENGTH": True,
    "LOCATE": True,
    "STARTSWITH": True,
    "ENDSWITH": True,
    "CONCAT": True,
    "SUBSTRING": True,
    "REPLACE": True,
    "UPPER": True,
    "LOWER": True,
    "TRIM": True,
    "STRCMP": True,
    "CAST": True,
    "CONST": False,
    "PI": False,
    "E": False,
    "TOP_K": False,
    "BOTTOM_K": False,
    "ZERO_DURATION": False,
    "NON_ZERO_DURATION": False,
    "ZERO_COUNT": False,
    "NON_ZERO_COUNT": False,
    "TIME_DIFFERENCE": False,
    "DIFFERENCE": False,
    "NON_NEGATIVE_DIFFERENCE": False,
    "DERIVATIVE": False,
    "NON_NEGATIVE_DERIVATIVE": False,
    "DIFF": False,
    "EQUAL_SIZE_BUCKET_RANDOM_SAMPLE": False,
    "EQUAL_SIZE_BUCKET_AGG_SAMPLE": False,
    "EQUAL_SIZE_BUCKET_M4_SAMPLE": False,
    "EQUAL_SIZE_BUCKET_OUTLIER_SAMPLE": False,
    "M4": False,
    "CHANGE_POINTS": False,
}


_missing_scalar_compatible_flags = (
    set(BUILTIN_TIMESERIES_FUNCTION_DEFINITIONS)
    - set(BUILTIN_TIMESERIES_SCALAR_COMPATIBLE_FLAGS)
)
if _missing_scalar_compatible_flags:
    raise ValueError(
        "Missing scalar_compatible flags for built-in time-series functions: "
        f"{sorted(_missing_scalar_compatible_flags)}"
    )

for _function_name, _definition in BUILTIN_TIMESERIES_FUNCTION_DEFINITIONS.items():
    _definition["scalar_compatible"] = BUILTIN_TIMESERIES_SCALAR_COMPATIBLE_FLAGS[
        _function_name
    ]


UDF_TIMESERIES_FUNCTION_DEFINITIONS = {
    # ------------------------------------------------------------------
    # 1. Data quality functions
    # ------------------------------------------------------------------
    "COMPLETENESS": {
        "function_name": "COMPLETENESS",
        "description": "Computes time-series completeness over non-overlapping windows, measuring whether data points are missing.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "window": "Window size. It can be a positive integer number of points or a duration with unit ms/s/m/h/d. If omitted, all input data belong to one window.",
            "downtime": "Whether downtime anomalies are considered. Allowed values: true/false. Default is true.",
        },
        "output_type": "DOUBLE",
    },
    "CONSISTENCY": {
        "function_name": "CONSISTENCY",
        "description": "Computes time-series consistency over non-overlapping windows, measuring whether the series changes smoothly and regularly.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "window": "Window size. It can be a positive integer number of points or a duration with unit ms/s/m/h/d. If omitted, all input data belong to one window.",
        },
        "output_type": "DOUBLE",
    },
    "TIMELINESS": {
        "function_name": "TIMELINESS",
        "description": "Computes time-series timeliness over non-overlapping windows, measuring whether data are collected or reported on time.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "window": "Window size. It can be a positive integer number of points or a duration with unit ms/s/m/h/d. If omitted, all input data belong to one window.",
        },
        "output_type": "DOUBLE",
    },
    "VALIDITY": {
        "function_name": "VALIDITY",
        "description": "Computes time-series validity over non-overlapping windows, measuring whether data are normal, usable, and free of abnormal values.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "window": "Window size. It can be a positive integer number of points or a duration with unit ms/s/m/h/d. If omitted, all input data belong to one window.",
        },
        "output_type": "DOUBLE",
    },

    # ------------------------------------------------------------------
    # 2. Data profiling and preprocessing functions
    # ------------------------------------------------------------------
    "ACF": {
        "function_name": "ACF",
        "description": "Computes the autocorrelation function of a single time series.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "PACF": {
        "function_name": "PACF",
        "description": "Computes the partial autocorrelation coefficients by solving the Yule-Walker equations.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "lag": "Maximum lag order. Default is min(10*log10(n), n-1), where n is the number of points.",
        },
        "output_type": "DOUBLE",
    },
    "HISTOGRAM": {
        "function_name": "HISTOGRAM",
        "description": "Computes the distribution histogram of a single numeric time series.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "min": "Lower bound of the value range. Default is -Double.MAX_VALUE.",
            "max": "Upper bound of the value range. Default is Double.MAX_VALUE.",
            "count": "Number of histogram buckets. It must be a positive integer. Default is 1.",
        },
        "output_type": "BUCKET_COUNT_SEQUENCE",
    },
    "DISTINCT": {
        "function_name": "DISTINCT",
        "description": "Returns all distinct elements that appear in the input series.",
        "parameters": ["expr"],
        "allowed_input_types": ["ANY_TYPE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
    "QLB": {
        "function_name": "QLB",
        "description": "Computes the Ljung-Box Q statistic p-values. Smaller p-values indicate the series is more likely to be non-stationary.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "lag": "Maximum lag order used in the calculation. It should be an integer between 1 and n-2. Default is n-2.",
        },
        "output_type": "DOUBLE",
    },
    "MINMAX": {
        "function_name": "MINMAX",
        "description": "Normalizes the input series using min-max normalization, mapping the minimum value to 0 and the maximum value to 1.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "compute": "Computation mode. Allowed values: batch or stream. Default is batch.",
            "min": "Minimum value used in stream mode.",
            "max": "Maximum value used in stream mode.",
        },
        "conditional_required_parameters": {
            "if compute == stream": ["min", "max"],
        },
        "output_type": "DOUBLE",
    },
    "ZSCORE": {
        "function_name": "ZSCORE",
        "description": "Normalizes the input series using z-score normalization.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "compute": "Computation mode. Allowed values: batch or stream. Default is batch.",
            "avg": "Mean value used in stream mode.",
            "sd": "Standard deviation used in stream mode.",
        },
        "conditional_required_parameters": {
            "if compute == stream": ["avg", "sd"],
        },
        "output_type": "DOUBLE",
    },
    "MVAVG": {
        "function_name": "MVAVG",
        "description": "Computes the moving average of the input numeric series.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "window": "Moving window length. Default is 10.",
        },
        "output_type": "DOUBLE",
    },
    "RESAMPLE": {
        "function_name": "RESAMPLE",
        "description": "Resamples the input series at a specified frequency, supporting both upsampling and downsampling.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": {
            "every": "Resampling frequency. It must be a positive duration with unit ms/s/m/h/d.",
        },
        "optional_attribute_parameters": {
            "interp": "Interpolation method for upsampling. Allowed values: NaN, FFill, BFill, Linear. Default is NaN.",
            "aggr": "Aggregation method for downsampling. Allowed values: Max, Min, First, Last, Mean, Median. Default is Mean.",
            "start": "Inclusive resampling start time, formatted as yyyy-MM-dd HH:mm:ss. Default is the timestamp of the first valid point.",
            "end": "Exclusive resampling end time, formatted as yyyy-MM-dd HH:mm:ss. Default is the timestamp of the last valid point.",
        },
        "output_type": "DOUBLE",
    },
    "SAMPLE": {
        "function_name": "SAMPLE",
        "description": "Samples a specified number of points from the input series using reservoir, isometric, or triangle sampling.",
        "parameters": ["expr"],
        "allowed_input_types": ["ANY_TYPE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "method": "Sampling method. Allowed values: reservoir, isometric, triangle. Default is reservoir.",
            "k": "Number of sampled points. It must be a positive integer. Default is 1.",
        },
        "output_type": "SAME_AS_INPUT",
    },
    "SEGMENT": {
        "function_name": "SEGMENT",
        "description": "Segments the input series according to linear trends and outputs fitted values of each segment.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "output": "Output mode. 'all' outputs all fitted values; 'first' outputs the fitted value at each segment start. Default is first.",
            "error": "Allowed fitting error threshold for linear trend detection. Default is 0.1.",
        },
        "output_type": "DOUBLE",
    },
    "SPLINE": {
        "function_name": "SPLINE",
        "description": "Performs cubic spline interpolation and resampling over the input series.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": {
            "points": "Number of resampled points.",
        },
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },

    # ------------------------------------------------------------------
    # 3. Anomaly detection functions
    # ------------------------------------------------------------------
    "IQR": {
        "function_name": "IQR",
        "description": "Detects distribution anomalies that exceed 1.5 times the interquartile range beyond Q1 or Q3.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "method": "Detection mode. Allowed values: batch or stream. Default is batch.",
            "q1": "First quartile used in stream mode.",
            "q3": "Third quartile used in stream mode.",
        },
        "conditional_required_parameters": {
            "if method == stream": ["q1", "q3"],
        },
        "output_type": "DOUBLE",
    },
    "KSIGMA": {
        "function_name": "KSIGMA",
        "description": "Detects anomalies using a dynamic K-Sigma algorithm. Points farther than k standard deviations from the mean within a window are output as anomalies.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "k": "Standard-deviation multiplier threshold. Default is 3.",
            "window": "Sliding window size. Default is 10000.",
        },
        "output_type": "SAME_AS_INPUT",
    },
    "LOF": {
        "function_name": "LOF",
        "description": "Computes local outlier factor values to detect density-based anomalies.",
        "parameters": ["expr1", "..."],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "method": "Detection method. Default is default for high-dimensional data; series converts a one-dimensional series into high-dimensional windows.",
            "k": "k-distance parameter for local outlier factor. Default is 3.",
            "window": "Read window length. Default is 10000.",
            "windowsize": "Window size used when method=series. Default is 5.",
        },
        "output_type": "DOUBLE",
    },
    "MISSDETECT": {
        "function_name": "MISSDETECT",
        "description": "Detects missing anomalies by finding long perfect linear segments that may be caused by interpolation-filled missing data.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "minlen": "Minimum length of a perfect linear segment to be marked as anomalous. It must be an integer >= 10. Default is 10.",
        },
        "output_type": "BOOLEAN",
    },
    "RANGE": {
        "function_name": "RANGE",
        "description": "Detects range anomalies by outputting points whose values are outside the specified lower and upper bounds.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": {
            "lower_bound": "Lower bound for range anomaly detection.",
            "upper_bound": "Upper bound for range anomaly detection. It must be greater than lower_bound.",
        },
        "optional_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },
    "TWOSIDEDFILTER": {
        "function_name": "TWOSIDEDFILTER",
        "description": "Filters abnormal points from the input series using a two-sided window detection method.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "len": "Two-sided window size. It must be a positive integer. Default is 5.",
            "threshold": "Anomaly degree threshold in (0, 1). Default is 0.3.",
        },
        "output_type": "SAME_AS_INPUT",
    },
    "OUTLIER": {
        "function_name": "OUTLIER",
        "description": "Detects distance-based outliers. A point is considered anomalous if the number of neighbors within distance r in the current window is less than k.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": {
            "r": "Distance threshold.",
            "k": "Density threshold.",
            "w": "Sliding window size.",
            "s": "Sliding window step.",
        },
        "optional_attribute_parameters": None,
        "output_type": "SAME_AS_INPUT",
    },

    # ------------------------------------------------------------------
    # 4. Frequency-domain functions
    # ------------------------------------------------------------------
    "CONV": {
        "function_name": "CONV",
        "description": "Computes the convolution of two numeric input series, equivalent to polynomial multiplication.",
        "parameters": ["expr1", "expr2"],
        "allowed_input_types": {
            "expr1": ["INT32", "INT64", "FLOAT", "DOUBLE"],
            "expr2": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        },
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "DECONV": {
        "function_name": "DECONV",
        "description": "Computes the deconvolution of two numeric input series, equivalent to polynomial division.",
        "parameters": ["expr1", "expr2"],
        "allowed_input_types": {
            "expr1": ["INT32", "INT64", "FLOAT", "DOUBLE"],
            "expr2": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        },
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "result": "Deconvolution result to output. Allowed values: quotient or remainder. Default is quotient.",
        },
        "output_type": "DOUBLE",
    },
    "DWT": {
        "function_name": "DWT",
        "description": "Performs one-dimensional discrete wavelet transform on the input series.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "method": "Wavelet filter type. Allowed values include Haar, DB4, DB6, DB8. Case-insensitive.",
            "coef": "Custom wavelet filter coefficients, separated by commas without spaces.",
            "layer": "Number of transform layers. Default is 1.",
        },
        "conditional_parameter_rule": "Either method or coef should be provided if a custom/default wavelet filter needs to be specified.",
        "output_type": "DOUBLE",
    },
    "IDWT": {
        "function_name": "IDWT",
        "description": "Performs inverse one-dimensional discrete wavelet transform, reconstructing original data from wavelet coefficients.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "method": "Wavelet filter type. Allowed values include Haar, DB4, DB6, DB8. Case-insensitive.",
            "coef": "Custom wavelet filter coefficients, separated by commas without spaces.",
            "layer": "Number of transform layers. Default is 1.",
        },
        "conditional_parameter_rule": "The method/coef/layer settings should be consistent with the corresponding DWT transform.",
        "output_type": "DOUBLE",
    },
    "FFT": {
        "function_name": "FFT",
        "description": "Performs fast Fourier transform on the input numeric series.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "method": "Fourier transform type. Allowed values: uniform or nonuniform. Default is uniform; nonuniform is documented as not implemented.",
            "result": "Output component. Allowed values: real, imag, abs, angle. Default is abs.",
            "compress": "Lossy compression energy-retention ratio in (0, 1]. If omitted, no compression is applied.",
        },
        "output_type": "DOUBLE",
    },
    "IFFT": {
        "function_name": "IFFT",
        "description": "Performs inverse fast Fourier transform by treating two input series as the real and imaginary parts of a complex sequence.",
        "parameters": ["real_expr", "imag_expr"],
        "allowed_input_types": {
            "real_expr": ["INT32", "INT64", "FLOAT", "DOUBLE"],
            "imag_expr": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        },
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "start": "Start time of the output series, formatted as yyyy-MM-dd HH:mm:ss. Default is 1970-01-01 08:00:00.",
            "interval": "Output time interval, as a positive duration with unit ms/s/m/h/d. Default is 1s.",
        },
        "output_type": "DOUBLE",
    },
    "HIGHPASS": {
        "function_name": "HIGHPASS",
        "description": "Applies a high-pass filter to extract components above the normalized cutoff frequency.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": {
            "wpass": "Normalized cutoff frequency in (0, 1).",
        },
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "LOWPASS": {
        "function_name": "LOWPASS",
        "description": "Applies a low-pass filter to extract components below the normalized cutoff frequency.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": {
            "wpass": "Normalized cutoff frequency in (0, 1).",
        },
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },
    "ENVELOPE": {
        "function_name": "ENVELOPE",
        "description": "Performs envelope analysis / demodulation over a numeric signal to extract amplitude variation trends.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "frequency": "Demodulation frequency. It should be positive. If omitted, the system infers it from the time interval.",
            "amplification": "Positive integer amplification factor, useful when the frequency is smaller than 1.",
        },
        "output_type": "DOUBLE",
    },

    # ------------------------------------------------------------------
    # 5. Local pattern matching and cross-correlation
    # ------------------------------------------------------------------
    "PTNSYM": {
        "function_name": "PTNSYM",
        "description": "Finds symmetric subsequences whose symmetry degree is less than or equal to a threshold. Symmetry is computed using DTW.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "window": "Length of the symmetric subsequence. It must be a positive integer. Default is 10.",
            "threshold": "Non-negative symmetry threshold. If omitted, all subsequences are output.",
        },
        "output_type": "DOUBLE",
    },
    "XCORR": {
        "function_name": "XCORR",
        "description": "Computes the cross-correlation function of two numeric time series under different alignments.",
        "parameters": ["expr1", "expr2"],
        "allowed_input_types": {
            "expr1": ["INT32", "INT64", "FLOAT", "DOUBLE"],
            "expr2": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        },
        "required_attribute_parameters": None,
        "optional_attribute_parameters": None,
        "output_type": "DOUBLE",
    },

    # ------------------------------------------------------------------
    # 6. Data repair functions
    # ------------------------------------------------------------------
    "TIMESTAMPREPAIR": {
        "function_name": "TIMESTAMPREPAIR",
        "description": "Repairs unstable timestamps into strictly equal intervals by slightly adjusting timestamps with minimum repair cost.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "interval": "Standard time interval in milliseconds. It must be a positive integer. If omitted, the interval is inferred.",
            "method": "Method for inferring the standard interval when interval is omitted. Allowed values: median, mode, cluster. Default is median.",
        },
        "output_type": "SAME_AS_INPUT",
    },
    "VALUEFILL": {
        "function_name": "VALUEFILL",
        "description": "Fills missing values in a single numeric time series.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "method": "Fill method. Allowed values: mean, previous, linear, likelihood, AR, MA, SCREEN. Default is linear.",
        },
        "output_type": "SAME_AS_INPUT",
    },
    "VALUEREPAIR": {
        "function_name": "VALUEREPAIR",
        "description": "Repairs abnormal values in a numeric time series using either a speed-threshold method or a likelihood-based greedy method.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "method": "Repair method. Allowed values: Screen, LsGreedy. Default is Screen.",
            "minSpeed": "Only valid for Screen. Values with speed smaller than this are repaired. Default is median minus 3*MAD.",
            "maxSpeed": "Only valid for Screen. Values with speed greater than this are repaired. Default is median plus 3*MAD.",
            "center": "Only valid for LsGreedy. Center of the Gaussian model for speed changes. Default is 0.",
            "sigma": "Only valid for LsGreedy. Standard deviation of the Gaussian model for speed changes. Default is MAD.",
        },
        "output_type": "SAME_AS_INPUT",
    },

    # ------------------------------------------------------------------
    # 7. Sequence discovery functions
    # ------------------------------------------------------------------
    "CONSECUTIVESEQUENCES": {
        "function_name": "CONSECUTIVESEQUENCES",
        "description": "Finds locally longest consecutive subsequences in multidimensional strictly equally spaced data.",
        "parameters": ["expr1", "..."],
        "allowed_input_types": ["ANY_TYPE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "gap": "Standard time interval as a positive duration with unit ms/s/m/h/d. If omitted, the function estimates the interval using the mode.",
        },
        "output_type": "INT32",
    },
    "CONSECUTIVEWINDOWS": {
        "function_name": "CONSECUTIVEWINDOWS",
        "description": "Finds consecutive windows with a specified length in multidimensional strictly equally spaced data.",
        "parameters": ["expr1", "..."],
        "allowed_input_types": ["ANY_TYPE"],
        "required_attribute_parameters": {
            "length": "Window length as a positive duration with unit ms/s/m/h/d.",
        },
        "optional_attribute_parameters": {
            "gap": "Standard time interval as a positive duration with unit ms/s/m/h/d. If omitted, the function estimates the interval using the mode.",
        },
        "output_type": "INT32",
    },

    # ------------------------------------------------------------------
    # 8. Machine learning functions
    # ------------------------------------------------------------------
    "AR": {
        "function_name": "AR",
        "description": "Learns autoregressive model coefficients from a single numeric time series.",
        "parameters": ["expr"],
        "allowed_input_types": ["INT32", "INT64", "FLOAT", "DOUBLE"],
        "required_attribute_parameters": None,
        "optional_attribute_parameters": {
            "p": "Order of the autoregressive model. It must be a positive integer. Default is 1.",
        },
        "output_type": "DOUBLE",
    },
}


UDF_TIMESERIES_SCALAR_COMPATIBLE_FLAGS = {
    "COMPLETENESS": False,
    "CONSISTENCY": False,
    "TIMELINESS": False,
    "VALIDITY": False,
    "ACF": False,
    "PACF": False,
    "HISTOGRAM": False,
    "DISTINCT": False,
    "QLB": False,
    "MINMAX": True,
    "ZSCORE": True,
    "MVAVG": False,
    "RESAMPLE": False,
    "SAMPLE": False,
    "SEGMENT": False,
    "SPLINE": False,
    "IQR": True,
    "KSIGMA": False,
    "LOF": False,
    "MISSDETECT": False,
    "RANGE": True,
    "TWOSIDEDFILTER": False,
    "OUTLIER": False,
    "CONV": False,
    "DECONV": False,
    "DWT": False,
    "IDWT": False,
    "FFT": False,
    "IFFT": False,
    "HIGHPASS": False,
    "LOWPASS": False,
    "ENVELOPE": False,
    "PTNSYM": False,
    "XCORR": False,
    "TIMESTAMPREPAIR": False,
    "VALUEFILL": False,
    "VALUEREPAIR": False,
    "CONSECUTIVESEQUENCES": False,
    "CONSECUTIVEWINDOWS": False,
    "AR": False,
}


_missing_udf_scalar_compatible_flags = (
    set(UDF_TIMESERIES_FUNCTION_DEFINITIONS)
    - set(UDF_TIMESERIES_SCALAR_COMPATIBLE_FLAGS)
)
if _missing_udf_scalar_compatible_flags:
    raise ValueError(
        "Missing scalar_compatible flags for UDF time-series functions: "
        f"{sorted(_missing_udf_scalar_compatible_flags)}"
    )

for _function_name, _definition in UDF_TIMESERIES_FUNCTION_DEFINITIONS.items():
    _definition["scalar_compatible"] = UDF_TIMESERIES_SCALAR_COMPATIBLE_FLAGS[
        _function_name
    ]


FUNCTION_TABLES_BY_CATEGORY = {
    BUILTIN_AGGREGATE: BUILTIN_AGGREGATE_FUNCTION_DEFINITIONS,
    UDF_AGGREGATE: UDF_AGGREGATE_FUNCTION_DEFINITIONS,
    BUILTIN_TIMESERIES: BUILTIN_TIMESERIES_FUNCTION_DEFINITIONS,
    UDF_TIMESERIES: UDF_TIMESERIES_FUNCTION_DEFINITIONS,
}


FUNCTION_NAME_INDEX_BY_CATEGORY = {
    category: {name.upper(): definition for name, definition in table.items()}
    for category, table in FUNCTION_TABLES_BY_CATEGORY.items()
}


FUNCTION_CATEGORY_INDEX = {
    function_name: tuple(
        category
        for category, table in FUNCTION_NAME_INDEX_BY_CATEGORY.items()
        if function_name in table
    )
    for function_name in {
        name
        for table in FUNCTION_NAME_INDEX_BY_CATEGORY.values()
        for name in table
    }
}


def normalize_function_name(function_name: str) -> str:
    """Normalize function names for case-insensitive lookup."""

    return str(function_name).strip().upper()


def get_function_definition(
    function_name: str,
    category: str | None = None,
) -> dict[str, Any] | None:
    """Return one function definition by name, optionally within one category."""

    normalized_name = normalize_function_name(function_name)
    if category is not None:
        definition = FUNCTION_NAME_INDEX_BY_CATEGORY.get(category, {}).get(
            normalized_name
        )
        return deepcopy(definition) if definition else None

    for matched_category in FUNCTION_CATEGORY_INDEX.get(normalized_name, ()):
        definition = FUNCTION_NAME_INDEX_BY_CATEGORY[matched_category][normalized_name]
        return deepcopy(definition)
    return None


_TYPE_ALIASES = {
    "BOOL": "BOOLEAN",
    "INT": "INT64",
    "INTEGER": "INT32",
    "LONG": "INT64",
    "STRING": "TEXT",
    "TEXT_OR_JSON_STRING": "TEXT",
}
_ANY_INPUT_TYPE_MARKERS = {"ANY_TYPE", "ALL_TYPES"}


def normalize_data_type(data_type: Any) -> str:
    """Normalize one IoTDB data type for compatibility checks."""

    normalized = str(data_type).strip().upper()
    return _TYPE_ALIASES.get(normalized, normalized)


def normalize_data_types(data_type: Any) -> tuple[str, ...]:
    """Flatten one expression data-type descriptor into concrete type names."""

    if data_type is None:
        return ()
    if isinstance(data_type, dict):
        values = data_type.values()
    elif isinstance(data_type, (list, tuple, set, frozenset)):
        values = data_type
    else:
        normalized = normalize_data_type(data_type)
        return (normalized,) if normalized else ()

    normalized_types = []
    for value in values:
        for normalized in normalize_data_types(value):
            if normalized not in normalized_types:
                normalized_types.append(normalized)
    return tuple(normalized_types)


def resolve_function_output_type(
    function_definition: dict[str, Any],
    input_data_type: Any,
    generated_expression: str = "",
) -> Any:
    """Resolve one instantiated function's output data type."""

    output_type = function_definition.get("output_type")
    if isinstance(output_type, dict):
        return _resolve_conditional_output_type(
            function_definition=function_definition,
            output_type=output_type,
            generated_expression=generated_expression,
        )

    normalized_output = str(output_type or "").strip().upper()
    if normalized_output == "SAME_AS_INPUT":
        return _normalize_expression_data_type(input_data_type)
    if normalized_output == "SAME_AS_FIRST_INPUT":
        input_types = normalize_data_types(input_data_type)
        return input_types[0] if input_types else None
    if normalized_output == "DETERMINED_BY_TYPE_PARAMETER":
        resolved_type = _extract_expression_attribute(
            generated_expression,
            "type",
        )
        if resolved_type is None and function_definition.get(
            "function_name"
        ) == "CAST":
            resolved_type = _extract_cast_target_type(generated_expression)
        return normalize_data_type(resolved_type) if resolved_type else None
    if not normalized_output:
        return None
    return normalize_data_type(normalized_output)


def _normalize_expression_data_type(data_type: Any) -> Any:
    normalized_types = normalize_data_types(data_type)
    if not normalized_types:
        return None
    if len(normalized_types) == 1:
        return normalized_types[0]
    return list(normalized_types)


def _resolve_conditional_output_type(
    function_definition: dict[str, Any],
    output_type: dict[str, Any],
    generated_expression: str,
) -> str | None:
    condition_attribute = None
    condition_value = None
    for condition in output_type:
        match = re.fullmatch(
            r"if\s+([A-Za-z_][A-Za-z0-9_]*)\s*==\s*([A-Za-z0-9_]+)",
            str(condition).strip(),
            flags=re.IGNORECASE,
        )
        if match:
            condition_attribute = match.group(1)
            break

    if condition_attribute:
        condition_value = _extract_expression_attribute(
            generated_expression,
            condition_attribute,
        )
    if condition_value is None:
        return None
    for condition, resolved_type in output_type.items():
        match = re.fullmatch(
            r"if\s+([A-Za-z_][A-Za-z0-9_]*)\s*==\s*([A-Za-z0-9_]+)",
            str(condition).strip(),
            flags=re.IGNORECASE,
        )
        if match and match.group(2).lower() == condition_value.lower():
            return normalize_data_type(resolved_type)
    return None


def _extract_expression_attribute(
    expression: str,
    attribute_name: str,
) -> str | None:
    if not expression:
        return None
    attribute = re.escape(attribute_name)
    match = re.search(
        rf"(?:['\"]?{attribute}['\"]?)\s*(?:=|=>|:)\s*"
        rf"['\"]?([A-Za-z0-9_]+)['\"]?",
        expression,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def _extract_cast_target_type(expression: str) -> str | None:
    if not expression:
        return None
    match = re.search(
        r"\bAS\s+([A-Za-z][A-Za-z0-9_]*)\s*\)?\s*$",
        expression,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def get_primary_allowed_input_types(
    function_definition: dict[str, Any],
) -> tuple[str, ...]:
    """Return accepted types for the function's first expression parameter."""

    allowed_input_types = function_definition.get("allowed_input_types")
    if isinstance(allowed_input_types, dict):
        parameters = function_definition.get("parameters") or []
        primary_parameter = parameters[0] if parameters else None
        if primary_parameter in allowed_input_types:
            allowed_input_types = allowed_input_types[primary_parameter]
        elif allowed_input_types:
            allowed_input_types = next(iter(allowed_input_types.values()))
        else:
            allowed_input_types = []

    return normalize_data_types(allowed_input_types)


def function_accepts_input_type(
    function_definition: dict[str, Any],
    data_type: Any,
) -> bool:
    """Return whether a function can accept the expression as its first input."""

    input_types = normalize_data_types(data_type)
    if not input_types:
        return False

    allowed_types = set(get_primary_allowed_input_types(function_definition))
    if allowed_types & _ANY_INPUT_TYPE_MARKERS:
        return True
    return all(input_type in allowed_types for input_type in input_types)


def get_function_categories(function_name: str) -> tuple[str, ...]:
    """Return the categories that contain a function name."""

    return FUNCTION_CATEGORY_INDEX.get(normalize_function_name(function_name), ())


def list_function_names(category: str | None = None) -> list[str]:
    """List available function names globally or inside one category."""

    if category is not None:
        return sorted(FUNCTION_NAME_INDEX_BY_CATEGORY.get(category, {}))
    return sorted(FUNCTION_CATEGORY_INDEX)


def list_function_definitions(category: str) -> list[dict[str, Any]]:
    """List function definitions for one category."""

    return [
        deepcopy(definition)
        for _, definition in sorted(
            FUNCTION_NAME_INDEX_BY_CATEGORY.get(category, {}).items()
        )
    ]


def normalize_operator_name(operator_name: str) -> str:
    """Normalize operator names for lookup."""

    return str(operator_name).strip().lower()


def get_operator_definition(operator_name: str) -> dict[str, Any] | None:
    """Return one IoTDB operator definition by operator name."""

    definition = IOTDB_OPERATOR_NAME_INDEX.get(normalize_operator_name(operator_name))
    return deepcopy(definition) if definition else None


def get_operator_definitions_by_symbol(symbol: str) -> list[dict[str, Any]]:
    """Return all IoTDB operator definitions using a given symbol."""

    operator_names = IOTDB_OPERATOR_SYMBOL_INDEX.get(str(symbol).strip(), ())
    return [
        deepcopy(IOTDB_OPERATOR_NAME_INDEX[operator_name])
        for operator_name in operator_names
    ]


def list_operator_definitions(category: str | None = None) -> list[dict[str, Any]]:
    """List IoTDB operator definitions globally or under one category path."""

    if category is None:
        definitions = IOTDB_OPERATOR_NAME_INDEX.values()
    else:
        normalized_category = normalize_operator_name(category)
        definitions = (
            definition
            for definition in IOTDB_OPERATOR_NAME_INDEX.values()
            if normalized_category
            in {normalize_operator_name(item) for item in definition["category_path"]}
        )
    return sorted(
        (deepcopy(definition) for definition in definitions),
        key=lambda definition: definition["operator_name"],
    )
