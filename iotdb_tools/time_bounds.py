"""Shared bounds for ignoring implausible IoTDB timestamps."""

from __future__ import annotations

import os
from datetime import datetime, timezone


DEFAULT_MINIMUM_DATA_YEAR = 2000
DEFAULT_MAXIMUM_DATA_YEAR = 2100
MINIMUM_YEAR_ENV = "TSSDA_MINIMUM_DATA_YEAR"
MAXIMUM_YEAR_ENV = "TSSDA_MAXIMUM_DATA_YEAR"


def year_bounds_ms(minimum_year: int, maximum_year: int) -> tuple[int, int]:
    """Return inclusive UTC epoch-millisecond bounds for whole calendar years."""

    if minimum_year < 1970:
        raise ValueError("minimum_year must be 1970 or later")
    if maximum_year < minimum_year:
        raise ValueError("maximum_year must not be earlier than minimum_year")
    if maximum_year >= 9999:
        raise ValueError("maximum_year must be below 9999")
    minimum = datetime(minimum_year, 1, 1, tzinfo=timezone.utc)
    maximum_exclusive = datetime(maximum_year + 1, 1, 1, tzinfo=timezone.utc)
    return (
        int(minimum.timestamp() * 1000),
        int(maximum_exclusive.timestamp() * 1000) - 1,
    )


def configured_year_bounds(
    minimum_year: int | None = None,
    maximum_year: int | None = None,
) -> tuple[int, int, int, int]:
    """Resolve explicit or environment-configured valid calendar-year bounds."""

    resolved_minimum_year = (
        int(os.getenv(MINIMUM_YEAR_ENV, DEFAULT_MINIMUM_DATA_YEAR))
        if minimum_year is None
        else int(minimum_year)
    )
    resolved_maximum_year = (
        int(os.getenv(MAXIMUM_YEAR_ENV, DEFAULT_MAXIMUM_DATA_YEAR))
        if maximum_year is None
        else int(maximum_year)
    )
    minimum_ms, maximum_ms = year_bounds_ms(
        resolved_minimum_year,
        resolved_maximum_year,
    )
    return (
        resolved_minimum_year,
        resolved_maximum_year,
        minimum_ms,
        maximum_ms,
    )
