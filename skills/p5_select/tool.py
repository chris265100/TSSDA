"""Tools for the P5 Select skill package."""

from __future__ import annotations

import random
from typing import Any

from iotdb_tools.schema_access import open_default_session


def get_random_point_time(
    timeseries_path: str,
    session: Any | None = None,
    random_generator: random.Random | None = None,
) -> int | None:
    """Return the timestamp of one random existing point in a timeseries.

    The returned timestamp is the IoTDB epoch-millisecond time value. If the
    timeseries has no data, this function returns None.
    """

    device_path, measurement_name = _split_timeseries_path(timeseries_path)
    owns_session = session is None
    session = session or open_default_session()
    try:
        total_count = _count_points(
            session=session,
            device_path=device_path,
            measurement_name=measurement_name,
        )
        if total_count <= 0:
            return None

        rng = random_generator or random
        offset = rng.randint(0, total_count - 1)
        return _read_point_time_at_offset(
            session=session,
            device_path=device_path,
            measurement_name=measurement_name,
            offset=offset,
        )
    finally:
        if owns_session:
            session.close()


def _split_timeseries_path(timeseries_path: str) -> tuple[str, str]:
    path = str(timeseries_path).strip()
    if "." not in path:
        raise ValueError(f"Invalid timeseries path: {timeseries_path!r}")
    device_path, measurement_name = path.rsplit(".", 1)
    if not device_path or not measurement_name:
        raise ValueError(f"Invalid timeseries path: {timeseries_path!r}")
    return device_path, measurement_name


def _count_points(
    session: Any,
    device_path: str,
    measurement_name: str,
) -> int:
    sql = f"SELECT COUNT({measurement_name}) FROM {device_path}"
    dataset = session.execute_query_statement(sql)
    try:
        if not dataset.has_next():
            return 0
        record = dataset.next()
        fields = record.get_fields()
        if not fields:
            return 0
        value = _field_value(fields[0])
        return int(value or 0)
    finally:
        dataset.close_operation_handle()


def _read_point_time_at_offset(
    session: Any,
    device_path: str,
    measurement_name: str,
    offset: int,
) -> int | None:
    sql = (
        f"SELECT {measurement_name} FROM {device_path} "
        f"WHERE {measurement_name} IS NOT NULL "
        f"ORDER BY TIME ASC LIMIT 1 OFFSET {offset}"
    )
    dataset = session.execute_query_statement(sql)
    try:
        if not dataset.has_next():
            return None
        return int(dataset.next().get_timestamp())
    finally:
        dataset.close_operation_handle()


def _field_value(field: Any) -> Any:
    try:
        return field.get_object_value(field.get_data_type())
    except TypeError:
        return field.get_object_value()
