"""Tools for the P2 Temporal skill package."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from iotdb_tools.schema_access import open_default_session
from iotdb_tools.time_bounds import configured_year_bounds
from iotdb_tools.temporal_sampling import read_timestamp_sample


def normalize_action_space(action_space: list[str]) -> list[str]:
    """Normalize candidate values while preserving order."""

    normalized = []
    seen = set()
    for item in action_space:
        if not item:
            continue
        value = str(item).strip()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def read_working_schema_time_ranges(
    state: dict | None = None,
    working_schema: dict | None = None,
    session: Any | None = None,
    minimum_year: int | None = None,
    maximum_year: int | None = None,
    max_measurements: int = 8,
    timestamp_sample_limit: int = 64,
) -> dict[str, Any]:
    """Read bounded temporal evidence from the current working schema.

    Args:
        state:
            Runtime state dict. When working_schema is omitted, this function
            reads state["schema"]["working_schema"].

        working_schema:
            Optional explicit working schema. This is useful for tests and for
            later runners that already have the scoped schema object.

        session:
            Optional opened IoTDB Session. If omitted, a local default session
            is opened and closed by this function.

        max_measurements, timestamp_sample_limit:
            Probe at most eight series and 64 timestamps per series by default.
            Bounds and samples are cached in trajectory State by the caller.

    Returns:
        A dict containing one item per probed measurement. scope_sampled marks
        a partial schema probe. Timestamps
        are formatted as YYYY-MM-DDTHH:MM:SS for prompt use, with raw
        epoch-millisecond values preserved in *_ms fields. If a measurement has
        no data, start_time and end_time are None.
    """

    (
        resolved_minimum_year,
        resolved_maximum_year,
        minimum_time_ms,
        maximum_time_ms,
    ) = configured_year_bounds(minimum_year, maximum_year)
    current_working_schema = working_schema or get_current_working_schema(state or {})
    if not isinstance(current_working_schema, dict) or not current_working_schema:
        return _empty_time_range_result(
            minimum_year=resolved_minimum_year,
            maximum_year=resolved_maximum_year,
            minimum_time_ms=minimum_time_ms,
            maximum_time_ms=maximum_time_ms,
        )

    measurements = extract_working_measurements(current_working_schema)
    total_measurement_count = len(measurements)
    if max_measurements < 1:
        raise ValueError("max_measurements must be positive")
    measurements = measurements[:max_measurements]
    if not measurements:
        return _empty_time_range_result(
            current_working_schema,
            minimum_year=resolved_minimum_year,
            maximum_year=resolved_maximum_year,
            minimum_time_ms=minimum_time_ms,
            maximum_time_ms=maximum_time_ms,
        )

    owns_session = session is None
    session = session or open_default_session()
    try:
        time_ranges = []
        for measurement in measurements:
            start_time = _query_boundary_time(
                session=session,
                device_path=measurement["device"],
                measurement_name=measurement["measurement"],
                order="ASC",
                minimum_time_ms=minimum_time_ms,
                maximum_time_ms=maximum_time_ms,
            )
            end_time = _query_boundary_time(
                session=session,
                device_path=measurement["device"],
                measurement_name=measurement["measurement"],
                order="DESC",
                minimum_time_ms=minimum_time_ms,
                maximum_time_ms=maximum_time_ms,
            )
            time_ranges.append(
                {
                    **measurement,
                    "start_time": format_epoch_millis(start_time),
                    "end_time": format_epoch_millis(end_time),
                    "start_time_ms": start_time,
                    "end_time_ms": end_time,
                }
            )
        valid = [item for item in time_ranges if item["start_time_ms"] is not None and item["end_time_ms"] is not None]
        if valid:
            # Every series uses the same observation interval. Disjoint ranges
            # remain detectable from the bounds without probing timestamps.
            start = max(item["start_time_ms"] for item in valid)
            end = min(item["end_time_ms"] for item in valid)
            if start <= end:
                for item in valid:
                    try:
                        item["timestamps_ms"] = read_timestamp_sample(
                            session, item, start=start, end=end, limit=timestamp_sample_limit,
                        )
                    except Exception:
                        # Timestamp evidence is optional; bounds remain useful.
                        pass
    finally:
        if owns_session:
            session.close()

    valid_measurement_count = sum(
        item["start_time_ms"] is not None and item["end_time_ms"] is not None
        for item in time_ranges
    )
    return {
        "schema_type": "iotdb_time_range",
        "time_format": "YYYY-MM-DDTHH:MM:SS",
        "database": (
            current_working_schema.get("database")
            or current_working_schema.get("root")
            or _infer_database(measurements)
        ),
        "measurement_count": len(time_ranges),
        "total_measurement_count": total_measurement_count,
        "scope_sampled": total_measurement_count > len(time_ranges),
        "timestamp_sample_limit": timestamp_sample_limit,
        "valid_measurement_count": valid_measurement_count,
        "excluded_measurement_count": len(time_ranges) - valid_measurement_count,
        "valid_time_filter": {
            "minimum_year": resolved_minimum_year,
            "maximum_year": resolved_maximum_year,
            "minimum_time": format_epoch_millis(minimum_time_ms),
            "maximum_time": format_epoch_millis(maximum_time_ms),
            "minimum_time_ms": minimum_time_ms,
            "maximum_time_ms": maximum_time_ms,
        },
        "measurements": time_ranges,
    }


def get_current_working_schema(state: dict, phase_id: str = "P1") -> dict | None:
    """Read the active working schema from a runtime state dict."""

    if not isinstance(state, dict):
        return None

    schema_context = state.get("schema", {})
    if isinstance(schema_context, dict):
        working_schema = schema_context.get("working_schema")
        if isinstance(working_schema, dict) and working_schema:
            return working_schema

    spo = state.get("spo", {})
    phase_spo = spo.get(phase_id, {}) if isinstance(spo, dict) else {}
    working_schema = phase_spo.get("working_schema")
    if isinstance(working_schema, dict) and working_schema:
        return working_schema

    return None


def extract_working_measurements(working_schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Return concrete measurement descriptors from a working schema."""

    seen_paths = set()
    measurements = []

    for path in working_schema.get("timeseries_paths", []) or []:
        descriptor = _descriptor_from_path(path)
        if descriptor and descriptor["path"] not in seen_paths:
            measurements.append(descriptor)
            seen_paths.add(descriptor["path"])

    devices = working_schema.get("devices", [])
    if isinstance(devices, list):
        for device in devices:
            for descriptor in _descriptors_from_device(device):
                if descriptor["path"] in seen_paths:
                    continue
                measurements.append(descriptor)
                seen_paths.add(descriptor["path"])

    return measurements


def _descriptors_from_device(device: Any) -> list[dict[str, Any]]:
    if not isinstance(device, dict):
        return []

    device_path = device.get("path")
    if not device_path:
        return []
    device_path = str(device_path)

    measurements = device.get("measurements", [])
    if not isinstance(measurements, list):
        return []

    descriptors = []
    for measurement in measurements:
        descriptor = _descriptor_from_measurement(device_path, measurement)
        if descriptor:
            descriptors.append(descriptor)
    return descriptors


def _descriptor_from_measurement(
    device_path: str,
    measurement: Any,
) -> dict[str, Any] | None:
    if isinstance(measurement, dict):
        path = measurement.get("path")
        name = measurement.get("name") or measurement.get("measurement")
        data_type = measurement.get("data_type")
    else:
        path = None
        name = measurement
        data_type = None

    if path:
        descriptor = _descriptor_from_path(path)
        if descriptor:
            descriptor["data_type"] = data_type
            return descriptor

    if name is None:
        return None
    name = str(name).strip()
    if not name:
        return None

    return {
        "path": f"{device_path}.{name}",
        "device": device_path,
        "measurement": name,
        "data_type": data_type,
    }


def _descriptor_from_path(path: Any) -> dict[str, Any] | None:
    if path is None:
        return None
    path = str(path).strip()
    if not path or "." not in path:
        return None
    device_path, measurement_name = path.rsplit(".", 1)
    return {
        "path": path,
        "device": device_path,
        "measurement": measurement_name,
        "data_type": None,
    }


def _query_boundary_time(
    session: Any,
    device_path: str,
    measurement_name: str,
    order: str,
    minimum_time_ms: int,
    maximum_time_ms: int,
) -> int | None:
    order = order.upper()
    if order not in {"ASC", "DESC"}:
        raise ValueError(f"Unsupported time boundary order: {order}")

    sql = (
        f"SELECT {measurement_name} FROM {device_path} "
        f"WHERE {measurement_name} IS NOT NULL "
        f"AND TIME >= {minimum_time_ms} AND TIME <= {maximum_time_ms} "
        f"ORDER BY TIME {order} LIMIT 1"
    )
    dataset = session.execute_query_statement(sql)
    try:
        if not dataset.has_next():
            return None
        timestamp_ms = int(dataset.next().get_timestamp())
        if timestamp_ms < minimum_time_ms or timestamp_ms > maximum_time_ms:
            return None
        return timestamp_ms
    finally:
        dataset.close_operation_handle()


def _empty_time_range_result(
    working_schema: dict[str, Any] | None = None,
    *,
    minimum_year: int | None = None,
    maximum_year: int | None = None,
    minimum_time_ms: int | None = None,
    maximum_time_ms: int | None = None,
) -> dict[str, Any]:
    working_schema = working_schema or {}
    return {
        "schema_type": "iotdb_time_range",
        "time_format": "YYYY-MM-DDTHH:MM:SS",
        "database": working_schema.get("database") or working_schema.get("root"),
        "measurement_count": 0,
        "valid_measurement_count": 0,
        "excluded_measurement_count": 0,
        "valid_time_filter": {
            "minimum_year": minimum_year,
            "maximum_year": maximum_year,
            "minimum_time": format_epoch_millis(minimum_time_ms),
            "maximum_time": format_epoch_millis(maximum_time_ms),
            "minimum_time_ms": minimum_time_ms,
            "maximum_time_ms": maximum_time_ms,
        },
        "measurements": [],
    }


def _infer_database(measurements: list[dict[str, Any]]) -> str | None:
    if not measurements:
        return None
    path = measurements[0].get("path", "")
    parts = str(path).split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return None


def format_epoch_millis(value: Any) -> str | None:
    """Convert an IoTDB epoch-millisecond timestamp to YYYY-MM-DDTHH:MM:SS."""

    if value is None:
        return None
    try:
        timestamp_ms = int(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%S")
