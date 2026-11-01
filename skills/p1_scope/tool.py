"""Tools for the P1 Scope skill package.

This file provides reusable helper functions used by P1 action-level skills.
The current version defines lightweight interfaces required by the MVP.
"""

from copy import deepcopy


def normalize_action_space(action_space: list[str]) -> list[str]:
    """Normalize the candidate action space.

    This function keeps the original order and removes empty or duplicated
    candidates.
    """
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


def build_working_schema(
    db_schema: dict,
    scope_decision: dict,
    previous_working_schema: dict | str | None = None,
) -> dict:
    """Build the current working schema from the global DB schema.

    Args:
        db_schema:
            Global database schema metadata. The exact internal format can be
            defined later.

        scope_decision:
            Current scope decision produced by an action-level skill.
            Expected fields include action_id, decision_name, and selected_action.

        previous_working_schema:
            Working schema produced by previous scope actions. In the current
            P1 design, later actions may overwrite the working_schema field.

    Returns:
        A lightweight working schema object for downstream phases.
    """
    action_id = scope_decision.get("action_id")
    decision_name = scope_decision.get("decision_name")
    selected_action = scope_decision.get("selected_action")

    working_schema = {
        "source": "P1",
        "updated_by": action_id,
        "decision_name": decision_name,
        "scope_value": selected_action,
        "working_root": None,
        "devices": [],
        "measurements": [],
        "tags": {},
        "schema_summary": {},
        "status": "placeholder",
    }

    if isinstance(db_schema, dict):
        working_schema["working_root"] = db_schema.get("root")
        working_schema["devices"] = db_schema.get("devices", [])
        working_schema["measurements"] = db_schema.get("measurements", [])
        working_schema["tags"] = db_schema.get("tags", {})

        working_schema["schema_summary"] = {
            "schema_type": db_schema.get("schema_type"),
            "root": db_schema.get("root"),
            "num_devices": len(db_schema.get("devices", []))
            if isinstance(db_schema.get("devices", []), list)
            else None,
            "num_measurements": len(db_schema.get("measurements", []))
            if isinstance(db_schema.get("measurements", []), list)
            else None,
        }

    return working_schema


def get_working_schema_meta(
    state: dict,
    phase_id: str = "P1",
) -> dict:
    """Extract current working schema metadata from the global state.

    Args:
        state:
            Current global generation state.

        phase_id:
            The phase that owns the working_schema field. Default is P1.

    Returns:
        A lightweight metadata object describing the current working schema.
        The detailed schema extraction logic can be implemented later.
    """
    working_schema = get_current_working_schema(state, phase_id=phase_id)

    return {
        "phase_id": phase_id,
        "working_schema": working_schema,
        "has_working_schema": working_schema is not None,
        "status": "placeholder",
    }


def get_current_working_schema(
    state: dict,
    phase_id: str = "P1",
) -> dict | None:
    """Read the current active working schema from state.

    The global schema working_schema is the runtime source of truth. The P1 SPO
    fallback keeps compatibility with earlier package-local prototypes.
    """

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


def get_tag_schema(
    state: dict | None = None,
    schema_context: dict | None = None,
) -> dict:
    """Read tag metadata from the current schema context.

    Returns an empty dict when the database has no tag metadata.
    """

    context = schema_context
    if context is None and isinstance(state, dict):
        context = state.get("schema", {})
    if not isinstance(context, dict):
        return {}

    tag_schema = context.get("tag_schema")
    if isinstance(tag_schema, dict) and tag_schema:
        return tag_schema

    detail_schema = context.get("detail_schema")
    if isinstance(detail_schema, dict):
        tag_schema = detail_schema.get("tag_schema")
        if isinstance(tag_schema, dict) and tag_schema:
            return tag_schema

    return {}


def get_working_tag_schema(
    state: dict | None = None,
    working_schema: dict | None = None,
    tag_schema: dict | None = None,
    phase_id: str = "P1",
) -> dict:
    """Return tag metadata only for timeseries inside the working schema."""

    current_working_schema = working_schema or get_current_working_schema(
        state or {},
        phase_id=phase_id,
    )
    current_tag_schema = tag_schema or get_tag_schema(state=state)
    if not isinstance(current_working_schema, dict) or not current_working_schema:
        return {}
    if not isinstance(current_tag_schema, dict) or not current_tag_schema:
        return {}

    tagged_timeseries = current_tag_schema.get("timeseries", [])
    if not isinstance(tagged_timeseries, list) or not tagged_timeseries:
        return {}

    working_paths = _working_timeseries_paths(current_working_schema)
    working_devices = _working_device_paths(current_working_schema)
    if not working_paths and not working_devices:
        return {}

    filtered = []
    tag_keys = set()
    for series in tagged_timeseries:
        if not isinstance(series, dict):
            continue
        series_path = series.get("path")
        if not _series_in_working_schema(series_path, working_paths, working_devices):
            continue
        series_copy = deepcopy(series)
        tags = series_copy.get("tags", {})
        if isinstance(tags, dict):
            tag_keys.update(tags)
        filtered.append(series_copy)

    if not filtered:
        return {}

    return {
        "schema_type": current_tag_schema.get("schema_type", "iotdb_tree_tags"),
        "database": (
            current_working_schema.get("database")
            or current_tag_schema.get("database")
            or current_working_schema.get("root")
        ),
        "timeseries_count": len(filtered),
        "tag_keys": sorted(tag_keys),
        "timeseries": filtered,
    }


def has_same_measurement_across_devices(
    state: dict | None = None,
    working_schema: dict | None = None,
    phase_id: str = "P1",
) -> bool:
    """Return whether different working-schema devices share a measurement name.

    If working_schema is empty, missing device details, or contains only one
    device, the result is False.
    """

    current_working_schema = working_schema or get_current_working_schema(
        state or {},
        phase_id=phase_id,
    )
    if not isinstance(current_working_schema, dict) or not current_working_schema:
        return False

    devices = current_working_schema.get("devices", [])
    if not isinstance(devices, list) or len(devices) <= 1:
        return False

    seen_measurements = set()
    for device in devices:
        names = _measurement_names_from_device(device)
        if seen_measurements.intersection(names):
            return True
        seen_measurements.update(names)

    return False


def _measurement_names_from_device(device) -> set[str]:
    """Extract measurement names from one working-schema device record."""

    if not isinstance(device, dict):
        return set()

    measurements = device.get("measurements", [])
    if not isinstance(measurements, list):
        return set()

    names = set()
    for measurement in measurements:
        name = _measurement_name(measurement)
        if name:
            names.add(name)
    return names


def _working_timeseries_paths(working_schema: dict) -> set[str]:
    paths = {
        str(path)
        for path in working_schema.get("timeseries_paths", [])
        if path
    }

    devices = working_schema.get("devices", [])
    if not isinstance(devices, list):
        return paths

    for device in devices:
        if not isinstance(device, dict):
            continue
        device_path = device.get("path")
        measurements = device.get("measurements", [])
        if not device_path or not isinstance(measurements, list):
            continue
        for measurement in measurements:
            measurement_path = _measurement_path_from_device(device_path, measurement)
            if measurement_path:
                paths.add(measurement_path)

    return paths


def _working_device_paths(working_schema: dict) -> set[str]:
    devices = working_schema.get("devices", [])
    if not isinstance(devices, list):
        return set()

    paths = set()
    for device in devices:
        if isinstance(device, dict) and device.get("path"):
            paths.add(str(device["path"]))
        elif isinstance(device, str) and device:
            paths.add(device)
    return paths


def _measurement_path_from_device(device_path: str, measurement) -> str | None:
    if isinstance(measurement, dict):
        path = measurement.get("path")
        if path:
            return str(path)
        name = measurement.get("name") or measurement.get("measurement")
    else:
        name = measurement

    if name is None:
        return None
    name = str(name).strip()
    if not name:
        return None
    return f"{device_path}.{name}"


def _series_in_working_schema(
    series_path,
    working_paths: set[str],
    working_devices: set[str],
) -> bool:
    if not series_path:
        return False
    series_path = str(series_path)
    if working_paths:
        return series_path in working_paths
    return any(series_path.startswith(f"{device_path}.") for device_path in working_devices)


def _measurement_name(measurement) -> str | None:
    """Normalize one measurement record into a comparable name."""

    if isinstance(measurement, dict):
        value = (
            measurement.get("name")
            or measurement.get("measurement")
            or measurement.get("path")
        )
    else:
        value = measurement

    if value is None:
        return None

    name = str(value).strip()
    if not name:
        return None
    return name.split(".")[-1]
