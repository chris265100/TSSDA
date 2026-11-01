import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from iotdb.Session import Session
from tssda.config import get_default_schema_root


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6667
DEFAULT_USERNAME = ""
DEFAULT_PASSWORD = ""
DEFAULT_SCHEMA_ROOT = get_default_schema_root()
DEFAULT_CONNECTION_TIMEOUT_MS = 60000
DEFAULT_SESSION_RETRY_NUM = 1


def create_session(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    connection_timeout_ms: int = DEFAULT_CONNECTION_TIMEOUT_MS,
    retry_num: int = DEFAULT_SESSION_RETRY_NUM,
) -> Session:
    """Create an IoTDB Session. The caller owns open/close."""

    if connection_timeout_ms <= 0:
        raise ValueError("IoTDB connection timeout must be positive")
    if retry_num < 0:
        raise ValueError("IoTDB session retry count must be non-negative")
    if not username or not password:
        raise ValueError("Set IOTDB_USERNAME and IOTDB_PASSWORD before connecting")
    session = Session(
        host,
        port,
        username,
        password,
        connection_timeout_in_ms=connection_timeout_ms,
    )
    # The Python SDK exposes RETRY_NUM as a class default but reads it through
    # the instance.  Override it per session so a stale RPC cannot spend many
    # socket-timeout windows reconnecting before control returns to TSSDA.
    session.RETRY_NUM = retry_num
    return session


def open_default_session(
    *,
    connection_timeout_ms: int | None = None,
    retry_num: int | None = None,
) -> Session:
    """Open a session using environment-provided credentials."""

    resolved_timeout_ms = (
        int(os.getenv("IOTDB_CONNECTION_TIMEOUT_MS", DEFAULT_CONNECTION_TIMEOUT_MS))
        if connection_timeout_ms is None
        else connection_timeout_ms
    )
    resolved_retry_num = (
        int(os.getenv("IOTDB_SESSION_RETRY_NUM", DEFAULT_SESSION_RETRY_NUM))
        if retry_num is None
        else retry_num
    )
    session = create_session(
        host=os.getenv("IOTDB_HOST", DEFAULT_HOST),
        port=int(os.getenv("IOTDB_PORT", DEFAULT_PORT)),
        username=os.getenv("IOTDB_USERNAME", DEFAULT_USERNAME),
        password=os.getenv("IOTDB_PASSWORD", DEFAULT_PASSWORD),
        connection_timeout_ms=resolved_timeout_ms,
        retry_num=resolved_retry_num,
    )
    session.open(False)
    return session


def _field_value(field: Any) -> Any:
    try:
        return field.get_object_value(field.get_data_type())
    except TypeError:
        return field.get_object_value()


def _query_dicts(session: Session, sql: str) -> list[dict[str, Any]]:
    dataset = session.execute_query_statement(sql)
    try:
        columns = list(dataset.get_column_names())
        rows = []
        while dataset.has_next():
            record = dataset.next()
            fields = record.get_fields()
            rows.append(
                {
                    column: _field_value(fields[index])
                    for index, column in enumerate(columns)
                }
            )
        return rows
    finally:
        dataset.close_operation_handle()


def _normalize_root(schema_root: str) -> str:
    return schema_root.rstrip(".")


def _timeseries_sql(schema_root: str) -> str:
    return f"SHOW TIMESERIES {_normalize_root(schema_root)}.**"


def _device_sql(schema_root: str) -> str:
    return f"SHOW DEVICES {_normalize_root(schema_root)}.** WITH DATABASE"


def _safe_query_devices(session: Session, schema_root: str) -> list[dict[str, Any]]:
    try:
        return _query_dicts(session, _device_sql(schema_root))
    except Exception:
        root = _normalize_root(schema_root)
        return [
            row
            for row in _query_dicts(session, "SHOW DEVICES WITH DATABASE")
            if str(row.get("Device", "")).startswith(root + ".")
        ]


def _device_path(row: dict[str, Any]) -> str:
    return str(row.get("Device", ""))


def _timeseries_path(row: dict[str, Any]) -> str:
    return str(row.get("Timeseries", ""))


def _measurement_path(series_path: str, device_path: str) -> str:
    prefix = device_path + "."
    if series_path.startswith(prefix):
        return series_path[len(prefix) :]
    return series_path.rsplit(".", 1)[-1]


def _find_device_for_timeseries(series_path: str, devices: list[str]) -> str:
    matches = [
        device
        for device in devices
        if series_path == device or series_path.startswith(device + ".")
    ]
    if matches:
        return max(matches, key=len)
    return series_path.rsplit(".", 1)[0]


def _bool_string(value: Any) -> bool | Any:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return value


def parse_iotdb_map_field(value: Any) -> dict[str, str]:
    """Parse IoTDB map-like metadata fields such as Tags or Attributes."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            str(key).strip().strip("'\""): str(item).strip().strip("'\"")
            for key, item in value.items()
            if str(key).strip().strip("'\"")
        }

    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "{}"}:
        return {}

    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    if not text:
        return {}

    parsed = {}
    for item in re.split(r"\s*,\s*", text):
        if not item:
            continue
        separator = "=" if "=" in item else ":" if ":" in item else None
        if separator is None:
            continue
        key, item_value = item.split(separator, 1)
        key = key.strip().strip("'\"")
        item_value = item_value.strip().strip("'\"")
        if key:
            parsed[key] = item_value
    return parsed


def _build_tag_schema(
    *,
    schema_root: str,
    timeseries_rows: list[dict[str, Any]],
    device_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build a standalone tag schema from SHOW TIMESERIES rows."""

    root = _normalize_root(schema_root)
    device_paths = device_paths or []
    tagged_timeseries = []
    tag_keys = set()

    for row in timeseries_rows:
        tags = parse_iotdb_map_field(row.get("Tags"))
        if not tags:
            continue

        series_path = _timeseries_path(row)
        device_path = _find_device_for_timeseries(series_path, device_paths)
        measurement = _measurement_path(series_path, device_path)
        tag_keys.update(tags)
        tagged_timeseries.append(
            {
                "path": series_path,
                "device": device_path,
                "measurement": measurement,
                "tags": tags,
            }
        )

    if not tagged_timeseries:
        return {}

    return {
        "schema_type": "iotdb_tree_tags",
        "database": root,
        "timeseries_count": len(tagged_timeseries),
        "tag_keys": sorted(tag_keys),
        "timeseries": tagged_timeseries,
    }


def get_schema_tags(
    schema_root: str = DEFAULT_SCHEMA_ROOT,
    session: Session | None = None,
) -> dict[str, Any]:
    """Return tagged timeseries metadata under one IoTDB schema root."""

    owns_session = session is None
    session = session or open_default_session()
    root = _normalize_root(schema_root)
    try:
        device_rows = _safe_query_devices(session, root)
        device_paths = [_device_path(row) for row in device_rows]
        timeseries_rows = _query_dicts(session, _timeseries_sql(root))
        return _build_tag_schema(
            schema_root=root,
            timeseries_rows=timeseries_rows,
            device_paths=device_paths,
        )
    finally:
        if owns_session:
            session.close()


def get_schema_detail(
    schema_root: str = DEFAULT_SCHEMA_ROOT,
    session: Session | None = None,
) -> dict[str, Any]:
    """Return detailed IoTDB tree schema information under a path."""

    owns_session = session is None
    session = session or open_default_session()
    root = _normalize_root(schema_root)
    try:
        device_rows = _safe_query_devices(session, root)
        timeseries_rows = _query_dicts(session, _timeseries_sql(root))
        device_paths = [_device_path(row) for row in device_rows]

        devices = {
            device_path: {
                "path": device_path,
                "database": row.get("Database"),
                "is_aligned": _bool_string(row.get("IsAligned")),
                "template": row.get("Template"),
                "ttl_ms": row.get("TTL(ms)"),
                "timeseries": [],
            }
            for row in device_rows
            for device_path in [_device_path(row)]
        }

        detailed_timeseries = []
        for row in timeseries_rows:
            series_path = _timeseries_path(row)
            device_path = _find_device_for_timeseries(series_path, device_paths)
            measurement = _measurement_path(series_path, device_path)
            tags = parse_iotdb_map_field(row.get("Tags"))
            attributes = parse_iotdb_map_field(row.get("Attributes"))
            detail = {
                "path": series_path,
                "device": device_path,
                "measurement": measurement,
                "alias": row.get("Alias"),
                "database": row.get("Database"),
                "data_type": row.get("DataType"),
                "encoding": row.get("Encoding"),
                "compression": row.get("Compression"),
                "tags": tags,
                "attributes": attributes,
                "deadband": row.get("Deadband"),
                "deadband_parameters": row.get("DeadbandParameters"),
                "view_type": row.get("ViewType"),
            }
            detailed_timeseries.append(detail)
            devices.setdefault(
                device_path,
                {
                    "path": device_path,
                    "database": detail["database"],
                    "is_aligned": None,
                    "template": None,
                    "ttl_ms": None,
                    "timeseries": [],
                },
            )["timeseries"].append(detail)

        return {
            "schema_type": "iotdb_tree",
            "root": root,
            "device_count": len(devices),
            "timeseries_count": len(detailed_timeseries),
            "devices": list(devices.values()),
            "timeseries": detailed_timeseries,
            "tag_schema": _build_tag_schema(
                schema_root=root,
                timeseries_rows=timeseries_rows,
                device_paths=device_paths,
            ),
        }
    finally:
        if owns_session:
            session.close()


def get_schema_meta(
    schema_root: str = DEFAULT_SCHEMA_ROOT,
    session: Session | None = None,
) -> dict[str, Any]:
    """Return lightweight schema metadata for data-agent scope decisions."""

    detail = get_schema_detail(schema_root=schema_root, session=session)
    devices = []
    unique_measurements = set()

    for device in detail["devices"]:
        measurements = [
            {
                "name": series["measurement"],
                "path": series["path"],
                "data_type": series["data_type"],
            }
            for series in device["timeseries"]
        ]
        unique_measurements.update(series["measurement"] for series in device["timeseries"])
        devices.append(
            {
                "path": device["path"],
                "database": device["database"],
                "is_aligned": device["is_aligned"],
                "measurements": measurements,
            }
        )

    return {
        "schema_type": detail["schema_type"],
        "root": detail["root"],
        "device_count": detail["device_count"],
        "timeseries_count": detail["timeseries_count"],
        "devices": devices,
        "measurements": sorted(unique_measurements),
        "timeseries_paths": [series["path"] for series in detail["timeseries"]],
    }


def get_schema_bundle(
    schema_root: str = DEFAULT_SCHEMA_ROOT,
    session: Session | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return `(meta_schema, detail_schema)` for one IoTDB schema root."""

    detail = get_schema_detail(schema_root=schema_root, session=session)
    meta = get_schema_meta_from_detail(detail)
    return meta, detail


def get_schema_meta_from_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Build lightweight schema metadata from an already fetched detail object."""

    devices = []
    unique_measurements = set()
    for device in detail["devices"]:
        measurements = [
            {
                "name": series["measurement"],
                "path": series["path"],
                "data_type": series["data_type"],
            }
            for series in device["timeseries"]
        ]
        unique_measurements.update(series["measurement"] for series in device["timeseries"])
        devices.append(
            {
                "path": device["path"],
                "database": device["database"],
                "is_aligned": device["is_aligned"],
                "measurements": measurements,
            }
        )

    return {
        "schema_type": detail["schema_type"],
        "root": detail["root"],
        "device_count": detail["device_count"],
        "timeseries_count": detail["timeseries_count"],
        "devices": devices,
        "measurements": sorted(unique_measurements),
        "timeseries_paths": [series["path"] for series in detail["timeseries"]],
    }


def write_schema_files(
    output_dir: str | Path,
    schema_root: str = DEFAULT_SCHEMA_ROOT,
    session: Session | None = None,
) -> tuple[Path, Path]:
    """Write meta and detailed schema JSON files, returning their paths."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    meta, detail = get_schema_bundle(schema_root=schema_root, session=session)
    safe_root = schema_root.replace(".", "_")
    meta_path = output_path / f"{safe_root}_schema_meta.json"
    detail_path = output_path / f"{safe_root}_schema_detail.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    detail_path.write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta_path, detail_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read IoTDB schema under a path.")
    parser.add_argument("--schema-root", default=DEFAULT_SCHEMA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime") / "schema",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path, detail_path = write_schema_files(
        output_dir=args.output_dir,
        schema_root=args.schema_root,
    )
    print(f"Schema meta written to: {meta_path}")
    print(f"Schema detail written to: {detail_path}")


if __name__ == "__main__":
    main()
