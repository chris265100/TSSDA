"""Schema state for the TSSDA data generation space.

The schema object has three responsibilities:

1. keep lightweight database meta information for agent decisions;
2. keep detailed database schema information for exact path/type lookup;
3. keep a mutable working schema that records the current SQL scope.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .config import get_default_schema_root


@dataclass
class Measurement:
    """A measurable field under a tree-model device."""

    name: str
    data_type: str | None = None
    path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Device:
    """A tree-model device with measurements and optional metadata."""

    path: str
    tags: dict[str, str] = field(default_factory=dict)
    database: str | None = None
    is_aligned: bool | None = None
    measurements: list[Measurement] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def measurement_by_name(self, name: str) -> Measurement | None:
        """Return a measurement by name when it exists on this device."""

        return next(
            (measurement for measurement in self.measurements if measurement.name == name),
            None,
        )


@dataclass
class WorkingSchema:
    """Mutable schema scope selected during one SQL generation trajectory."""

    root: str | None = None
    schema_view: dict[str, Any] | None = None
    scope_pattern: str | None = None
    working_path: str | None = None
    devices: list[str] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)
    timeseries_paths: list[str] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    updated_by_phase: str | None = None
    updated_by_action: str | None = None

    def update(self, patch: dict[str, Any], phase_id: str | None = None) -> None:
        """Merge a patch into the working schema."""

        if not patch:
            return

        if self._is_schema_view_patch(patch):
            schema_view = deepcopy(patch)
            schema_view.pop("action_id", None)
            schema_view.pop("updated_by_action", None)
            schema_view.pop("updated_by_phase", None)
            self.schema_view = schema_view
            self.root = schema_view.get("database") or schema_view.get("root") or self.root
            self.devices = [
                device.get("path")
                for device in schema_view.get("devices", [])
                if isinstance(device, dict) and device.get("path")
            ]
            action_id = patch.get("action_id") or patch.get("updated_by_action")
            if phase_id is not None:
                self.updated_by_phase = phase_id
            if action_id is not None:
                self.updated_by_action = str(action_id)
            return

        action_id = patch.get("action_id") or patch.get("updated_by_action")
        for key, value in patch.items():
            if key in {"action_id", "updated_by_action", "updated_by_phase"}:
                continue
            if hasattr(self, key):
                setattr(self, key, deepcopy(value))
            else:
                self.extra[key] = deepcopy(value)

        if phase_id is not None:
            self.updated_by_phase = phase_id
        if action_id is not None:
            self.updated_by_action = str(action_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict for skills and LLM prompts."""

        if self.schema_view is not None:
            return deepcopy(self.schema_view)

        return {
            "root": self.root,
            "scope_pattern": self.scope_pattern,
            "working_path": self.working_path,
            "devices": deepcopy(self.devices),
            "measurements": deepcopy(self.measurements),
            "timeseries_paths": deepcopy(self.timeseries_paths),
            "tags": deepcopy(self.tags),
            "extra": deepcopy(self.extra),
            "updated_by_phase": self.updated_by_phase,
            "updated_by_action": self.updated_by_action,
        }

    def is_active(self) -> bool:
        """Return whether this working schema has been formally populated."""

        return bool(
            self.schema_view
            or self.scope_pattern
            or self.working_path
            or self.devices
            or self.measurements
            or self.timeseries_paths
            or self.tags
            or self.extra
            or self.updated_by_phase
            or self.updated_by_action
        )

    def _is_schema_view_patch(self, patch: dict[str, Any]) -> bool:
        """Return whether a patch is a meta-schema-shaped working schema."""

        return bool(
            isinstance(patch, dict)
            and "schema_type" in patch
            and "database" in patch
            and "devices" in patch
        )


@dataclass
class DatabaseSchema:
    """Database schema visible to the data agent and TSSDA phases."""

    schema_root: str = field(default_factory=get_default_schema_root)
    schema_type: str = "iotdb_tree"
    meta_schema: dict[str, Any] = field(default_factory=dict)
    detail_schema: dict[str, Any] = field(default_factory=dict)
    tag_schema: dict[str, Any] = field(default_factory=dict)
    working_schema: WorkingSchema = field(default_factory=WorkingSchema)

    @classmethod
    def from_access(
        cls,
        schema_root: str | None = None,
        session: Any | None = None,
    ) -> "DatabaseSchema":
        """Create a schema object using iotdb_tools.schema_access."""

        from iotdb_tools.schema_access import get_schema_bundle

        schema_root = schema_root or get_default_schema_root()
        meta_schema, detail_schema = get_schema_bundle(
            schema_root=schema_root,
            session=session,
        )
        return cls.from_raw(
            meta_schema=meta_schema,
            detail_schema=detail_schema,
            schema_root=schema_root,
        )

    @classmethod
    def from_raw(
        cls,
        meta_schema: dict[str, Any],
        detail_schema: dict[str, Any] | None = None,
        tag_schema: dict[str, Any] | None = None,
        schema_root: str | None = None,
    ) -> "DatabaseSchema":
        """Create a schema object from already fetched schema dictionaries."""

        detail_schema = deepcopy(detail_schema or {})
        root = (
            schema_root
            or meta_schema.get("root")
            or meta_schema.get("database")
            or detail_schema.get("root")
            or detail_schema.get("database")
            or get_default_schema_root()
        )
        schema_type = meta_schema.get("schema_type") or "iotdb_tree"
        return cls(
            schema_root=root,
            schema_type=schema_type,
            meta_schema=deepcopy(meta_schema),
            detail_schema=detail_schema,
            tag_schema=deepcopy(
                tag_schema
                if tag_schema is not None
                else detail_schema.get("tag_schema", {})
            ),
            working_schema=WorkingSchema(root=root),
        )

    def to_agent_context(self, include_detail: bool = False) -> dict[str, Any]:
        """Return schema context safe for skills to inspect."""

        context = {
            "schema_root": self.schema_root,
            "schema_type": self.schema_type,
            "meta_schema": self._build_agent_meta_schema(),
            "tag_schema": deepcopy(self.tag_schema),
            "working_schema": self.get_active_working_schema(),
        }
        if include_detail:
            context["detail_schema"] = deepcopy(self.detail_schema)
        return context

    def update_working_schema(
        self,
        patch: dict[str, Any],
        phase_id: str | None = None,
    ) -> None:
        """Update the mutable working schema during agent execution."""

        self.working_schema.update(patch=patch, phase_id=phase_id)

    def get_working_schema(self) -> dict[str, Any]:
        """Return the current working schema as a plain dict."""

        return self.working_schema.to_dict()

    def get_active_working_schema(self) -> dict[str, Any] | None:
        """Return working schema only after it has been formally populated."""

        if not self.working_schema.is_active():
            return None
        return self.working_schema.to_dict()

    def reset_working_schema(self) -> None:
        """Reset working scope while keeping database schema information."""

        self.working_schema = WorkingSchema(root=self.schema_root)

    @property
    def devices(self) -> list[Device]:
        """Return meta devices as typed objects for convenient lookup."""

        devices = []
        for device_data in self.meta_schema.get("devices", []):
            measurements = [
                Measurement(
                    name=item.get("name") or item.get("measurement", ""),
                    data_type=item.get("data_type"),
                    path=item.get("path"),
                    meta=deepcopy(item),
                )
                for item in device_data.get("measurements", [])
            ]
            devices.append(
                Device(
                    path=device_data.get("path", ""),
                    tags=deepcopy(device_data.get("tags", {})),
                    database=device_data.get("database"),
                    is_aligned=device_data.get("is_aligned"),
                    measurements=measurements,
                    meta=deepcopy(device_data),
                )
            )
        return devices

    def device_by_path(self, path: str) -> Device | None:
        """Return a device by full path."""

        return next((device for device in self.devices if device.path == path), None)

    def timeseries_by_path(self, path: str) -> dict[str, Any] | None:
        """Return a detailed timeseries record by full path."""

        for series in self.detail_schema.get("timeseries", []):
            if series.get("path") == path:
                return deepcopy(series)
        return None

    def devices_under(self, prefix: str | None = None) -> list[str]:
        """Return device paths under a prefix, defaulting to the schema root."""

        root = (prefix or self.schema_root).rstrip(".")
        return [
            device.get("path", "")
            for device in self.meta_schema.get("devices", [])
            if device.get("path", "").startswith(root)
        ]

    def timeseries_under(self, prefix: str | None = None) -> list[str]:
        """Return timeseries paths under a prefix, defaulting to the schema root."""

        root = (prefix or self.schema_root).rstrip(".")
        return [
            path
            for path in self.meta_schema.get("timeseries_paths", [])
            if str(path).startswith(root)
        ]

    def _build_agent_meta_schema(self) -> dict[str, Any]:
        """Return a compact meta schema for LLM decision inputs."""

        meta_schema = deepcopy(self.meta_schema)
        root = meta_schema.pop("root", self.schema_root)
        meta_schema.pop("measurements", None)
        meta_schema.pop("timeseries_paths", None)
        return {
            "schema_type": meta_schema.get("schema_type"),
            "database": root,
            **{
                key: value
                for key, value in meta_schema.items()
                if key != "schema_type"
            },
        }


class TreeSchema(DatabaseSchema):
    """Backward-compatible constructor for the earlier minimal runner."""

    def __init__(
        self,
        schema_id: str,
        model: str = "tree",
        devices: list[Device] | None = None,
        functions: dict[str, list[str]] | None = None,
    ) -> None:
        devices = devices or []
        root = self._infer_root(devices)
        meta_devices = []
        detail_devices = []
        detail_timeseries = []

        for device in devices:
            meta_measurements = []
            detail_series = []
            for measurement in device.measurements:
                series_path = measurement.path or f"{device.path}.{measurement.name}"
                meta_measurement = {
                    "name": measurement.name,
                    "path": series_path,
                    "data_type": measurement.data_type,
                }
                detail = {
                    "path": series_path,
                    "device": device.path,
                    "measurement": measurement.name,
                    "data_type": measurement.data_type,
                    "tags": deepcopy(device.tags),
                }
                meta_measurements.append(meta_measurement)
                detail_series.append(detail)
                detail_timeseries.append(detail)

            meta_devices.append(
                {
                    "path": device.path,
                    "database": device.database or root,
                    "is_aligned": device.is_aligned,
                    "tags": deepcopy(device.tags),
                    "measurements": meta_measurements,
                }
            )
            detail_devices.append(
                {
                    "path": device.path,
                    "database": device.database or root,
                    "is_aligned": device.is_aligned,
                    "tags": deepcopy(device.tags),
                    "timeseries": detail_series,
                }
            )

        meta_schema = {
            "schema_id": schema_id,
            "schema_type": model,
            "root": root,
            "device_count": len(meta_devices),
            "timeseries_count": len(detail_timeseries),
            "devices": meta_devices,
            "measurements": sorted(
                {
                    measurement["name"]
                    for device in meta_devices
                    for measurement in device["measurements"]
                }
            ),
            "timeseries_paths": [series["path"] for series in detail_timeseries],
            "functions": deepcopy(functions or {}),
        }
        detail_schema = {
            "schema_id": schema_id,
            "schema_type": model,
            "root": root,
            "device_count": len(detail_devices),
            "timeseries_count": len(detail_timeseries),
            "devices": detail_devices,
            "timeseries": detail_timeseries,
            "functions": deepcopy(functions or {}),
        }
        tagged_timeseries = [
            {
                "path": series["path"],
                "device": series["device"],
                "measurement": series["measurement"],
                "tags": deepcopy(series["tags"]),
            }
            for series in detail_timeseries
            if series.get("tags")
        ]
        tag_schema = {}
        if tagged_timeseries:
            tag_schema = {
                "schema_type": "iotdb_tree_tags",
                "database": root,
                "timeseries_count": len(tagged_timeseries),
                "tag_keys": sorted(
                    {
                        tag_key
                        for series in tagged_timeseries
                        for tag_key in series["tags"]
                    }
                ),
                "timeseries": tagged_timeseries,
            }
            detail_schema["tag_schema"] = tag_schema

        super().__init__(
            schema_root=root,
            schema_type=model,
            meta_schema=meta_schema,
            detail_schema=detail_schema,
            tag_schema=tag_schema,
            working_schema=WorkingSchema(root=root),
        )
        self.schema_id = schema_id
        self.model = model
        self.functions = deepcopy(functions or {})

    @staticmethod
    def _infer_root(devices: list[Device]) -> str:
        if not devices:
            return "root"
        path_parts = devices[0].path.split(".")
        return ".".join(path_parts[:2]) if len(path_parts) >= 2 else devices[0].path
