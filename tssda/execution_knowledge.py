"""Validated repair experience, indexed by IoTDB error code."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_default_execution_knowledge_path


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class ExecutionKnowledgeBase:
    """Admit successful repair rules and track outcomes of retrieved advice."""

    def __init__(
        self, path: str | Path | None = None, *,
        deletion_confidence_threshold: float = 0.35,
        min_failures_before_deletion: int = 3,
    ):
        if not 0 <= deletion_confidence_threshold <= 1:
            raise ValueError("Deletion confidence threshold must be between 0 and 1")
        if min_failures_before_deletion < 1:
            raise ValueError("Minimum failures before deletion must be positive")
        self.storage_path = Path(path or get_default_execution_knowledge_path())
        self.deletion_confidence_threshold = deletion_confidence_threshold
        self.min_failures_before_deletion = min_failures_before_deletion
        self.error_types: dict[str, dict[str, Any]] = {}
        self._legacy_bytes: bytes | None = None
        if self.storage_path.exists():
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if data.get("version") == 1 and isinstance(data.get("errors"), dict):
                # Old failures are not successful experience. Preserve them in a
                # backup before the first v2 write, without admitting any as rules.
                self._legacy_bytes = self.storage_path.read_bytes()
            elif data.get("version") == 2 and isinstance(data.get("error_types"), dict):
                self.error_types = data["error_types"]
                self._validate_loaded_rules()
            else:
                raise ValueError("Unsupported execution knowledge-base format")

    @staticmethod
    def normalize_error_type(error_code: Any) -> str:
        if isinstance(error_code, bool) or not re.fullmatch(r"\d+", str(error_code).strip()):
            raise ValueError("Knowledge error_type must be a numeric IoTDB error code")
        return str(int(str(error_code).strip()))

    @classmethod
    def error_type_for_failure(cls, failure: dict[str, Any]) -> str | None:
        if failure.get("stage") != "execution":
            return None
        try:
            return cls.normalize_error_type(failure.get("error_code"))
        except ValueError:
            return None

    @property
    def record_count(self) -> int:
        return sum(len(group["rules"]) for group in self.error_types.values())

    def lookup(self, error_type: str | int) -> dict[str, Any] | None:
        return deepcopy(self.error_types.get(self.normalize_error_type(error_type)))

    def top_rules(self, error_type: str | int, limit: int = 5) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("Rule retrieval limit must be positive")
        group = self.error_types.get(self.normalize_error_type(error_type), {})
        records = sorted(group.get("rules", {}).values(), key=lambda record: (
            -record["success_count"], record["failure_count"], record["rule_id"],
        ))
        return deepcopy(records[:limit])

    def settle_repair(
        self, *, retrieved: dict[str, list[dict[str, Any]]],
        outcomes: dict[str, bool], rule: str | None,
        run_id: str, attempt: int,
    ) -> dict[str, list[str]]:
        """Credit each retrieved rule once per repair round, then admit new advice.

        Outcomes concern the current repair targets, not the trajectory's earlier
        successful samples. A shared one-sentence rule is admitted only after the
        caller confirms that the whole set of repair targets has passed.
        """

        updates: dict[str, list[str]] = {"updated": [], "admitted": [], "deleted": []}
        now = datetime.now(timezone.utc).isoformat()
        for code, success in outcomes.items():
            code = self.normalize_error_type(code)
            group = self.error_types.get(code, {})
            records = group.get("rules", {})
            credited = set()
            for supplied in retrieved.get(code, []):
                rule_id = supplied["rule_id"]
                record = records.get(rule_id)
                if record is None or rule_id in credited:
                    continue
                credited.add(rule_id)
                record["success_count" if success else "failure_count"] += 1
                record.update(last_used_at=now, last_run_id=run_id, last_attempt=attempt)
                updates["updated"].append(rule_id)
                confidence = (record["success_count"] + 1) / (
                    record["success_count"] + record["failure_count"] + 2
                )
                if not success and (
                    record["failure_count"] >= self.min_failures_before_deletion
                    and confidence <= self.deletion_confidence_threshold
                ):
                    del records[rule_id]
                    updates["deleted"].append(rule_id)
            if success and rule:
                text = " ".join(rule.split())
                rule_id = _fingerprint([code, text.casefold()])
                if rule_id not in credited:
                    group = self.error_types.setdefault(code, {"error_type": code, "rules": {}})
                    record = group["rules"].setdefault(rule_id, {
                        "rule_id": rule_id, "rule": text, "success_count": 0,
                        "failure_count": 0, "created_at": now,
                    })
                    record["success_count"] += 1
                    record.update(last_used_at=now, last_run_id=run_id, last_attempt=attempt)
                    updates["admitted"].append(rule_id)
            if code in self.error_types and not self.error_types[code]["rules"]:
                del self.error_types[code]
        if any(updates.values()):
            self.save()
        return updates

    def _validate_loaded_rules(self) -> None:
        for code, group in self.error_types.items():
            if self.normalize_error_type(code) != code or group.get("error_type") != code:
                raise ValueError("Invalid error-code bucket in knowledge base")
            if not isinstance(group.get("rules"), dict):
                raise ValueError("Knowledge rules must be indexed by rule_id")
            for rule_id, record in group["rules"].items():
                text = record.get("rule")
                if not isinstance(text, str) or not text.strip() or record.get("rule_id") != rule_id:
                    raise ValueError("Invalid knowledge rule")
                if _fingerprint([code, " ".join(text.split()).casefold()]) != rule_id:
                    raise ValueError("Knowledge rule_id does not match its rule")
                for key, minimum in (("success_count", 1), ("failure_count", 0)):
                    value = record.get(key)
                    if type(value) is not int or value < minimum:
                        raise ValueError(f"Invalid knowledge {key}")

    def save(self) -> Path:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self._legacy_bytes is not None:
            suffix = hashlib.sha256(self._legacy_bytes).hexdigest()[:12]
            backup = self.storage_path.with_name(f"{self.storage_path.name}.v1-{suffix}.bak")
            if not backup.exists():
                with backup.open("xb") as handle:
                    handle.write(self._legacy_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            elif backup.read_bytes() != self._legacy_bytes:
                raise ValueError("Existing legacy knowledge backup has different contents")
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.storage_path.parent,
                prefix=f".{self.storage_path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump({"version": 2, "error_types": self.error_types}, handle,
                          ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.storage_path)
            self._legacy_bytes = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return self.storage_path
