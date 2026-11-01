"""Indexed in-memory and JSON-persisted storage for TSSDA reuse records."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from ..config import get_default_memory_path
from .expressions import ExpressionMemoryRecord, ExpressionMemoryStore
from .matcher import MemoryMatchConfig, MemoryRetrievalResult, StateMemoryMatcher
from .models import (
    MemoryIndex,
    MemoryStatistics,
    StateMemoryProfile,
    StateMemoryRecord,
    ValidatedStateTransition,
)


class StateMemoryStore:
    """Store records under a fast phase/skill/P0-intent primary index."""

    def __init__(
        self,
        records: Iterable[StateMemoryRecord] | None = None,
        *,
        storage_path: str | Path | None = None,
    ):
        self._index: dict[MemoryIndex, dict[str, StateMemoryRecord]] = {}
        self.expressions = ExpressionMemoryStore()
        self.storage_path = Path(storage_path or get_default_memory_path())
        for record in records or []:
            self.add(record)

    def admit(
        self,
        *,
        index: MemoryIndex,
        profile: StateMemoryProfile,
        selected_action: Any,
        state_delta: dict[str, Any] | None = None,
        statistics: MemoryStatistics | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateMemoryRecord:
        """Create and add one validated transition record."""

        record = StateMemoryRecord(
            index=index,
            profile=profile,
            transition=ValidatedStateTransition(
                selected_action=deepcopy(selected_action),
                state_delta=deepcopy(state_delta or {}),
            ),
            statistics=statistics or MemoryStatistics(),
            metadata=deepcopy(metadata or {}),
        )
        return self.add(record)

    def add(self, record: StateMemoryRecord) -> StateMemoryRecord:
        """Add a record, merging counters for an identical transition."""

        stored = StateMemoryRecord.from_dict(record.to_dict())
        bucket = self._index.setdefault(stored.index, {})
        existing = bucket.get(stored.record_key)
        if existing is not None:
            existing.statistics.merge(stored.statistics)
            existing.metadata.update(deepcopy(stored.metadata))
            return StateMemoryRecord.from_dict(existing.to_dict())
        bucket[stored.record_key] = stored
        return StateMemoryRecord.from_dict(stored.to_dict())

    def candidates(
        self,
        *,
        phase_id: str,
        skill_id: str,
        intent: str,
        action_space_key: str | None = None,
    ) -> list[StateMemoryRecord]:
        """Return candidates from the primary index, optionally action-filtered."""

        index = MemoryIndex(phase_id=phase_id, skill_id=skill_id, intent=intent)
        records = self._index.get(index, {}).values()
        if action_space_key is not None:
            records = (
                record
                for record in records
                if record.profile.action_space_key == action_space_key
            )
        return [StateMemoryRecord.from_dict(record.to_dict()) for record in records]

    def profile_candidates(
        self,
        *,
        index: MemoryIndex,
        profile_key: str,
    ) -> list[StateMemoryRecord]:
        """Return every transition stored for an exact non-index profile."""

        return [
            StateMemoryRecord.from_dict(record.to_dict())
            for record in self._index.get(index, {}).values()
            if record.profile_key == profile_key
        ]

    def retrieve(
        self,
        *,
        phase_id: str,
        skill_id: str,
        intent: str,
        profile: StateMemoryProfile,
        config: MemoryMatchConfig | None = None,
        top_k: int | None = None,
    ) -> MemoryRetrievalResult:
        """Run strict indexed lookup and compatibility-gated profile matching."""

        index = MemoryIndex(
            phase_id=phase_id,
            skill_id=skill_id,
            intent=intent,
        )
        records = [
            StateMemoryRecord.from_dict(record.to_dict())
            for record in self._index.get(index, {}).values()
        ]
        return StateMemoryMatcher(config=config).retrieve(
            index=index,
            current_profile=profile,
            candidates=records,
            top_k=top_k,
        )

    def record_validation(
        self,
        *,
        index: MemoryIndex,
        record_id: str | None = None,
        record_key: str | None = None,
        success: bool,
        count: int = 1,
    ) -> MemoryStatistics:
        record = self._require_record(
            index=index,
            record_id=self._resolve_record_id(record_id, record_key),
        )
        record.statistics.record_validation(success=success, count=count)
        return MemoryStatistics.from_dict(record.statistics.to_dict())

    def record_reuse(
        self,
        *,
        index: MemoryIndex,
        record_id: str | None = None,
        record_key: str | None = None,
        count: int = 1,
    ) -> MemoryStatistics:
        record = self._require_record(
            index=index,
            record_id=self._resolve_record_id(record_id, record_key),
        )
        record.statistics.record_reuse(count=count)
        return MemoryStatistics.from_dict(record.statistics.to_dict())

    def refine_temporal_profile(
        self,
        *,
        index: MemoryIndex,
        record_id: str,
        failure_signature: dict[str, Any],
        max_contexts: int = 8,
        similarity_threshold: float = 0.95,
    ) -> StateMemoryRecord:
        """Tighten one record with a failed normalized temporal context."""

        record = self._require_record(index=index, record_id=record_id)
        record.profile.temporal.record_failure_context(
            failure_signature,
            max_contexts=max_contexts,
            similarity_threshold=similarity_threshold,
        )
        return StateMemoryRecord.from_dict(record.to_dict())

    def get(self, *, index: MemoryIndex, record_id: str) -> StateMemoryRecord:
        """Return one record by its stable identifier."""

        return StateMemoryRecord.from_dict(
            self._require_record(index=index, record_id=record_id).to_dict()
        )

    def remove(
        self,
        *,
        index: MemoryIndex,
        record_id: str | None = None,
        record_key: str | None = None,
    ) -> bool:
        resolved_id = self._resolve_record_id(record_id, record_key)
        bucket = self._index.get(index)
        if bucket is None or resolved_id not in bucket:
            return False
        del bucket[resolved_id]
        if not bucket:
            del self._index[index]
        return True

    def clear(self) -> None:
        self._index.clear()

    @property
    def record_count(self) -> int:
        return sum(len(records) for records in self._index.values())

    @property
    def index_count(self) -> int:
        return len(self._index)

    def to_dict(self) -> dict[str, Any]:
        records = [
            record.to_dict()
            for bucket in self._index.values()
            for record in bucket.values()
        ]
        records.sort(key=lambda record: record["record_key"])
        return {
            "memory_type": "hierarchical",
            "format_version": 2,
            "records": records,
            "expression_records": self.expressions.to_list(),
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        storage_path: str | Path | None = None,
    ) -> "StateMemoryStore":
        records = [
            StateMemoryRecord.from_dict(item)
            for item in value.get("records", [])
        ]
        store = cls(records=records, storage_path=storage_path)
        store.expressions = ExpressionMemoryStore(
            ExpressionMemoryRecord.from_dict(item)
            for item in value.get("expression_records", [])
        )
        return store

    def save(self, path: str | Path | None = None) -> Path:
        """Atomically persist every memory record to one local JSON file.

        When ``path`` is omitted, the store writes back to the path it was
        loaded from, or to ``TSSDA_MEMORY_PATH``/the workspace default for a
        newly created store.
        """

        target = Path(path) if path is not None else self.storage_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(payload)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, target)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

        self.storage_path = target
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> "StateMemoryStore":
        """Load a local memory file, or return an empty path-bound store."""

        source = Path(path) if path is not None else get_default_memory_path()
        if not source.exists():
            return cls(storage_path=source)
        return cls.from_dict(
            json.loads(source.read_text(encoding="utf-8")),
            storage_path=source,
        )

    @classmethod
    def load_or_create(
        cls,
        path: str | Path | None = None,
    ) -> "StateMemoryStore":
        """Load persisted state memory, creating an empty file when absent."""

        source = Path(path) if path is not None else get_default_memory_path()
        store = cls.load(source)
        if not source.exists():
            store.save(source)
        return store

    def _require_record(
        self,
        *,
        index: MemoryIndex,
        record_id: str,
    ) -> StateMemoryRecord:
        record = self._index.get(index, {}).get(record_id)
        if record is None:
            raise KeyError(
                "Unknown memory record "
                f"{record_id!r} under index {index.to_dict()}"
            )
        return record

    @staticmethod
    def _resolve_record_id(
        record_id: str | None,
        record_key: str | None,
    ) -> str:
        resolved = str(record_id or record_key or "").strip()
        if not resolved:
            raise ValueError("record_id must not be empty")
        return resolved
