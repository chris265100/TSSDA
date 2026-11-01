"""Execute final SELECT samples on IoTDB before admitting them to the dataset."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from threading import Event, Thread
from time import perf_counter
from typing import Any, Callable


_KNOWN_GROUP_BY_TAG_LIMITATIONS = (
    (
        "group_by_tags_align_by_device",
        re.compile(
            r"^701:\s*GROUP BY TAGS does not support align by device now\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "group_by_tags_expression",
        re.compile(
            r"^701:\s*.+ can't be used in group by tag\.\s*"
            r"It will be supported in the future\.?$",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "group_by_tags_series_pagination",
        re.compile(
            r"^701:\s*Limit or slimit are not supported yet in GROUP BY TAGS\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "group_by_tags_having",
        re.compile(
            r"^701:\s*Having clause is not supported yet in GROUP BY TAGS query\.?$",
            re.IGNORECASE,
        ),
    ),
)

_CONNECTION_ERROR_TYPES = {
    "connecterror",
    "connecttimeout",
    "connectionerror",
    "iotdbconnectionexception",
    "networkerror",
    "readtimeout",
    "timeouterror",
    "ttransportexception",
    "writetimeout",
}
_CONNECTION_MESSAGE_MARKERS = (
    "could not connect",
    "connection aborted",
    "connection refused",
    "connection reset",
    "network error",
    "server is down",
    "socket timeout",
    "timed out",
    "transport not open",
)


@dataclass
class SQLValidationResult:
    passed: bool
    error_message: str = ""
    error_type: str = ""
    error_code: str | int | None = None
    stage: str = "execution"
    row_count: int = 0
    elapsed_ms: float = 0.0
    cleanup_errors: list[str] = field(default_factory=list)
    unsupported_but_valid: bool = False
    unsupported_feature: str = ""
    support_probe_sql: str = ""
    support_probe_row_count: int = 0
    support_probe_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IoTDBSQLValidator:
    """Drain the actual query, including later result pages and execution errors.

    Empty results are valid. Only one read-only SELECT statement is executed.
    """

    def __init__(
        self,
        session_factory: Callable[[], Any] | None = None,
        *,
        timeout_ms: int = 60000,
        wall_timeout_ms: int | None = None,
        connection_failure_limit: int = 3,
    ):
        if timeout_ms <= 0:
            raise ValueError("SQL validation timeout must be positive")
        if wall_timeout_ms is not None and wall_timeout_ms <= 0:
            raise ValueError("SQL validation wall timeout must be positive")
        if connection_failure_limit < 1:
            raise ValueError("Connection failure limit must be at least 1")
        self.session_factory = session_factory
        self.timeout_ms = timeout_ms
        self.wall_timeout_ms = wall_timeout_ms
        self.connection_failure_limit = connection_failure_limit

    def validate(self, sql: str) -> SQLValidationResult:
        """Validate one statement within an optional whole-call deadline.

        The IoTDB ``timeout`` request field covers server-side query execution,
        but it does not bound opening/reconnecting a Thrift socket or waiting
        for a stale RPC response.  The daemon watchdog returns control to the
        trajectory even if an SDK call fails to return.  Socket timeouts still
        clean up the background call shortly afterwards.
        """

        if self.wall_timeout_ms is None:
            return self._validate_direct(sql)

        started = perf_counter()
        completed = Event()
        outcome: dict[str, Any] = {}

        def run_validation() -> None:
            try:
                outcome["result"] = self._validate_direct(sql)
            except BaseException as exc:  # pragma: no cover - defensive guard
                outcome["exception"] = exc
            finally:
                completed.set()

        worker = Thread(
            target=run_validation,
            name="iotdb-sql-validation",
            daemon=True,
        )
        worker.start()
        if not completed.wait(self.wall_timeout_ms / 1000):
            elapsed_ms = round((perf_counter() - started) * 1000, 3)
            return SQLValidationResult(
                passed=False,
                error_message=(
                    "IoTDB validation exceeded the complete wall-clock "
                    f"timeout of {self.wall_timeout_ms} ms"
                ),
                error_type="TimeoutError",
                stage="wall_timeout",
                elapsed_ms=elapsed_ms,
            )
        if "exception" in outcome:
            exc = outcome["exception"]
            return SQLValidationResult(
                passed=False,
                error_message=str(exc),
                error_type=type(exc).__name__,
                stage="validator_worker",
                elapsed_ms=round((perf_counter() - started) * 1000, 3),
            )
        return outcome["result"]

    def _validate_direct(self, sql: str) -> SQLValidationResult:
        started = perf_counter()
        result = SQLValidationResult(passed=False, stage="statement_guard")
        session = dataset = None
        try:
            self._check_statement(sql)
            result.stage = "connection"
            if self.session_factory is None:
                from iotdb_tools.schema_access import open_default_session

                session = open_default_session()
            else:
                session = self.session_factory()
            result.stage = "execution"
            dataset = session.execute_query_statement(sql, timeout=self.timeout_ms)
            while dataset.has_next():
                dataset.next()
                result.row_count += 1
            result.passed = True
        except Exception as exc:
            result.error_message = str(exc)
            result.error_type = type(exc).__name__
            status = getattr(exc, "status", None)
            # SDK versions expose TSStatus either as .status or an exception arg.
            if status is None and exc.args:
                status = exc.args[0]
            result.error_code = getattr(status, "code", None)
            if result.error_code is None and result.error_type == "StatementExecutionException":
                code = re.match(r"^(\d+):", result.error_message)
                if code:
                    result.error_code = int(code.group(1))
            limitation = self._known_server_limitation(result)
            if limitation and session is not None and not self._contains_into(sql):
                probe_sql = self._build_group_by_tags_support_probe(sql)
                if probe_sql:
                    result.support_probe_sql = probe_sql
                    probe_dataset = None
                    try:
                        probe_dataset = session.execute_query_statement(
                            probe_sql,
                            timeout=self.timeout_ms,
                        )
                        while probe_dataset.has_next():
                            probe_dataset.next()
                            result.support_probe_row_count += 1
                        result.unsupported_but_valid = True
                        result.unsupported_feature = limitation
                    except Exception as probe_exc:
                        result.support_probe_error = str(probe_exc)
                    finally:
                        if probe_dataset is not None:
                            try:
                                probe_dataset.close_operation_handle()
                            except Exception as cleanup_exc:
                                result.cleanup_errors.append(str(cleanup_exc))
        finally:
            # A cleanup failure must not hide the original database error or
            # cause a successfully executed SELECT INTO to be sent again.
            for resource, method in (
                (dataset, "close_operation_handle"), (session, "close")
            ):
                if resource is not None:
                    try:
                        getattr(resource, method)()
                    except Exception as exc:
                        result.cleanup_errors.append(str(exc))
            result.elapsed_ms = round((perf_counter() - started) * 1000, 3)
        return result

    @staticmethod
    def is_connection_failure(result: SQLValidationResult | dict[str, Any]) -> bool:
        """Recognize transport failures regardless of their recorded stage."""

        if isinstance(result, SQLValidationResult):
            error_type = result.error_type
            error_message = result.error_message
            stage = result.stage
            passed = result.passed
        else:
            error_type = str(result.get("error_type") or "")
            error_message = str(result.get("error_message") or "")
            stage = str(result.get("stage") or "")
            passed = bool(result.get("passed"))
        if passed:
            return False
        if stage.strip().lower() == "connection":
            return True
        normalized_type = error_type.rsplit(".", 1)[-1].lower()
        if normalized_type in _CONNECTION_ERROR_TYPES:
            return True
        if stage.strip().lower() == "wall_timeout":
            return True
        normalized_message = error_message.lower()
        return any(
            marker in normalized_message for marker in _CONNECTION_MESSAGE_MARKERS
        )

    @staticmethod
    def _check_statement(sql: str) -> None:
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("Expected a non-empty SELECT statement")
        # Ignore quoted literals/identifiers and comments for the safety guard;
        # syntax and semantic validation itself belongs to the IoTDB parser.
        masked = re.sub(
            r"'(?:''|\\.|[^'\\])*'|\"(?:\"\"|\\.|[^\"\\])*\""
            r"|`(?:``|[^`])*`|--[^\r\n]*|/\*[\s\S]*?\*/",
            " ",
            sql,
        ).strip()
        if not re.match(r"SELECT\b", masked, re.IGNORECASE):
            raise ValueError("Only SELECT is allowed")
        if re.search(r"\bINTO\b", masked, re.IGNORECASE):
            raise ValueError("SELECT INTO is disabled in this workflow")
        if ";" in masked.rstrip().removesuffix(";"):
            raise ValueError("Multiple SQL statements are not allowed")

    @staticmethod
    def _known_server_limitation(result: SQLValidationResult) -> str | None:
        """Return a narrow server-limitation label for a known IoTDB message."""

        if result.stage != "execution" or str(result.error_code).strip() != "701":
            return None
        for limitation, pattern in _KNOWN_GROUP_BY_TAG_LIMITATIONS:
            if pattern.fullmatch(result.error_message.strip()):
                return limitation
        return None

    @staticmethod
    def _contains_into(sql: str) -> bool:
        """Avoid compatibility probes that could repeat a SELECT INTO write."""

        masked = re.sub(
            r"'(?:''|\\.|[^'\\])*'|\"(?:\"\"|\\.|[^\"\\])*\""
            r"|`(?:``|[^`])*`|--[^\r\n]*|/\*[\s\S]*?\*/",
            " ",
            sql,
        )
        return re.search(r"\bINTO\b", masked, re.IGNORECASE) is not None

    @classmethod
    def _build_group_by_tags_support_probe(cls, sql: str) -> str | None:
        """Remove only the unsupported grouping feature for a validity probe.

        IoTDB can report the GROUP BY TAGS limitation before checking the SELECT
        expression itself.  Executing this read-only probe prevents malformed
        functions or bad argument types from being mislabeled as valid SQL.
        """

        statement = sql.strip().removesuffix(";").rstrip()
        statement, tag_count = re.subn(
            r",\s*TAGS\s*\([^()]*\)",
            "",
            statement,
            count=1,
            flags=re.IGNORECASE,
        )
        if tag_count != 1:
            return None
        statement = re.sub(
            r"\s+ALIGN\s+BY\s+DEVICE\s*$",
            "",
            statement,
            count=1,
            flags=re.IGNORECASE,
        ).rstrip()
        return cls._remove_device_ordering(statement)

    @staticmethod
    def _remove_device_ordering(sql: str) -> str:
        """Drop DEVICE from ORDER BY after removing ALIGN BY DEVICE."""

        match = re.search(
            r"\bORDER\s+BY\s+(?P<items>.*?)"
            r"(?=\s+(?:FILL|SLIMIT|SOFFSET|LIMIT|OFFSET)\b|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            return sql
        items = [item.strip() for item in match.group("items").split(",")]
        retained = [
            item
            for item in items
            if not re.fullmatch(r"DEVICE(?:\s+(?:ASC|DESC))?", item, re.IGNORECASE)
        ]
        replacement = f"ORDER BY {', '.join(retained)}" if retained else ""
        before = sql[: match.start()].rstrip()
        after = sql[match.end() :].lstrip()
        return " ".join(part for part in (before, replacement, after) if part)
