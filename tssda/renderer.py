"""Tree-model SQL renderer for the minimal TSSDA query frame."""

from __future__ import annotations

from .state import QueryFrame


class TreeSQLRenderer:
    """Render a QueryFrame into a single Timecho/IoTDB tree-model SQL string."""

    def render(self, query_frame: QueryFrame) -> str:
        """Render SQL using the fixed MVP clause order."""

        self._validate(query_frame)
        clauses = [self._render_select(query_frame)]

        if query_frame.into:
            clauses.append(self._render_into(query_frame.into))
        clauses.append(f"FROM {', '.join(query_frame.from_paths)}")
        if query_frame.where:
            clauses.append(f"WHERE {' AND '.join(query_frame.where)}")
        if query_frame.group_by:
            clauses.append(
                "GROUP BY "
                + ", ".join(
                    self._strip_clause_keyword(item, "GROUP BY")
                    for item in query_frame.group_by
                )
            )
        if query_frame.having:
            clauses.append(f"HAVING {' AND '.join(query_frame.having)}")
        if query_frame.order_by:
            clauses.append(f"ORDER BY {', '.join(item.upper() if item.lower() in {'asc', 'desc'} else item.replace(' asc', ' ASC').replace(' desc', ' DESC') for item in query_frame.order_by)}")
        if query_frame.fill:
            clauses.append(self._render_fill(query_frame.fill))
        for field_name, keyword in (
            ("slimit", "SLIMIT"),
            ("soffset", "SOFFSET"),
            ("limit", "LIMIT"),
            ("offset", "OFFSET"),
        ):
            value = getattr(query_frame, field_name)
            if value is not None:
                clauses.append(f"{keyword} {value}")
        if query_frame.align_by and query_frame.align_by.lower() == "device":
            clauses.append("ALIGN BY DEVICE")

        return " ".join(clauses)

    def _validate(self, query_frame: QueryFrame) -> None:
        if not query_frame.select:
            raise ValueError("QueryFrame.select must not be empty before rendering")
        if not query_frame.from_paths:
            raise ValueError("QueryFrame.from_paths must not be empty before rendering")

    def _render_select(self, query_frame: QueryFrame) -> str:
        if isinstance(query_frame.select, str):
            keyword = (
                "SELECT LAST"
                if query_frame.sql_query_mode == "select_last"
                else "SELECT"
            )
            return f"{keyword} {query_frame.select.strip()}"

        if query_frame.select and all(
            isinstance(value, str) for value in query_frame.select
        ):
            if len(query_frame.select) != 1:
                raise ValueError(
                    "TreeSQLRenderer renders one SQL statement and requires "
                    "exactly one QueryFrame.select group"
                )
            keyword = (
                "SELECT LAST"
                if query_frame.sql_query_mode == "select_last"
                else "SELECT"
            )
            return f"{keyword} {query_frame.select[0].strip()}"

        if query_frame.sql_query_mode == "select_last":
            exprs = [target["expr"] for target in query_frame.select]
            return f"SELECT LAST {', '.join(exprs)}"

        items = []
        for target in query_frame.select:
            expr = target["expr"]
            alias = target.get("alias")
            items.append(f"{expr} AS {alias}" if alias else expr)
        return f"SELECT {', '.join(items)}"

    @staticmethod
    def _strip_clause_keyword(value: str, keyword: str) -> str:
        text = str(value).strip()
        if text.upper().startswith(keyword):
            return text[len(keyword) :].strip()
        return text

    def _render_into(self, into: dict) -> str:
        clause = into.get("clause")
        if clause:
            return str(clause).strip()
        mappings = into.get("mappings")
        if isinstance(mappings, list) and mappings:
            mapping_clause = mappings[0].get("clause")
            if mapping_clause:
                return str(mapping_clause).strip()
        if "target_device" in into and "target_measurement" in into:
            return (
                f"INTO {into['target_device']}"
                f"({into['target_measurement']})"
            )
        raise ValueError("QueryFrame.into does not contain a renderable INTO clause")

    def _render_fill(self, fill: dict) -> str:
        clause = fill.get("clause")
        if clause:
            return str(clause).strip()
        mode = fill.get("mode")
        if mode == "PREVIOUS":
            interval = fill.get("interval")
            if interval is None:
                interval = fill.get("parameter")
            return f"FILL(PREVIOUS, {interval})" if interval else "FILL(PREVIOUS)"
        if mode == "LINEAR":
            return "FILL(LINEAR)"
        if mode == "constant":
            return f"FILL({fill.get('value')})"
        return f"FILL({mode})"
