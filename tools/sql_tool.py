"""Run read-only DuckDB queries against the movies table."""
from __future__ import annotations

import re
from typing import Any

from config import SQL_ROW_LIMIT, SQL_ROWS_TO_LLM
from data.loader import SCHEMA_DESCRIPTION, get_conn
from tools._util import normalize_rows

SQL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "sql_query",
        "description": (
            "Run a read-only DuckDB SELECT/WITH query against the `movies` table. "
            "Use for all structured questions: top-N, filters, sorts, aggregations, "
            "counts, GROUP BY / HAVING. Returns a list of row dicts.\n\n"
            + SCHEMA_DESCRIPTION
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single DuckDB SELECT (or WITH) statement. No DDL/DML.",
                },
            },
            "required": ["sql"],
        },
    },
}

_ALLOWED_START = re.compile(r"^\s*(with|select)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|copy|pragma|call|truncate)\b",
    re.IGNORECASE,
)


def _validate(sql: str) -> str | None:
    if not sql or not sql.strip():
        return "empty query"
    if ";" in sql.rstrip().rstrip(";"):
        return "multiple statements are not allowed"
    if not _ALLOWED_START.match(sql):
        return "only SELECT / WITH queries are allowed"
    if _FORBIDDEN.search(sql):
        return "query contains a forbidden keyword"
    return None


def _inject_limit(sql: str, limit: int) -> str:
    if re.search(r"\blimit\s+\d+\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip().rstrip(';')} LIMIT {limit}"


def sql_query(sql: str) -> dict[str, Any]:
    err = _validate(sql)
    if err:
        return {"error": err, "sql": sql}

    safe_sql = _inject_limit(sql, SQL_ROW_LIMIT)
    conn = get_conn()
    try:
        df = conn.execute(safe_sql).fetch_df()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "sql": safe_sql}

    full_rows = normalize_rows(df.to_dict(orient="records"))

    return {
        "sql": safe_sql,
        "row_count": len(full_rows),
        "rows": full_rows[:SQL_ROWS_TO_LLM],
        "truncated": len(full_rows) > SQL_ROWS_TO_LLM,
        "full_rows": full_rows,
    }
