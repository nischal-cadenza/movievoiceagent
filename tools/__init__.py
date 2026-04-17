"""Tool registry for the OpenAI function-calling agent.

Exposes:
  TOOL_SCHEMAS — list of OpenAI tool definitions
  dispatch(name, args_json) — run a tool by name and return its result
"""
from __future__ import annotations

import json
from typing import Any

from tools.clarify_tool import CLARIFY_SCHEMA, request_clarification
from tools.recommend_tool import RECOMMEND_SCHEMA, recommend_similar
from tools.semantic_tool import SEMANTIC_SCHEMA, semantic_search
from tools.sql_tool import SQL_SCHEMA, sql_query

TOOL_SCHEMAS = [SQL_SCHEMA, SEMANTIC_SCHEMA, RECOMMEND_SCHEMA, CLARIFY_SCHEMA]

_HANDLERS = {
    "sql_query": sql_query,
    "semantic_search": semantic_search,
    "recommend_similar": recommend_similar,
    "request_clarification": request_clarification,
}


def dispatch(name: str, args_json: str) -> dict[str, Any]:
    try:
        kwargs = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        return {"error": f"invalid args JSON: {e}"}
    handler = _HANDLERS.get(name)
    if not handler:
        return {"error": f"unknown tool: {name}"}
    try:
        return handler(**kwargs)
    except Exception as e:  # surfacing errors to the model so it can retry
        return {"error": f"{type(e).__name__}: {e}"}
