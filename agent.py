"""OpenAI function-calling agent that orchestrates SQL + semantic + recommendations."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from config import CHAT_MODEL, MAX_AGENT_ITERATIONS, OPENAI_API_KEY
from data.loader import SCHEMA_DESCRIPTION
from tools import TOOL_SCHEMAS, dispatch
from tools.clarify_tool import CLARIFY_MARKER
from tools.recommend_tool import recommend_similar as _recommend

MAX_REC_REFERENCES = 5
MAX_REC_RESULTS = 5

SYSTEM_PROMPT = f"""You are IMDB-Voice, a helpful conversational agent that answers questions about the IMDB Top-1000 movies dataset. You will be queried both by voice (transcribed) and text.

You have four tools:

1. `sql_query(sql)` — Run read-only DuckDB queries for all STRUCTURED questions: top-N, filters, sorts, aggregations, GROUP BY, HAVING. Use this whenever the user asks for counts, rankings, numeric filters, exact year/genre/director matches.

2. `semantic_search(query, k, year_min?, year_max?, genre_contains?)` — Embedding search over plot OVERVIEWS. Use when the answer depends on PLOT CONTENT or themes that may not appear verbatim (e.g. 'police involvement', 'coming of age', 'revenge', 'death'). You can pre-filter by year range or genre substring.

3. `recommend_similar(reference_titles, limit)` — Similar-movie recommender. Available if you want to include suggestions inline, but the orchestrator also calls it automatically after your final answer when specific movie titles were returned. You normally don't need to call it yourself.

4. `request_clarification(question, options)` — Ask the user a BEFORE-ANSWERING clarifying question when the query is genuinely ambiguous in a way that changes the filter. The canonical example: 'Al Pacino movies' needs lead-actor vs. any-role disambiguation. Do NOT call this for trivial clarifications.

Dataset schema:
{SCHEMA_DESCRIPTION}

Guidelines:
- Prefer `sql_query` for numeric/aggregation work. Prefer `semantic_search` for plot-content questions.
- For hybrid questions like 'comedy movies with death in plot', call `semantic_search(query='death dying dead', genre_contains='Comedy')`.
- For cast questions like '<Actor> movies', call `request_clarification` first (lead vs. any-role) unless the user has already clarified.
- Always present results with a short **reasoning line** explaining what you did (e.g. 'Queried movies with imdb_rating > 8 and meta_score > 85 in the Horror genre...').
- Keep answers conversational — you will be read aloud. Use short sentences. The UI will show the full table.
- If a table is long, name the top few and reference the count.
- When citing numeric values, use human formats: $500M not 500000000, runtime in 'h m', year as 1994.
- A 'You might also like' block will be auto-appended by the UI from orchestrator-computed recommendations. Don't try to hand-roll recommendations in your text.
- If `sql_query` returns 0 rows, don't hallucinate — say no movies matched and suggest relaxed criteria.
"""


@dataclass
class AgentResponse:
    text: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    clarification: dict[str, Any] | None = None
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    reasoning_trace: list[str] = field(default_factory=list)


def _client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _message_to_dict(msg) -> dict[str, Any]:
    d: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def initial_history() -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def run(messages: list[dict[str, Any]], user_text: str) -> AgentResponse:
    """Append user_text to `messages` (mutated) and run the tool loop.

    Returns an AgentResponse summarising the assistant's final reply and the tool
    artefacts (tables, recommendations, clarification request) for the UI layer.
    """
    messages.append({"role": "user", "content": user_text})

    resp = AgentResponse(text="")
    client = _client()

    for _ in range(MAX_AGENT_ITERATIONS):
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = completion.choices[0].message
        messages.append(_message_to_dict(msg))

        if not msg.tool_calls:
            resp.text = (msg.content or "").strip()
            _auto_recommend(resp)
            return resp

        short_circuit_clarify = False
        for tc in msg.tool_calls:
            name = tc.function.name
            resp.tools_used.append(name)
            result = dispatch(name, tc.function.arguments)

            # Stash UI-usable artefacts
            if name == "sql_query" and "full_rows" in result:
                resp.tables.append({
                    "tool": "sql_query",
                    "sql": result.get("sql"),
                    "rows": result["full_rows"],
                })
                resp.reasoning_trace.append(
                    f"SQL: {result.get('sql','')} → {result.get('row_count',0)} rows"
                )
            elif name == "semantic_search":
                resp.tables.append({
                    "tool": "semantic_search",
                    "query": result.get("query"),
                    "filters": result.get("filters"),
                    "rows": result.get("hits", []),
                })
                resp.reasoning_trace.append(
                    f"Semantic search: '{result.get('query','')}' "
                    f"(filters={result.get('filters')}) → {len(result.get('hits', []))} hits"
                )
            elif name == "recommend_similar":
                resp.recommendations = result.get("recommendations", [])
                basis = result.get("basis", {})
                resp.reasoning_trace.append(
                    f"Recommendations based on {basis.get('reference_titles',[])} "
                    f"(avg IMDB {basis.get('avg_imdb')}, genres {basis.get('genres')})"
                )
            elif name == "request_clarification" and result.get(CLARIFY_MARKER):
                resp.clarification = {
                    "question": result["question"],
                    "options": result["options"],
                }
                short_circuit_clarify = True

            llm_payload = {
                k: v for k, v in result.items()
                if k not in {"full_rows"}  # keep token count manageable
            }
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(llm_payload, default=str),
            })

        if short_circuit_clarify:
            resp.text = resp.clarification["question"]
            return resp

    resp.text = "(The agent hit its iteration limit. Please rephrase or try again.)"
    return resp


def _auto_recommend(resp: AgentResponse) -> None:
    """Populate `resp.recommendations` from the titles returned in the last answer.

    Runs only if the model didn't already populate recommendations and at least
    one table has a `series_title` or `title` column with rows.
    """
    if resp.recommendations or resp.clarification:
        return

    titles: list[str] = []
    for t in resp.tables:
        for row in (t.get("rows") or [])[:MAX_REC_REFERENCES]:
            title = row.get("series_title") or row.get("title")
            if title and title not in titles:
                titles.append(title)
            if len(titles) >= MAX_REC_REFERENCES:
                break
        if len(titles) >= MAX_REC_REFERENCES:
            break

    if not titles:
        return

    try:
        result = _recommend(reference_titles=titles, limit=MAX_REC_RESULTS)
        resp.recommendations = result.get("recommendations", [])
        basis = result.get("basis", {})
        if basis:
            resp.reasoning_trace.append(
                f"Auto-recommendations from {basis.get('reference_titles')} "
                f"(avg IMDB {basis.get('avg_imdb')}, genres {basis.get('genres')})"
            )
    except Exception as e:
        resp.reasoning_trace.append(f"Auto-recommendation skipped: {e}")
