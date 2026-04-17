# Architectural Decisions

## 2026-04-16 — Hybrid tool-calling over pure RAG / pure Text-to-SQL
The query set spans structured aggregations (*top 5 by meta score*, *directors with 2+ $500M films*), text keyword search (*comedy with death in Overview*), semantic similarity (*pre-1990 police involvement*), and summarisation (*Spielberg sci-fi plots*). No single approach handles all of these:

- **Pure RAG** fails on aggregations — embeddings can't compute `COUNT(*) HAVING`.
- **Pure text-to-SQL** fails on plot-content queries where the user's word isn't in the Overview (*"police"* vs a plot about *"detective investigating"*).

Decision: the agent exposes four tools and an LLM router picks per question. Each tool owns one concern:
- `sql_query` — DuckDB over the cleaned in-memory table
- `semantic_search` — Chroma over plot embeddings, with optional metadata pre-filters
- `recommend_similar` — post-answer recommendations
- `request_clarification` — disambiguation turn

## 2026-04-16 — OpenAI native SDK, no LangChain/LlamaIndex
`openai>=1.54` supports tool calling natively. Adding an orchestration framework would add hundreds of MB and more abstraction without functional gain for a 4-tool agent. The loop in `agent.py` is ~80 lines and easier to debug.

## 2026-04-16 — DuckDB in-memory vs. SQLite on disk
DuckDB supports typed arrays (`VARCHAR[]`) natively, which makes genre/cast ANY queries trivial (`'Comedy' = ANY(genres)`). It's also columnar, fast for the 1000-row analytical workload, and doesn't require an on-disk file — the CSV is the source of truth. Re-load on startup is < 50 ms.

## 2026-04-16 — Chroma vs. FAISS
Chroma gives persistence + metadata filtering out of the box (`where={"year": {"$lte": 1989}}`), which is required for Q9 (pre-1990 police). FAISS would need a parallel metadata store. Chroma's on-disk footprint is ~6 MB for 1000 plots — ships easily with the zip.

## 2026-04-16 — Clarification as a tool, not a flag
Modelling `request_clarification` as a tool lets the LLM decide when disambiguation helps. The orchestrator short-circuits on this tool call and the Streamlit UI renders radio options; the user's choice becomes the next user message. This keeps disambiguation turn-shaped (consistent with the rest of the conversation) rather than a parallel UI state machine.

## 2026-04-16 — Press-to-talk voice UX
Real-time streaming mic + barge-in is overkill for a QA agent and painful in Streamlit's rerun model. Press-to-talk + TTS playback covers the assignment's "conversational voice" requirement cleanly with `streamlit-mic-recorder` + `st.audio(autoplay=True)`.

## 2026-04-16 — `gpt-4o-mini` as the default chat model
Strong function-calling, ~10x cheaper than `gpt-4o`, and fast enough for a responsive voice feel. Configurable via `CHAT_MODEL` env for users who want to upgrade.

## 2026-04-16 — Auto-append recommendations in the orchestrator, not the LLM
During verification `gpt-4o-mini` frequently declined to call `recommend_similar` as a follow-up, even with strong prompt guidance. Rather than fight the model, the orchestrator now calls the recommender deterministically in `agent._auto_recommend()` using the titles pulled from any `sql_query`/`semantic_search` result. Benefits: guaranteed "You might also like" block on every query that returned titles; fewer tokens (no extra tool round-trip); LLM stays focused on the primary answer.

## 2026-04-16 — Teach the LLM better SQL via schema-embedded query patterns
`gpt-4o-mini` initially wrote a flat `WHERE gross_usd > 500M` query for "directors with 2+ $500M films" — missing the `GROUP BY ... HAVING` + self-join. Rather than overhaul the system prompt, the canonical CTE pattern is baked into `SCHEMA_DESCRIPTION` inside `data/loader.py`. Since that schema description is passed inside every `sql_query` tool description, the model sees it on every call and picks up the pattern. Worked on first retry.

