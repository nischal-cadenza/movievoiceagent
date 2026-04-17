# Bug Tracker

Format: `[Date] [Severity] Title — Status`

## Open
_None yet._

## Resolved

### [2026-04-16] Medium — `TypeError: boolean value of NA is ambiguous` in build_index
**Where:** `data/build_index.py` when iterating nullable Int64 columns (`gross_usd`, `released_year`).
**Cause:** `if row.gross_usd` triggers `__bool__` on pandas NA.
**Fix:** Use `pd.notna(value)` everywhere for nullable checks.

### [2026-04-16] Medium — "truth value of an array is ambiguous" in recommend / sql row serialisation
**Where:** `tools/recommend_tool.py` and `tools/sql_tool.py` while post-processing rows that contain array columns (`genres`, `all_stars`).
**Cause:** `if v == v` / `if v` on numpy ndarrays returns an element-wise array, not a bool.
**Fix:** Factored a `tools/_util.py::to_jsonable()` helper that handles ndarray/NaN/numpy scalars explicitly; sql and recommend tools now use `normalize_rows()` on all outbound dicts.

### [2026-04-16] Low — Model did not call `recommend_similar` reliably
**Where:** `gpt-4o-mini` often skipped the follow-up tool call even when prompted to.
**Fix:** Made recommendations deterministic — `agent._auto_recommend()` runs after the LLM's final text and pulls reference titles from any `sql_query` / `semantic_search` table that returned specific movies. Removes a source of LLM flakiness and guarantees the 'You might also like' block appears.
**Commit:** see agent.py changes.
