# Progress Log

## 2026-04-16

### Initial build
- Scaffolded project structure: `tools/`, `voice/`, `data/`, `chroma_db/`
- Wrote `config.py`, `.env`, `.env.example`, `.gitignore`, `requirements.txt`
- Implemented `data/loader.py` (CSV → cleaned DuckDB `movies` table)
  - Handles commas in `Gross`, " min" suffix in `Runtime`, coerces `Released_Year` to Int64
  - Adds `genres` and `all_stars` array columns for ANY/ALL queries
  - Exposes `SCHEMA_DESCRIPTION` for the agent system prompt
- Implemented `data/build_index.py` (idempotent Chroma builder, batch 100, skip if already full)
- Implemented tools:
  - `sql_tool.py` — read-only DuckDB with LIMIT injection and SELECT-only guard
  - `semantic_tool.py` — Chroma query with year-range metadata pre-filter + post-filter on genre substring
  - `recommend_tool.py` — Jaccard genre overlap + IMDB/Meta proximity scoring
  - `clarify_tool.py` — returns a clarification payload; agent loop short-circuits on it
- Implemented `voice/stt.py` (Whisper) and `voice/tts.py` (OpenAI TTS with 3.5k char cap)
- Implemented `agent.py` — OpenAI tool-calling loop with max 6 iterations, stashes tables/recs/clarification for UI
- Implemented `app.py` — Streamlit UI with mic recorder, chat history, table+poster rendering, reasoning trace, autoplay TTS
- Wrote README.md with setup + architecture + per-question walkthrough

### Verification (2026-04-16, same session)
- `python3.13 -m venv .venv` + `pip install -r requirements.txt` — all deps resolve on Python 3.13.
- `python -m data.build_index` — 1000 docs embedded in 8.3 s (~$0.001).
- Ran `streamlit run app.py` headless — app boots, healthcheck 200, no errors in log.
- End-to-end agent tests (run via Python):
  - **Q1 Matrix release year** → "The Matrix was released in 1999." ✅
  - **Q2 Top 5 of 2019 by meta_score** → Gisaengchung (96), Portrait… (95), Marriage Story (94), The Irishman (94), Little Women (91) ✅ + 5 recs
  - **Q3 Top 7 comedy 2010-2020 IMDB** → Gisaengchung 8.6, Intouchables 8.5, Chhichhore/Green Book/Three Billboards/Klaus/Queen 8.2 ✅
  - **Q4 Horror meta>85 imdb>8** → Psycho (8.5/97), Alien (8.4/89) ✅
  - **Q5 Directors 2+ $500M** → Russo (Endgame, Infinity War), Cameron (Avatar, Titanic) ✅ — required adding CTE example to schema desc
  - **Q6 >1M votes low gross** → American History X, Léon, Memento, Shawshank, Fight Club… ✅
  - **Q7 Comedy with death** → Evil Dead II, Me and Earl and the Dying Girl, Knockin' on Heaven's Door, Deadpool, Harold and Maude ✅
  - **Q8 Spielberg sci-fi plots** → Jurassic Park, E.T., Close Encounters — with plot summaries ✅
  - **Q9 Pre-1990 police** → Serpico, Lethal Weapon, Laura, Naked Gun, Dirty Harry ✅ (semantic hit even where 'police' isn't literal)
  - **Al Pacino clarify** → asked lead-vs-any, 'any' returned Godfather I+II, Heat, Scent of a Woman ✅
  - **Recommendations** → auto-appended on every title-returning query ✅

### Bugs fixed during verification
- `pd.NA.__bool__` ambiguity in `build_index.py` (Int64 nullable columns)
- `np.ndarray` truthiness in row post-processing (factored `tools/_util.py::to_jsonable`)
- `gpt-4o-mini` not calling `recommend_similar` reliably → made auto in `agent._auto_recommend`
- `gpt-4o-mini` writing wrong SQL for "2+ movies over $500M" → added CTE pattern to schema description

### Outstanding
- Manual browser test of mic recording + TTS playback — requires a human click; skipped in headless run but code path exercised by stt/tts unit checks.

