# Movie Voice Agent

A Gen-AI powered **conversational voice agent** over the IMDB Top-1000 dataset. Ask questions by speaking or typing; the agent routes each question to the right tool (SQL, semantic search, recommendations), explains its reasoning, and replies in a natural voice.

## Quick start

```bash
# 1. Clone / enter the project
git clone https://github.com/nischal-cadenza/movievoiceagent.git
cd movievoiceagent

# 2. Create and activate a Python 3.10+ virtual env
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. API key — already provided in .env for reviewers.
#    If you are starting from .env.example, paste the key there.
cat .env | head -1             # should show OPENAI_API_KEY=sk-proj-...

# 5. Build the Chroma vector index (one-time, ~30s, ~$0.001 of embeddings).
#    Skip this step if chroma_db/ is already populated (shipped in the zip).
python -m data.build_index

# 6. Run
streamlit run app.py
```

Browser opens at `http://localhost:8501`.

## Architecture

```
Voice (mic) ─▶ Whisper STT ─┐
                            ├─▶  Tool-calling Agent (gpt-4o-mini)
Text (chat input) ──────────┘      │
                                   ├─ sql_query       (DuckDB over cleaned CSV)
                                   ├─ semantic_search (Chroma over plot embeddings)
                                   ├─ recommend_similar (genre + rating proximity)
                                   └─ request_clarification (Al-Pacino-style disambiguation)
                                   │
                            ┌──────▼──────┐
                            │  response   │───▶  Streamlit (tables + posters + reasoning)
                            └──────┬──────┘
                                   │
                             OpenAI TTS ──▶ autoplay audio reply
```

**Why hybrid?** A pure-RAG approach fails on aggregations ("top 5 by meta score") and a pure text-to-SQL fails on plot-content queries where the user's word isn't literally in the Overview ("police involvement" doesn't match a plot that says "detective investigating"). The agent picks the right tool per question.

## Models

| Role | Model | Notes |
|---|---|---|
| Chat / tool routing | `gpt-4o-mini` | Fast, cheap, strong function-calling |
| Embeddings | `text-embedding-3-small` | 1536-dim, cosine distance |
| Speech-to-text | `whisper-1` | |
| Text-to-speech | `tts-1` voice `nova` | |

All models are configurable via `.env` (`CHAT_MODEL`, `EMBED_MODEL`, `STT_MODEL`, `TTS_MODEL`, `TTS_VOICE`).

## Repo layout

```
movievoiceagent/
├── app.py                    Streamlit UI + conversation loop
├── agent.py                  OpenAI tool-calling orchestrator
├── config.py                 env-driven config
├── tools/
│   ├── sql_tool.py           DuckDB read-only SELECT runner
│   ├── semantic_tool.py      Chroma semantic search w/ metadata filters
│   ├── recommend_tool.py     "Similar movies" scorer (Jaccard genres + score proximity)
│   └── clarify_tool.py       Ask-user-a-question short-circuit
├── voice/
│   ├── stt.py                Whisper wrapper
│   └── tts.py                OpenAI TTS wrapper
├── data/
│   ├── loader.py             CSV cleaning → DuckDB
│   └── build_index.py        One-shot Chroma builder
├── imdb_dataset/imdb_top_1000.csv
├── chroma_db/                persisted vector store (~6 MB, shipped with the zip)
├── .env / .env.example
├── requirements.txt
├── progress.md / bugs.md / decisions.md
└── README.md
```

## How it handles each test question

| # | Question | How |
|---|---|---|
| 1 | When did The Matrix release? | `sql_query` on `series_title='The Matrix'` |
| 2 | Top 5 movies of 2019 by meta score | `sql_query` with `ORDER BY meta_score DESC LIMIT 5` |
| 3 | Top 7 comedy movies 2010–2020 by IMDB rating | `sql_query` with `'Comedy' = ANY(genres) AND released_year BETWEEN 2010 AND 2020` |
| 4 | Top horror with meta>85 AND imdb>8 | `sql_query` with multi-condition WHERE |
| 5 | Directors with 2+ movies over $500M | `sql_query` with `GROUP BY director HAVING COUNT(*) >= 2` on gross-filtered set |
| 6 | >1M votes but low gross | `sql_query` `WHERE no_of_votes > 1000000 ORDER BY gross_usd ASC` |
| 7 | Comedy with death in plot | `semantic_search("death dying dead", genre_contains="Comedy")` |
| 8 | Summarize Spielberg sci-fi plots | `sql_query` for Spielberg+Sci-Fi → LLM summarizes each Overview |
| 9 | Pre-1990 police involvement | `semantic_search("police officer detective investigation", year_max=1989)` |
| N1 | Al Pacino movies $50M+ rating 8+ | `request_clarification` (lead vs any role) → `sql_query` |
| N2 | Recommendations | `recommend_similar` is called after each answer |

## Notes

- **Dataset cleaning**: the raw CSV has `"28,341,469"`-style strings in `Gross` and `"142 min"` in `Runtime`; `data/loader.py` normalises these to `BIGINT` / `INTEGER`. `Released_Year` has a few non-numeric entries that are coerced to NULL.
- **Vector store**: Chroma `PersistentClient` at `./chroma_db`. Cosine distance. ~6 MB on disk. Ships with the zip so reviewers don't have to re-embed.
- **Voice UX**: press-to-talk. Click the mic, speak your question, click stop; Whisper transcribes; the agent answers; the reply autoplays via OpenAI TTS. Works offline-fallback-free because TTS is server-side.
- **SQL safety**: `sql_tool` rejects anything that isn't a single `SELECT`/`WITH`, blocks DDL/DML keywords, and injects a default `LIMIT 200`.
- **Clarifying question**: the agent uses the `request_clarification` tool to pause and ask the user; the UI renders a radio + Send button; the answer is fed back as the next user message, and the agent continues.

## Troubleshooting

- **"OPENAI_API_KEY is missing"** — check `.env` has a line `OPENAI_API_KEY=sk-...` and restart `streamlit run app.py`.
- **"vector index is empty"** — run `python -m data.build_index` once.
- **Mic button doesn't appear / no audio permission** — Streamlit requires HTTPS or localhost; browsers block mic on non-secure origins. localhost is fine.
- **TTS fails with HTTP 400 or 413** — response was too long; the app trims to 3500 chars; kept for headroom. Check `.env` for `TTS_MODEL`.
- **Chroma "DuplicateIDError" on rebuild** — use "Rebuild vector index" in the sidebar or `python -m data.build_index --force`.
