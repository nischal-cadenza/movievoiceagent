"""Load and clean the IMDB CSV into a DuckDB in-memory table.

Exposes a singleton `get_conn()` that returns a DuckDB connection with a
`movies` table ready for querying.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import duckdb
import numpy as np
import pandas as pd

from config import CSV_PATH

_LOCK = threading.Lock()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["released_year"] = pd.to_numeric(df["Released_Year"], errors="coerce").astype("Int64")

    df["runtime_min"] = (
        df["Runtime"].astype(str).str.replace(" min", "", regex=False).str.strip()
    )
    df["runtime_min"] = pd.to_numeric(df["runtime_min"], errors="coerce").astype("Int64")

    df["gross_usd"] = (
        df["Gross"].astype(str).str.replace(",", "", regex=False).str.replace("nan", "", regex=False).str.strip()
    )
    df["gross_usd"] = pd.to_numeric(df["gross_usd"], errors="coerce").astype("Int64")

    df["imdb_rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")
    df["meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")
    df["no_of_votes"] = pd.to_numeric(df["No_of_Votes"], errors="coerce").astype("Int64")

    df["genres"] = df["Genre"].fillna("").apply(
        lambda s: [g.strip() for g in s.split(",") if g.strip()]
    )
    df["all_stars"] = df[["Star1", "Star2", "Star3", "Star4"]].apply(
        lambda row: [s for s in row.tolist() if isinstance(s, str) and s.strip()],
        axis=1,
    )

    for col in ["Series_Title", "Certificate", "Genre", "Overview", "Director",
                "Star1", "Star2", "Star3", "Star4", "Poster_Link"]:
        df[col] = df[col].fillna("").astype(str)

    out = pd.DataFrame({
        "series_title": df["Series_Title"],
        "released_year": df["released_year"],
        "certificate": df["Certificate"],
        "runtime_min": df["runtime_min"],
        "genre": df["Genre"],
        "genres": df["genres"],
        "imdb_rating": df["imdb_rating"],
        "overview": df["Overview"],
        "meta_score": df["meta_score"],
        "director": df["Director"],
        "star1": df["Star1"],
        "star2": df["Star2"],
        "star3": df["Star3"],
        "star4": df["Star4"],
        "all_stars": df["all_stars"],
        "no_of_votes": df["no_of_votes"],
        "gross_usd": df["gross_usd"],
        "poster_link": df["Poster_Link"],
    })
    return out


@lru_cache(maxsize=1)
def load_df() -> pd.DataFrame:
    raw = pd.read_csv(CSV_PATH)
    return _clean(raw)


_conn: duckdb.DuckDBPyConnection | None = None


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    with _LOCK:
        if _conn is None:
            _conn = duckdb.connect(":memory:")
            df = load_df()  # noqa: F841 — referenced by DuckDB via variable name
            _conn.register("movies_df", df)
            _conn.execute("CREATE OR REPLACE TABLE movies AS SELECT * FROM movies_df")
            _conn.execute("CREATE INDEX idx_title ON movies(series_title)")
    return _conn


SCHEMA_DESCRIPTION = """
Table: movies  (1000 rows, one per top-rated IMDB movie)

Columns:
  series_title    VARCHAR   -- movie title, e.g. 'The Matrix'
  released_year   INTEGER   -- year released (nullable; some rows missing)
  certificate     VARCHAR   -- rating certificate (A, UA, U, PG-13, R, ...)
  runtime_min     INTEGER   -- runtime in minutes
  genre           VARCHAR   -- original comma-joined genre, e.g. 'Crime, Drama'
  genres          VARCHAR[] -- array form for ANY/ALL queries, e.g. ['Crime','Drama']
  imdb_rating     DOUBLE    -- 0..10
  overview        VARCHAR   -- plot summary
  meta_score      DOUBLE    -- 0..100 (nullable)
  director        VARCHAR
  star1,star2,star3,star4  VARCHAR   -- top-billed cast
  all_stars       VARCHAR[] -- [star1,star2,star3,star4] for ANY queries
  no_of_votes     BIGINT
  gross_usd       BIGINT    -- US gross in dollars (nullable)
  poster_link     VARCHAR

Tips for the agent:
  * For genre filters prefer `'Comedy' = ANY(genres)` (exact) rather than `genre LIKE '%Comedy%'`.
  * For cast filters prefer `'Al Pacino' = ANY(all_stars)` unless the user specifies lead-only (then use `star1 = 'Al Pacino'`).
  * `gross_usd` is in dollars — '$500M' = 500000000.
  * Always handle NULLs: `WHERE meta_score IS NOT NULL AND meta_score > 85`.
  * Table name is `movies` (lowercase).

Useful query patterns:

  -- "Directors who had at least 2 movies grossing over $500M, and list those movies":
  WITH big_hits AS (
      SELECT director, series_title, gross_usd
      FROM movies
      WHERE gross_usd > 500000000
  ),
  qualifying_directors AS (
      SELECT director
      FROM big_hits
      GROUP BY director
      HAVING COUNT(*) >= 2
  )
  SELECT b.director, b.series_title, b.gross_usd
  FROM big_hits b
  JOIN qualifying_directors q USING (director)
  ORDER BY b.director, b.gross_usd DESC;

  -- "Top N of YEAR by meta_score" — always ORDER BY with NULLS LAST and add `meta_score IS NOT NULL`.
  -- "Semantic" style questions ('comedy with death', 'police in plot') go to semantic_search, not SQL.
""".strip()
