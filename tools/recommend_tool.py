"""Recommend movies similar to a given set of references.

Similarity score = 0.5*genre_overlap + 0.25*imdb_proximity + 0.25*meta_proximity
"""
from __future__ import annotations

from typing import Any

from data.loader import get_conn
from tools._util import normalize_rows

RECOMMEND_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "recommend_similar",
        "description": (
            "Given one or more reference movie titles, recommend similar movies from the dataset "
            "based on genre overlap and IMDB/Meta score proximity. Call this AFTER answering the "
            "user's main question to append a 'You might also like' block."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reference_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Movie titles the user was just shown (use their Series_Title exactly).",
                },
                "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            },
            "required": ["reference_titles"],
        },
    },
}


def recommend_similar(reference_titles: list[str], limit: int = 5) -> dict[str, Any]:
    if not reference_titles:
        return {"recommendations": []}

    conn = get_conn()
    placeholders = ",".join("?" for _ in reference_titles)
    refs = conn.execute(
        f"SELECT series_title, genres, imdb_rating, meta_score FROM movies "
        f"WHERE series_title IN ({placeholders})",
        reference_titles,
    ).fetch_df().to_dict(orient="records")

    if not refs:
        return {"recommendations": [], "note": "no reference titles matched the dataset"}

    ref_genres: set[str] = set()
    ratings: list[float] = []
    metas: list[float] = []
    for r in refs:
        g = r.get("genres")
        if g is not None:
            ref_genres.update(str(x) for x in g)
        rating = r.get("imdb_rating")
        if rating is not None and rating == rating:
            ratings.append(float(rating))
        meta = r.get("meta_score")
        if meta is not None and meta == meta:
            metas.append(float(meta))

    avg_rating = sum(ratings) / len(ratings) if ratings else 8.0
    avg_meta = sum(metas) / len(metas) if metas else 75.0

    candidates = conn.execute(
        """
        SELECT series_title, released_year, genre, genres, director,
               imdb_rating, meta_score, gross_usd, poster_link
        FROM movies
        WHERE series_title NOT IN ({placeholders})
          AND imdb_rating IS NOT NULL
        """.format(placeholders=placeholders),
        reference_titles,
    ).fetch_df().to_dict(orient="records")

    scored = []
    for c in candidates:
        g = c.get("genres")
        cg = set(str(x) for x in g) if g is not None else set()
        if not cg or not ref_genres:
            overlap = 0.0
        else:
            overlap = len(cg & ref_genres) / len(cg | ref_genres)  # Jaccard
        rating_prox = 1 - min(abs((c.get("imdb_rating") or 0) - avg_rating) / 3, 1)
        meta = c.get("meta_score")
        if meta is None or meta != meta:
            meta_prox = 0.5  # neutral when missing
        else:
            meta_prox = 1 - min(abs(meta - avg_meta) / 50, 1)
        score = 0.5 * overlap + 0.25 * rating_prox + 0.25 * meta_prox
        if overlap == 0:
            continue  # must share at least one genre
        scored.append({**c, "score": round(score, 3)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    recs = normalize_rows(scored[:limit])

    return {
        "recommendations": recs,
        "basis": {
            "reference_titles": reference_titles,
            "avg_imdb": round(avg_rating, 2),
            "avg_meta": round(avg_meta, 1) if metas else None,
            "genres": sorted(ref_genres),
        },
    }
