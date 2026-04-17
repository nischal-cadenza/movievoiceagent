"""Semantic search over movie plot overviews via Chroma + OpenAI embeddings."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from openai import OpenAI

from config import CHROMA_COLLECTION, CHROMA_DIR, EMBED_MODEL, OPENAI_API_KEY

SEMANTIC_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "semantic_search",
        "description": (
            "Semantic similarity search over movie plot overviews. "
            "Use when the question asks about plot content or themes that may not appear verbatim "
            "(e.g. 'police involvement', 'coming of age', 'revenge story'). "
            "Supports optional metadata pre-filters for year range and genre substring. "
            "Returns top-k matches with similarity scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language meaning to search for."},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                "year_min": {"type": "integer", "description": "Filter: released_year >= this. Optional."},
                "year_max": {"type": "integer", "description": "Filter: released_year <= this. Optional."},
                "genre_contains": {
                    "type": "string",
                    "description": "Filter: the movie's Genre string must contain this substring (case-insensitive). E.g. 'Comedy'. Optional.",
                },
            },
            "required": ["query"],
        },
    },
}


@lru_cache(maxsize=1)
def _collection():
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})


@lru_cache(maxsize=1)
def _openai() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _embed(text: str) -> list[float]:
    resp = _openai().embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def _build_where(year_min: int | None, year_max: int | None) -> dict | None:
    clauses = []
    if year_min is not None:
        clauses.append({"year": {"$gte": int(year_min)}})
    if year_max is not None:
        clauses.append({"year": {"$lte": int(year_max)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def semantic_search(
    query: str,
    k: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    genre_contains: str | None = None,
) -> dict[str, Any]:
    if not query or not query.strip():
        return {"error": "empty query"}

    coll = _collection()
    if coll.count() == 0:
        return {"error": "vector index is empty — run `python -m data.build_index` first"}

    where = _build_where(year_min, year_max)
    over_fetch = max(k * 3, 20) if genre_contains else k

    results = coll.query(
        query_embeddings=[_embed(query)],
        n_results=over_fetch,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    hits = []
    for meta, doc, dist in zip(
        results["metadatas"][0], results["documents"][0], results["distances"][0]
    ):
        if genre_contains and genre_contains.lower() not in meta.get("genre", "").lower():
            continue
        overview = doc.split(". ", 1)[1] if ". " in doc else doc
        hits.append({
            "title": meta.get("title"),
            "year": meta.get("year"),
            "genre": meta.get("genre"),
            "director": meta.get("director"),
            "imdb_rating": meta.get("imdb_rating"),
            "meta_score": meta.get("meta_score"),
            "gross_usd": meta.get("gross_usd"),
            "overview": overview[:400],
            "similarity": round(1 - dist, 4),
        })
        if len(hits) >= k:
            break

    return {
        "query": query,
        "filters": {"year_min": year_min, "year_max": year_max, "genre_contains": genre_contains},
        "hits": hits,
    }
