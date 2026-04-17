"""One-shot script to build the Chroma vector index over movie plots.

Run:
    python -m data.build_index

Idempotent — re-running skips if the collection already has all 1000 docs.
"""
from __future__ import annotations

import sys
import time

import chromadb
import pandas as pd

from config import CHROMA_COLLECTION, CHROMA_DIR, EMBED_MODEL, OPENAI_API_KEY
from data.loader import load_df
from openai_client import get_client

BATCH = 100


def _embed(texts: list[str]) -> list[list[float]]:
    resp = get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def build(force: bool = False) -> None:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set. Populate .env first.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    df = load_df()
    total = len(df)

    if not force and collection.count() >= total:
        print(f"[build_index] Already built ({collection.count()} docs). Use force=True to rebuild.")
        return

    if force and collection.count() > 0:
        print("[build_index] Force rebuild — clearing collection.")
        chroma.delete_collection(CHROMA_COLLECTION)
        collection = chroma.get_or_create_collection(
            name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    start = time.time()

    for batch_start in range(0, total, BATCH):
        batch = df.iloc[batch_start : batch_start + BATCH]
        ids = [f"m_{batch_start + i}" for i in range(len(batch))]
        docs = [
            f"{row.series_title} ({int(row.released_year) if pd.notna(row.released_year) else 'n.d.'}) - "
            f"{row.genre} - directed by {row.director}. {row.overview}"
            for row in batch.itertuples()
        ]
        metas = [
            {
                "title": str(row.series_title),
                "year": int(row.released_year) if pd.notna(row.released_year) else 0,
                "genre": str(row.genre),
                "director": str(row.director),
                "imdb_rating": float(row.imdb_rating) if pd.notna(row.imdb_rating) else 0.0,
                "meta_score": float(row.meta_score) if pd.notna(row.meta_score) else 0.0,
                "gross_usd": int(row.gross_usd) if pd.notna(row.gross_usd) else 0,
            }
            for row in batch.itertuples()
        ]
        embeddings = _embed(docs)
        collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
        print(f"[build_index] upserted {batch_start + len(batch)}/{total}")

    elapsed = time.time() - start
    print(f"[build_index] Done — {collection.count()} docs in {elapsed:.1f}s.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    build(force=force)
