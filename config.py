"""Central config loaded from .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "nova")

CSV_PATH = ROOT / "imdb_dataset" / "imdb_top_1000.csv"
CHROMA_DIR = ROOT / "chroma_db"
CHROMA_COLLECTION = "movies"

MAX_AGENT_ITERATIONS = 6
SQL_ROW_LIMIT = 200
SQL_ROWS_TO_LLM = 50
