"""Text-to-speech via OpenAI TTS. Returns MP3 bytes."""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from config import OPENAI_API_KEY, TTS_MODEL, TTS_VOICE

MAX_CHARS = 3500  # OpenAI TTS limit is 4096; keep headroom


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def synthesize(text: str) -> bytes:
    if not text or not text.strip():
        return b""
    clipped = text.strip()[:MAX_CHARS]
    resp = _client().audio.speech.create(model=TTS_MODEL, voice=TTS_VOICE, input=clipped)
    return resp.content
