"""Text-to-speech via OpenAI TTS. Returns MP3 bytes."""
from __future__ import annotations

from config import TTS_MODEL, TTS_VOICE
from openai_client import get_client

MAX_CHARS = 3500  # OpenAI TTS limit is 4096; keep headroom


def synthesize(text: str) -> bytes:
    if not text or not text.strip():
        return b""
    clipped = text.strip()[:MAX_CHARS]
    resp = get_client().audio.speech.create(model=TTS_MODEL, voice=TTS_VOICE, input=clipped)
    return resp.content
