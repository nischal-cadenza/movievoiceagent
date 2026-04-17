"""Speech-to-text via OpenAI Whisper."""
from __future__ import annotations

import io
from functools import lru_cache

from openai import OpenAI

from config import OPENAI_API_KEY, STT_MODEL


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def transcribe(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    if not audio_bytes:
        return ""
    buf = io.BytesIO(audio_bytes)
    buf.name = "clip.wav" if "wav" in mime else "clip.webm"
    resp = _client().audio.transcriptions.create(model=STT_MODEL, file=buf)
    return resp.text.strip()
