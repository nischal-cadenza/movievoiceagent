"""Speech-to-text via OpenAI Whisper."""
from __future__ import annotations

import io

from config import STT_MODEL
from openai_client import get_client


def transcribe(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    if not audio_bytes:
        return ""
    buf = io.BytesIO(audio_bytes)
    buf.name = "clip.wav" if "wav" in mime else "clip.webm"
    resp = get_client().audio.transcriptions.create(model=STT_MODEL, file=buf)
    return resp.text.strip()
