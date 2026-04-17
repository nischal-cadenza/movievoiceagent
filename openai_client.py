"""Shared OpenAI client singleton.

Centralises the `OpenAI(api_key=...)` construction so that callers don't each
instantiate and cache their own. Reused by agent, voice (STT/TTS), semantic
search, and the index builder.
"""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from config import OPENAI_API_KEY


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)
