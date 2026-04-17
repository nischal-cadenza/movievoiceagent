"""Request a clarifying choice from the user.

When the agent calls this tool, the orchestrator short-circuits and returns a
special payload to the UI. The UI renders clickable options; the user's choice
becomes the next user message.
"""
from __future__ import annotations

from typing import Any

CLARIFY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "request_clarification",
        "description": (
            "Ask the user a clarifying multiple-choice question BEFORE attempting to answer. "
            "Use when the question is ambiguous in a way that would meaningfully change the answer. "
            "Example: 'Al Pacino movies' — is that lead-actor only (Star1) or any role (Star1..Star4)? "
            "Do not use for trivial clarifications; only when the choice changes the SQL/filter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question shown to the user.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 4,
                    "description": "Concrete, distinct choices the user can pick.",
                },
            },
            "required": ["question", "options"],
        },
    },
}

CLARIFY_MARKER = "__needs_clarification__"


def request_clarification(question: str, options: list[str]) -> dict[str, Any]:
    return {
        CLARIFY_MARKER: True,
        "question": question,
        "options": options,
    }
