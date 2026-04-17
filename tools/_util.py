"""Shared helpers for the tools."""
from __future__ import annotations

from typing import Any

import numpy as np


def to_jsonable(v: Any) -> Any:
    """Convert numpy / pandas scalars and arrays to plain Python values.

    Handles NaN → None, numpy int/float → Python int/float, ndarray → list.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return [to_jsonable(x) for x in v.tolist()]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if f != f else f
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except Exception:
            return str(v)
    return v


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: to_jsonable(v) for k, v in r.items()} for r in rows]
