"""Timestamp <-> seconds conversions shared by the LLM-based scene methods
(azure_llm and gemini_multimodal), which both get HH:MM:SS.mmm strings back
from a model and need to cut clips from them."""
from __future__ import annotations

import re

_TS = re.compile(r"^\s*(?:(\d+):)?(?:(\d+):)?(\d+(?:[.,]\d+)?)\s*$")


def parse_ts(value) -> float | None:
    """'00:04:07.500' | '4:07' | 247.5 -> seconds. None if unparseable.

    Models drop the hours field on short videos often enough that accepting
    MM:SS is cheaper than a retry.
    """
    if isinstance(value, (int, float)):
        return float(value)
    match = _TS.match(str(value))
    if not match:
        return None
    first, second, last = match.groups()
    parts = [p for p in (first, second) if p is not None]
    seconds = float(last.replace(",", "."))
    if len(parts) == 2:                       # H:M:S
        seconds += int(parts[0]) * 3600 + int(parts[1]) * 60
    elif len(parts) == 1:                     # M:S
        seconds += int(parts[0]) * 60
    return seconds


def hhmmss(seconds: float) -> str:
    """3661.5 -> '01:01:01.500' (ffmpeg-friendly, and readable in a filename)."""
    hours, rest = divmod(max(0.0, seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"
