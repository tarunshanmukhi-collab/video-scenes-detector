"""Shared helpers for reading a scenes_<method>.json manifest and normalising
it into one canonical shape.

Used by both faces.py (to know which scene each track falls in, for per-scene
clustering) and assemble.py (to build the final output.json) - both need to
agree on the exact same 0-based scene ids, since faces.py writes
faces/scene_00/, faces/scene_01/, ... and assemble.py has to reference those
same folder names.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def pick_scene_manifest(output_dir: Path, method: str | None) -> tuple[Path | None, str | None]:
    """The scenes_<method>.json to use. An explicit method wins; otherwise the
    manifest with the most written descriptions - only the LLM-based methods
    write them, and that's the strongest signal of which run was the intended
    submission one."""
    if method:
        candidate = output_dir / f"scenes_{method}.json"
        return (candidate, method) if candidate.is_file() else (None, method)

    best: tuple[int, Path, str] | None = None
    for candidate in sorted(output_dir.glob("scenes_*.json")):
        scenes = load_json(candidate)
        if not isinstance(scenes, list) or not scenes:
            continue
        described = sum(1 for scene in scenes if scene.get("description"))
        name = candidate.stem[len("scenes_"):]
        if best is None or described > best[0]:
            best = (described, candidate, name)
    return (best[1], best[2]) if best else (None, None)


def scene_bounds(scene: dict) -> tuple[float, float]:
    """start/end in seconds, tolerating either manifest field naming
    (scenes_<method>.json uses start_sec/end_sec; the assignment's own schema
    uses start_seconds/end_seconds)."""
    start = scene.get("start_sec", scene.get("start_seconds", 0.0)) or 0.0
    end = scene.get("end_sec", scene.get("end_seconds", 0.0)) or 0.0
    return float(start), float(end)


def normalize_scenes(raw_scenes: list[dict]) -> list[dict]:
    """[{"id", "start_seconds", "end_seconds", "description"}, ...], 0-based
    ids in file order - the canonical shape everything downstream keys its
    per-scene output against."""
    normalized = []
    for index, scene in enumerate(raw_scenes):
        start, end = scene_bounds(scene)
        normalized.append({
            "id": index,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "description": scene.get("description"),
        })
    return normalized


def load_scenes(output_dir: str | Path, video_duration: float | None = None,
                method: str | None = None) -> list[dict]:
    """Normalised scenes from whichever manifest is on disk, or - if none
    exist - a single scene spanning the whole video, so faces.py and
    assemble.py still work when scene detection hasn't been run."""
    out_dir = Path(output_dir)
    manifest_path, _ = pick_scene_manifest(out_dir, method)
    raw = load_json(manifest_path) if manifest_path else None
    if isinstance(raw, list) and raw:
        return normalize_scenes(raw)
    if video_duration:
        return [{"id": 0, "start_seconds": 0.0, "end_seconds": round(video_duration, 3),
                 "description": None}]
    return []
