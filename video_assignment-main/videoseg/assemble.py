"""Assemble the final submission output: one JSON in the required schema.

Reads whatever the individual steps left in output_dir - shots.json,
scenes_<method>.json, faces/by_scene.json - and produces output.json, the
combined shots + scenes + faces JSON in the assignment's exact schema. The
face crops themselves (faces/scene_00/group1/*.jpg, ...) are already written
by faces.py, which clusters independently within each scene; this module just
reads its per-scene group summary (id, screen time, timecodes) and folds it
into each scene's "faces" map.
"""
from __future__ import annotations

import json
from pathlib import Path

from .scene_bounds import load_json, normalize_scenes, pick_scene_manifest


def generate(output_dir: str | Path, video_path: str | Path | None = None,
             video_url: str | None = None, method: str | None = None) -> Path:
    """Write output_dir/output.json."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shots = load_json(out_dir / "shots.json") or []

    manifest_path, method_used = pick_scene_manifest(out_dir, method)
    raw_scenes = load_json(manifest_path) if manifest_path else None
    scenes = normalize_scenes(raw_scenes) if isinstance(raw_scenes, list) else []

    by_scene_data = load_json(out_dir / "faces" / "by_scene.json") or {}
    groups_by_scene_id = {
        scene["id"]: {group["group_id"]: group["timecodes"] for group in scene.get("groups", [])}
        for scene in by_scene_data.get("scenes", [])
    }
    for scene in scenes:
        scene["faces"] = groups_by_scene_id.get(scene["id"], {})

    payload = {
        "video": {
            "url": video_url,
            "path": str(video_path) if video_path else None,
            "duration_seconds": round(scenes[-1]["end_seconds"], 3) if scenes else None,
        },
        "scene_method": method_used,
        "shots": shots,
        "scenes": scenes,
    }

    output_path = out_dir / "output.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    missing = [s["id"] for s in scenes if not s["description"]]
    if missing:
        print(f"  ! {len(missing)} of {len(scenes)} scenes have no description "
              f"(scene_method={method_used!r}). The spec requires one per scene - "
              f"run with SCENE_METHOD = \"azure_llm\" or \"gemini_multimodal\".")
    if not shots:
        print("  ! no shots.json found - run with SHOTS_RUN=true (.env) to populate the "
              "\"shots\" array.")
    if not groups_by_scene_id:
        print("  ! no faces/by_scene.json found - run with FACES_RUN=true (.env) to "
              "populate per-scene face groups.")

    return output_path
