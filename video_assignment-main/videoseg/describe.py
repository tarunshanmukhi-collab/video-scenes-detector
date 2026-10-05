"""Refine each scene's description against what was actually said.

A vision-based scene method (gemini_multimodal, clip_embeddings) samples the
video sparsely and reasons over the whole thing in one shot - which is
exactly how it can misattribute a conversation to the wrong scene (see
docs/APPROACHES.md's writeup on the scene-5 and scene-13 mismatches this step
exists to catch). This module grounds every scene's description against
Whisper's transcript for that scene's exact time range, then asks a
text-only Gemini call (no video re-upload - cheap, one call covers every
scene) to keep, extend, or correct each description based on what the
transcript actually supports.

Overwrites outputs/scenes_<method>.json's "description" fields in place;
everything else (timestamps, scene_number) is left untouched. pipeline.py
runs this before assemble.py/report.py, so the correction cascades into
output.json and report.html automatically - they always read the manifest
fresh off disk.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from google import genai
from google.genai import types

from .scene_bounds import scene_bounds
from .timecode import parse_ts
from .transcript import save_transcript

_TRANSCRIPT_LINE = re.compile(
    r"^\[(?P<start>[\d:.,]+)\s*-->\s*(?P<end>[\d:.,]+)\]\s*(?P<text>.*)$")


def load_transcript_segments(transcript_path: str) -> list[dict]:
    """Parse transcript.txt's "[start --> end] text" lines (written by
    transcript.save_transcript) back into {"start_sec", "end_sec", "text"}."""
    path = Path(transcript_path)
    if not path.is_file():
        return []
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _TRANSCRIPT_LINE.match(line.strip())
        if not match:
            continue
        start = parse_ts(match.group("start"))
        end = parse_ts(match.group("end"))
        if start is None or end is None:
            continue
        segments.append({"start_sec": start, "end_sec": end, "text": match.group("text").strip()})
    return segments


def transcript_for_range(segments: list[dict], start: float, end: float) -> str:
    """Dialogue spoken in [start, end) - any segment overlapping the range at
    all, not just ones fully inside it, joined in chronological order."""
    lines = [seg["text"] for seg in sorted(segments, key=lambda s: s["start_sec"])
             if seg["start_sec"] < end and seg["end_sec"] > start]
    return " ".join(lines).strip()


RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "scene_number": {"type": "INTEGER"},
            "description": {"type": "STRING"},
        },
        "required": ["scene_number", "description"],
    },
}

PROMPT_TEMPLATE = """You are fact-checking a set of scene descriptions for a video against \
its actual spoken dialogue (transcribed from the audio).

For each scene below you are given: its time range, the description a \
different (vision-based) model wrote for it, and the real transcript of \
what was said during that time range.

Decide whether the description is well supported by the transcript:
- If the transcript confirms it and adds nothing important, keep the \
description EXACTLY as-is.
- If the transcript reveals something the description missed (a name, a \
detail, the actual topic), add it.
- If the transcript CONTRADICTS the description (wrong topic, wrong people, \
events that don't match what's actually said), rewrite the description to \
match what the transcript actually supports. Prefer the transcript's \
evidence over the original description whenever they conflict - the \
original was written by a model that samples video at roughly 1 frame/second \
across a very long recording and can misattribute which scene a \
conversation belongs to.
- The transcript may be empty or sparse for a visual-only scene (no \
dialogue) - in that case you have no evidence to change anything, so keep \
the original description.

Return a JSON array with exactly one entry per scene below (matched by \
scene_number), each with the final "description" to use.

SCENES:
{scenes_block}
"""


def _build_scenes_block(scenes: list[dict], segments: list[dict]) -> str:
    blocks = []
    for scene in scenes:
        start, end = scene_bounds(scene)
        transcript = transcript_for_range(segments, start, end) or "(no dialogue in this range)"
        blocks.append(
            f"Scene {scene.get('scene_number')} "
            f"({scene.get('start_time', start)} -> {scene.get('end_time', end)}):\n"
            f"  current description: {scene.get('description') or '(none)'}\n"
            f"  transcript: {transcript}\n")
    return "\n".join(blocks)


def _write_io_log(log_path: Path, model: str, prompt: str, raw_response: str,
                  scenes: list[dict], updated: list[dict]) -> None:
    """Dump exactly what was sent to and received from the model, plus a
    before/after summary - so a wrong correction (or a missed one) can be
    debugged without re-running the API call."""
    changes = []
    for before, after in zip(scenes, updated):
        old, new = before.get("description"), after.get("description")
        tag = "CHANGED" if old != new else "unchanged"
        changes.append(
            f"Scene {after.get('scene_number')} [{tag}]\n"
            f"  before: {old}\n"
            f"  after:  {new}\n")

    log_path.write_text(
        f"=== MODEL ===\n{model}\n\n"
        f"=== PROMPT (input sent to the model) ===\n{prompt}\n\n"
        f"=== RAW RESPONSE (output from the model) ===\n{raw_response}\n\n"
        f"=== BEFORE / AFTER ===\n" + "\n".join(changes),
        encoding="utf-8")


def refine_descriptions(scenes: list[dict], segments: list[dict], model: str,
                        log_path: Path | None = None) -> list[dict]:
    """A new scenes list with each "description" reviewed against the
    transcript for its time range; every other field is untouched. If
    ``log_path`` is given, the exact prompt sent and response received are
    written there alongside a before/after summary."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("description refinement needs GEMINI_API_KEY (text-only Gemini call).")

    prompt = PROMPT_TEMPLATE.format(scenes_block=_build_scenes_block(scenes, segments))
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,               # fact-checking wants consistency, not creativity
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("description refinement: model returned no text")
    corrections = {item["scene_number"]: item["description"] for item in json.loads(text)}

    updated = []
    changed = 0
    for scene in scenes:
        scene = dict(scene)
        new_description = corrections.get(scene.get("scene_number"))
        if new_description is not None and new_description != scene.get("description"):
            changed += 1
            scene["description"] = new_description
        updated.append(scene)
    print(f"  refined {changed}/{len(scenes)} scene descriptions against the transcript")

    if log_path is not None:
        _write_io_log(log_path, model, prompt, text, scenes, updated)
        print(f"  wrote {log_path} (full input/output)")

    return updated


def run(video_path: str, output_dir: str, transcript_path: str, method: str, model: str) -> Path | None:
    """Load outputs/scenes_<method>.json, refine its descriptions against the
    transcript (transcribing the video first if transcript_path doesn't
    exist yet), and overwrite the manifest in place. Returns the manifest
    path, or None if there was nothing to refine (no manifest to begin with)."""
    out_dir = Path(output_dir)
    manifest_path = out_dir / f"scenes_{method}.json"
    if not manifest_path.is_file():
        print(f"  ! no {manifest_path.name} to refine")
        return None

    if not os.path.isfile(transcript_path):
        Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
        save_transcript(video_path, transcript_path)

    segments = load_transcript_segments(transcript_path)
    if not segments:
        print(f"  ! transcript at {transcript_path} has no usable segments - "
              f"skipping description refinement")
        return None

    scenes = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated = refine_descriptions(scenes, segments, model, log_path=out_dir / "describe_io.txt")
    manifest_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {manifest_path} (descriptions refined against transcript)")
    return manifest_path
