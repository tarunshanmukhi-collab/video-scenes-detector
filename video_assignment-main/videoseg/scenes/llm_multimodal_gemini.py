"""Method 3: multimodal video understanding (Gemini).

Uploads the video itself (not a transcript) to Gemini and asks it for a JSON
list of scene boundaries, schema-enforced so there's no markdown to strip. The
video is also run through ffmpeg's own scene filter to get the exact timestamp
of every hard cut; each scene boundary the model returns is then snapped onto
the nearest real cut, which compensates for Gemini sampling video at roughly
1 frame/second (measured on this project's sample video: its raw boundaries
sat a median 1.0s, and up to 4.6s, away from the real cut).

This is the only one of the three methods that reasons over the *whole* video
(sight and sound together) in one model call, rather than either a fixed
transcript (azure_llm) or a simple windowed-embedding signal with no semantic
understanding (clip_embeddings). The cost is real: the whole video is
uploaded and priced as input tokens (roughly 300k tokens for a 20-minute
video), and it still needs the snap-to-cut step to be frame-accurate.

Requires (all in a local .env - see .env.example - or the environment; every
one is optional, each falls back to the default shown):
    GEMINI_API_KEY             no default, must be set
    GEMINI_MODEL               gemini-2.5-flash
    GEMINI_CUT_THRESHOLD       0.20  - ffmpeg scene-filter sensitivity (lower = more cuts)
    GEMINI_CUT_SCALE_WIDTH     320   - detection runs downscaled; ~1.5 min for 24 min of video
    GEMINI_SNAP_WINDOW_SEC     5.0   - a boundary further than this from any cut is left alone
    GEMINI_MIN_SCENE_SEC       1.0   - anything shorter is merged into the scene before it
    GEMINI_UPLOAD_TIMEOUT_SEC  900   - a 20-minute video can sit in PROCESSING for a while
    FFMPEG_DIR                 ""    - only needed if ffmpeg/ffprobe aren't on PATH

Only ever called through pipeline.py's run() - no standalone CLI. To try it
ad hoc, call run()/detect_scenes() directly from a quick script instead.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

from ..timecode import hhmmss, parse_ts

# Evaluated once at import time. In the normal pipeline path, config.py has
# already loaded .env into os.environ by the time this module is (lazily)
# imported - see scenes/__init__.py. Running this file standalone relies on
# these being set in the real environment instead (same as GEMINI_API_KEY
# always has).
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_CUT_THRESHOLD = float(os.environ.get("GEMINI_CUT_THRESHOLD", 0.20))
DEFAULT_CUT_SCALE_WIDTH = int(os.environ.get("GEMINI_CUT_SCALE_WIDTH", 320))
DEFAULT_SNAP_WINDOW_SEC = float(os.environ.get("GEMINI_SNAP_WINDOW_SEC", 5.0))
DEFAULT_MIN_SCENE_SEC = float(os.environ.get("GEMINI_MIN_SCENE_SEC", 1.0))
UPLOAD_TIMEOUT_SEC = int(os.environ.get("GEMINI_UPLOAD_TIMEOUT_SEC", 900))


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe
# --------------------------------------------------------------------------- #
def _tool(name: str, ffmpeg_dir: str = "") -> str:
    """Absolute path to ffmpeg/ffprobe, preferring ffmpeg_dir over PATH."""
    if ffmpeg_dir:
        local = Path(ffmpeg_dir) / f"{name}.exe"
        if local.is_file():
            return str(local)
    from shutil import which
    found = which(name)
    if not found:
        sys.exit(f"{name} not found in {ffmpeg_dir!r} or on PATH")
    return found


def video_duration(path: str, ffmpeg_dir: str = "") -> float:
    """Length of the video in seconds, via ffprobe."""
    out = subprocess.run(
        [_tool("ffprobe", ffmpeg_dir), "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def detect_shot_cuts(video: str, out_dir: Path, threshold: float, ffmpeg_dir: str = "") -> list[float]:
    """Exact timestamps of every hard cut, from ffmpeg's scene filter.

    Cached in ``<out_dir>/cuts.json`` against the video's size/mtime and the
    threshold, because a full decode costs a couple of minutes and nothing
    about it changes while you are iterating on the prompt.
    """
    source = Path(video)
    key = {"video": str(source), "size": source.stat().st_size,
           "mtime": int(source.stat().st_mtime), "threshold": threshold}
    cache = out_dir / "cuts.json"
    if cache.is_file():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            print(f"shot cuts: {len(saved['cuts'])} (cached)")
            return saved["cuts"]

    print(f"detecting shot cuts with ffmpeg (threshold {threshold}) ...")
    result = subprocess.run(
        [_tool("ffmpeg", ffmpeg_dir), "-hide_banner", "-nostats", "-loglevel", "info",
         "-i", video, "-an", "-filter:v",
         f"scale={DEFAULT_CUT_SCALE_WIDTH}:-2,select='gt(scene,{threshold})',"
         f"metadata=print:file=-",
         "-f", "null", "-"],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ! cut detection failed, continuing without it:\n"
              f"{result.stderr.strip()[-500:]}")
        return []

    cuts = sorted({round(float(t), 3)
                   for t in re.findall(r"pts_time:([\d.]+)", result.stdout)})
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "cuts": cuts}), encoding="utf-8")
    print(f"  {len(cuts)} shot cuts")
    return cuts


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = """You are an expert film editor. Segment this {duration_str} ({duration_sec:.2f} s) video into its SCENES.

### DEFINITION OF A SCENE
- A **SHOT** is an unbroken camera take, from one cut to the next.
- A **SCENE** is a sequence of one or more shots related by the same location, time, event and context -- a single continuous part of the story or activity. It begins, it plays out, and it ends.
- A scene is a chapter of the video, NOT a shot and NOT a moment. It must stand on its own: someone watching that clip alone should see a complete piece of action from its beginning to its end.

### SCENE CRITERIA
Shots belong to the SAME scene when they share all four of:
1. **Place:** one location, or one continuous physical space.
2. **Time:** continuous chronological flow, with no jump forward or backward.
3. **Event/Activity:** one action, conversation or event, running from its start to its resolution.
4. **Context:** the same participants and the same circumstances throughout.

Start a NEW scene only when one of those four genuinely breaks: the video moves to a different place, time jumps, a different event begins, or the participants and situation change.

### WHAT IS NOT A NEW SCENE
- Camera angle changes, close-ups, wide shots, over-the-shoulder angles during one conversation.
- Cutaways, inserts and reaction shots that return to the same action.
- Pans, zooms, tracking shots, or reframing within the same space.
- A shot of a different person in the same place at the same moment.
These are shots INSIDE a scene. Group them into the scene they belong to.

### BOUNDARY PLACEMENT -- BE EXACT
- **Opening frame:** a scene starts on the first frame of its own content -- immediately after the cut into it, once a fade or dissolve has fully resolved, or the frame a title card becomes legible.
- **Closing frame:** a scene ends on the last frame that belongs to it, which is the frame before the next scene starts. The `end_time` of one scene MUST equal the `start_time` of the next, exactly.
- **Never cut mid-action:** if a line of dialogue, a gesture or a movement is still finishing, the scene has not ended yet. Carry it to its natural stopping point.
- **Transitions:** a fade, dissolve or wipe belongs to the scene it is leaving. The next scene begins where its content is established.
- Opening black, logos, standalone title cards and end credits are scenes in their own right.

### MANDATORY RULES
1. **Complete coverage:** the first scene starts at `00:00:00.000`, the last ends at `{duration_str}`, with zero gaps and zero overlaps.
2. **Timestamp format:** strictly `HH:MM:SS.mmm`, zero-padded (e.g. `00:01:23.400`).
3. **Playback order:** scenes appear in chronological order.
4. **Minimum length:** no scene shorter than {min_scene} seconds, unless it is genuinely standalone (a title card or logo). If a short stretch shares place, time and event with what surrounds it, merge it in.
5. **Be a scene list, not a shot list.** Emitting one entry per camera cut is the single most common way to get this task wrong. Expect scenes measured in tens of seconds to several minutes.

### FINAL CHECK BEFORE ANSWERING
Re-read your list. For every neighbouring pair ask: do these two share the same place, time, event and participants? If yes they are ONE scene -- merge them. Then confirm each `end_time` equals the next `start_time`, and that the last one is `{duration_str}`.

### FIELDS
Return exactly three fields per scene:
- `start_time` -- when the scene begins, `HH:MM:SS.mmm`.
- `end_time` -- when it ends, `HH:MM:SS.mmm`.
- `description` -- 5-6 sentences: where it takes place, who is in it, and what happens from the opening of the scene through to its close, including any dialogue or on-screen text that identifies it. Describe only what is actually in the video; never guess or invent.

{cut_list_block}
### OUTPUT REQUIREMENT
Output ONLY a valid JSON array matching the required schema. No introductory text, no commentary, no markdown outside the JSON.
"""

CUT_LIST_BLOCK = """### DETECTED SHOT CUTS (authoritative timestamps)
You are shown this video at about one frame per second, so you cannot see where
inside a second a cut falls. The list below is the exact timestamp of every hard
cut in the file, measured frame by frame. Use it instead of estimating.

- Every `start_time` (except the first, which is `00:00:00.000`) and every
  `end_time` (except the last) MUST be copied character-for-character from this
  list. Never invent a value that falls between two entries.
- This is a SHOT list. Most of these cuts are camera changes *inside* a scene
  and are NOT scene boundaries -- pick out only the few where a new scene
  begins. There are far more cuts here than there are scenes.
- If a scene truly begins where no cut was detected (a fade, a dissolve, or a
  gradual change), use the most accurate timestamp you can and say so in the
  first words of the description.

{cuts}
"""


def format_cut_list(cuts: list[float], per_line: int = 6) -> str:
    """The cut list as prompt text (~2k tokens for a 24-minute video)."""
    stamps = [hhmmss(c) for c in cuts]
    return "\n".join("  " + "  ".join(stamps[i:i + per_line])
                     for i in range(0, len(stamps), per_line))


SCENE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "start_time": {"type": "STRING", "description": "HH:MM:SS.mmm"},
            "end_time": {"type": "STRING", "description": "HH:MM:SS.mmm"},
            "description": {"type": "STRING",
                            "description": "5-6 sentences: where it takes "
                                           "place, who is in it, what happens "
                                           "from the scene's opening to its "
                                           "close"},
        },
        "required": ["start_time", "end_time", "description"],
        "propertyOrdering": ["start_time", "end_time", "description"],
    },
}


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def upload_and_wait(client: genai.Client, path: str):
    """Upload the video and block until the Files API finishes processing it."""
    print(f"uploading {path} ...")
    handle = client.files.upload(file=path)
    deadline = time.time() + UPLOAD_TIMEOUT_SEC

    while str(getattr(handle.state, "name", handle.state)).upper() == "PROCESSING":
        if time.time() > deadline:
            sys.exit(f"upload still PROCESSING after {UPLOAD_TIMEOUT_SEC}s")
        print("  processing ...")
        time.sleep(5)
        handle = client.files.get(name=handle.name)

    state = str(getattr(handle.state, "name", handle.state)).upper()
    if state != "ACTIVE":
        sys.exit(f"upload failed (state={state}): {getattr(handle, 'error', '')}")
    print(f"  ready: {handle.name}")
    return handle


_TOKEN_LABELS = {
    "prompt_token_count": "input (video+prompt)",
    "cached_content_token_count": "cached",
    "candidates_token_count": "output",
    "thoughts_token_count": "thinking",
    "total_token_count": "total",
}


def token_usage(response) -> dict:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    return {field: getattr(meta, field) for field in _TOKEN_LABELS
            if getattr(meta, field, None)}


def print_token_usage(usage: dict) -> None:
    if not usage:
        print("tokens used: not reported by the API")
        return
    print("tokens used:")
    width = max(len(label) for label in _TOKEN_LABELS.values())
    for field, count in usage.items():
        print(f"  {_TOKEN_LABELS[field]:<{width}}  {count:>10,}")


def detect_scenes(client: genai.Client, handle, duration: float, model: str,
                  min_scene: float, cuts: list[float] | None = None) -> tuple[list[dict], dict]:
    """Ask the model for the scene list; return it parsed, with token counts."""
    cut_block = (CUT_LIST_BLOCK.format(cuts=format_cut_list(cuts))
                 if cuts else "")
    prompt = PROMPT_TEMPLATE.format(duration_str=hhmmss(duration),
                                    duration_sec=duration,
                                    min_scene=min_scene,
                                    cut_list_block=cut_block)
    print(f"asking {model} for scene boundaries ...")
    response = client.models.generate_content(
        model=model,
        contents=[handle, prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=SCENE_SCHEMA,
        ),
    )
    usage = token_usage(response)
    print_token_usage(usage)

    text = (response.text or "").strip()
    if not text:
        sys.exit("model returned no text (check the safety / finish reason)")
    try:
        scenes = json.loads(text)
    except json.JSONDecodeError as exc:
        sys.exit(f"could not parse model JSON: {exc}\n{text[:2000]}")
    if not isinstance(scenes, list) or not scenes:
        sys.exit(f"expected a non-empty JSON array, got: {text[:500]}")
    return scenes, usage


# --------------------------------------------------------------------------- #
# Normalisation + snapping
# --------------------------------------------------------------------------- #
def snap_starts(scenes: list[dict], cuts: list[float], window: float) -> None:
    """Move each scene start onto the real shot cut it was aiming at.

    Closest pairs are assigned first, and a cut is used once: when two
    boundaries sit near the same cut (a 4-second title card between two
    scenes, say) the closest boundary wins the cut and the other keeps the
    model's own timestamp, rather than first-come-first-served collapsing the
    title card to nothing.

    A boundary further than ``window`` from every cut is left alone: it is
    usually a fade or dissolve, which the scene filter scores too low to
    report.
    """
    pairs = sorted(
        (abs(cut - scene["start"]), index, cut)
        for index, scene in enumerate(scenes) if index > 0   # scene 1 is 0.0
        for cut in cuts if abs(cut - scene["start"]) <= window)

    taken_scenes: set[int] = set()
    taken_cuts: set[float] = set()
    for _, index, cut in pairs:
        if index in taken_scenes or cut in taken_cuts:
            continue
        scenes[index]["model_start"] = scenes[index]["start"]
        scenes[index]["snapped_by"] = round(cut - scenes[index]["start"], 3)
        scenes[index]["start"] = cut
        taken_scenes.add(index)
        taken_cuts.add(cut)

    scenes.sort(key=lambda s: s["start"])


def normalise(raw: list[dict], duration: float, min_scene: float,
              cuts: list[float] | None = None,
              window: float = DEFAULT_SNAP_WINDOW_SEC) -> list[dict]:
    """Turn the model's list into a clean, gapless, in-bounds timeline.

    The model is right about *where* scenes change far more often than it is
    exact about the timecode, so the fixes here are deliberately conservative:
    snap each start to a real cut, clamp to the real duration, remove overlaps
    by trusting each start, drop slivers - never invent or reorder content.

    Ends are rebuilt from the next scene's start rather than taken from the
    model, which is what makes the timeline gapless by construction instead of
    by the model's arithmetic.
    """
    scenes: list[dict] = []
    for index, item in enumerate(raw, start=1):
        start = parse_ts(item.get("start_time"))
        end = parse_ts(item.get("end_time"))
        if start is None or end is None:
            print(f"  ! scene {index}: unparseable timestamps "
                  f"({item.get('start_time')!r} -> {item.get('end_time')!r}), skipped")
            continue
        scenes.append({**item, "start": start, "end": end})

    if not scenes:
        sys.exit("no scene had usable timestamps")

    scenes.sort(key=lambda s: s["start"])
    scenes[0]["start"] = 0.0
    if cuts:
        snap_starts(scenes, cuts, window)

    for current, following in zip(scenes, scenes[1:]):
        current["end"] = following["start"]
    scenes[-1]["end"] = duration

    cleaned: list[dict] = []
    for scene in scenes:
        start = min(max(0.0, scene["start"]), duration)
        end = min(max(0.0, scene["end"]), duration)
        if cleaned:
            start = max(start, cleaned[-1]["end"])      # no overlaps
            end = max(end, start)
        if end - start < min_scene:
            if cleaned:                                 # absorb the sliver
                cleaned[-1]["end"] = max(cleaned[-1]["end"], end)
            continue
        scene["start"], scene["end"] = start, end
        cleaned.append(scene)

    if not cleaned:
        sys.exit("no usable scenes after validation")

    cleaned[0]["start"] = 0.0
    cleaned[-1]["end"] = duration
    for number, scene in enumerate(cleaned, start=1):
        scene["scene_number"] = number
        scene["start_time"] = hhmmss(scene["start"])
        scene["end_time"] = hhmmss(scene["end"])
        scene["duration_sec"] = round(scene["end"] - scene["start"], 3)
    return cleaned


# --------------------------------------------------------------------------- #
# Cutting
# --------------------------------------------------------------------------- #
def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_").lower()
    return slug[:limit] or "scene"


def cut_scene(video: str, scene: dict, out_dir: Path, copy: bool, ffmpeg_dir: str = "") -> Path:
    """Write one clip. ``-c copy`` is much faster but can only cut on a
    keyframe, wrong for scene boundaries, hence the opt-in flag."""
    name = (f"scene_{scene['scene_number']:02d}"
            f"_{scene['start_time'].replace(':', '-').replace('.', '-')}"
            f"_{_slug(scene.get('description'))}.mp4")
    out_path = out_dir / name

    command = [_tool("ffmpeg", ffmpeg_dir), "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{scene['start']:.3f}", "-i", video,
               "-t", f"{scene['duration_sec']:.3f}"]
    if copy:
        command += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart"]
    command.append(str(out_path))

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on scene {scene['scene_number']}:\n"
                           f"{result.stderr.strip()}")
    return out_path


# --------------------------------------------------------------------------- #
# Pipeline entry point: run(video_path, transcript_path, output_dir, cfg, split_clips)
# --------------------------------------------------------------------------- #
def run(video_path: str, transcript_path, output_dir: str, cfg: dict, split_clips: bool = True) -> list[dict]:
    params = cfg.get("gemini_multimodal", {})
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "gemini_multimodal needs GEMINI_API_KEY. Copy .env.example to .env and fill it in."
        )

    # cfg["gemini_multimodal"] already has every env var resolved (config.py
    # is the one place that reads them); no need to re-check os.environ here.
    model = params.get("model", DEFAULT_MODEL)
    cut_threshold = params.get("cut_threshold", DEFAULT_CUT_THRESHOLD)
    snap_window = params.get("snap_window_sec", DEFAULT_SNAP_WINDOW_SEC)
    min_scene = params.get("min_scene_sec", DEFAULT_MIN_SCENE_SEC)
    ffmpeg_dir = params.get("ffmpeg_dir", "")

    out_dir = Path(output_dir) / "scenes_gemini_multimodal"
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = video_duration(video_path, ffmpeg_dir)
    print(f"{Path(video_path).name}  duration {hhmmss(duration)}")
    cuts = detect_shot_cuts(video_path, out_dir, cut_threshold, ffmpeg_dir)

    client = genai.Client(api_key=api_key)
    handle = upload_and_wait(client, video_path)
    try:
        raw, usage = detect_scenes(client, handle, duration, model, min_scene, cuts)
    finally:
        try:
            client.files.delete(name=handle.name)
        except Exception as exc:
            print(f"  (could not delete upload: {exc})")

    scenes = normalise(raw, duration, min_scene, cuts, snap_window)
    for scene in scenes:
        moved = (f"{scene['snapped_by']:+.2f}s" if "snapped_by" in scene else "  --  ")
        summary = " ".join(str(scene.get("description", "")).split())[:60]
        print(f"  {scene['scene_number']:>2}. {scene['start_time']} -> "
              f"{scene['end_time']}  ({scene['duration_sec']:>7.2f}s)  "
              f"[{moved:>8}]  {summary}")

    manifest = {"video": video_path, "duration_sec": round(duration, 3),
                "model": model, "token_usage": usage,
                "cut_threshold": cut_threshold, "shot_cuts_detected": len(cuts),
                "scenes": scenes, "raw_response": raw}
    (out_dir / "scenes_raw.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if split_clips:
        for scene in scenes:
            path = cut_scene(video_path, scene, out_dir, copy=False, ffmpeg_dir=ffmpeg_dir)
            print(f"  {path.name}")

    return [{
        "scene_number": s["scene_number"],
        "start_time": s["start_time"],
        "end_time": s["end_time"],
        "start_sec": s["start"],
        "end_sec": s["end"],
        "description": s.get("description"),
    } for s in scenes]
