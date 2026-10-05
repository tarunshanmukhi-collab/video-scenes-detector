"""Build a local HTML report of whatever the pipeline has produced.

Structured scene-first: each scene shows its own clip + description, then the
shots and face groups that fall inside it, nested underneath. Runs
automatically at the end of pipeline.run(), and scans output_dir fresh each
time rather than only what the current run touched - so if you've run several
SCENE_METHODs over time, one report shows all of them, side by side, each
with their own shots/faces nested in.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from .scene_bounds import load_json, normalize_scenes

STYLE = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       padding: 24px 32px 64px; background: Canvas; color: CanvasText; }
h1 { font-size: 1.5rem; margin-bottom: 4px; }
.subtitle { color: GrayText; margin-top: 0; font-size: 0.9rem; }
h2 { font-size: 1.15rem; margin-top: 2.5rem; border-bottom: 1px solid color-mix(in srgb, CanvasText 20%, transparent);
     padding-bottom: 6px; }
.count { color: GrayText; font-weight: normal; font-size: 0.85em; }
.method { color: GrayText; font-size: 0.85rem; margin-top: -6px; }
.empty { color: GrayText; font-style: italic; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 16px; margin-top: 12px; }
.clip, .person-card { border: 1px solid color-mix(in srgb, CanvasText 15%, transparent);
        border-radius: 8px; padding: 8px; background: color-mix(in srgb, Canvas 96%, CanvasText 4%); }
video, img { width: 100%; border-radius: 4px; display: block; background: #000; }
.person-card img { aspect-ratio: 1; object-fit: cover; }
.clip-name { font-size: 0.75rem; color: GrayText; margin-top: 6px; word-break: break-all; }
.person-id { font-weight: 600; margin-top: 8px; }
.person-stats { font-size: 0.8rem; color: GrayText; }
.no-clip { padding: 40px 0; text-align: center; color: GrayText; font-size: 0.8rem;
           background: color-mix(in srgb, CanvasText 8%, transparent); border-radius: 4px; }
.thumbs { display: flex; flex-wrap: wrap; gap: 4px; }
.thumbs img { width: 64px; height: 64px; object-fit: cover; }

.scene { border: 1px solid color-mix(in srgb, CanvasText 15%, transparent);
         border-radius: 10px; padding: 16px; margin-top: 16px; }
.scene-head { display: flex; justify-content: space-between; align-items: baseline;
              flex-wrap: wrap; gap: 8px; }
.scene-title { font-weight: 700; font-size: 1.05rem; }
.scene-time { color: GrayText; font-size: 0.85rem; }
.scene-desc { margin: 8px 0 0; font-size: 0.9rem; }
.scene-clip { max-width: 360px; margin-top: 12px; }
.sub { margin-top: 16px; }
.sub-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
             letter-spacing: 0.04em; color: GrayText; margin: 0 0 6px; }
.sub .grid { grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; }
.none { color: GrayText; font-size: 0.8rem; font-style: italic; }
"""


def _rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _find_clip(clip_dir: Path, number: int, *, zero_pad: int = 2) -> Path | None:
    """A clip file for the given 1-based number, under either naming
    convention in use across this project: azure_llm/gemini_multimodal name
    clips "scene_01_...mp4" (clip_writer.py / llm_multimodal_gemini.py's own
    cut_scene), while clip_embeddings/shots.py hand off to PySceneDetect's
    own split_video_ffmpeg, which names them "<video-name>-Scene-001.mp4"."""
    if not clip_dir.is_dir():
        return None
    for pattern in (f"scene_{number:0{zero_pad}d}_*.mp4", f"*-Scene-{number:03d}.mp4"):
        matches = sorted(clip_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _shot_card(output_dir: Path, shots_dir: Path, shot: dict) -> str:
    clip = _find_clip(shots_dir, shot["id"] + 1)  # PySceneDetect numbers shots from 1
    media = (f'<video controls preload="metadata" src="{_rel(clip, output_dir)}"></video>'
             if clip else '<div class="no-clip">no clip file</div>')
    return f"""
          <div class="clip">
            {media}
            <div class="clip-name">shot {shot['id']} &middot; {shot['start_seconds']:.1f}s&ndash;{shot['end_seconds']:.1f}s</div>
          </div>"""


def _group_card(output_dir: Path, group_dir: Path, group_id: str, timecodes: list) -> str:
    thumbs = "".join(f'<img src="{_rel(crop, output_dir)}" alt="{html.escape(group_id)}">'
                     for crop in sorted(group_dir.glob("*.jpg"))) if group_dir.is_dir() else ""
    stats = (f"{len(timecodes)} detections &middot; {min(timecodes):.1f}s&ndash;{max(timecodes):.1f}s"
             if timecodes else "")
    return f"""
          <div class="person-card">
            <div class="thumbs">{thumbs}</div>
            <div class="person-id">{html.escape(group_id)}</div>
            <div class="person-stats">{stats}</div>
          </div>"""


def _scene_block(output_dir: Path, scene: dict, clip_dir: Path, shots: list[dict],
                 faces_scene_dir: Path | None, faces: dict) -> str:
    scene_id, start, end = scene["id"], scene["start_seconds"], scene["end_seconds"]

    clip = _find_clip(clip_dir, scene_id + 1)
    clip_html = (f'<video class="scene-clip" controls preload="metadata" '
                f'src="{_rel(clip, output_dir)}"></video>' if clip else "")
    desc_html = (f'<p class="scene-desc">{html.escape(scene["description"])}</p>'
                if scene.get("description") else "")

    scene_shots = [s for s in shots if start <= s.get("start_seconds", 0) < end]
    shots_html = ("".join(_shot_card(output_dir, output_dir / "shots", s) for s in scene_shots)
                 if scene_shots else '<p class="none">no shots recorded for this range</p>')
    shots_grid = (f'<div class="grid">{shots_html}</div>' if scene_shots else shots_html)

    if faces:
        groups_html = "".join(
            _group_card(output_dir, faces_scene_dir / group_id, group_id, timecodes)
            for group_id, timecodes in faces.items())
        faces_block = f'<div class="grid">{groups_html}</div>'
    else:
        faces_block = '<p class="none">no face groups in this scene</p>'

    return f"""
    <div class="scene">
      <div class="scene-head">
        <span class="scene-title">Scene {scene_id}</span>
        <span class="scene-time">{start:.1f}s &rarr; {end:.1f}s</span>
      </div>
      {desc_html}
      {clip_html}
      <div class="sub">
        <p class="sub-title">Shots <span class="count">{len(scene_shots)}</span></p>
        {shots_grid}
      </div>
      <div class="sub">
        <p class="sub-title">Faces <span class="count">{len(faces)}</span></p>
        {faces_block}
      </div>
    </div>"""


def _scenes_sections(output_dir: Path) -> str:
    shots = load_json(output_dir / "shots.json") or []
    output_json = load_json(output_dir / "output.json") or {}
    canonical_method = output_json.get("scene_method")
    faces_by_scene_id = {s["id"]: s.get("faces", {}) for s in output_json.get("scenes", [])}

    sections = []
    for manifest_path in sorted(output_dir.glob("scenes_*.json")):
        method = manifest_path.stem[len("scenes_"):]
        raw = load_json(manifest_path)
        if not isinstance(raw, list) or not raw:
            continue
        scenes = normalize_scenes(raw)

        clip_dir = output_dir / manifest_path.stem
        is_canonical = method == canonical_method
        blocks = "".join(
            _scene_block(
                output_dir, scene, clip_dir, shots,
                output_dir / "faces" / f"scene_{scene['id']:02d}" if is_canonical else None,
                faces_by_scene_id.get(scene["id"], {}) if is_canonical else {},
            )
            for scene in scenes
        )
        note = "" if is_canonical else '<p class="method">(face groups only shown for the scene method output.json was built from)</p>'
        sections.append(f"""
    <section>
      <h2>Scenes &mdash; <code>{html.escape(method)}</code> <span class="count">{len(scenes)}</span></h2>
      {note}
      {blocks}
    </section>""")
    return "\n".join(sections)


def generate(output_dir: str | Path, video_path: str | Path | None = None) -> Path:
    """Scan output_dir for shots.json/scenes_*.json/output.json and write
    output_dir/report.html."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = (_scenes_sections(out_dir)
               or '<p class="empty">Nothing found in this output folder yet - run the pipeline first.</p>')

    title = Path(video_path).name if video_path else "Video shot & scene detection"
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} - results</title>
<style>{STYLE}</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="subtitle">generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from {html.escape(str(out_dir))}</p>
  {sections}
</body>
</html>"""

    report_path = out_dir / "report.html"
    report_path.write_text(body, encoding="utf-8")
    return report_path
