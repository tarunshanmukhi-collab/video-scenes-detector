"""End-to-end pipeline: transcript -> shots -> scenes(selected method) ->
describe (refine descriptions against the transcript) -> faces -> report.

Which scene-detection method runs is picked in scripts/run.py's SCENE_METHOD
(or a direct override here), so switching between the approaches this project
tried is a one-line change - see docs/APPROACHES.md for what each one does
and why. Every other tuning knob comes from config.load_config() (.env - see
config.py's own docstring). Shots and faces are both independent of the scene
*method* and run purely off RUN_SHOTS/RUN_FACES in scripts/run.py - but faces
does depend on scenes now: it clusters faces separately *within* each scene,
so it needs scene boundaries before it can run (falling back to shot
boundaries, or the whole video as one scene, if RUN_SCENES is off). Every run
ends by regenerating outputs/output.json and outputs/report.html from
whatever is actually on disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import assemble as assemble_module
from . import faces as faces_module
from . import report as report_module
from . import scene_bounds
from . import shots as shots_module
from .config import load_config, resolve_path
from .scenes import load_method
from .transcript import save_transcript

# Only azure_llm reads the transcript to detect scenes in the first place;
# clip_embeddings and gemini_multimodal look at the video's pixels instead.
# Description refinement (RUN_DESCRIBE) needs the transcript regardless of
# which of the three ran.
_NEEDS_TRANSCRIPT = {"azure_llm"}


def ensure_transcript(video_path: str, transcript_path: str) -> str:
    if os.path.isfile(transcript_path):
        print(f"using existing transcript: {transcript_path}")
        return transcript_path
    Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
    return save_transcript(video_path, transcript_path)


def run_shots(video_path: str, output_dir: str, split_clips: bool = True):
    shot_list = shots_module.find_shots(video_path)

    manifest = [{"id": index, "start_seconds": round(start.seconds, 3),
                 "end_seconds": round(end.seconds, 3)}
                for index, (start, end) in enumerate(shot_list)]
    manifest_path = Path(output_dir) / "shots.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")

    if split_clips:
        shots_module.split_shots(video_path, shot_list, output_dir=f"{output_dir}/shots")
    return shot_list


def run_faces(video_path: str, output_dir: str, cfg: dict, scenes: list[dict] | None) -> list[dict]:
    return faces_module.run(video_path, output_dir, cfg, scenes=scenes)


def run_scenes(method: str, video_path: str, transcript_path: str, output_dir: str,
               cfg: dict, split_clips: bool = True) -> list[dict]:
    scenes = load_method(method).run(video_path, transcript_path, output_dir, cfg, split_clips=split_clips)

    manifest_path = Path(output_dir) / f"scenes_{method}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(scenes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return scenes


def run(scene_method_override: str | None = None,
        video_path_override: str | None = None, run_shots_override: bool | None = None,
        run_faces_override: bool | None = None, run_scenes_override: bool | None = None,
        run_describe_override: bool | None = None, video_url: str | None = None) -> list[dict]:
    cfg = load_config()

    video_path = resolve_path(video_path_override or cfg["video_path"])
    output_dir = resolve_path(cfg.get("output_dir", "outputs"))
    transcript_path = resolve_path(cfg.get("transcript_path", "outputs/transcript.txt"))
    method = scene_method_override or cfg["scenes"]["method"]
    video_url = video_url or cfg.get("video_url")

    do_scenes = cfg.get("scenes", {}).get("run", True) if run_scenes_override is None else run_scenes_override
    do_describe = cfg.get("describe", {}).get("run", True) if run_describe_override is None else run_describe_override
    # azure_llm needs the transcript to detect scenes at all, so it's ensured
    # up front. describe (below) needs one too, but generates its own if
    # missing - see describe.run() - so it isn't duplicated here.
    if do_scenes and method in _NEEDS_TRANSCRIPT:
        transcript_path = ensure_transcript(video_path, transcript_path)

    do_shots = cfg.get("shots", {}).get("run", True) if run_shots_override is None else run_shots_override
    if do_shots:
        run_shots(video_path, output_dir, split_clips=cfg["shots"].get("split_clips", True))

    scenes = []
    if do_scenes:
        scenes = run_scenes(method, video_path, transcript_path, output_dir, cfg,
                            split_clips=cfg["scenes"].get("split_clips", True))

        if do_describe:
            # Local import: pulls in google-genai, which every other run
            # (RUN_DESCRIBE = False, or a method that doesn't need it) should
            # not have to load - see scenes/__init__.py's load_method() for
            # the same reasoning.
            from . import describe as describe_module

            # Grounds each scene's description against what was actually said
            # (Whisper transcript for that scene's time range) and overwrites
            # scenes_<method>.json in place - assemble.py/report.py always
            # re-read that file fresh, so this cascades automatically.
            refined_path = describe_module.run(video_path, output_dir, transcript_path, method,
                                               cfg["gemini_multimodal"]["model"])
            if refined_path is not None:
                scenes = json.loads(refined_path.read_text(encoding="utf-8"))

    do_faces = cfg.get("faces", {}).get("run", True) if run_faces_override is None else run_faces_override
    if do_faces:
        # faces.py clusters within each scene, so it needs scene boundaries -
        # from this run if scenes just ran, otherwise from whatever
        # scenes_<method>.json is already on disk (or one whole-video scene).
        normalized_scenes = (scene_bounds.normalize_scenes(scenes) if scenes
                             else scene_bounds.load_scenes(output_dir, method=method))
        run_faces(video_path, output_dir, cfg, normalized_scenes)

    # Always reassembled, even on a partial run: it stitches together whatever
    # shots/scenes/faces artefacts are on disk into the submission JSON.
    output_path = assemble_module.generate(output_dir, video_path, video_url,
                                           method if do_scenes else None)
    print(f"wrote {output_path}")

    report_path = report_module.generate(output_dir, video_path)
    print(f"wrote {report_path}")

    return scenes
