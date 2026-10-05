"""Cut a video into one clip per scene, given start/end seconds.

Shared by the LLM-based scene methods (azure_llm, gemini_multimodal).
clip_embeddings instead uses PySceneDetect's own split_video_ffmpeg, since it
already has FrameTimecode objects rather than plain seconds.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from shutil import which


def _tool(name: str) -> str:
    found = which(name)
    if not found:
        raise RuntimeError(f"{name} not found on PATH - install ffmpeg or add it to PATH")
    return found


def _slug(text, limit: int = 40) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_").lower()
    return slug[:limit] or "scene"


def cut_clips(video_path: str, scenes: list[dict], output_dir: str, copy: bool = False) -> list[Path]:
    """scenes: [{"scene_number": int, "start_sec": float, "end_sec": float, "description"?: str}, ...]

    Re-encodes by default so cuts are frame-accurate; ``copy=True`` is much
    faster but can only cut on a keyframe, shifting the start of a clip by up
    to a GOP (often 1-4s).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for scene in scenes:
        start = scene["start_sec"]
        duration = scene["end_sec"] - start
        label = _slug(scene.get("description") or scene.get("start_time"))
        name = f"scene_{scene['scene_number']:02d}_{label}.mp4"
        out_path = out_dir / name

        command = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                   "-ss", f"{start:.3f}", "-i", video_path, "-t", f"{duration:.3f}"]
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
        written.append(out_path)
    return written
