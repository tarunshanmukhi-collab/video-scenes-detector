from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value and value[0] in "\"'":
            # Quoted value: take exactly what's between the quotes, so a
            # trailing `# comment` after the closing quote is discarded.
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            # Unquoted value: an inline ` #comment` ends it.
            value = value.split(" #", 1)[0].rstrip()
        # Always overwrite: .env is authoritative, so a stray leftover
        # $env: value from an earlier terminal session can't silently shadow it.
        os.environ[key.strip()] = value


def resolve_path(path: str) -> str:
    """Anchor a relative path to the repo root, not the current working
    directory - so results land in the same outputs/ regardless of whether
    the pipeline is launched from the repo root, scripts/, or an IDE's "Run
    Python File" (which uses the file's own directory as cwd)."""
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else REPO_ROOT / candidate)


def _env(key: str, default, cast=str):
    """os.environ[key], cast to the type of ``default`` - or ``default``
    itself if the var is unset/empty. ``cast=bool`` understands the usual
    true/false/yes/no/1/0/on/off spellings, not just Python's bool(str)."""
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    if cast is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    return cast(value)


def load_config() -> dict:
    """Read .env, then build the settings dict every module reads from -
    entirely from environment variables. Same shape regardless of how many
    of them are actually set; every value has a sane default."""
    _load_dotenv(REPO_ROOT / ".env")

    return {
        "video_path": _env("VIDEO_PATH", "data/video.mp4"),
        "video_url": _env("VIDEO_URL", None),
        "transcript_path": _env("TRANSCRIPT_PATH", "outputs/transcript.txt"),
        "output_dir": _env("OUTPUT_DIR", "outputs"),

        "shots": {
            "run": _env("SHOTS_RUN", True, bool),
            "split_clips": _env("SHOTS_SPLIT_CLIPS", True, bool),
        },
        "faces": {
            "run": _env("FACES_RUN", True, bool),
            "backend": _env("FACES_BACKEND", "auto"),          # auto | insightface | opencv
            "sample_fps": _env("FACES_SAMPLE_FPS", 2.0, float),
            "frame_height": _env("FACES_FRAME_HEIGHT", 720, int),
            "num_persons": _env("FACES_NUM_PERSONS", None, int),
            "min_appearances": _env("FACES_MIN_APPEARANCES", 1, int),
        },
        "scenes": {
            "run": _env("SCENES_RUN", True, bool),
            "method": "azure_llm",                              # scripts/run.py's SCENE_METHOD decides this in practice
            "split_clips": _env("SCENES_SPLIT_CLIPS", True, bool),
        },
        "describe": {
            "run": _env("DESCRIBE_RUN", True, bool),
        },

        "clip_embeddings": {
            "window_seconds": _env("CLIP_WINDOW_SECONDS", 5.0, float),
            "min_scene_seconds": _env("CLIP_MIN_SCENE_SECONDS", 30.0, float),
            "prominence_factor": _env("CLIP_PROMINENCE_FACTOR", 1.0, float),
        },
        "azure_llm": {
            "deployment": _env("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            "api_version": _env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        },
        "gemini_multimodal": {
            "model": _env("GEMINI_MODEL", "gemini-2.5-flash"),
            "cut_threshold": _env("GEMINI_CUT_THRESHOLD", 0.20, float),
            "cut_scale_width": _env("GEMINI_CUT_SCALE_WIDTH", 320, int),
            "snap_window_sec": _env("GEMINI_SNAP_WINDOW_SEC", 5.0, float),
            "min_scene_sec": _env("GEMINI_MIN_SCENE_SEC", 1.0, float),
            "upload_timeout_sec": _env("GEMINI_UPLOAD_TIMEOUT_SEC", 900, int),
            "ffmpeg_dir": _env("FFMPEG_DIR", ""),
        },
    }
