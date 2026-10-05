import os
from pathlib import Path

from videoseg.config import load_config, resolve_path
from videoseg.download import download
from videoseg.pipeline import run

# --- pick the scene-detection method: uncomment exactly one ---
# SCENE_METHOD = "clip_embeddings"       # visual: CLIP frame embeddings, novelty peaks
# SCENE_METHOD = "azure_llm"           # text: Whisper transcript -> Azure OpenAI -> scenes
SCENE_METHOD = "gemini_multimodal"   # visual+audio: Gemini watches the whole video

if __name__ == "__main__":
    cfg = load_config()
    video_path = resolve_path(cfg["video_path"])
    video_url = cfg["video_url"]

    if not os.path.isfile(video_path):
        if not video_url:
            raise SystemExit(
                f"{video_path} not found and VIDEO_URL is not set in .env - "
                "either place a video at VIDEO_PATH or set VIDEO_URL to fetch one."
            )
        print(f"{video_path} not found - downloading from {video_url} ...")
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        download(video_url, video_path)

    run(scene_method_override=SCENE_METHOD)
