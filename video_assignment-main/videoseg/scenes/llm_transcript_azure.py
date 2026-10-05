"""Method 2: LLM-on-transcript (Azure OpenAI).

Sends the full timestamped transcript to a chat model and asks it to split the
dialogue into scenes, starting a new one whenever EITHER the location or the
topic changes - even between two people who never move. Returns start/end/
description per scene in one shot.

Understands narrative and topic shifts well and needs no signal-processing
parameters to tune, but it is still text-only: a scene change with no
dialogue cue at all (clip_embeddings' whole reason for existing) produces no
signal here, and its timestamps are the model's best guess rather than
measured from the video.

Requires these environment variables (put them in a local .env - see
.env.example - never commit real values):
    AZURE_OPENAI_ENDPOINT     e.g. https://<resource-name>.openai.azure.com/
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT   name of your chat-model deployment; defaults to gpt-4o
    AZURE_OPENAI_API_VERSION  optional, defaults to 2024-10-21
"""
from __future__ import annotations

import json
import os

from openai import AzureOpenAI

from ..clip_writer import cut_clips
from ..timecode import parse_ts

SYSTEM_PROMPT = """You are given a movie/video transcript as timestamped lines \
in the form [HH:MM:SS.mmm --> HH:MM:SS.mmm] text.

Read through the dialogue and split it into distinct scenes. Start a new scene \
whenever EITHER of these changes, even if the other stays the same:
- location (the characters move to a different place)
- topic (the conversation moves to a different subject), even if it's the same \
characters in the same location continuing to talk

Do not merge multiple topics into one scene just because it's the same characters \
or the same room - a scene is a single continuous location AND topic together. \
For each scene, write a detailed description of what happens.

Respond with ONLY a JSON object of this exact shape, with the scenes covering \
the entire transcript from its first timestamp to its last with no gaps or overlaps:

{"scenes": [{"start_time": "HH:MM:SS.mmm", "end_time": "HH:MM:SS.mmm", "description": "..."}]}
"""


def _client() -> AzureOpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError(
            "azure_llm needs AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY. "
            "Copy .env.example to .env and fill them in."
        )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def detect_scenes(transcript_text: str, deployment: str) -> list[dict]:
    response = _client().chat.completions.create(
        model=deployment,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
    )
    return json.loads(response.choices[0].message.content)["scenes"]


def run(video_path: str, transcript_path: str, output_dir: str, cfg: dict, split_clips: bool = True) -> list[dict]:
    if not transcript_path or not os.path.isfile(transcript_path):
        raise RuntimeError(
            "azure_llm needs a transcript file. Calling this via pipeline.run() generates one "
            "automatically (Whisper); if calling this module directly, pass an existing "
            "transcript_path or run the transcript method first."
        )
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    params = cfg.get("azure_llm", {})
    # cfg["azure_llm"] already has AZURE_OPENAI_DEPLOYMENT resolved by
    # config.py, the one place that reads env vars.
    deployment = params.get("deployment", "gpt-4o")
    raw_scenes = detect_scenes(transcript_text, deployment)

    scenes = []
    for i, scene in enumerate(raw_scenes, start=1):
        start_sec = parse_ts(scene.get("start_time"))
        end_sec = parse_ts(scene.get("end_time"))
        scenes.append({
            "scene_number": i,
            "start_time": scene.get("start_time"),
            "end_time": scene.get("end_time"),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "description": scene.get("description"),
        })
        print(f"Scene {i:3d}: {scene.get('start_time')} -> {scene.get('end_time')}  "
              f"{str(scene.get('description', ''))[:60]}")

    if split_clips:
        cut_clips(video_path, scenes, output_dir=f"{output_dir}/scenes_azure_llm")
    return scenes
