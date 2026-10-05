"""Method: CLIP visual embedding novelty (local, no API calls).

Samples one representative frame per fixed-size time window, embeds each frame
with CLIP's image encoder, and marks a scene boundary wherever the visual
embedding jumps well above the track's typical fluctuation.

Purely visual - no transcript, no LLM call - so it catches a silent scene
change (no dialogue cue at all) that the transcript-based azure_llm method
would miss entirely.

Weaknesses: only ever looks at one frame per window, so a quick visual change
that happens to land between two sampled frames can be missed; a fast pan or a
brightness/lighting change within the same real scene can also register as a
false novelty spike. Tune `window_seconds` and `prominence_factor` per video.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from scenedetect import FrameTimecode, split_video_ffmpeg
from scipy.signal import find_peaks
from transformers import CLIPModel, CLIPProcessor

_MODEL_NAME = "openai/clip-vit-base-patch32"
_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(_MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _video_fps_and_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count / fps


def _sample_window_frames(video_path, window_seconds, duration):
    """One frame from the middle of each fixed-size time window, as RGB arrays.

    Stops early - and returns a shorter ``usable_duration`` - the first time a
    seek can't read a frame. A truncated file's container metadata can claim a
    longer duration than the file actually holds; padding out to that claimed
    length with a repeated last frame would just add dead, content-free
    windows instead of real signal, and would misreport the final scene's end
    time as covering footage that was never in the file to begin with.
    """
    num_windows = max(int(duration // window_seconds), 1)
    cap = cv2.VideoCapture(video_path)
    frames = []
    usable_duration = duration
    for i in range(num_windows):
        mid_time = (i + 0.5) * window_seconds
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
        ok, frame = cap.read()
        if not ok:
            usable_duration = i * window_seconds
            print(f"  ! only {usable_duration:.1f}s of {duration:.1f}s reported "
                  f"duration is actually readable - the source file looks "
                  f"truncated. Treating the video as {usable_duration:.1f}s long.")
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, usable_duration


def _embed_frames(frames):
    model, processor = _load_clip()
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    # Newer transformers wraps the projected embedding in a model-output object
    # (its .pooler_output) instead of returning the tensor directly.
    features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


def detect_scenes(video_path, window_seconds=5.0, min_scene_seconds=30.0, prominence_factor=1.0):
    """Return a list of (FrameTimecode start, FrameTimecode end) scene boundaries."""
    fps, duration = _video_fps_and_duration(video_path)
    frames, duration = _sample_window_frames(video_path, window_seconds, duration)
    if len(frames) < 2:
        # Not enough usable video to compare anything against anything -
        # report it as one scene covering whatever is actually readable.
        return [(FrameTimecode(timecode=0.0, fps=fps), FrameTimecode(timecode=duration, fps=fps))]
    embeddings = _embed_frames(frames)

    novelty = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    threshold = novelty.mean() + prominence_factor * novelty.std()
    min_distance = max(int(min_scene_seconds / window_seconds), 1)
    boundaries, _ = find_peaks(novelty, height=threshold, distance=min_distance)

    cut_times = [0.0] + [(idx + 1) * window_seconds for idx in boundaries.tolist()] + [duration]

    scenes = []
    for i in range(len(cut_times) - 1):
        start = FrameTimecode(timecode=cut_times[i], fps=fps)
        end = FrameTimecode(timecode=cut_times[i + 1], fps=fps)
        scenes.append((start, end))
    return scenes


def split_scenes(video_path, scene_list, output_dir="outputs/scenes_clip_embeddings/"):
    split_video_ffmpeg(video_path, scene_list, output_dir=output_dir)


def _to_manifest(scene_list) -> list[dict]:
    manifest = []
    for i, (start, end) in enumerate(scene_list, start=1):
        manifest.append({
            "scene_number": i,
            "start_time": start.get_timecode(),
            "end_time": end.get_timecode(),
            "start_sec": start.seconds,
            "end_sec": end.seconds,
            "description": None,
        })
    return manifest


def run(video_path: str, transcript_path, output_dir: str, cfg: dict, split_clips: bool = True) -> list[dict]:
    params = cfg.get("clip_embeddings", {})
    scene_list = detect_scenes(
        video_path,
        window_seconds=params.get("window_seconds", 5.0),
        min_scene_seconds=params.get("min_scene_seconds", 30.0),
        prominence_factor=params.get("prominence_factor", 1.0),
    )
    for i, (start, end) in enumerate(scene_list):
        print(f"Scene {i+1:3d}: {start.get_timecode()} -> {end.get_timecode()}")
    if split_clips:
        split_scenes(video_path, scene_list, output_dir=f"{output_dir}/scenes_clip_embeddings")
    return _to_manifest(scene_list)
