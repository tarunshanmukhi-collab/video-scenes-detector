"""Shot (camera-cut) detection.

Only one method is used for shots across this whole project: PySceneDetect's
AdaptiveDetector, which flags a cut wherever a frame differs from its
neighbours by well more than the clip's own recent average (as opposed to a
fixed threshold, which is what ContentDetector uses). Every scene-detection
method downstream works at the scene level, not the shot level, but shots are
still detected/split independently since they're a useful sanity-check unit
(one shot = one uninterrupted camera take) and gemini_multimodal reuses the
same idea - via ffmpeg's scene filter - to snap scene boundaries onto real cuts.
"""
from __future__ import annotations

from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg


def find_shots(video_path: str):
    shot_list = detect(video_path, AdaptiveDetector())
    for i, (start, end) in enumerate(shot_list):
        print(f"Shot {i+1:3d}: {start.get_timecode()} -> {end.get_timecode()}")
    return shot_list


def split_shots(video_path: str, shot_list, output_dir: str = "outputs/shots/"):
    split_video_ffmpeg(video_path, shot_list, output_dir=output_dir)
