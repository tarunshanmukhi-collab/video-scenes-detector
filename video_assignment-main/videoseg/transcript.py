"""Whisper transcription. azure_llm needs it to detect scenes in the first
place; describe.py needs it (regardless of scene method) to fact-check
descriptions against what was actually said. clip_embeddings and
gemini_multimodal don't read it themselves - they watch the video directly."""
from __future__ import annotations

_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe(video_path: str) -> list[dict]:
    """Return Whisper's segment list: [{"start": s, "end": s, "text": ...}, ...]."""
    return _load_whisper().transcribe(video_path)["segments"]


def format_timestamp(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"


def save_transcript(video_path: str, output_path: str = "transcript.txt") -> str:
    """Transcribe the video's audio and write a timestamped transcript to a text file."""
    print(f"Transcribing {video_path} - this can take a while for long videos...")
    segments = transcribe(video_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            f.write(f"[{start} --> {end}] {seg['text'].strip()}\n")
    return output_path
