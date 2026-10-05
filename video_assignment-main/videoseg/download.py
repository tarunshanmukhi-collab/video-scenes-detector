"""Download a source video with yt-dlp, ready to point VIDEO_PATH at."""

import yt_dlp


def download(url: str, out: str = "data/video.mp4") -> str:
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out
