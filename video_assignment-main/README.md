# Video shot & scene detection

Splits a video into **shots** (individual camera takes) and **scenes**
(groups of shots that belong to the same place/time/topic), trying multiple
approaches for scene detection and letting you pick which one runs by editing
a variable in `main.py` — no CLI flags. Also finds the **distinct
on-screen characters within each scene** (one folder of face crops per
group per scene).

See [`APPROACHES.md`](APPROACHES.md) for what each approach does, why
it was tried, and how they compare — that's the write-up for the assessment.

## Assignment requirements → where they're met

| Requirement | Where |
|---|---|
| 25-30 min YouTube video | `data/video.mp4` (25.7 min); URL in `.env`'s `VIDEO_URL`, auto-fetched by `main.py` via `videoseg/download.py` if missing, and recorded in `outputs/output.json` under `video.url` |
| Identify shots | `videoseg/shots.py` → `outputs/shots.json` + clips in `outputs/shots/` |
| Identify scenes by common theming | `videoseg/scenes/` (3 methods) → `outputs/scenes_<method>.json` + clips |
| Description per scene | `azure_llm` or `gemini_multimodal` scene method (`clip_embeddings` leaves them `null`) |
| Faces detected + grouped by similarity, per scene | `videoseg/faces.py` clusters independently **within each scene** → `outputs/faces/by_scene.json` |
| Faces extracted, one folder per group per scene | `videoseg/faces.py` → `outputs/faces/scene_NN/groupN/*.jpg` |
| Combined output JSON | `videoseg/assemble.py` → **`outputs/output.json`** (`shots[]` + `scenes[]` with `id`, `start_seconds`, `end_seconds`, `description`, `faces{groupN: [timecodes]}`) |

To produce a complete submission run, leave `SHOTS_RUN`, `FACES_RUN`,
`SCENES_RUN`, and `DESCRIBE_RUN` at their default `true` in `.env`, and set
this in `main.py` — the description requirement means the scene
method **must** be one of the two LLM ones:

```python
SCENE_METHOD = "gemini_multimodal"   # or "azure_llm" - clip_embeddings has no descriptions
```

## Folder structure

```
main.py                         # entry point - SET THE METHOD HERE, then `python main.py`
.env.example                   # copy to .env and fill in - the ONLY settings file (secrets + tuning knobs)
requirements.txt

videoseg/
  shots.py                     # shot detection (single method: PySceneDetect AdaptiveDetector)
  faces.py                     # character detection: tracks faces once, clusters them PER SCENE
  scene_bounds.py               # shared: read/normalise a scenes_<method>.json manifest
  transcript.py                # shared Whisper transcription
  timecode.py                  # shared HH:MM:SS.mmm <-> seconds helpers
  clip_writer.py               # shared ffmpeg clip cutting for the LLM-based methods
  download.py                  # pull a source video with yt-dlp; called by main.py if VIDEO_PATH is missing
  config.py                    # reads .env, builds the settings dict everything else uses; anchors relative paths to the repo root
  pipeline.py                  # orchestrates: transcript -> shots -> scenes -> describe -> faces -> assemble -> report
  describe.py                  # fact-checks each scene's description against the transcript (text-only Gemini call)
  assemble.py                  # builds outputs/output.json from shots.json + scenes_<method>.json + faces/by_scene.json
  report.py                    # builds outputs/report.html from whatever's actually in outputs/
  scenes/
    clip_embeddings.py         # method: CLIP visual-frame embedding novelty (local)
    llm_transcript_azure.py    # method: Whisper transcript -> Azure OpenAI -> scenes  -> key "azure_llm"
    llm_multimodal_gemini.py   # method: multimodal video understanding (Gemini)       -> key "gemini_multimodal"

APPROACHES.md                   # write-up: all methods tried, tradeoffs, comparison table

data/                           # input videos (gitignored)
outputs/                        # everything the pipeline generates (gitignored)
models/                         # yunet.onnx + sface.onnx (OpenCV face backend),
                                 # auto-downloaded here on first run - not required at checkout
```

## Setup

```
pip install -r requirements.txt
copy .env.example .env      # then fill in AZURE_OPENAI_*, GEMINI_API_KEY, and any settings you want to change
```

`.env` is the only settings file — everything (API keys, which backend, window
sizes, thresholds, and now which steps run: `SHOTS_RUN`/`FACES_RUN`/
`SCENES_RUN`/`DESCRIBE_RUN`) lives there, with sane defaults for anything you
don't set (see `.env.example`'s comments, or `config.py`'s `load_config()`
for the full list). The one exception is *which scene method runs* — that's
`main.py`'s own `SCENE_METHOD` variable, kept as plain Python
(comment/uncomment) rather than config, by design.

ffmpeg must be on PATH (or set `FFMPEG_DIR` in `.env`). `clip_embeddings` downloads CLIP's weights
(~600MB) from Hugging Face the first time it runs. `faces` works out of the
box with `opencv-python` (already installed; it fetches two small ONNX models
into `models/` on first use) but gets noticeably better identity matching with
`pip install insightface onnxruntime`.

### `.env` variable reference

Copy `.env.example` to `.env` and fill it in. Nothing is hard-required except
credentials for whichever `SCENE_METHOD` you pick in `main.py` (and
`GEMINI_API_KEY` regardless, since `describe.py` uses it too) — every other
variable has a working default and can be left blank.

**General**

| Variable | Default | Meaning |
|---|---|---|
| `VIDEO_PATH` | `data/video.mp4` | Path to the source video (relative to repo root). |
| `VIDEO_URL` | *(empty)* | YouTube URL used to auto-download the video to `VIDEO_PATH` if it's not already on disk. |
| `OUTPUT_DIR` | `outputs` | Where all generated files (shots, scenes, faces, `output.json`, `report.html`) are written. |
| `TRANSCRIPT_PATH` | `outputs/transcript.txt` | Where the Whisper transcript is cached; generated automatically if missing. |
| `FFMPEG_DIR` | *(empty)* | Only needed if `ffmpeg`/`ffprobe` are not already on PATH. |

**Shots** (`videoseg/shots.py`)

| Variable | Default | Meaning |
|---|---|---|
| `SHOTS_RUN` | `true` | Whether the shot-detection step runs. |
| `SHOTS_SPLIT_CLIPS` | `true` | Whether individual shot clips are written to `outputs/shots/` (set `false` to only produce `shots.json`). |

**Faces** (`videoseg/faces.py`)

| Variable | Default | Meaning |
|---|---|---|
| `FACES_RUN` | `true` | Whether face detection/clustering runs. |
| `FACES_BACKEND` | `auto` | `auto` \| `insightface` \| `opencv`. `insightface` (optional install) gives better identity matching; `opencv` works out of the box. |
| `FACES_SAMPLE_FPS` | `2.0` | Frames per second sampled for face tracking. |
| `FACES_FRAME_HEIGHT` | `720` | Frames are resized to this height before detection. |
| `FACES_NUM_PERSONS` | *(empty = auto-detect)* | Force a known number of distinct people per scene instead of letting clustering decide. |
| `FACES_MIN_APPEARANCES` | `1` | Minimum times a face must appear in a scene to count as its own group. |

**Scenes** (`videoseg/scenes/`)

| Variable | Default | Meaning |
|---|---|---|
| `SCENES_RUN` | `true` | Whether scene detection runs (*which method* runs is set in `main.py`'s `SCENE_METHOD`, not `.env`). |
| `SCENES_SPLIT_CLIPS` | `true` | Whether scene clips are written to `outputs/scenes_<method>/`. |

**Describe** (`videoseg/describe.py` — fact-checks scene descriptions against the transcript; needs `GEMINI_API_KEY` even when `SCENE_METHOD` isn't Gemini)

| Variable | Default | Meaning |
|---|---|---|
| `DESCRIBE_RUN` | `true` | Whether the fact-check pass runs. |

**`clip_embeddings` scene method** (local, no API key)

| Variable | Default | Meaning |
|---|---|---|
| `CLIP_WINDOW_SECONDS` | `5.0` | Sliding window size for CLIP frame-embedding comparison. |
| `CLIP_MIN_SCENE_SECONDS` | `30.0` | Minimum scene length. |
| `CLIP_PROMINENCE_FACTOR` | `1.0` | Peak-prominence sensitivity for detecting novelty (scene-change) points. |

**`azure_llm` scene method** (Whisper transcript → Azure OpenAI) — **required if `SCENE_METHOD = "azure_llm"`**

| Variable | Default | Meaning |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | *(empty)* | e.g. `https://<resource-name>.openai.azure.com/`. **Required.** |
| `AZURE_OPENAI_API_KEY` | *(empty)* | Your Azure OpenAI API key. **Required.** |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Deployment/model name. |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | Azure OpenAI API version. |

**`gemini_multimodal` scene method** (Gemini watches the whole video) — **required if `SCENE_METHOD = "gemini_multimodal"`, and always required for `DESCRIBE_RUN`**

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | Your Google Gemini API key. **Required.** |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for scene understanding and the describe fact-check. |
| `GEMINI_CUT_THRESHOLD` | `0.20` | Sensitivity for detecting shot cuts from Gemini's frame analysis. |
| `GEMINI_CUT_SCALE_WIDTH` | `320` | Frame width Gemini analyzes cuts at (smaller = cheaper/faster). |
| `GEMINI_SNAP_WINDOW_SEC` | `5.0` | How far scene boundaries can snap to align with nearby shot cuts. |
| `GEMINI_MIN_SCENE_SEC` | `1.0` | Minimum scene length. |
| `GEMINI_UPLOAD_TIMEOUT_SEC` | `900` | Timeout for uploading the video to Gemini's API. |

Where to get keys: Azure OpenAI key/endpoint come from the Azure OpenAI
resource in the Azure Portal; the Gemini API key comes from
[Google AI Studio](https://aistudio.google.com/apikey).

Put your source video at `data/video.mp4` (or edit `VIDEO_PATH` in `.env`) -
if it's missing, `python main.py` downloads it automatically from `VIDEO_URL`
(set in `.env`) using `videoseg/download.py` (`yt-dlp` under the hood), so
there's no separate download step to run by hand.

## Running

Open [`main.py`](main.py) and pick the scene method (this is
the one thing still set in Python, not `.env`):

```python
# SCENE_METHOD = "clip_embeddings"       # visual: CLIP frame embeddings, novelty peaks
# SCENE_METHOD = "azure_llm"           # text: Whisper transcript -> Azure OpenAI -> scenes
SCENE_METHOD = "gemini_multimodal"   # visual+audio: Gemini watches the whole video
```

Which steps run is controlled in `.env` instead:

```
SHOTS_RUN=true       # PySceneDetect AdaptiveDetector
FACES_RUN=true        # distinct on-screen characters
SCENES_RUN=true
DESCRIBE_RUN=true     # fact-check descriptions against the real transcript
```

`SHOTS_RUN`, `FACES_RUN`, `SCENES_RUN`, `DESCRIBE_RUN` (`.env`), and
`SCENE_METHOD` (`main.py`) toggle independently, but `pipeline.py`
always runs them in this order: **shots → scenes → describe → faces**.
That's a real dependency, not just a nice-to-have: `faces.py` clusters
separately *within each scene*, so it needs scene boundaries to partition
tracks by, and it needs shot boundaries to cut tracks correctly in the first
place. If `SCENES_RUN=false`, it falls back to whatever `scenes_<method>.json`
is already on disk, or treats the whole video as one scene if nothing's there
at all — so `FACES_RUN` alone still works, just with a coarser scene split.

`DESCRIBE_RUN` catches a real failure mode: a vision-based scene method
reasoning over a long video in one shot can misattribute *which* scene a
conversation belongs to (this happened twice on the full 25-minute episode —
see `APPROACHES.md`'s writeup). It re-grounds every scene's description
against Whisper's transcript for that exact time range and asks a text-only
Gemini call to fix anything that doesn't match, overwriting
`scenes_<method>.json`'s descriptions in place — so it needs `GEMINI_API_KEY`
even if `SCENE_METHOD` is `azure_llm` or `clip_embeddings`. Then:

```
python main.py
```

Per-method parameters (window size, thresholds, model names, ...) live in
`.env` — see `.env.example` for the full list, one section per method/step.

Each run writes shot clips to `outputs/shots/` (+ `shots.json`), scene clips
to `outputs/scenes_<method>/` (+ `scenes_<method>.json`), face crops to
`outputs/faces/scene_NN/groupM/` (+ `faces/by_scene.json`), and combines all
of that into **`outputs/output.json`** in the assignment's schema. `azure_llm`
also needs `outputs/transcript.txt`, which is generated automatically the
first time it runs. `outputs/` is always resolved against the repo root, not
whatever the current working directory happens to be — so results land in the
same place whether you run `python main.py` from the repo root or via an
IDE's "Run Python File".

Every run finishes by regenerating **`outputs/report.html`** — a single local
page, structured scene-first: one block per scene, each showing its own clip
and description with its shots and face groups **nested inside it**. It
covers every `scenes_*.json` manifest it finds, so running more than one
`SCENE_METHOD` over time builds up one combined report (face groups only
render for whichever method `output.json` was last built from, since that's
the only one with matching per-scene face data). Open it directly in a
browser; no server needed.

None of the modules under `videoseg/` have their own CLI — everything is
called through `pipeline.run()` (i.e. `python main.py`). For an ad hoc
one-off (e.g. testing a face-detection backend in isolation), call the
function directly from a throwaway script instead of adding a CLI for it.