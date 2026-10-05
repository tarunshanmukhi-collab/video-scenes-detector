# Shot & scene detection — approaches tried

This document is the write-up for the assessment: what was tried, why, and how
each approach compares. Three methods are wired into the runnable pipeline
under [`videoseg/scenes/`](videoseg/scenes/) — pick which one runs
by editing `SCENE_METHOD` in [`main.py`](main.py), see the
[README](README.md) for details. A fourth was tried and later removed once
fully superseded (see the end of this section).

## Shot detection

**Method:** PySceneDetect's `AdaptiveDetector`
([`videoseg/shots.py`](videoseg/shots.py)).

A shot is one uninterrupted camera take. `AdaptiveDetector` flags a cut
wherever a frame differs from its recent rolling average by well more than
usual, rather than against one fixed threshold — which holds up better across
a video with both static dialogue and fast-motion sections. Only one shot
method was tried; it isn't user-selectable, since it consistently reproduced
the manual "detect-adaptive" CLI result it was built to match.

Shots are the finest-grained unit in this project: every scene-detection
method below either groups shots into scenes (conceptually) or, in
`gemini_multimodal`'s case, actively uses the exact shot-cut timestamps to
correct its own output.

## Character (face) detection

**Method:** detect + embed + track + cluster
([`videoseg/faces.py`](videoseg/faces.py), ported from
`face-detect.py`).

Independent of both shots and scenes — same idea as shot detection, it just
needs the raw video and runs (or doesn't, via `FACES_RUN` in `.env`)
regardless of which `SCENE_METHOD` is active. The goal is different too: not
*where* something changes, but *who* is on screen, with one representative
image saved per distinct person.

Detecting a face is the easy part; a 20-minute video yields tens of thousands
of crops of the same handful of people. The pipeline is:

1. Sample frames (default 2/s) and detect + embed every face (InsightFace's
   SCRFD+ArcFace if installed, else OpenCV's bundled YuNet+SFace as a
   no-extra-install fallback).
2. Link detections into **tracks** within one camera take (by box overlap and
   embedding similarity). Averaging a track's embeddings cancels
   expression/blur/lighting noise and shrinks the problem ~20x before
   clustering.

   A track must never span a camera cut: across a cut, box overlap is
   meaningless and two different actors can occupy the same corner of the
   frame, which glues two people into one track and averages them into a
   garbage identity. So cuts come from the **shot detection step** — if
   `outputs/shots.json` exists, `faces.py` reads those exact boundaries
   (`load_shot_boundaries`) and each track is tagged with its `shot_id`. With
   no shots.json it falls back to a coarse HSV-histogram comparison on the
   sampled frames, which both misses cuts between similarly-lit locations and
   fires spuriously on flashes and fast pans — so running shots before faces
   measurably improves the grouping.
3. Cluster tracks into people with average-linkage hierarchical clustering —
   **separately within each scene**, not once globally across the whole
   video. A track is assigned to whichever scene its start time falls in
   (`_partition_tracks_by_scene`; exact, since scene boundaries are always a
   subset of shot boundaries and tracks already never cross those), then
   `cannot_link_pairs` + clustering run again per scene. The cut point is
   calibrated per scene, not hard-coded: two faces in the *same frame* are
   provably different people (a "cannot-link" pair), and the loosest
   dendrogram cut that violates none of those pairs is chosen automatically.
   Groups are named `group1`, `group2`, ... by screen time *within that
   scene* — `group1` in scene 0 is not necessarily the same person as
   `group1` in scene 3, since identity isn't tracked across scenes.
4. Save every appearance's crop into `faces/scene_NN/groupM/` — one folder
   per group per scene — plus `by_scene.json` (each scene's groups: screen
   time, every timecode). `assemble.py` folds this straight into
   `output.json`'s per-scene `"faces"` map.

**Strengths:** fully local, no LLM calls, the cannot-link calibration removes
the usual "pick a similarity threshold by eye" guesswork. Clustering within a
scene is also an easier problem than clustering the whole video at once —
fewer, more visually-similar people to tell apart per run.
**Weaknesses:** no cross-scene identity — a recurring character gets a fresh
(and not necessarily matching) group number in every scene they appear in.
Identity quality also still depends on the detector backend — the no-install
OpenCV fallback (YuNet/SFace) splits one person into more clusters than
InsightFace's ArcFace embeddings would, and a scene with very few
co-occurring faces has little cannot-link evidence to calibrate its threshold
against. Like `clip_embeddings`, its parameters (`sample_fps`, cluster
distances) are tuned per video.

## Scene detection

A **scene** here means a group of one or more shots that share the same
place, time, and topic/event — the thing a human editor would call "one part
of the story", as opposed to a single camera cut.

### 1. CLIP visual embedding novelty — key `clip_embeddings`

[`videoseg/scenes/clip_embeddings.py`](videoseg/scenes/clip_embeddings.py)
— local, no API calls.

- Sample one representative frame from the middle of each fixed-size time
  window (default 5s).
- Embed each frame with CLIP's image encoder (`openai/clip-vit-base-patch32`,
  via `transformers`).
- Compute "novelty" as the distance between each window's frame embedding and
  the next window's.
- A scene boundary is a novelty peak significantly above the track's own mean
  (`prominence_factor` standard deviations), spaced at least
  `min_scene_seconds` apart.

Watches the video directly, so it catches a scene change with no dialogue cue
at all (a silent location change, a montage, an establishing shot) — the exact
thing the transcript-based methods below cannot.

**Strengths:** fast, free, fully local, catches purely visual scene changes.
**Weaknesses:** only looks at one frame per window, so a quick visual change
landing between two sampled frames can be missed; a fast pan, a flash, or a
lighting change within the *same* real scene can register as a false novelty
spike. Window size and threshold need tuning per video.

### 2. LLM-on-transcript — key `azure_llm`

[`videoseg/scenes/llm_transcript_azure.py`](videoseg/scenes/llm_transcript_azure.py)
— Whisper transcript → one Azure OpenAI chat-completion call → scenes → ffmpeg split.

- Transcribe the audio with Whisper (generated automatically the first time
  this method runs).
- Feed the entire timestamped transcript to a chat model in one prompt, and
  ask it to start a new scene whenever *either* the location or the topic
  changes — even between the same two people in the same room.
- The model returns `start_time` / `end_time` / `description` per scene
  directly, no further signal processing needed.
- Cut one clip per scene with ffmpeg ([`clip_writer.py`](videoseg/clip_writer.py)).

**Strengths:** genuinely understands topic and narrative shifts (e.g. "the
conversation moved from work gossip to a personal confession" reads as a scene
break, correctly, even with zero change in camera or location). No parameters
to tune. Produces a human-readable description per scene for free.
**Weaknesses:** text-only — a scene change with no dialogue cue produces no
signal (`clip_embeddings` is the method that covers this gap). Timestamps are
the model's best guess against the transcript, not measured from the video, so
they can drift by a few seconds. Costs one LLM call per video and is not
perfectly deterministic run-to-run.

### 3. Multimodal video understanding — key `gemini_multimodal`

[`videoseg/scenes/llm_multimodal_gemini.py`](videoseg/scenes/llm_multimodal_gemini.py)
— uploads the whole video to Gemini.

- Upload the raw video file (not a transcript) to Gemini via the Files API.
- Prompt it with an explicit scene-vs-shot definition (what counts as "the
  same scene": place + time + event + participants all continuous) and ask
  for a JSON array of scene boundaries + descriptions, schema-enforced.
- Separately, run the video through ffmpeg's `scene` filter to get the exact
  timestamp of every hard cut (cached in `cuts.json`, since a full decode
  takes a couple of minutes).
- **Snap** each model-returned boundary onto the nearest real cut within a
  window (default 5s) — this matters because Gemini effectively samples video
  at ~1 frame/second, so it can say *which second* a scene changes but not
  precisely where inside it. Measured on this project's sample video, raw
  model boundaries sat a median 1.0s (up to 4.6s) away from the true cut;
  snapping removes that error.

**Strengths:** sees and hears the whole video at once (not just windowed
frames like `clip_embeddings`), so it catches scene changes with no dialogue
cue and also reasons about longer-range context than a fixed window can.
Descriptions are grounded in what's actually shown, not inferred from dialogue
alone.
**Weaknesses:** most expensive by far — the whole video is priced as input
tokens (~300k tokens for a 20-minute clip on this project's test video, printed
per run). Needs an API key and a video upload/processing round-trip. Still
needs the ffmpeg snapping step to be frame-accurate on its own. Observed in
practice on the full 25-minute episode: reasoning over a long video in one
shot can misattribute *which* scene a conversation belongs to — see the
description-refinement step below, which exists specifically to catch this.

### Description refinement (fact-check against the transcript) — `describe.py`

[`videoseg/describe.py`](videoseg/describe.py), controlled by
`DESCRIBE_RUN` in `.env` — runs after whichever `SCENE_METHOD`
produced descriptions, regardless of which one it was.

On the full 25-minute episode, `gemini_multimodal` got two things
demonstrably wrong: one scene's cut-snapping pulled its boundary 2.6s earlier
than the content it described actually ended (so the last few seconds of
"Chandler's reaction" ended up in the next clip instead), and a separate
scene's description read as though it was talking about the wrong stretch of
the episode entirely — a plausible failure mode when one model call reasons
about 25 minutes of ~1fps-sampled frames at once, especially when several
scenes share a similar setting (multiple Central Perk / apartment scenes
recur throughout).

This step re-grounds every scene's description in something the vision model
never explicitly reasoned over per-scene: the real, timestamped Whisper
transcript.

1. Transcribe the audio (reuses `transcript.py`, same as `azure_llm`) if not
   already done.
2. For each scene, slice the transcript to just its `[start, end)` window
   (`transcript_for_range`) — the dialogue that was actually spoken during
   that scene.
3. Send every scene's `{time range, current description, transcript slice}`
   to a **text-only** Gemini call (no video re-upload — one cheap call covers
   every scene) with instructions to keep the description if the transcript
   supports it, extend it if the transcript adds detail, or **rewrite it** if
   the transcript contradicts it outright.
4. Overwrite `scenes_<method>.json`'s `description` fields in place.
   `assemble.py`/`report.py` always re-read that file fresh, so the
   correction cascades into `output.json` and `report.html` automatically.

**Strengths:** directly targets the specific failure mode observed above —
misattribution the vision model can't self-correct without a second, more
targeted look. Cheap (text-only, one call for all scenes) compared to
re-uploading the video. Model-agnostic on which side it distrusts: a scene
with no dialogue is explicitly left untouched (no evidence to correct
against), so it can't make a silent scene *worse*.
**Weaknesses:** only as good as the transcript — it can't fix a scene with no
dialogue at all (a silent visual sequence), and a bad transcription (Whisper
mishearing a name, say) could introduce a *new* error instead of fixing one.
Adds one more API call and requires `GEMINI_API_KEY` even when the scene
method itself was `azure_llm` or `clip_embeddings`.

### Comparison

| | `clip_embeddings` | `azure_llm` | `gemini_multimodal` |
|---|---|---|---|
| Input it actually sees | sampled video frames | transcript text | the whole video |
| Catches silent (visual-only) scene changes | ✓ | ✗ | ✓ |
| Per-scene description | ✗ | ✓ | ✓ |
| Needs an API key | ✗ | ✓ (Azure OpenAI) | ✓ (Gemini) |
| Relative cost | free / local | one chat completion | highest (whole video as tokens) |
| Determinism | high (same signal → same peaks) | medium | medium (mitigated by cut-snapping) |
| Boundary precision | window-sized (±window/2) | model's guess, unsnapped | snapped to an exact detected cut |

### Also tried, since removed: transcript embedding novelty

Was `videoseg/scenes/embedding_novelty.py` — deleted once nothing
referenced it, since it was fully superseded rather than an alternative
worth keeping around.

Same idea as `clip_embeddings` (novelty peaks in an embedding sequence,
spaced `min_scene_seconds` apart) but applied to windowed transcript text
embedded with a sentence-transformer (`all-MiniLM-L6-v2`) instead of video
frames. It was tried early on as a fully local, no-LLM way to turn a
transcript into scenes; `azure_llm` replaced it as the transcript-based method
that's actually part of the pipeline, since an LLM reading the transcript
directly produced noticeably better boundaries (understands topic shifts, not
just embedding-distance spikes) and free-text descriptions, at the cost of
needing an API call.
