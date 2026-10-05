"""Character detection: find the *distinct people* in a video, one image each.

Entry point is ``run()``, called by pipeline.py (RUN_FACES in
scripts/run.py). Detects and tracks faces across the whole video in one
pass, but then **clusters separately within each scene**: a character's
identity doesn't need to persist across scenes for this project's
output.json (each scene's "faces" map is independent), and clustering a
smaller, more homogeneous population per scene is an easier problem than
clustering everyone in the video at once. Output goes to
``<output_dir>/faces/scene_00/group1/``, one folder per group per scene, plus
``by_scene.json`` (each scene's groups: screen time, timecodes) which
assemble.py folds into the final output.json.

Detecting faces is the easy half; a 20-minute episode yields tens of
thousands of face crops of the same handful of people, which is useless to
look at, so the real work is deciding *who is who*:

    sample frames -> detect + embed faces -> link into tracks -> cluster tracks
    -> one representative image per person, per scene

Two things make the grouping hold up. First, faces are linked into **tracks**
inside a continuous camera take (never spanning a shot cut - see
detect_tracks) and clustered as tracks, not as individual detections:
averaging a track's embeddings cancels expression, blur and lighting noise,
and shrinks the problem ~20x. Second, two faces visible in the *same frame*
are provably different people, which is free supervision -- those cannot-link
pairs both calibrate the clustering threshold and measure the result
afterwards.

Uses InsightFace (SCRFD + ArcFace) if it's installed - much better embeddings -
and falls back to OpenCV's YuNet/SFace (bundled with opencv-python, two small
ONNX files fetched on first use) otherwise, so no extra install is required to
run this at all:

    pip install insightface onnxruntime   # optional, but much better
"""

from __future__ import annotations

import bisect
import json
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

from .config import REPO_ROOT
from .scene_bounds import load_scenes

# --------------------------------------------------------------------------- #
# Settings -- edit these (or .env's FACES_*)
# --------------------------------------------------------------------------- #
SAMPLE_FPS = 2.0             # frames analysed per second of video
FRAME_HEIGHT = 720           # downscale tall frames for detection; 0 = native
MIN_DET_SCORE = 0.60         # detector confidence floor
MIN_FACE_PX = 40             # drop background extras and faces on posters
MIN_SHARPNESS = 25.0         # absolute variance-of-Laplacian floor (motion blur)
SHARPNESS_PERCENTILE = 15.0  # ... plus the blurriest N% of this video's own faces
DET_SIZE = 640               # InsightFace detector input square

# Tracking: link a face to the same face in the next sampled frame.
CUT_CORRELATION = 0.55       # histogram correlation below this = camera cut
TRACK_IOU = 0.20             # box overlap that alone justifies continuing a track
TRACK_COSINE = 0.45          # ... or appearance similarity that alone justifies it
TRACK_MAX_GAP = 2            # tolerate this many sampled frames of detector dropout
MIN_TRACK_DETECTIONS = 2     # a one-frame "track" is usually a false positive
MIN_FRONTALITY = 0.35        # profiles are detected but kept out of the average

# Clustering: group tracks into people.
CLUSTER_DISTANCE = 0.62      # fallback cosine cut when calibration is impossible
CLUSTER_DISTANCE_MIN = 0.40  # calibration sweeps this range
CLUSTER_DISTANCE_MAX = 0.85
CLUSTER_MERGE_DISTANCE = 0.45  # second pass: merge near-identical centroids
MIN_TRACKS_PER_PERSON = 3    # smaller clusters are flagged minor (extras, not leads)

CROP_MARGIN = 0.25           # extra context around the box, as a fraction of size
CROP_SIZE = 224              # crops are square and uniform
MODEL_DIR = str(REPO_ROOT / "models")  # YuNet / SFace are cached here on first use


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale to unit length so cosine similarity is a plain dot product."""
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, 1e-9)


# --------------------------------------------------------------------------- #
# Detector backends -- each returns boxes, 5 landmarks and an identity vector
# --------------------------------------------------------------------------- #
class InsightFaceDetector:
    """SCRFD detection + ArcFace ``w600k_r50`` embedding (CPU)."""

    name = "insightface"

    def __init__(self, det_size: int = DET_SIZE):
        from insightface.app import FaceAnalysis

        print("loading InsightFace buffalo_l (CPU) ...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def detect(self, image: np.ndarray) -> list[dict]:
        results = []
        for face in self.app.get(image):
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = l2_normalize(np.asarray(face.embedding, dtype=np.float32))
            results.append(
                {
                    "bbox": np.asarray(face.bbox, dtype=np.float32),
                    "score": float(face.det_score),
                    "landmarks": np.asarray(face.kps, dtype=np.float32),
                    "embedding": np.asarray(embedding, dtype=np.float32),
                }
            )
        return results


class OpenCVFaceDetector:
    """Fallback: YuNet + SFace, both shipped in ``cv2`` bar two small ONNX files.

    Weaker embeddings than ArcFace, so expect one person to split into more
    clusters -- but nothing large is downloaded and no extra package is needed.
    """

    name = "opencv-yunet"

    DETECTOR_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                    "face_detection_yunet/face_detection_yunet_2023mar.onnx")
    RECOGNIZER_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                      "face_recognition_sface/face_recognition_sface_2021dec.onnx")

    def __init__(self, model_dir: str = MODEL_DIR):
        root = Path(model_dir)
        det = self._fetch(self.DETECTOR_URL, root / "yunet.onnx")
        rec = self._fetch(self.RECOGNIZER_URL, root / "sface.onnx")
        # The detector applies its own floor; keep it below ours so the score
        # gate stays in one place further down.
        self.detector = cv2.FaceDetectorYN.create(
            str(det), "", (320, 320),
            score_threshold=max(0.5, MIN_DET_SCORE - 0.1),
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(rec), "")

    @staticmethod
    def _fetch(url: str, target: Path) -> Path:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {target.name} ...")
            urllib.request.urlretrieve(url, target)
        return target

    def detect(self, image: np.ndarray) -> list[dict]:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, raw = self.detector.detect(image)
        if raw is None:
            return []

        results = []
        for row in raw:
            x, y, w, h = row[:4]
            try:
                aligned = self.recognizer.alignCrop(image, row)
                embedding = self.recognizer.feature(aligned).flatten()
            except cv2.error:
                continue
            results.append(
                {
                    "bbox": np.array([x, y, x + w, y + h], dtype=np.float32),
                    "score": float(row[14]),
                    "landmarks": row[4:14].reshape(5, 2).astype(np.float32),
                    "embedding": l2_normalize(embedding.astype(np.float32)),
                }
            )
        return results


def build_detector(backend: str = "auto"):
    """Explicit backend wins; otherwise InsightFace if it imports."""
    if backend == "opencv":
        return OpenCVFaceDetector()
    try:
        return InsightFaceDetector()
    except Exception as exc:
        if backend == "insightface":
            raise
        print(f"InsightFace unavailable ({exc}) -- falling back to YuNet/SFace")
        return OpenCVFaceDetector()


# --------------------------------------------------------------------------- #
# Quality measures and cropping
# --------------------------------------------------------------------------- #
def measure_sharpness(image: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Variance of the Laplacian inside the face box; low value == motion blur."""
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] > 120:                       # comparable across face sizes
        gray = cv2.resize(gray, (120, 120), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_frontality(landmarks: np.ndarray | None) -> float:
    """Cheap yaw proxy in [0, 1] from the 5-point landmarks.

    A frontal face puts the nose equidistant from both eyes; on a profile one of
    those distances collapses. The ratio of the smaller to the larger is 1.0
    head-on and ~0.0 in full profile, with no pose model needed.
    """
    if landmarks is None or len(landmarks) < 3:
        return 0.5
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    d_left = abs(float(nose[0] - left_eye[0]))
    d_right = abs(float(right_eye[0] - nose[0]))
    if max(d_left, d_right) < 1e-3:
        return 0.0
    return float(np.clip(min(d_left, d_right) / max(d_left, d_right), 0.0, 1.0))


def face_quality(detection: dict) -> float:
    """One number for "how good a portrait is this", used to pick the keeper.

    Sharpness is unbounded and dominates a raw sum, so it is squashed through a
    saturating curve first: past a few hundred, extra sharpness stops mattering
    and pose and detector confidence should decide.
    """
    sharp = detection["sharpness"] / (detection["sharpness"] + 100.0)
    return float(0.4 * sharp + 0.4 * detection["frontality"]
                 + 0.2 * detection["det_score"])


def clip_bbox(bbox: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Round a float box to pixels and pull it inside the frame."""
    height, width = shape[:2]
    x1 = int(max(0, min(width - 1, round(float(bbox[0])))))
    y1 = int(max(0, min(height - 1, round(float(bbox[1])))))
    x2 = int(max(0, min(width, round(float(bbox[2])))))
    y2 = int(max(0, min(height, round(float(bbox[3])))))
    return x1, y1, x2, y2


def crop_face(image: np.ndarray, bbox: tuple[int, int, int, int],
              margin: float = CROP_MARGIN,
              size: int = CROP_SIZE) -> np.ndarray | None:
    """Square crop around a face box, padded with margin and resized.

    Squaring on the *longer* side keeps the aspect ratio intact, so faces near a
    frame edge come out un-stretched even though the crop is clipped short.
    """
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None

    centre_x, centre_y = x1 + width / 2, y1 + height / 2
    half = max(width, height) * (1 + margin) / 2

    frame_h, frame_w = image.shape[:2]
    cx1 = int(max(0, round(centre_x - half)))
    cy1 = int(max(0, round(centre_y - half)))
    cx2 = int(min(frame_w, round(centre_x + half)))
    cy2 = int(min(frame_h, round(centre_y + half)))

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / max(union, 1e-6)


# --------------------------------------------------------------------------- #
# Tracks -- one on-screen appearance of one face
# --------------------------------------------------------------------------- #
class Track:
    """A face followed across consecutive sampled frames of one camera take.

    Only the *best* crop is held, and on disk rather than in memory: keeping one
    full frame per detection is what makes the naive version of this die on a
    feature-length file, and nothing downstream needs more than the keeper.
    """

    __slots__ = ("track_id", "detections", "embedding_sum", "last_index",
                 "last_bbox", "last_embedding", "best", "crop_path", "shot_id")

    def __init__(self, track_id: int, detection: dict, crop_path: Path,
                 shot_id: int | None = None):
        self.track_id = track_id
        self.detections = [detection]
        self.embedding_sum = detection["embedding"].astype(np.float32).copy()
        self.last_index = detection["sample_index"]
        self.last_bbox = detection["bbox"]
        self.last_embedding = detection["embedding"]
        self.best = detection
        self.crop_path = crop_path
        self.shot_id = shot_id

    def add(self, detection: dict) -> None:
        self.detections.append(detection)
        self.embedding_sum += detection["embedding"]
        self.last_index = detection["sample_index"]
        self.last_bbox = detection["bbox"]
        self.last_embedding = l2_normalize(self.embedding_sum / len(self.detections))

    @property
    def start_sec(self) -> float:
        return self.detections[0]["time_sec"]

    @property
    def end_sec(self) -> float:
        return self.detections[-1]["time_sec"]

    def mean_embedding(self) -> np.ndarray:
        """Quality-weighted mean over the frontal-enough detections.

        ArcFace embeddings degrade sharply in profile, and letting those frames
        into the average is what makes one person drift into two clusters. A
        profile-only track still yields a vector -- it just gets no say in
        setting the centroid it lands in.
        """
        usable = [d for d in self.detections if d["frontality"] >= MIN_FRONTALITY]
        pool = usable or self.detections
        weights = np.array([face_quality(d) for d in pool], dtype=np.float32)
        if weights.sum() <= 0:
            weights = np.ones(len(pool), dtype=np.float32)
        stacked = np.stack([d["embedding"] for d in pool]).astype(np.float32)
        return l2_normalize((stacked * weights[:, None]).sum(axis=0) / weights.sum())


def detect_tracks(video_path: str, detector, staging: Path,
                  sample_fps: float = SAMPLE_FPS,
                  frame_height: int = FRAME_HEIGHT, start: float = 0.0,
                  duration: float | None = None,
                  shot_boundaries: list[float] | None = None) -> list[Track]:
    """Sample the video and return one Track per continuous face appearance.

    Sampling skips frames on a sequential read rather than seeking per sample:
    seeking in a long H.264 file lands on keyframes and is far slower than
    decoding straight through.

    ``shot_boundaries`` (cut timestamps in seconds, from shots.py) is what ends
    a track: a track must never span a camera cut, because across a cut box
    overlap is meaningless and two different actors can occupy the same corner
    of the frame - which is exactly how two people get glued into one track and
    averaged into a garbage identity. Given the real boundaries this is exact;
    without them it falls back to a coarse HSV-histogram guess on the sampled
    frames, which both misses cuts between similarly-lit locations and fires
    spuriously on flashes and fast pans.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(source_fps / max(sample_fps, 0.01))))
    if start > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    end = None if duration is None else start + duration
    staging.mkdir(parents=True, exist_ok=True)

    print(f"{video_path}: {source_fps:.2f} fps, detecting every {stride} frame(s) "
          f"with {detector.name}")

    cuts = sorted(shot_boundaries) if shot_boundaries else None
    print(f"  shot boundaries: {len(cuts)} from shot detection"
          if cuts else "  shot boundaries: none supplied, guessing cuts from "
                       "frame histograms")

    active: list[Track] = []
    finished: list[Track] = []
    next_track_id = 0
    previous_hist = None
    current_shot = None
    rejected = {"score": 0, "size": 0, "blur": 0}
    frame_index = 0
    sampled = 0
    detections_total = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        time_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if end is not None and time_sec > end:
            break
        if frame_index % stride:
            frame_index += 1
            continue
        frame_index += 1
        sample_index = sampled
        sampled += 1

        # A camera cut ends every open track (see the docstring). With real
        # shot boundaries this is just "which shot is this timestamp in?";
        # otherwise fall back to comparing consecutive frame histograms.
        if cuts is not None:
            shot_id = bisect.bisect_right(cuts, time_sec)
            if current_shot is not None and shot_id != current_shot:
                finished.extend(active)
                active = []
            current_shot = shot_id
        else:
            hist = _frame_histogram(frame)
            if previous_hist is not None:
                similarity = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL)
                if similarity < CUT_CORRELATION:
                    finished.extend(active)
                    active = []
            previous_hist = hist

        # Detect on a downscaled copy but crop from the full-resolution frame:
        # detection cost falls with the pixels while the saved faces keep every
        # pixel the source actually has.
        scale = 1.0
        image = frame
        if frame_height and frame.shape[0] > frame_height:
            scale = frame_height / frame.shape[0]
            image = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        current: list[dict] = []
        for raw in detector.detect(image):
            if raw["score"] < MIN_DET_SCORE:
                rejected["score"] += 1
                continue
            bbox = clip_bbox(np.asarray(raw["bbox"]) / scale, frame.shape)
            x1, y1, x2, y2 = bbox
            if min(x2 - x1, y2 - y1) < MIN_FACE_PX:
                rejected["size"] += 1
                continue
            sharpness = measure_sharpness(frame, bbox)
            if sharpness < MIN_SHARPNESS:
                rejected["blur"] += 1
                continue

            landmarks = raw.get("landmarks")
            current.append(
                {
                    "sample_index": sample_index,
                    "frame_index": frame_index - 1,
                    "time_sec": round(time_sec, 3),
                    "bbox": bbox,
                    "det_score": float(raw["score"]),
                    "sharpness": sharpness,
                    "frontality": measure_frontality(
                        None if landmarks is None
                        else np.asarray(landmarks) / scale),
                    "embedding": l2_normalize(
                        np.asarray(raw["embedding"], dtype=np.float32)),
                }
            )
        detections_total += len(current)

        # Retire tracks the detector has not refreshed for a while.
        still_active = []
        for track in active:
            if sample_index - track.last_index > TRACK_MAX_GAP:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

        matches, unmatched = _match(active, current)
        for track_position, detection in matches:
            track = active[track_position]
            track.add(detection)
            if face_quality(detection) > face_quality(track.best):
                track.best = detection
                _write_crop(frame, detection["bbox"], track.crop_path)
        for detection in unmatched:
            crop_path = staging / f"track{next_track_id:06d}.jpg"
            if not _write_crop(frame, detection["bbox"], crop_path):
                continue
            active.append(Track(next_track_id, detection, crop_path, current_shot))
            next_track_id += 1

        if sampled % 200 == 0:
            print(f"  {sampled} frames sampled, {detections_total} faces, "
                  f"{len(finished) + len(active)} tracks, t={time_sec:.1f}s")

    capture.release()
    finished.extend(active)

    kept = [t for t in finished if len(t.detections) >= MIN_TRACK_DETECTIONS]
    keep_ids = {t.track_id for t in kept}
    for track in finished:
        if track.track_id not in keep_ids:
            track.crop_path.unlink(missing_ok=True)

    print(f"sampled {sampled} frames; {detections_total} faces "
          f"({rejected['score']} below score, {rejected['size']} too small, "
          f"{rejected['blur']} blurred)")
    print(f"built {len(kept)} face tracks "
          f"({len(finished) - len(kept)} single-frame tracks dropped)")
    return kept


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    """Coarse HSV histogram -- enough to notice a camera cut, cheap to compute."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _write_crop(frame: np.ndarray, bbox: tuple[int, int, int, int],
                target: Path) -> bool:
    crop = crop_face(frame, bbox)
    if crop is None:
        return False
    cv2.imwrite(str(target), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def _match(active: list[Track],
           current: list[dict]) -> tuple[list[tuple[int, dict]], list[dict]]:
    """Optimal one-to-one assignment of this frame's faces to the open tracks.

    Geometry and appearance are combined rather than chosen between: fast motion
    breaks the overlap but the face still looks the same, and two look-alikes
    standing together confuse appearance but sit in different parts of the frame.
    """
    if not active or not current:
        return [], list(current)

    similarity = np.zeros((len(active), len(current)), dtype=np.float32)
    acceptable = np.zeros_like(similarity, dtype=bool)
    for i, track in enumerate(active):
        for j, detection in enumerate(current):
            iou = bbox_iou(track.last_bbox, detection["bbox"])
            appearance = float(np.dot(track.last_embedding, detection["embedding"]))
            similarity[i, j] = 0.5 * iou + 0.5 * max(0.0, appearance)
            acceptable[i, j] = iou >= TRACK_IOU or appearance >= TRACK_COSINE

    rows, cols = linear_sum_assignment(-similarity)
    matches: list[tuple[int, dict]] = []
    matched: set[int] = set()
    for i, j in zip(rows, cols):
        if acceptable[i, j]:
            matches.append((int(i), current[j]))
            matched.add(int(j))
    return matches, [d for j, d in enumerate(current) if j not in matched]


# --------------------------------------------------------------------------- #
# Cannot-link constraints
# --------------------------------------------------------------------------- #
def cannot_link_pairs(tracks: list[Track]) -> set[tuple[int, int]]:
    """Pairs of track positions that are provably *different* people.

    Two faces visible in the same frame cannot be one person. That costs one
    pass over the detections and is the only ground truth available without
    labelling anything by hand.
    """
    frame_to_tracks: dict[int, set[int]] = {}
    for position, track in enumerate(tracks):
        for detection in track.detections:
            frame_to_tracks.setdefault(detection["frame_index"], set()).add(position)

    pairs: set[tuple[int, int]] = set()
    for members in frame_to_tracks.values():
        ordered = sorted(members)
        for a in range(len(ordered)):
            for b in range(a + 1, len(ordered)):
                pairs.add((ordered[a], ordered[b]))

    print(f"derived {len(pairs)} cannot-link pairs from co-occurring faces")
    return pairs


def _violations(group: list[int], cannot_link: set[tuple[int, int]]) -> int:
    count = 0
    for a in range(len(group)):
        for b in range(a + 1, len(group)):
            if (min(group[a], group[b]), max(group[a], group[b])) in cannot_link:
                count += 1
    return count


# --------------------------------------------------------------------------- #
# Clustering tracks into people
# --------------------------------------------------------------------------- #
def cluster_tracks(tracks: list[Track], cannot_link: set[tuple[int, int]],
                   num_persons: int | None = None) -> list[list[int]]:
    """Average-linkage agglomerative clustering plus two correction passes."""
    if len(tracks) < 2:
        return [[0]] if tracks else []

    embeddings = l2_normalize(
        np.stack([t.mean_embedding() for t in tracks]).astype(np.float32))
    tree = linkage(pdist(embeddings, metric="cosine"), method="average")

    if num_persons:
        labels = fcluster(tree, t=min(num_persons, len(tracks)),
                          criterion="maxclust")
        print(f"cutting dendrogram at a fixed k={min(num_persons, len(tracks))}")
    else:
        cut = calibrate_cut(tree, cannot_link)
        labels = fcluster(tree, t=cut, criterion="distance")

    groups = _to_groups(labels)
    print(f"initial cut -> {len(groups)} clusters")

    if cannot_link:
        groups = _repair(groups, embeddings, cannot_link)
        print(f"after cannot-link repair -> {len(groups)} clusters")
    if not num_persons:
        groups = _merge_close(groups, embeddings, cannot_link)
        print(f"after centroid merge -> {len(groups)} clusters")
    return groups


def calibrate_cut(tree: np.ndarray, cannot_link: set[tuple[int, int]]) -> float:
    """Derive the dendrogram cut from evidence instead of hard-coding it.

    Sweep the cut upward and keep the *loosest* one that still contradicts none
    of the cannot-link pairs. Loosest matters because the failure this actually
    suffers is over-splitting -- one actor arriving as five "characters" -- and
    the co-occurrence pairs are what stop a loose cut from going too far.
    """
    if not cannot_link:
        print(f"no cannot-link pairs; using the default cut {CLUSTER_DISTANCE:.2f}")
        return CLUSTER_DISTANCE

    best = CLUSTER_DISTANCE_MIN
    for candidate in np.round(np.arange(CLUSTER_DISTANCE_MIN,
                                        CLUSTER_DISTANCE_MAX + 1e-9, 0.01), 4):
        groups = _to_groups(fcluster(tree, t=float(candidate), criterion="distance"))
        if any(_violations(g, cannot_link) for g in groups):
            break                                  # ascending: last clean cut wins
        best = float(candidate)

    print(f"calibrated cluster distance = {best:.2f} (loosest cut consistent with "
          f"{len(cannot_link)} cannot-link pairs)")
    return best


def _to_groups(labels: np.ndarray) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return [sorted(members) for members in groups.values()]


def _repair(groups: list[list[int]], embeddings: np.ndarray,
            cannot_link: set[tuple[int, int]]) -> list[list[int]]:
    """Re-cut any cluster that merged two people seen in the same frame."""
    repaired: list[list[int]] = []
    for group in groups:
        if len(group) < 2 or not _violations(group, cannot_link):
            repaired.append(group)
            continue

        subset = embeddings[group]
        tree = linkage(pdist(subset, metric="cosine"), method="average")
        for k in range(2, len(group) + 1):         # smallest consistent k
            labels = fcluster(tree, t=k, criterion="maxclust")
            parts = [[group[i] for i in np.where(labels == label)[0]]
                     for label in np.unique(labels)]
            if all(not _violations(part, cannot_link) for part in parts):
                repaired.extend(parts)
                break
        else:
            repaired.extend([[index] for index in group])
    return repaired


def _merge_close(groups: list[list[int]], embeddings: np.ndarray,
                 cannot_link: set[tuple[int, int]]) -> list[list[int]]:
    """Greedily merge the closest compatible pair until nothing is close enough.

    The distance cut is deliberately conservative, because it is far easier to
    merge two halves of one person than to unpick two people who were fused.
    This pass buys that back for the clusters that are plainly the same face.
    """
    groups = [list(g) for g in groups]
    while len(groups) > 1:
        centroids = l2_normalize(
            np.stack([embeddings[g].mean(axis=0) for g in groups]))
        distance = squareform(pdist(centroids, metric="cosine"))
        np.fill_diagonal(distance, np.inf)

        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if distance[i, j] >= CLUSTER_MERGE_DISTANCE:
                    continue
                if _violations(groups[i] + groups[j], cannot_link):
                    continue
                if best is None or distance[i, j] < distance[best[0], best[1]]:
                    best = (i, j)
        if best is None:
            break
        i, j = best
        groups[i] = sorted(groups[i] + groups[j])
        groups.pop(j)
    return groups


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def build_persons(groups: list[list[int]], tracks: list[Track],
                  sample_fps: float) -> list[dict]:
    """Rank clusters by screen time and describe each one."""
    persons = []
    for group in groups:
        members = sorted((tracks[i] for i in group),
                         key=lambda t: -face_quality(t.best))
        detections = sum(len(t.detections) for t in members)
        persons.append(
            {
                "num_appearances": len(members),
                "num_detections": detections,
                "screen_time_sec": round(detections / max(sample_fps, 1e-6), 2),
                "first_seen_sec": round(min(t.start_sec for t in members), 2),
                "last_seen_sec": round(max(t.end_sec for t in members), 2),
                "best_quality": round(face_quality(members[0].best), 3),
                "_tracks": members,
            }
        )

    persons.sort(key=lambda p: (-p["screen_time_sec"], -p["num_detections"]))
    for rank, person in enumerate(persons, start=1):
        person["person_id"] = f"person_{rank:02d}"
        person["is_minor"] = person["num_appearances"] < MIN_TRACKS_PER_PERSON
    return persons



# --------------------------------------------------------------------------- #
# Pipeline entry point: run(video_path, output_dir, cfg)
# --------------------------------------------------------------------------- #
def load_shot_boundaries(output_dir: str | Path) -> list[float] | None:
    """Cut timestamps from shots.json, if shot detection has been run.

    Face tracks are cut at these, so running shots first makes the grouping
    measurably better - see detect_tracks. Returns None when there's no
    shots.json to read, which is what makes RUN_SHOTS = False still work.
    """
    path = Path(output_dir) / "shots.json"
    if not path.is_file():
        return None
    try:
        shots = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(shots, list) or len(shots) < 2:
        return None
    # The boundaries are every shot's start except the first (which is 0.0).
    return [float(shot["start_seconds"]) for shot in shots[1:]
            if "start_seconds" in shot]


def _video_duration_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps


def _partition_tracks_by_scene(tracks: list[Track], scenes: list[dict]) -> dict[int, list[Track]]:
    """Assign each track to the scene its start falls in.

    A track never spans a shot cut (detect_tracks), and every scene boundary
    is also a shot boundary (a scene is a contiguous run of shots), so this
    partition is exact - no track can straddle two scenes.
    """
    boundaries = [scene["start_seconds"] for scene in scenes[1:]]
    by_scene: dict[int, list[Track]] = {scene["id"]: [] for scene in scenes}
    for track in tracks:
        index = bisect.bisect_right(boundaries, track.start_sec)
        scene_id = scenes[min(index, len(scenes) - 1)]["id"]
        by_scene[scene_id].append(track)
    return by_scene


def _save_scene_groups(persons: list[dict], scene_dir: Path) -> list[dict]:
    """Save every appearance's crop into scene_dir/<group_id>/ - the "one
    folder per group per scene" the assignment asks for - and return this
    scene's group summary (id, screen time, every timecode the group appeared
    at) for output.json."""
    groups = []
    for person in persons:
        group_id = person["person_id"]
        group_dir = scene_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)

        timecodes = []
        for index, track in enumerate(person["_tracks"]):
            crop_name = f"{group_id}_a{index:02d}.jpg"
            shutil.copyfile(track.crop_path, group_dir / crop_name)
            timecodes.extend(round(d["time_sec"], 2) for d in track.detections)

        groups.append({
            "group_id": group_id,
            "num_appearances": person["num_appearances"],
            "num_detections": person["num_detections"],
            "screen_time_sec": person["screen_time_sec"],
            "timecodes": sorted(timecodes),
        })
    return groups


def run(video_path: str, output_dir: str, cfg: dict | None = None,
        scenes: list[dict] | None = None) -> list[dict]:
    """Find the distinct people **within each scene** and save their face
    crops one folder per group per scene: outputs/faces/scene_00/group1/, ...

    Grouping is scene-scoped rather than one clustering pass over the whole
    video: a character's identity doesn't need to persist across scenes for
    this assignment's output (each scene's "faces" map is independent), and
    clustering a smaller, more homogeneous population per scene is an easier
    problem than clustering everyone in the video at once.

    Called from pipeline.py, after shots and scenes (it needs both: shot
    boundaries to cut tracks correctly, scene boundaries to know which scene
    each track belongs to). ``scenes`` is normally passed in directly from
    that same pipeline run (scene_bounds.normalize_scenes on whatever
    run_scenes() just produced); if omitted it's loaded from whatever
    scenes_<method>.json is on disk, falling back to one scene spanning the
    whole video.
    """
    params = (cfg or {}).get("faces", {})
    out_dir = Path(f"{output_dir}/faces")
    backend = params.get("backend", "auto")
    sample_fps = params.get("sample_fps", SAMPLE_FPS)
    frame_height = params.get("frame_height", FRAME_HEIGHT)
    num_persons = params.get("num_persons")
    min_appearances = params.get("min_appearances", 1)

    staging = out_dir / "_tracks"
    detector = build_detector(backend)
    tracks = detect_tracks(video_path, detector, staging, sample_fps, frame_height,
                           shot_boundaries=load_shot_boundaries(output_dir))
    if not tracks:
        print("no faces found")
        return []

    if not scenes:
        scenes = load_scenes(output_dir, video_duration=_video_duration_seconds(video_path))
    by_scene = _partition_tracks_by_scene(tracks, scenes)

    scene_results = []
    for scene in scenes:
        scene_tracks = by_scene.get(scene["id"], [])
        groups: list[dict] = []
        if scene_tracks:
            pairs = cannot_link_pairs(scene_tracks)
            group_indices = cluster_tracks(scene_tracks, pairs, num_persons)
            persons = build_persons(group_indices, scene_tracks, sample_fps)
            if min_appearances > 1:
                persons = [p for p in persons if p["num_appearances"] >= min_appearances]
            for rank, person in enumerate(persons, start=1):
                person["person_id"] = f"group{rank}"
            groups = _save_scene_groups(persons, out_dir / f"scene_{scene['id']:02d}")

        print(f"  scene {scene['id']} ({scene['start_seconds']:.1f}s-"
              f"{scene['end_seconds']:.1f}s): {len(scene_tracks)} tracks -> "
              f"{len(groups)} groups")
        scene_results.append({**scene, "groups": groups})

    (out_dir / "by_scene.json").write_text(
        json.dumps({"video": video_path, "scenes": scene_results}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    shutil.rmtree(staging, ignore_errors=True)

    total_groups = sum(len(s["groups"]) for s in scene_results)
    print(f"\n{total_groups} face groups across {len(scenes)} scenes -> {out_dir}")
    return scene_results
