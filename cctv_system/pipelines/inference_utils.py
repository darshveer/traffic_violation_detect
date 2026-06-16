"""Inference utilities: crops, batching, frame iteration, checkpoints, labels.

Pure helper functions shared by the detector and training/eval code. Kept free
of heavy model imports so they're cheap to unit-test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# COCO class ids emitted by the YOLO11 base model that we care about.
COCO_CLASSES: Dict[str, int] = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "traffic_light": 9,
}
VEHICLE_CLASSES: Tuple[str, ...] = ("car", "motorcycle", "bus", "truck", "bicycle")
ID_TO_NAME: Dict[int, str] = {v: k for k, v in COCO_CLASSES.items()}


def clip_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp an ``[x1, y1, x2, y2]`` box to the image bounds and cast to int.

    Parameters
    ----------
    box : sequence of float
        Box coordinates ``[x1, y1, x2, y2]``.
    width, height : int
        Image dimensions.

    Returns
    -------
    tuple[int, int, int, int]
        The clipped integer box.
    """
    x1, y1, x2, y2 = box[:4]
    x1 = int(max(0, min(x1, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    x2 = int(max(0, min(x2, width)))
    y2 = int(max(0, min(y2, height)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def crop(image: np.ndarray, box: Sequence[float], pad: float = 0.0) -> np.ndarray:
    """Crop ``box`` from ``image`` with optional fractional padding.

    Parameters
    ----------
    image : np.ndarray
        Source image (H, W, C).
    box : sequence of float
        ``[x1, y1, x2, y2]`` in pixel coords.
    pad : float
        Fraction of box size to expand on each side (e.g. ``0.1`` -> +10%).

    Returns
    -------
    np.ndarray
        The cropped sub-image (may be empty if the box is degenerate).
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box[:4]
    if pad > 0:
        bw, bh = (x2 - x1), (y2 - y1)
        x1, x2 = x1 - bw * pad, x2 + bw * pad
        y1, y2 = y1 - bh * pad, y2 + bh * pad
    x1, y1, x2, y2 = clip_box([x1, y1, x2, y2], w, h)
    return image[y1:y2, x1:x2]


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    """Return the ``(cx, cy)`` centre of a box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def point_in_box(point: Tuple[float, float], box: Sequence[float]) -> bool:
    """Whether ``point`` lies within ``box`` (inclusive)."""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def overlap_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of ``inner``'s area that intersects ``outer`` (0..1)."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    inner_area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return float(inter / inner_area)


def batched(items: Sequence, batch_size: int) -> Iterator[List]:
    """Yield ``items`` in chunks of up to ``batch_size``.

    Parameters
    ----------
    items : sequence
        Items to batch.
    batch_size : int
        Maximum batch size (must be >= 1).

    Yields
    ------
    list
        Successive sub-lists of ``items``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def iter_video_frames(
    video_path: str, skip_frames: int = 5, start_frame: int = 0
) -> Iterator[Tuple[int, "np.ndarray", float]]:
    """Iterate over (kept) frames of a video, skipping for speed.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    skip_frames : int
        Process every ``skip_frames``-th frame (1 == every frame).
    start_frame : int
        Frame index to resume from (for checkpoint resume).

    Yields
    ------
    tuple[int, np.ndarray, float]
        ``(frame_index, frame_bgr, timestamp_seconds)`` for kept frames.

    Raises
    ------
    RuntimeError
        If the video cannot be opened.
    """
    import cv2  # local import; heavy + only needed for video

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(skip_frames))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = start_frame
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start_frame) % step == 0:
                yield idx, frame, idx / fps
            idx += 1
    finally:
        cap.release()


def video_metadata(video_path: str) -> Dict[str, float]:
    """Return basic video metadata: ``fps``, ``frame_count``, ``width``, ``height``."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


def save_checkpoint(path: str, state: Dict) -> None:
    """Atomically write a JSON checkpoint (for video resume).

    Parameters
    ----------
    path : str
        Destination JSON path.
    state : dict
        Serializable progress state (e.g. ``{"last_frame": 1200}``).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    tmp.replace(p)


def load_checkpoint(path: str) -> Optional[Dict]:
    """Load a JSON checkpoint if it exists, else return ``None``."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def parse_yolo_label(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Parse a YOLO-format label file.

    Each non-empty line is ``class x_center y_center width height`` with
    normalised coordinates.

    Parameters
    ----------
    label_path : str
        Path to a ``.txt`` label file.

    Returns
    -------
    list[tuple]
        ``(class_id, x_center, y_center, width, height)`` rows.

    Raises
    ------
    ValueError
        If a line is malformed (wrong field count or non-numeric values).
    """
    rows: List[Tuple[int, float, float, float, float]] = []
    p = Path(label_path)
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}:{lineno}: expected 5 fields, got {len(parts)}"
                )
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:])
            except ValueError as exc:
                raise ValueError(f"{label_path}:{lineno}: non-numeric value: {exc}") from exc
            rows.append((cls, cx, cy, bw, bh))
    return rows
