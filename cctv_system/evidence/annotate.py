"""Draw detections and violations onto frames for court-ready evidence.

Annotations use red boxes/labels for violations and green for normal objects,
with the class label, confidence percentage, an optional track id, and a frame
timestamp burned into the corner.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# BGR colours (OpenCV).
RED = (0, 0, 255)
GREEN = (0, 200, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _put_label(img: np.ndarray, text: str, x: int, y: int, color, cv2) -> None:
    """Draw ``text`` with a filled background box anchored at ``(x, y)`` (top-left)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y_top = max(0, y - th - base - 2)
    cv2.rectangle(img, (x, y_top), (x + tw + 4, y_top + th + base + 2), color, -1)
    cv2.putText(img, text, (x + 2, y_top + th), font, scale, WHITE, thick, cv2.LINE_AA)


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[Dict],
    timestamp: Optional[str] = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw detection/violation boxes and labels on a single frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate.
    detections : sequence of dict
        Each detection dict may contain:

        - ``box``: ``[x1, y1, x2, y2]`` (required)
        - ``label``: class or violation name (required)
        - ``confidence``: float 0..1 (optional)
        - ``is_violation``: bool (optional, default False)
        - ``track_id``: int (optional)
        - ``plate``: str (optional)
    timestamp : str, optional
        Timestamp string drawn in the top-left corner.
    copy : bool
        If True (default), annotate a copy and leave ``frame`` untouched.

    Returns
    -------
    np.ndarray
        The annotated image.
    """
    import cv2

    img = frame.copy() if copy else frame
    h, w = img.shape[:2]

    for det in detections:
        box = det.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        is_violation = bool(det.get("is_violation", False))
        color = RED if is_violation else GREEN

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        parts: List[str] = [str(det.get("label", "obj"))]
        conf = det.get("confidence")
        if conf is not None:
            parts.append(f"{float(conf):.2f}")
        if det.get("track_id") is not None:
            parts.append(f"#{det['track_id']}")
        if det.get("plate"):
            parts.append(str(det["plate"]))
        _put_label(img, " ".join(parts), x1, y1, color, cv2)

    if timestamp:
        _put_label(img, str(timestamp), 5, 22, BLACK, cv2)

    return img


def annotate_batch(
    frames: Sequence[np.ndarray],
    detections_per_frame: Sequence[Sequence[Dict]],
    timestamps: Optional[Sequence[str]] = None,
) -> List[np.ndarray]:
    """Annotate a batch of frames.

    Parameters
    ----------
    frames : sequence of np.ndarray
        Frames to annotate.
    detections_per_frame : sequence of sequence of dict
        Detections for each corresponding frame.
    timestamps : sequence of str, optional
        Per-frame timestamps.

    Returns
    -------
    list[np.ndarray]
        Annotated frames.
    """
    out: List[np.ndarray] = []
    for i, frame in enumerate(frames):
        ts = timestamps[i] if timestamps is not None and i < len(timestamps) else None
        dets = detections_per_frame[i] if i < len(detections_per_frame) else []
        out.append(annotate_frame(frame, dets, ts))
    return out
