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


def draw_violation_overlay(img: np.ndarray, counts: Dict[str, int], cv2) -> None:
    """Draw a red box with red background in the top-left corner displaying active violations."""
    if not counts:
        return

    lines = ["VIOLATIONS DETECTED:"]
    for vtype, count in counts.items():
        name = vtype.replace("_", " ").upper()
        lines.append(f"- {name}: {count}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    line_spacing = 6

    # Calculate size of the box
    max_w = 0
    total_h = 0
    sizes = []
    for line in lines:
        (w, h), base = cv2.getTextSize(line, font, scale, thick)
        sizes.append((w, h, base))
        if w > max_w:
            max_w = w
        total_h += h + base + line_spacing

    pad_x = 12
    pad_y = 12
    box_w = max_w + 2 * pad_x
    box_h = total_h - line_spacing + 2 * pad_y

    # Position: top-left corner below the timestamp (which is drawn at y=22)
    x, y = 10, 40

    overlay = img.copy()
    # Draw filled red rectangle (background) - BGR (0, 0, 220)
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 220), -1)
    # Draw border (white)
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (255, 255, 255), 2)

    current_y = y + pad_y
    for i, line in enumerate(lines):
        w, h, base = sizes[i]
        # Title is drawn slightly thicker
        text_thick = 2 if i == 0 else 1
        cv2.putText(
            overlay,
            line,
            (x + pad_x, current_y + h),
            font,
            scale,
            (255, 255, 255),
            text_thick,
            cv2.LINE_AA,
        )
        current_y += h + base + line_spacing

    # Blend with original frame (85% overlay opacity for glassmorphism effect)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[Dict],
    timestamp: Optional[str] = None,
    violations: Optional[Sequence[Dict]] = None,
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
    violations : sequence of dict, optional
        Active violation events to overlay in the top-left corner red box.
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

    if violations:
        active_violations = [v for v in violations if v.get("is_violation", True)]
        if active_violations:
            counts: Dict[str, int] = {}
            for v in active_violations:
                vtype = v["type"]
                counts[vtype] = counts.get(vtype, 0) + 1
            draw_violation_overlay(img, counts, cv2)

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
