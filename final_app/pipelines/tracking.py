"""Lightweight multi-object tracking.

Provides a self-contained SORT tracker (constant-velocity Kalman filter +
IoU association via the Hungarian algorithm) used by the wrong-side-driving
logic to obtain stable per-vehicle tracks and motion vectors.

The implementation is numpy-only (no ``filterpy`` dependency); SciPy is used for
optimal assignment when available, with a greedy fallback otherwise. A thin
:class:`Tracker` wrapper lets callers swap in Ultralytics' built-in ByteTrack.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a listed dependency
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute the IoU matrix between two sets of ``[x1, y1, x2, y2]`` boxes.

    Parameters
    ----------
    boxes_a : np.ndarray
        Array of shape ``(N, 4)``.
    boxes_b : np.ndarray
        Array of shape ``(M, 4)``.

    Returns
    -------
    np.ndarray
        IoU matrix of shape ``(N, M)``.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    inter_w = np.clip(xx2 - xx1, 0.0, None)
    inter_h = np.clip(yy2 - yy1, 0.0, None)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-9
    return (inter / union).astype(np.float32)


def _box_to_z(box: np.ndarray) -> np.ndarray:
    """Convert ``[x1,y1,x2,y2]`` to measurement ``[cx, cy, s, r]``."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w / 2.0
    cy = box[1] + h / 2.0
    s = w * h
    r = w / (h + 1e-9)
    return np.array([cx, cy, s, r], dtype=np.float32)


def _z_to_box(state: np.ndarray) -> np.ndarray:
    """Convert state ``[cx, cy, s, r, ...]`` back to ``[x1,y1,x2,y2]``."""
    cx, cy, s, r = state[0], state[1], state[2], state[3]
    s = max(float(s), 1e-6)
    r = max(float(r), 1e-6)
    w = np.sqrt(s * r)
    h = s / (w + 1e-9)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Kalman box tracker
# --------------------------------------------------------------------------- #
class KalmanBoxTracker:
    """Constant-velocity Kalman filter tracking a single bounding box.

    State vector is ``[cx, cy, s, r, vcx, vcy, vs]`` (centre, scale, aspect and
    their velocities except aspect). Implemented directly with numpy.
    """

    _count = 0

    def __init__(self, box: np.ndarray, cls_id: int = -1) -> None:
        dim_x, dim_z = 7, 4
        # State transition (constant velocity for cx, cy, s).
        self._F = np.eye(dim_x, dtype=np.float32)
        for i in range(3):
            self._F[i, i + 4] = 1.0
        # Measurement matrix (observe cx, cy, s, r).
        self._H = np.zeros((dim_z, dim_x), dtype=np.float32)
        self._H[:4, :4] = np.eye(4, dtype=np.float32)

        self._P = np.eye(dim_x, dtype=np.float32)
        self._P[4:, 4:] *= 1000.0  # high uncertainty for unobserved velocities
        self._P *= 10.0
        self._Q = np.eye(dim_x, dtype=np.float32)
        self._Q[4:, 4:] *= 0.01
        self._R = np.eye(dim_z, dtype=np.float32)
        self._R[2:, 2:] *= 10.0

        self._x = np.zeros((dim_x, 1), dtype=np.float32)
        self._x[:4, 0] = _box_to_z(box)

        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        self.cls_id = cls_id
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        # Trajectory of centroids for motion-direction analysis.
        self.centroids: List[Tuple[float, float]] = [
            (float(self._x[0, 0]), float(self._x[1, 0]))
        ]

    def predict(self) -> np.ndarray:
        """Advance the state one step and return the predicted box."""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_box(self._x[:, 0])

    def update(self, box: np.ndarray, cls_id: Optional[int] = None) -> None:
        """Correct the state with a matched detection ``box``."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if cls_id is not None:
            self.cls_id = cls_id

        z = _box_to_z(box).reshape(4, 1)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        identity = np.eye(self._P.shape[0], dtype=np.float32)
        self._P = (identity - K @ self._H) @ self._P
        self.centroids.append((float(self._x[0, 0]), float(self._x[1, 0])))

    @property
    def box(self) -> np.ndarray:
        """Current bounding box estimate ``[x1, y1, x2, y2]``."""
        return _z_to_box(self._x[:, 0])

    def motion_vector(self, window: int = 8) -> Tuple[float, float]:
        """Net displacement (dx, dy) over the last ``window`` centroids."""
        if len(self.centroids) < 2:
            return (0.0, 0.0)
        recent = self.centroids[-window:]
        x0, y0 = recent[0]
        x1, y1 = recent[-1]
        return (x1 - x0, y1 - y0)


# --------------------------------------------------------------------------- #
# SORT
# --------------------------------------------------------------------------- #
def _associate(
    detections: np.ndarray, trackers: np.ndarray, iou_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Associate detections to trackers via IoU + Hungarian assignment.

    Returns
    -------
    matches : list[tuple[int, int]]
        ``(detection_idx, tracker_idx)`` pairs.
    unmatched_dets : list[int]
    unmatched_trks : list[int]
    """
    if trackers.shape[0] == 0 or detections.shape[0] == 0:
        return [], list(range(detections.shape[0])), list(range(trackers.shape[0]))

    iou = iou_batch(detections, trackers)
    if _HAS_SCIPY:
        row, col = linear_sum_assignment(-iou)
        candidates = list(zip(row.tolist(), col.tolist()))
    else:  # greedy fallback
        candidates = []
        used_d, used_t = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for d, t in order:
            if d in used_d or t in used_t:
                continue
            used_d.add(d)
            used_t.add(t)
            candidates.append((int(d), int(t)))

    matches: List[Tuple[int, int]] = []
    matched_d, matched_t = set(), set()
    for d, t in candidates:
        if iou[d, t] >= iou_threshold:
            matches.append((d, t))
            matched_d.add(d)
            matched_t.add(t)

    unmatched_dets = [d for d in range(detections.shape[0]) if d not in matched_d]
    unmatched_trks = [t for t in range(trackers.shape[0]) if t not in matched_t]
    return matches, unmatched_dets, unmatched_trks


class Sort:
    """Simple Online and Realtime Tracker (SORT).

    Parameters
    ----------
    max_age : int
        Frames a track survives without a matching detection before deletion.
    min_hits : int
        Minimum consecutive hits before a track is reported.
    iou_threshold : float
        Minimum IoU for a detection<->track match.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: Optional[np.ndarray] = None) -> List[KalmanBoxTracker]:
        """Update the tracker set with the current frame's detections.

        Parameters
        ----------
        detections : np.ndarray, optional
            Array of shape ``(N, 5)`` as ``[x1, y1, x2, y2, cls_id]``
            (cls_id optional; pass ``(N, 4)`` to omit). ``None`` == no dets.

        Returns
        -------
        list[KalmanBoxTracker]
            The active, confirmed trackers after this update.
        """
        self.frame_count += 1
        if detections is None or len(detections) == 0:
            dets = np.empty((0, 5), dtype=np.float32)
        else:
            dets = np.asarray(detections, dtype=np.float32)
            if dets.shape[1] == 4:
                dets = np.hstack([dets, -np.ones((dets.shape[0], 1), dtype=np.float32)])

        # Predict existing trackers and drop any that become invalid.
        predicted = np.zeros((len(self.trackers), 4), dtype=np.float32)
        to_del = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            predicted[i] = pred
            if np.any(np.isnan(pred)):
                to_del.append(i)
        for i in reversed(to_del):
            self.trackers.pop(i)
            predicted = np.delete(predicted, i, axis=0)

        matches, unmatched_dets, _ = _associate(dets[:, :4], predicted, self.iou_threshold)

        for d, t in matches:
            self.trackers[t].update(dets[d, :4], int(dets[d, 4]))
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d, :4], int(dets[d, 4])))

        # Reap stale trackers and collect confirmed ones.
        confirmed: List[KalmanBoxTracker] = []
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
                continue
            if trk.time_since_update == 0 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                confirmed.append(trk)
        return confirmed


class Tracker:
    """Thin facade over the tracking backend selected in the config.

    Currently wraps :class:`Sort`. The ``bytetrack`` backend is handled directly
    by the Ultralytics ``model.track`` call in the detector, so this wrapper
    always provides the SORT path used for wrong-side motion analysis.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.sort = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def update(self, detections: Optional[np.ndarray] = None) -> List[KalmanBoxTracker]:
        """Update and return confirmed tracks (see :meth:`Sort.update`)."""
        return self.sort.update(detections)
