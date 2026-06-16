"""Unit tests for the SORT tracker and IoU helper."""

import numpy as np

from pipelines.tracking import Sort, iou_batch


def test_iou_batch_identical_boxes():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    iou = iou_batch(boxes, boxes)
    assert iou.shape == (1, 1)
    assert iou[0, 0] == 1.0


def test_iou_batch_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[20, 20, 30, 30]], dtype=np.float32)
    assert iou_batch(a, b)[0, 0] == 0.0


def test_iou_batch_empty():
    a = np.zeros((0, 4), dtype=np.float32)
    b = np.array([[0, 0, 1, 1]], dtype=np.float32)
    assert iou_batch(a, b).shape == (0, 1)


def test_sort_tracks_single_object_across_frames():
    sort = Sort(max_age=5, min_hits=1, iou_threshold=0.2)
    box = np.array([[100, 100, 140, 180, 2]], dtype=np.float32)
    last = None
    for step in range(5):
        moved = box.copy()
        moved[:, [0, 2]] += step * 10  # drift right
        last = sort.update(moved)
    assert len(last) == 1
    # The track should have accumulated a rightward (+x) motion vector.
    dx, _dy = last[0].motion_vector(window=5)
    assert dx > 0


def test_sort_handles_no_detections():
    sort = Sort(max_age=2, min_hits=1, iou_threshold=0.3)
    sort.update(np.array([[0, 0, 10, 10, 2]], dtype=np.float32))
    confirmed = sort.update(None)  # no detections this frame
    assert isinstance(confirmed, list)


def test_sort_reaps_stale_tracks():
    sort = Sort(max_age=1, min_hits=1, iou_threshold=0.3)
    sort.update(np.array([[0, 0, 10, 10, 2]], dtype=np.float32))
    sort.update(None)
    sort.update(None)
    sort.update(None)
    assert len(sort.trackers) == 0
