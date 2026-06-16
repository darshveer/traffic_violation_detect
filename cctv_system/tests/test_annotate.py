"""Tests for frame annotation drawing."""

import numpy as np

from evidence.annotate import annotate_batch, annotate_frame


def _blank(h=120, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_annotate_frame_returns_same_shape():
    frame = _blank()
    dets = [{"box": [10, 10, 50, 60], "label": "helmet_absent",
             "confidence": 0.87, "is_violation": True}]
    out = annotate_frame(frame, dets, timestamp="00:00:01")
    assert out.shape == frame.shape


def test_annotate_frame_does_not_mutate_original_when_copy():
    frame = _blank()
    before = frame.copy()
    annotate_frame(frame, [{"box": [0, 0, 30, 30], "label": "car"}], copy=True)
    assert np.array_equal(frame, before)


def test_annotate_frame_draws_something():
    frame = _blank()
    out = annotate_frame(frame, [{"box": [5, 5, 40, 40], "label": "car",
                                  "is_violation": True}], copy=True)
    # A red violation box should introduce non-zero (coloured) pixels.
    assert out.sum() > 0


def test_annotate_frame_handles_missing_box():
    frame = _blank()
    out = annotate_frame(frame, [{"label": "no_box"}])
    assert out.shape == frame.shape


def test_annotate_batch_length_matches():
    frames = [_blank(), _blank()]
    dets = [[{"box": [0, 0, 10, 10], "label": "a"}], []]
    out = annotate_batch(frames, dets, timestamps=["t0", "t1"])
    assert len(out) == 2
