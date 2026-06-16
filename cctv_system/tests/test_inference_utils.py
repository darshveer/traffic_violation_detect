"""Unit tests for geometry / batching / label-parsing helpers."""

import numpy as np
import pytest

from pipelines.inference_utils import (
    batched,
    box_center,
    clip_box,
    crop,
    overlap_ratio,
    parse_yolo_label,
    point_in_box,
)


def test_clip_box_clamps_to_bounds():
    assert clip_box([-5, -5, 50, 50], 40, 30) == (0, 0, 40, 30)


def test_clip_box_degenerate_box_made_valid():
    x1, y1, x2, y2 = clip_box([10, 10, 10, 10], 100, 100)
    assert x2 > x1 and y2 > y1


def test_box_center():
    assert box_center([0, 0, 10, 20]) == (5.0, 10.0)


def test_point_in_box():
    assert point_in_box((5, 5), [0, 0, 10, 10])
    assert not point_in_box((15, 5), [0, 0, 10, 10])


def test_overlap_ratio_full_containment():
    inner = [2, 2, 4, 4]
    outer = [0, 0, 10, 10]
    assert overlap_ratio(inner, outer) == pytest.approx(1.0)


def test_overlap_ratio_no_overlap():
    assert overlap_ratio([0, 0, 1, 1], [5, 5, 6, 6]) == pytest.approx(0.0)


def test_crop_returns_region():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:20, 30:40] = 255
    region = crop(img, [30, 10, 40, 20])
    assert region.shape[0] > 0 and region.shape[1] > 0


def test_batched_chunks():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_batched_rejects_zero():
    with pytest.raises(ValueError):
        list(batched([1, 2], 0))


def test_parse_yolo_label_valid(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.2 0.1 0.1\n")
    rows = parse_yolo_label(str(p))
    assert rows == [(0, 0.5, 0.5, 0.3, 0.4), (1, 0.2, 0.2, 0.1, 0.1)]


def test_parse_yolo_label_skips_blank_lines(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("\n0 0.5 0.5 0.3 0.4\n\n")
    assert len(parse_yolo_label(str(p))) == 1


def test_parse_yolo_label_missing_file_returns_empty(tmp_path):
    assert parse_yolo_label(str(tmp_path / "nope.txt")) == []


def test_parse_yolo_label_malformed_raises(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("0 0.5 0.5 0.3\n")  # only 4 fields
    with pytest.raises(ValueError):
        parse_yolo_label(str(p))


def test_parse_yolo_label_non_numeric_raises(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("0 a b c d\n")
    with pytest.raises(ValueError):
        parse_yolo_label(str(p))
