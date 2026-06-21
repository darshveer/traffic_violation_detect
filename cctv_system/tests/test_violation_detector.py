"""Integration tests for ViolationDetector.

These tests use a fully mocked ModelLoader and OCR handler so they run
without GPU, model weights, or real video footage. They exercise the complete
per-frame detection pipeline — base detection → violation detection → summary
→ report output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the package root is on the path.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


# ---------------------------------------------------------------------------
# Fake model helpers
# ---------------------------------------------------------------------------

def _make_fake_result(boxes: List[Dict]) -> Any:
    """Build a fake Ultralytics result with the given box dicts.

    The violation_detector calls `boxes.xyxy.cpu().numpy()`, so we need
    objects that support the `.cpu().numpy()` chain. We wrap numpy arrays
    in a tiny CpuArray shim that makes `.cpu()` return self.
    """
    import numpy as _np

    class _CpuArray:
        """Wraps a numpy array so .cpu() returns self and .numpy() returns the array."""
        def __init__(self, arr: _np.ndarray) -> None:
            self._arr = arr

        def cpu(self) -> "_CpuArray":
            return self

        def numpy(self) -> _np.ndarray:
            return self._arr

        def astype(self, dtype) -> "_CpuArray":
            return _CpuArray(self._arr.astype(dtype))

        @property
        def shape(self):
            return self._arr.shape

        def __len__(self):
            return len(self._arr)

    class FakeBoxes:
        def __init__(self, data):
            if data:
                raw_xyxy = _np.array([d["box"] for d in data], dtype=_np.float32).reshape(-1, 4)
                raw_cls = _np.array([d["cls_id"] for d in data], dtype=_np.float32)
                raw_conf = _np.array([d["conf"] for d in data], dtype=_np.float32)
                self.xyxy = _CpuArray(raw_xyxy)
                self.cls = _CpuArray(raw_cls)
                self.conf = _CpuArray(raw_conf)
                # shape[0] is used for the empty-check in _predict
                self.shape = (len(data),)
            else:
                self.xyxy = _CpuArray(_np.zeros((0, 4), dtype=_np.float32))
                self.cls = _CpuArray(_np.array([], dtype=_np.float32))
                self.conf = _CpuArray(_np.array([], dtype=_np.float32))
                self.shape = (0,)

    class FakeResult:
        def __init__(self, data):
            self.boxes = FakeBoxes(data)

    return [FakeResult(boxes)]


def _make_loader(base_boxes: List[Dict], helmet_boxes: List[Dict] | None = None) -> Any:
    """Create a ModelLoader mock that returns configured fake detections."""
    from pipelines.model_loader import LoadedModel

    def _predict_side_effect(source, conf, iou, imgsz, device, half, verbose):
        return _make_fake_result(base_boxes)

    base_model = MagicMock()
    base_model.predict.side_effect = _predict_side_effect
    base_model.names = {0: "person", 2: "car", 3: "motorcycle", 9: "traffic_light"}

    helmet_model = MagicMock()
    if helmet_boxes is not None:
        def _h_predict(source, conf, iou, imgsz, device, half, verbose):
            return _make_fake_result(helmet_boxes)
        helmet_model.predict.side_effect = _h_predict
        helmet_model.names = {0: "helmet_absent", 1: "helmet_present"}
        helmet_available = True
    else:
        helmet_available = False

    loader = MagicMock()
    loader.models = {
        "base": LoadedModel(name="base", model=base_model, available=True, device="cpu", source="fake"),
        "helmet": LoadedModel(name="helmet", model=helmet_model, available=helmet_available, device="cpu", source="fake"),
        "seatbelt": LoadedModel(name="seatbelt", available=False, source="missing"),
        "triple_rider": LoadedModel(name="triple_rider", available=False, source="built-in"),
        "red_light": LoadedModel(name="red_light", available=False, source="built-in"),
        "wrong_side": LoadedModel(name="wrong_side", available=False, source="built-in"),
    }

    def _get(name):
        return loader.models.get(name, LoadedModel(name=name, available=False, source="missing"))

    loader.get.side_effect = _get
    return loader


BLANK_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

# Fake green frame with a red region simulating a traffic light (for red-light test)
RED_LIGHT_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
RED_LIGHT_FRAME[:, :, 2] = 220   # predominantly red (BGR: B=0, G=0, R=220)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestViolationDetectorInit:
    """ViolationDetector initializes without model weights."""

    def test_init_no_crash(self):
        from pipelines.violation_detector import ViolationDetector

        loader = _make_loader([])
        with patch("evidence.ocr_handler.OCRHandler") as MockOCR:
            mock_ocr = MagicMock()
            mock_ocr.available = False
            MockOCR.return_value = mock_ocr
            det = ViolationDetector(model_loader=loader, ocr_handler=mock_ocr)

        assert det is not None
        assert det.device in ("cpu", "cuda:0", "cuda", "mps")

    def test_config_loaded(self):
        from pipelines.violation_detector import ViolationDetector

        loader = _make_loader([])
        ocr = MagicMock()
        ocr.available = False
        det = ViolationDetector(model_loader=loader, ocr_handler=ocr)
        assert det.violation_conf > 0


class TestInferFrame:
    """infer_frame returns the correct schema regardless of what's detected."""

    def _make_det(self) -> "ViolationDetector":
        from pipelines.violation_detector import ViolationDetector

        loader = _make_loader([])
        ocr = MagicMock()
        ocr.available = False
        return ViolationDetector(model_loader=loader, ocr_handler=ocr)

    def test_schema_keys_present(self):
        det = self._make_det()
        result = det.infer_frame(BLANK_FRAME, frame_id=0)
        assert set(result.keys()) >= {"frame_id", "timestamp", "summary", "violations", "detections", "counts"}

    def test_summary_has_all_violation_keys(self):
        det = self._make_det()
        result = det.infer_frame(BLANK_FRAME)
        summary = result["summary"]
        expected = {
            "helmet_absent", "helmet_present",
            "seatbelt_absent", "seatbelt_present",
            "triple_rider", "red_light_violation",
            "wrong_side_driving", "license_plate",
        }
        assert expected <= set(summary.keys())

    def test_no_detections_empty_violations(self):
        det = self._make_det()
        result = det.infer_frame(BLANK_FRAME, frame_id=7)
        assert result["violations"] == []
        assert result["frame_id"] == 7

    def test_timestamp_default(self):
        det = self._make_det()
        result = det.infer_frame(BLANK_FRAME)
        assert isinstance(result["timestamp"], str)
        assert len(result["timestamp"]) > 0

    def test_custom_timestamp(self):
        det = self._make_det()
        result = det.infer_frame(BLANK_FRAME, timestamp="2024-12-05T10:30:00")
        assert result["timestamp"] == "2024-12-05T10:30:00"


class TestTripleRiderDetector:
    """Triple-rider detection: ≥3 persons overlapping a motorcycle."""

    def test_triple_rider_flagged(self):
        from pipelines.violation_detector import ViolationDetector

        # A motorcycle at [100, 100, 300, 400] and 3 persons overlapping it.
        moto_box = [100.0, 100.0, 300.0, 400.0]
        # Persons centred inside the motorcycle box.
        base_boxes = [
            {"box": moto_box, "cls_id": 3, "conf": 0.85},   # motorcycle
            {"box": [120.0, 120.0, 180.0, 350.0], "cls_id": 0, "conf": 0.9},  # person 1
            {"box": [150.0, 120.0, 210.0, 350.0], "cls_id": 0, "conf": 0.9},  # person 2
            {"box": [180.0, 120.0, 240.0, 350.0], "cls_id": 0, "conf": 0.9},  # person 3
        ]
        loader = _make_loader(base_boxes)
        ocr = MagicMock(); ocr.available = False
        det = ViolationDetector(model_loader=loader, ocr_handler=ocr)
        # Lower the threshold so the derived confidence passes.
        det.violation_conf = 0.5
        result = det.infer_frame(BLANK_FRAME)
        types = [v["type"] for v in result["violations"]]
        assert "triple_rider" in types

    def test_two_riders_not_flagged(self):
        from pipelines.violation_detector import ViolationDetector

        moto_box = [100.0, 100.0, 300.0, 400.0]
        base_boxes = [
            {"box": moto_box, "cls_id": 3, "conf": 0.85},
            {"box": [120.0, 120.0, 180.0, 350.0], "cls_id": 0, "conf": 0.9},
            {"box": [180.0, 120.0, 240.0, 350.0], "cls_id": 0, "conf": 0.9},
        ]
        loader = _make_loader(base_boxes)
        ocr = MagicMock(); ocr.available = False
        det = ViolationDetector(model_loader=loader, ocr_handler=ocr)
        result = det.infer_frame(BLANK_FRAME)
        types = [v["type"] for v in result["violations"]]
        assert "triple_rider" not in types


class TestHelmetDetector:
    """Helmet detection: helmet model runs on rider/motorcycle crops."""

    def test_helmet_absent_flagged(self):
        from pipelines.violation_detector import ViolationDetector

        # Motorcycle with rider.
        base_boxes = [{"box": [100.0, 100.0, 300.0, 400.0], "cls_id": 3, "conf": 0.85}]
        # Helmet model returns "helmet_absent" (class 0) with high confidence.
        helmet_boxes = [{"box": [0.0, 0.0, 50.0, 50.0], "cls_id": 0, "conf": 0.92}]
        loader = _make_loader(base_boxes, helmet_boxes)
        ocr = MagicMock(); ocr.available = False
        det = ViolationDetector(model_loader=loader, ocr_handler=ocr)
        det.violation_conf = 0.5
        result = det.infer_frame(BLANK_FRAME)
        types = [v["type"] for v in result["violations"]]
        assert "helmet_absent" in types

    def test_helmet_present_not_violation(self):
        from pipelines.violation_detector import ViolationDetector

        base_boxes = [{"box": [100.0, 100.0, 300.0, 400.0], "cls_id": 3, "conf": 0.85}]
        helmet_boxes = [{"box": [0.0, 0.0, 50.0, 50.0], "cls_id": 1, "conf": 0.88}]
        loader = _make_loader(base_boxes, helmet_boxes)
        ocr = MagicMock(); ocr.available = False
        det = ViolationDetector(model_loader=loader, ocr_handler=ocr)
        result = det.infer_frame(BLANK_FRAME)
        violations = [v for v in result["violations"] if v.get("is_violation")]
        assert not any(v["type"] == "helmet_absent" for v in violations)


class TestBuildSummary:
    """_build_summary aggregates max confidence per violation type."""

    def test_max_confidence_per_type(self):
        from pipelines.violation_detector import ViolationDetector

        loader = _make_loader([])
        ocr = MagicMock(); ocr.available = False
        det = ViolationDetector(model_loader=loader, ocr_handler=ocr)

        violations = [
            {"type": "helmet_absent", "confidence": 0.7, "is_violation": True},
            {"type": "helmet_absent", "confidence": 0.9, "is_violation": True},
            {"type": "triple_rider", "confidence": 0.8, "is_violation": True},
        ]
        summary = det._build_summary(violations, "KA01AB1234")
        assert summary["helmet_absent"] == pytest.approx(0.9)
        assert summary["triple_rider"] == pytest.approx(0.8)
        assert summary["license_plate"] == "KA01AB1234"


class TestAbsenceClassMapping:
    """Class-name -> violation (absence) mapping for helmet/seatbelt models."""

    @pytest.mark.parametrize("name", [
        "no-helmet", "no_helmet", "no helmet", "without helmet", "helmet_absent",
        "no-seatbelt", "no_seatbelt", "no seatbelt", "not wearing seatbelt", "missing helmet",
    ])
    def test_absence_classes(self, name):
        from pipelines.violation_detector import ViolationDetector
        assert ViolationDetector._is_absence_class(name) is True

    @pytest.mark.parametrize("name", [
        "helmet", "moto-helmet", "seatbelt", "with helmet", "wearing seatbelt",
    ])
    def test_presence_classes(self, name):
        from pipelines.violation_detector import ViolationDetector
        assert ViolationDetector._is_absence_class(name) is False
