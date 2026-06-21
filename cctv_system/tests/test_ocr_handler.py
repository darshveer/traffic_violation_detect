"""Tests for the OCRHandler (evidence/ocr_handler.py).

All tests mock the PaddleOCR and EasyOCR backends so no actual OCR
libraries need to be installed for the test suite to pass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_crop(h: int = 60, w: int = 120) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_handler_with_mock_paddle(ocr_result) -> "OCRHandler":
    """Create an OCRHandler backed by a mocked PaddleOCR engine."""
    from evidence.ocr_handler import OCRHandler

    with patch("evidence.ocr_handler.OCRHandler._init_engine"):
        handler = OCRHandler.__new__(OCRHandler)
        handler.lang = "en"
        handler.use_gpu = False
        handler.min_conf = 0.5
        handler._backend = "paddleocr"

        mock_engine = MagicMock()
        mock_engine.ocr.return_value = ocr_result
        handler._engine = mock_engine

    return handler


def _make_handler_with_mock_easy(ocr_result) -> "OCRHandler":
    """Create an OCRHandler backed by a mocked EasyOCR engine."""
    from evidence.ocr_handler import OCRHandler

    with patch("evidence.ocr_handler.OCRHandler._init_engine"):
        handler = OCRHandler.__new__(OCRHandler)
        handler.lang = "en"
        handler.use_gpu = False
        handler.min_conf = 0.5
        handler._backend = "easyocr"

        mock_engine = MagicMock()
        mock_engine.readtext.return_value = ocr_result
        handler._engine = mock_engine

    return handler


def _make_disabled_handler() -> "OCRHandler":
    """Create an OCRHandler with no engine loaded."""
    from evidence.ocr_handler import OCRHandler

    with patch("evidence.ocr_handler.OCRHandler._init_engine"):
        handler = OCRHandler.__new__(OCRHandler)
        handler.lang = "en"
        handler.use_gpu = False
        handler.min_conf = 0.5
        handler._backend = "none"
        handler._engine = None
    return handler


# ---------------------------------------------------------------------------
# Tests: OCRHandler.available / backend
# ---------------------------------------------------------------------------

class TestOCRHandlerProperties:
    def test_available_false_when_no_engine(self):
        handler = _make_disabled_handler()
        assert handler.available is False

    def test_available_true_when_engine_set(self):
        from evidence.ocr_handler import OCRHandler
        with patch("evidence.ocr_handler.OCRHandler._init_engine"):
            handler = OCRHandler.__new__(OCRHandler)
            handler.lang = "en"
            handler.use_gpu = False
            handler.min_conf = 0.5
            handler._backend = "paddleocr"
            handler._engine = MagicMock()
        assert handler.available is True

    def test_backend_property(self):
        handler = _make_disabled_handler()
        assert handler.backend == "none"


# ---------------------------------------------------------------------------
# Tests: OCRHandler.read_plate — PaddleOCR backend
# ---------------------------------------------------------------------------

class TestReadPlatePaddle:
    def test_returns_empty_on_disabled(self):
        handler = _make_disabled_handler()
        assert handler.read_plate(_blank_crop()) == ""

    def test_returns_empty_on_none_crop(self):
        handler = _make_handler_with_mock_paddle([[]])
        assert handler.read_plate(None) == ""

    def test_returns_empty_on_zero_size_crop(self):
        handler = _make_handler_with_mock_paddle([[]])
        assert handler.read_plate(np.zeros((0, 0, 3), dtype=np.uint8)) == ""

    def test_valid_plate_returned(self):
        # Simulate PaddleOCR returning a valid Indian plate.
        ocr_result = [
            [[None, ("KA01AB1234", 0.95)]]
        ]
        handler = _make_handler_with_mock_paddle(ocr_result)
        result = handler.read_plate(_blank_crop())
        assert "KA01AB1234" in result

    def test_low_confidence_filtered(self):
        # Confidence below min_conf (0.5) should be rejected.
        ocr_result = [
            [[None, ("DL3CAB1234", 0.3)]]
        ]
        handler = _make_handler_with_mock_paddle(ocr_result)
        result = handler.read_plate(_blank_crop())
        assert result == ""

    def test_multiple_lines_best_selected(self):
        # Two candidates: low-conf non-plate and high-conf plate.
        ocr_result = [
            [
                [None, ("GARBAGE", 0.6)],
                [None, ("MH12AB1234", 0.91)],
            ]
        ]
        handler = _make_handler_with_mock_paddle(ocr_result)
        result = handler.read_plate(_blank_crop())
        assert "MH12AB1234" in result

    def test_engine_exception_returns_empty(self):
        from evidence.ocr_handler import OCRHandler
        with patch("evidence.ocr_handler.OCRHandler._init_engine"):
            handler = OCRHandler.__new__(OCRHandler)
            handler.lang = "en"
            handler.use_gpu = False
            handler.min_conf = 0.5
            handler._backend = "paddleocr"
            engine = MagicMock()
            engine.ocr.side_effect = RuntimeError("OCR crash")
            handler._engine = engine
        # Should not raise; returns empty string instead.
        result = handler.read_plate(_blank_crop())
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: OCRHandler.read_plate — EasyOCR backend
# ---------------------------------------------------------------------------

class TestReadPlateEasy:
    def test_valid_plate_returned_easyocr(self):
        # EasyOCR returns [(box, text, conf), ...]
        ocr_result = [
            ([None], "TN 09 BE 1234", 0.88)
        ]
        handler = _make_handler_with_mock_easy(ocr_result)
        result = handler.read_plate(_blank_crop())
        assert "TN09BE1234" in result

    def test_low_conf_filtered_easyocr(self):
        ocr_result = [([None], "XX 99 YY 9999", 0.2)]
        handler = _make_handler_with_mock_easy(ocr_result)
        assert handler.read_plate(_blank_crop()) == ""


# ---------------------------------------------------------------------------
# Tests: OCRHandler._clean / _best
# ---------------------------------------------------------------------------

class TestCleanAndBest:
    def test_clean_strips_non_alphanumeric(self):
        from evidence.ocr_handler import OCRHandler
        assert OCRHandler._clean("KA-01 AB 1234") == "KA01AB1234"

    def test_clean_uppercases(self):
        from evidence.ocr_handler import OCRHandler
        assert OCRHandler._clean("ka01ab1234") == "KA01AB1234"

    def test_best_prefers_plate_regex(self):
        from evidence.ocr_handler import OCRHandler
        candidates = [
            ("GARBAGE", 0.99),
            ("KA01AB1234", 0.7),
        ]
        assert OCRHandler._best(candidates) == "KA01AB1234"

    def test_best_returns_longest_when_no_regex_match(self):
        from evidence.ocr_handler import OCRHandler
        candidates = [
            ("ABC", 0.9),
            ("LONGSTRING", 0.7),
        ]
        assert OCRHandler._best(candidates) == "LONGSTRING"

    def test_best_returns_empty_on_no_candidates(self):
        from evidence.ocr_handler import OCRHandler
        assert OCRHandler._best([]) == ""
