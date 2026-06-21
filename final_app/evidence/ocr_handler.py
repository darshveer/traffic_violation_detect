"""License-plate OCR via PaddleOCR with graceful degradation.

PaddleOCR / PaddlePaddle can be awkward to install and is GPU/driver sensitive.
This handler isolates that risk: if the library fails to import or initialise,
:meth:`OCRHandler.read_plate` simply returns an empty string so the rest of the
pipeline keeps running. An optional EasyOCR fallback is used if present.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import numpy as np

# Disable oneDNN/MKLDNN before paddle imports: paddle 3.x's CPU PIR executor
# raises NotImplementedError (ConvertPirAttribute2RuntimeAttribute ... onednn)
# on the OCR models otherwise. Must be set before paddlepaddle is imported.
os.environ.setdefault("FLAGS_use_mkldnn", "0")

logger = logging.getLogger("cctv")

# Indian plate sanity pattern, e.g. "KA01AB1234" (used only for light cleanup).
_PLATE_RE = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}")


class OCRHandler:
    """Read license-plate text from vehicle crops.

    Parameters
    ----------
    lang : str
        OCR language (default ``"en"``).
    use_gpu : bool
        Request GPU inference where supported.
    min_conf : float
        Minimum per-line confidence to accept a recognised string.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = True, min_conf: float = 0.5) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.min_conf = min_conf
        self._engine = None
        self._backend = "none"
        self._init_engine()

    # ------------------------------------------------------------------ setup
    def _init_engine(self) -> None:
        """Try PaddleOCR first, then EasyOCR; disable OCR if neither loads."""
        try:
            from paddleocr import PaddleOCR

            # Try newest signature first (3.x with mkldnn off), then older ones.
            for kwargs in (
                {"lang": self.lang, "enable_mkldnn": False},
                {"use_angle_cls": True, "lang": self.lang},
                {"lang": self.lang},
            ):
                try:
                    self._engine = PaddleOCR(**kwargs)
                    break
                except Exception:  # noqa: BLE001 - try next signature
                    continue
            if self._engine is None:
                raise RuntimeError("no compatible PaddleOCR signature")
            self._backend = "paddleocr"
            logger.info("OCRHandler: PaddleOCR initialised (lang=%s).", self.lang)
            return
        except Exception as exc:  # noqa: BLE001 - any failure -> try fallback
            logger.warning("OCRHandler: PaddleOCR unavailable (%s); trying EasyOCR.", exc)

        try:
            import easyocr

            self._engine = easyocr.Reader([self.lang], gpu=self.use_gpu)
            self._backend = "easyocr"
            logger.info("OCRHandler: EasyOCR fallback initialised.")
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCRHandler: no OCR backend available (%s); OCR disabled.", exc)
            self._engine = None
            self._backend = "none"

    @property
    def available(self) -> bool:
        """Whether an OCR backend successfully initialised."""
        return self._engine is not None

    @property
    def backend(self) -> str:
        """Name of the active backend (``paddleocr`` / ``easyocr`` / ``none``)."""
        return self._backend

    # ------------------------------------------------------------------- read
    def read_plate(self, crop: np.ndarray) -> str:
        """Recognise plate text from a vehicle/plate crop.

        Parameters
        ----------
        crop : np.ndarray
            BGR image region likely containing a plate.

        Returns
        -------
        str
            The best-effort plate string (uppercased, alphanumerics only), or
            ``""`` if nothing confident was found or OCR is disabled.
        """
        if self._engine is None or crop is None or crop.size == 0:
            return ""
        try:
            if self._backend == "paddleocr":
                return self._read_paddle(crop)
            return self._read_easy(crop)
        except Exception as exc:  # noqa: BLE001 - never let OCR crash the pipeline
            logger.debug("OCRHandler.read_plate failed: %s", exc)
            return ""

    def _read_paddle(self, crop: np.ndarray) -> str:
        candidates: list[tuple[str, float]] = []
        # PaddleOCR 3.x: .predict() -> list of dict-like results with
        # 'rec_texts' / 'rec_scores'. Preferred path.
        try:
            for res in (self._engine.predict(crop) or []):
                d = res if hasattr(res, "get") else getattr(res, "res", {})
                texts = d.get("rec_texts") if hasattr(d, "get") else None
                scores = d.get("rec_scores") if hasattr(d, "get") else None
                if texts:
                    for t, s in zip(texts, scores or [1.0] * len(texts)):
                        c = float(s) if s is not None else 1.0
                        if c >= self.min_conf:
                            candidates.append((self._clean(t), c))
                    return self._best(candidates)
        except (AttributeError, TypeError):
            pass  # fall back to legacy 2.x .ocr() API below
        result = self._engine.ocr(crop, cls=True)
        # 2.x: [[ [box, (text, conf)], ... ]] (nested per image).
        for page in result or []:
            for line in page or []:
                text, conf = line[1][0], float(line[1][1])
                if conf >= self.min_conf:
                    candidates.append((self._clean(text), conf))
        return self._best(candidates)

    def _read_easy(self, crop: np.ndarray) -> str:
        result = self._engine.readtext(crop)
        candidates = [
            (self._clean(text), float(conf))
            for (_box, text, conf) in result
            if float(conf) >= self.min_conf
        ]
        return self._best(candidates)

    @staticmethod
    def _clean(text: str) -> str:
        """Uppercase and strip to alphanumerics."""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @staticmethod
    def _best(candidates: list[tuple[str, float]]) -> str:
        """Pick the best candidate, preferring plate-shaped strings.

        A string matching the Indian plate regex wins; otherwise the longest
        high-confidence string is returned.
        """
        if not candidates:
            return ""
        for text, _conf in sorted(candidates, key=lambda c: -c[1]):
            if _PLATE_RE.search(text):
                return text
        # No regex match: return the longest cleaned string.
        return max((c[0] for c in candidates), key=len, default="")
