#!/usr/bin/env python3
"""Standalone PaddleOCR plate-recognition test (runs on the fine-tune box).

Pipeline: trained plate detector (YOLO) localises plates in each frame ->
PaddleOCR reads the cropped plate. Reports read-rate + sample texts so we can
compare recognition quality against EasyOCR.
"""
from __future__ import annotations
import os
# Disable oneDNN/MKLDNN: paddle 3.x CPU PIR executor crashes on it
# (NotImplementedError ConvertPirAttribute2RuntimeAttribute ... onednn).
os.environ["FLAGS_use_mkldnn"] = "0"
import re
import sys

import cv2

PLATE_DET = "/home/ubuntu/ft/license_plate_detector.pt"
VIDEO = "/home/ubuntu/ft/anpr_plates.mp4"
MAX_FRAMES = 30


def clean(t: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(t).upper())


def paddle_read(ocr, crop):
    """Read text from a crop, supporting PaddleOCR 3.x (.predict) and 2.x (.ocr)."""
    # 3.x API
    try:
        res = ocr.predict(crop)
        out = []
        for r in (res or []):
            d = r if isinstance(r, dict) else getattr(r, "res", r)
            texts = d.get("rec_texts") if hasattr(d, "get") else None
            scores = d.get("rec_scores") if hasattr(d, "get") else None
            if texts:
                for t, s in zip(texts, scores or [1.0] * len(texts)):
                    if s is None or s >= 0.3:
                        out.append(clean(t))
        if out:
            return max(out, key=len)
    except Exception:
        pass
    # 2.x fallback
    try:
        res = ocr.ocr(crop, cls=True)
        cands = []
        for page in (res or []):
            for line in (page or []):
                txt, sc = line[1][0], float(line[1][1])
                if sc >= 0.3:
                    cands.append(clean(txt))
        if cands:
            return max(cands, key=len)
    except Exception:
        pass
    return ""


def main():
    from ultralytics import YOLO
    from paddleocr import PaddleOCR
    import paddleocr as _p
    print("paddleocr", getattr(_p, "__version__", "?"))
    det = YOLO(PLATE_DET)
    ocr = None
    for kw in ({"lang": "en", "enable_mkldnn": False},
               {"use_angle_cls": True, "lang": "en"},
               {"lang": "en"}):
        try:
            ocr = PaddleOCR(**kw)
            break
        except Exception as e:
            print(f"PaddleOCR init {kw} failed: {e}")
    if ocr is None:
        raise SystemExit("could not init PaddleOCR")

    cap = cv2.VideoCapture(VIDEO)
    n = 0; loc = 0; reads = []
    while n < MAX_FRAMES:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        r = det.predict(fr, conf=0.35, verbose=False)[0]
        for b in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [int(v) for v in b]
            crop = fr[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            loc += 1
            t = paddle_read(ocr, crop)
            if t:
                reads.append(t)
    cap.release()
    rate = len(reads) / max(1, loc)
    print(f"frames={n} plates_localised={loc} plates_read={len(reads)} read_rate={rate:.2%}")
    print(f"sample reads: {reads[:30]}")


if __name__ == "__main__":
    main()
