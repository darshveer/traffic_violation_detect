#!/usr/bin/env python3
"""Inspect the raw PaddleOCR return format on a real plate crop."""
from __future__ import annotations
import sys
from pathlib import Path
PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
import cv2  # noqa: E402


def main(video):
    import paddleocr
    from ultralytics import YOLO
    from pipelines.common import select_device
    from pipelines.inference_utils import COCO_CLASSES, crop
    print("paddleocr version:", getattr(paddleocr, "__version__", "?"))
    dev = select_device("auto")
    m = YOLO(str(PKG / "models" / "yolo11n.pt"))
    cap = cv2.VideoCapture(video)
    # find a frame with a large car crop
    best = None
    for _ in range(60):
        ok, fr = cap.read()
        if not ok:
            break
        r = m.predict(fr, device=dev, conf=0.4, verbose=False)[0]
        xy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(cl)):
            if cl[i] in (COCO_CLASSES["car"], COCO_CLASSES["truck"], COCO_CLASSES["bus"]):
                c = crop(fr, xy[i].tolist(), pad=0.0)
                area = c.shape[0] * c.shape[1]
                if best is None or area > best[0]:
                    best = (area, c)
    cap.release()
    if best is None:
        print("no vehicle crop found"); return
    c = best[1]
    print("crop shape:", c.shape)
    eng = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
    # Try legacy .ocr API
    try:
        res = eng.ocr(c, cls=True)
        print("=== .ocr(cls=True) type:", type(res))
        print(repr(res)[:1500])
    except Exception as exc:
        print(".ocr(cls=True) FAILED:", type(exc).__name__, exc)
    # Try new .predict API
    try:
        res2 = eng.predict(c)
        print("=== .predict() type:", type(res2))
        r0 = res2[0] if isinstance(res2, list) and res2 else res2
        print("keys:", list(r0.keys()) if hasattr(r0, "keys") else "n/a")
        if hasattr(r0, "get"):
            print("rec_texts:", r0.get("rec_texts"))
            print("rec_scores:", r0.get("rec_scores"))
    except Exception as exc:
        print(".predict() FAILED:", type(exc).__name__, exc)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(PKG / "datasets/test_data/anpr_plates.mp4"))
