#!/usr/bin/env python3
"""Inspect PaddleOCR 3.x return structure on one plate crop."""
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

det = YOLO("/home/ubuntu/ft/license_plate_detector.pt")
ocr = PaddleOCR(lang="en")
cap = cv2.VideoCapture("/home/ubuntu/ft/anpr_plates.mp4")
crop = None
for _ in range(40):
    ok, fr = cap.read()
    if not ok:
        break
    r = det.predict(fr, conf=0.35, verbose=False)[0]
    if len(r.boxes):
        x1, y1, x2, y2 = [int(v) for v in r.boxes.xyxy.cpu().numpy()[0]]
        crop = fr[max(0, y1):y2, max(0, x1):x2]
        break
cap.release()
print("crop shape:", None if crop is None else crop.shape)
res = ocr.predict(crop)
print("=== type(res):", type(res), "len:", len(res) if hasattr(res, "__len__") else "n/a")
r0 = res[0]
print("=== type(r0):", type(r0))
print("=== has keys:", hasattr(r0, "keys"))
try:
    print("=== keys:", list(r0.keys()))
except Exception as e:
    print("no keys:", e)
for k in ("rec_texts", "rec_scores", "rec_text", "text"):
    try:
        print(f"  r0[{k}] =", r0[k])
    except Exception as e:
        print(f"  r0[{k}] err: {e}")
print("=== repr (truncated) ===")
print(repr(r0)[:800])
