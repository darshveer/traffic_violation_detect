#!/usr/bin/env python3
"""Estimate the dominant traffic-flow direction from a sample clip.

Wrong-side detection needs the *allowed* direction of travel for each camera.
This tool tracks vehicles over a clip, aggregates their motion vectors, and
prints a suggested ``allowed_direction`` for ``configs/pipeline.yaml`` plus a
heading histogram (so you can tell one-way from two-way roads).

    venv\\Scripts\\python.exe scripts\\calibrate_wrong_side.py <video> [--frames N]
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from collections import Counter

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
import numpy as np  # noqa: E402
import cv2  # noqa: E402


def main(video, max_frames=300):
    from ultralytics import YOLO
    from pipelines.common import select_device
    from pipelines.tracking import Tracker
    from pipelines.inference_utils import COCO_CLASSES, VEHICLE_CLASSES, ID_TO_NAME

    dev = select_device("auto")
    m = YOLO(str(PKG / "models" / "yolo11n.pt"))
    trk = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    veh_ids = {COCO_CLASSES[c] for c in VEHICLE_CLASSES}

    cap = cv2.VideoCapture(video)
    headings = []          # (dx,dy) net per track step, displacement-weighted
    angle_hist = Counter()
    prev_centroid = {}
    n = 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        r = m.predict(fr, device=dev, conf=0.35, verbose=False)[0]
        xy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
        dets = np.array([[*xy[i], cl[i]] for i in range(len(cl)) if cl[i] in veh_ids],
                        dtype=np.float32) if len(cl) else np.empty((0, 5), np.float32)
        for t in trk.update(dets):
            cx, cy = float(t.box[0] + t.box[2]) / 2, float(t.box[1] + t.box[3]) / 2
            if t.id in prev_centroid:
                dx, dy = cx - prev_centroid[t.id][0], cy - prev_centroid[t.id][1]
                d = math.hypot(dx, dy)
                if d >= 1.0:
                    headings.append((dx, dy))
                    angle_hist[round(math.degrees(math.atan2(dy, dx)) / 30) * 30] += 1
            prev_centroid[t.id] = (cx, cy)
    cap.release()

    if not headings:
        print("No vehicle motion detected; cannot calibrate."); return
    arr = np.array(headings)
    resultant = arr.sum(axis=0)
    rmag = float(np.hypot(*resultant))
    unit = resultant / (rmag + 1e-9)
    total = float(np.abs(arr).sum())
    coherence = rmag / (total + 1e-9)   # ~1 one-way, ~0 balanced two-way
    print(f"frames={n} motion_samples={len(headings)}")
    print(f"heading histogram (deg bucket: count): {dict(sorted(angle_hist.items()))}")
    print(f"resultant flow vector: [{unit[0]:.2f}, {unit[1]:.2f}]  (angle={math.degrees(math.atan2(unit[1],unit[0])):.0f} deg)")
    print(f"flow coherence: {coherence:.2f}  ({'one-way' if coherence>0.4 else 'two-way / mixed — set ROI per lane'})")
    print(f"\nSuggested configs/pipeline.yaml:\n  wrong_side:\n    allowed_direction: [{unit[0]:.2f}, {unit[1]:.2f}]")


if __name__ == "__main__":
    args = sys.argv[1:]
    mf = 300
    if "--frames" in args:
        i = args.index("--frames"); mf = int(args[i + 1]); args = args[:i] + args[i + 2:]
    main(args[0], mf)
