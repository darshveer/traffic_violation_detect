#!/usr/bin/env python3
"""Diagnostic harness: profile videos and exercise each pipeline module.

Run on the GPU box:
    venv\\Scripts\\python.exe scripts\\diagnose.py profile <video> [<video> ...]
    venv\\Scripts\\python.exe scripts\\diagnose.py pipeline <video> [--frames N]
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

import cv2  # noqa: E402


def _dev():
    from pipelines.common import select_device
    return select_device("auto")


def profile(videos):
    """Print resolution/length + COCO class histogram per video using the base model."""
    from ultralytics import YOLO
    dev = _dev()
    m = YOLO(str(PKG / "models" / "yolo11n.pt"))
    for v in videos:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print(f"{v}: CANNOT OPEN"); continue
        fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tot = Counter(); sampled = 0
        step = max(1, n // 10)
        for i in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                continue
            sampled += 1
            r = m.predict(fr, device=dev, conf=0.3, verbose=False)[0]
            for c in r.boxes.cls.tolist():
                tot[m.names[int(c)]] += 1
        cap.release()
        print(f"{v} | {w}x{h} {n}f @{fps:.0f}fps | {sampled} frames sampled | {dict(tot)}")


def pipeline(video, max_frames=40):
    """Run the full ViolationDetector and tally per-module outputs."""
    from pipelines.violation_detector import ViolationDetector
    from pipelines.inference_utils import iter_video_frames
    det = ViolationDetector(config_path=str(PKG / "configs" / "pipeline.yaml"))
    print(f"device={det.device} ocr_backend={getattr(det.ocr,'backend','off')}")
    loaded = {k: (v.available, v.source) for k, v in det.loader.models.items()}
    print("models:", loaded)

    counts = Counter(); plates = []; n = 0
    for idx, frame, ts in iter_video_frames(video, skip_frames=5, start_frame=0):
        res = det.infer_frame(frame, frame_id=idx, timestamp=str(ts), use_tracking=True)
        for k, v in res["counts"].items():
            counts[k] += v
        lp = res["summary"].get("license_plate")
        if lp:
            plates.append(lp)
        n += 1
        if n >= max_frames:
            break
    print(f"processed {n} frames")
    print("violation/observation counts:", dict(counts))
    print("plates read:", plates[:20])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "profile":
        profile(sys.argv[2:])
    elif cmd == "pipeline":
        args = sys.argv[2:]
        mf = 40
        if "--frames" in args:
            i = args.index("--frames"); mf = int(args[i + 1]); args = args[:i] + args[i + 2:]
        pipeline(args[0], mf)
    else:
        print(__doc__)
