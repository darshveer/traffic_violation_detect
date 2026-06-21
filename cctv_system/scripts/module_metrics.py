#!/usr/bin/env python3
"""Per-module correctness metrics for the violation pipeline.

Modes:
  detect  - YOLO.val() mAP/P/R for helmet, seatbelt, triple_rider detectors
  logic   - functional tests for wrong_side + red_light built-in logic
  ocr     - run OCR on real vehicle/plate crops from a video, report read-rate
  all     - run everything
"""
from __future__ import annotations
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
import numpy as np  # noqa: E402


def _fixed_yaml(dataset_dir: Path, nc: int, names: list, split_map: dict) -> Path:
    import yaml
    out = dataset_dir / "data_metrics.yaml"
    cfg = {"path": str(dataset_dir), **split_map, "nc": nc, "names": names}
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return out


def eval_detect():
    from ultralytics import YOLO
    from pipelines.common import select_device
    dev = select_device("auto")
    jobs = [
        ("helmet", PKG / "models/helmet/helmet_finetuned.pt",
         PKG / "datasets/helmet_data", 3, ["helmet", "moto-helmet", "no-helmet"], "test"),
        ("seatbelt", PKG / "models/seatbelt/seatbelt_finetuned.pt",
         PKG / "datasets/seatbelt_data", 2, ["no-seatbelt", "seatbelt"], "valid"),
        ("triple_rider", PKG / "models/plugins/triple_rider.pt",
         PKG / "datasets/triple_rider_data", 1, ["tripleriding"], "test"),
    ]
    print("\n========== DETECTION METRICS (YOLO.val) ==========")
    for name, weights, dd, nc, names, split in jobs:
        if not weights.exists():
            print(f"[{name}] weights missing: {weights}"); continue
        # build a yaml whose 'test' entry points at the actual folder we evaluate
        sm = {"train": "train/images", "val": f"{split}/images", "test": f"{split}/images"}
        yml = _fixed_yaml(dd, nc, names, sm)
        m = YOLO(str(weights))
        try:
            res = m.val(data=str(yml), split="test", device=dev, verbose=False, plots=False)
            p, r = float(res.box.mp), float(res.box.mr)
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            print(f"[{name}] split={split} P={p:.3f} R={r:.3f} F1={f1:.3f} "
                  f"mAP50={float(res.box.map50):.3f} mAP50-95={float(res.box.map):.3f}")
        except Exception as exc:
            print(f"[{name}] val failed: {type(exc).__name__}: {exc}")


def _detector_no_models():
    """ViolationDetector with an empty loader + OCR off, for logic-only tests."""
    from unittest.mock import MagicMock
    from pipelines.violation_detector import ViolationDetector
    loader = MagicMock()
    lm = MagicMock(); lm.available = False; lm.source = "built-in"; lm.model = None
    loader.get.return_value = lm
    loader.models = {}
    ocr = MagicMock(); ocr.available = False; ocr.backend = "off"
    return ViolationDetector(model_loader=loader, ocr_handler=ocr)


def eval_logic():
    print("\n========== LOGIC TESTS (wrong_side, red_light) ==========")
    det = _detector_no_models()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # --- wrong_side: vehicle moving UP (against allowed [0,1]=down) should flag ---
    det.tracker.sort.trackers.clear()
    flagged_wrong = False
    for step in range(14):
        y = 300 - step * 18  # moving upward
        veh = [{"box": [300, y, 360, y + 40], "cls_id": 2, "conf": 0.9}]
        evs = det._detect_wrong_side(frame, veh, {})
        if any(e["type"] == "wrong_side_driving" for e in evs):
            flagged_wrong = True
    # --- control: vehicle moving DOWN (with allowed) must NOT flag ---
    det.tracker.sort.trackers.clear()
    flagged_right = False
    for step in range(14):
        y = 20 + step * 18  # moving downward
        veh = [{"box": [300, y, 360, y + 40], "cls_id": 2, "conf": 0.9}]
        evs = det._detect_wrong_side(frame, veh, {})
        if any(e["type"] == "wrong_side_driving" for e in evs):
            flagged_right = True
    print(f"[wrong_side] against-flow flagged={flagged_wrong} (expect True); "
          f"with-flow flagged={flagged_right} (expect False) -> "
          f"{'PASS' if flagged_wrong and not flagged_right else 'FAIL'}")

    # --- red_light: red light + vehicle in stop band should flag ---
    red_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    red_frame[40:80, 300:330] = (0, 0, 255)  # BGR red patch where the light box is
    lights = [{"box": [300, 40, 330, 80], "cls_id": 9, "conf": 0.9}]
    # stop band default [0,0.55,1,0.65] -> y in [198,234]; place vehicle there
    veh_in = [{"box": [280, 205, 360, 230], "cls_id": 2, "conf": 0.9}]
    ev_red = det._detect_red_light(red_frame, lights, veh_in, {})
    # green light control
    green_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    green_frame[40:80, 300:330] = (0, 255, 0)
    ev_green = det._detect_red_light(green_frame, lights, veh_in, {})
    red_ok = any(e["type"] == "red_light_violation" for e in ev_red)
    green_ok = not any(e["type"] == "red_light_violation" for e in ev_green)
    print(f"[red_light] red+in-band flagged={red_ok} (expect True); "
          f"green+in-band flagged={not green_ok} (expect False) -> "
          f"{'PASS' if red_ok and green_ok else 'FAIL'}")


def eval_ocr(video, max_frames=30):
    import cv2
    from ultralytics import YOLO
    from pipelines.common import select_device
    from pipelines.inference_utils import COCO_CLASSES, VEHICLE_CLASSES, crop
    from evidence.ocr_handler import OCRHandler
    dev = select_device("auto")
    print(f"\n========== OCR METRICS ({video}) ==========")
    m = YOLO(str(PKG / "models" / "yolo11n.pt"))
    ocr = OCRHandler(use_gpu=(dev != "cpu"), min_conf=0.25)
    print(f"OCR backend: {ocr.backend}")
    # Optional plate detector -> localise plates before OCR.
    plate_path = PKG / "models/plugins/license_plate_detector.pt"
    plate = YOLO(str(plate_path)) if plate_path.exists() else None
    print(f"plate_detector: {'loaded' if plate else 'absent (whole-vehicle OCR)'}")
    veh_ids = {COCO_CLASSES[c] for c in VEHICLE_CLASSES}
    cap = cv2.VideoCapture(video)
    n = 0; vehicles_seen = 0; plates_loc = 0; reads = []
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        r = m.predict(fr, device=dev, conf=0.4, verbose=False)[0]
        xy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(cl)):
            if cl[i] not in veh_ids:
                continue
            vehicles_seen += 1
            vcrop = crop(fr, xy[i].tolist(), pad=0.05)
            if vcrop.size == 0:
                continue
            if plate is not None:
                pr = plate.predict(vcrop, device=dev, conf=0.25, verbose=False)[0]
                pbx = pr.boxes.xyxy.cpu().numpy()
                pcf = pr.boxes.conf.cpu().numpy()
                order = pcf.argsort()[::-1]
                for j in order:
                    plates_loc += 1
                    t = ocr.read_plate(crop(vcrop, pbx[j].tolist(), pad=0.12))
                    if t:
                        reads.append(t); break
            else:
                t = ocr.read_plate(vcrop)
                if t:
                    reads.append(t)
    cap.release()
    denom = plates_loc if plate is not None else vehicles_seen
    rate = len(reads) / max(1, denom)
    print(f"frames={n} vehicle_crops={vehicles_seen} plates_localised={plates_loc} "
          f"plates_read={len(reads)} read_rate={rate:.2%}")
    print(f"sample plates: {reads[:25]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("detect", "all"):
        eval_detect()
    if mode in ("logic", "all"):
        eval_logic()
    if mode in ("ocr", "all"):
        vid = sys.argv[2] if len(sys.argv) > 2 else str(PKG / "datasets/test_data/anpr_plates.mp4")
        eval_ocr(vid)
