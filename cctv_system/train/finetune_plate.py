#!/usr/bin/env python3
"""Fine-tune a license-plate DETECTOR (localises plate boxes for OCR).

Standalone: downloads a Roboflow plate dataset, trains yolo11n, and writes
``license_plate_detector.pt`` next to this script's output dir. Designed to run
on a fine-tuning box via nohup; the resulting .pt is copied back to the Mac and
wired into the pipeline so OCR runs on tight plate crops instead of whole cars.
"""
from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

BASE = Path(os.environ.get("FT_BASE", str(Path.home() / "ft")))
API_KEY = "qmB9vC2MU6LTHe1JwBpp"
DATA = BASE / "plate_data"
OUT = BASE / "license_plate_detector.pt"

# (workspace, project, version, format) candidates, tried in order.
CANDIDATES = [
    ("roboflow-universe-projects", "license-plate-recognition-rxg4e", 4, "yolov11"),
    ("roboflow-universe-projects", "license-plate-recognition-rxg4e", 11, "yolov8"),
]


def get_data():
    import yaml
    fixed = DATA / "data_fixed.yaml"
    if not ((DATA / "data.yaml").exists() and (DATA / "train" / "images").exists()):
        from roboflow import Roboflow
        rf = Roboflow(api_key=API_KEY)
        for ws, proj, ver, fmt in CANDIDATES:
            try:
                print(f"downloading {ws}/{proj} v{ver} ({fmt})", flush=True)
                p = rf.workspace(ws).project(proj)
                p.version(ver).download(fmt, location=str(DATA), overwrite=True)
                break
            except Exception as exc:
                print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
        else:
            raise SystemExit("all plate dataset candidates failed")
    else:
        print("reusing existing plate dataset", flush=True)
    # ALWAYS rewrite data_fixed.yaml with the CURRENT absolute path (the dataset
    # may have moved between machines, so a stale path: must not be reused).
    cfg = yaml.safe_load(open(DATA / "data.yaml"))
    cfg.update({"path": str(DATA), "train": "train/images",
                "val": "valid/images", "test": "test/images"})
    fixed = DATA / "data_fixed.yaml"
    yaml.safe_dump(cfg, open(fixed, "w"), sort_keys=False)
    print("classes:", cfg.get("names"), flush=True)
    return fixed


def main(epochs=15, batch=32, workers=16, patience=5):
    from ultralytics import YOLO
    data = get_data()
    model = YOLO("yolo11n.pt")
    res = model.train(data=str(data), epochs=epochs, batch=batch, imgsz=640,
                      device=0, workers=workers, patience=patience, cache=False, plots=False,
                      project=str(BASE / "runs_plate"),
                      name="train", exist_ok=True, save=True, save_period=5, verbose=True)
    best = Path(getattr(res, "save_dir", BASE / "runs_plate/train")) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(str(best), str(OUT))
        print(f"COPIED best -> {OUT}", flush=True)
    else:
        print(f"WARNING best.pt missing at {best}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
