#!/usr/bin/env python3
"""Retrain the triple-rider detector on the fine-tuning box (higher recall).

Uses yolo11s (vs the original n) + more epochs to lift recall above the 0.55 of
the first model. Downloads the Roboflow dataset, trains, and copies best.pt to
/home/oem/ft/triple_rider.pt for pulling back to the Mac.
"""
from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

BASE = Path(os.environ.get("FT_BASE", str(Path.home() / "ft")))
API_KEY = "qmB9vC2MU6LTHe1JwBpp"
DATA = BASE / "triple_data"
OUT = BASE / "triple_rider.pt"


def get_data():
    import yaml
    if (DATA / "data.yaml").exists() and (DATA / "train" / "images").exists():
        print("reusing existing triple dataset", flush=True)
    else:
        from roboflow import Roboflow
        rf = Roboflow(api_key=API_KEY)
        print("downloading pran/triple-riding-detection-kbulg v1", flush=True)
        rf.workspace("pran").project("triple-riding-detection-kbulg").version(1).download(
            "yolov8", location=str(DATA), overwrite=True)
    cfg = yaml.safe_load(open(DATA / "data.yaml"))
    cfg.update({"path": str(DATA), "train": "train/images",
                "val": "valid/images", "test": "test/images"})
    fixed = DATA / "data_fixed.yaml"
    yaml.safe_dump(cfg, open(fixed, "w"), sort_keys=False)
    print("classes:", cfg.get("names"), flush=True)
    return fixed


def main(epochs=80, batch=32, workers=16):
    from ultralytics import YOLO
    data = get_data()
    model = YOLO("yolo11s.pt")
    res = model.train(data=str(data), epochs=epochs, batch=batch, imgsz=640,
                      device=0, workers=workers, patience=30, cache="ram", plots=False,
                      project=str(BASE / "runs_triple"), name="train",
                      exist_ok=True, save=True, save_period=20, verbose=True)
    best = Path(getattr(res, "save_dir", BASE / "runs_triple/train")) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(str(best), str(OUT))
        print(f"COPIED best -> {OUT}", flush=True)
    else:
        print(f"WARNING best.pt missing at {best}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
