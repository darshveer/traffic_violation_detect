#!/usr/bin/env python3
"""Train a dedicated single-class triple-riding detector (triple_rider plugin).

Trains yolo11n on the Roboflow ``pran/triple-riding-detection`` dataset and
copies the best weights to ``models/plugins/triple_rider.pt`` so the pipeline's
plug-in path picks it up automatically.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
DATA_DIR = PKG / "datasets" / "triple_rider_data"
OUT = PKG / "models" / "plugins" / "triple_rider.pt"


def _write_fixed_yaml() -> Path:
    """Write a data.yaml with an absolute ``path`` so ultralytics resolves splits."""
    import yaml
    fixed = DATA_DIR / "data_fixed.yaml"
    cfg = {
        "path": str(DATA_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["tripleriding"],
    }
    with open(fixed, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return fixed


def main(epochs: int = 100, batch: int = 16, imgsz: int = 640):
    from ultralytics import YOLO
    from pipelines.common import select_device
    dev = select_device("auto")
    data = _write_fixed_yaml()

    run_dir = PKG / "runs" / "triple" / "train"
    last = run_dir / "weights" / "last.pt"

    # --- checkpoint versioning / resume -------------------------------------
    # Ultralytics writes last.pt + best.pt every epoch, and a versioned
    # epoch{N}.pt every `save_period` epochs. If a prior run was interrupted,
    # resume from last.pt so no epochs are lost.
    common = dict(
        epochs=epochs, batch=batch, imgsz=imgsz, device=dev, patience=25,
        project=str(PKG / "runs" / "triple"), name="train", exist_ok=True,
        save=True, save_period=10, verbose=True,
    )
    if last.exists():
        print(f"device={dev} | RESUMING from {last}")
        model = YOLO(str(last))
        results = model.train(resume=True)
    else:
        print(f"device={dev} data={data} | fresh training")
        model = YOLO(str(PKG / "models" / "yolo11n.pt"))
        results = model.train(data=str(data), **common)

    # Locate best.pt
    save_dir = Path(getattr(results, "save_dir", run_dir))
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(best), str(OUT))
        print(f"COPIED best -> {OUT}")
    else:
        print(f"WARNING best.pt not found at {best}")


if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(epochs=ep)
