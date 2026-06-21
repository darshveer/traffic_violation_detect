#!/usr/bin/env python3
"""Download a triple-riding dataset from Roboflow Universe for training."""
from __future__ import annotations
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
OUT = PKG / "datasets" / "triple_rider_data"
API_KEY = "qmB9vC2MU6LTHe1JwBpp"

# (workspace, project, version, format)
CANDIDATES = [
    ("pran", "triple-riding-detection-kbulg", 1, "yolov8"),
]


def main():
    from roboflow import Roboflow
    rf = Roboflow(api_key=API_KEY)
    for ws, proj, ver, fmt in CANDIDATES:
        try:
            print(f"=== {ws}/{proj} v{ver} ({fmt}) ===")
            project = rf.workspace(ws).project(proj)
            version = project.version(ver)
            ds = version.download(fmt, location=str(OUT), overwrite=True)
            print("downloaded to:", ds.location if hasattr(ds, "location") else OUT)
            return
        except Exception as exc:
            print(f"FAILED {ws}/{proj}: {type(exc).__name__}: {exc}")
    print("all candidates failed")


if __name__ == "__main__":
    main()
