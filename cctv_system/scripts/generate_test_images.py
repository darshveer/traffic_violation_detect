#!/usr/bin/env python3
"""Generate synthetic test images and a short test video for pipeline validation.

Creates simple CCTV-scene images using OpenCV (colored rectangles simulating
vehicles, persons, traffic lights), writes matching YOLO-format label files,
and assembles them into a short MP4 video.

All output goes into ``datasets/test_data/``.

Usage
-----
    python scripts/generate_test_images.py
    python scripts/generate_test_images.py --n_images 50 --video_frames 100
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

OUT_DIR = PACKAGE_ROOT / "datasets" / "test_data"
IMG_DIR = OUT_DIR / "images" / "test"
LBL_DIR = OUT_DIR / "labels" / "test"

# YOLO class indices for the test dataset (matches general traffic classes).
# We use COCO-compatible IDs so the base detector can score them:
# 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck, 9=traffic_light
CLASS_NAMES = {
    "person": 0,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "traffic_light": 9,
}
CLASS_COLORS = {
    "person": (180, 100, 80),       # blue-ish
    "car": (60, 160, 240),           # orange
    "motorcycle": (120, 200, 120),   # green
    "bus": (200, 180, 60),           # teal
    "truck": (80, 80, 200),          # red
    "traffic_light": (0, 220, 220),  # yellow
}

W, H = 1280, 720    # synthetic frame size


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rand_box(
    cls: str, rng: random.Random
) -> Tuple[int, int, int, int]:
    """Return a random (x1,y1,x2,y2) box appropriate for the given class."""
    if cls == "person":
        w = rng.randint(30, 60)
        h = rng.randint(80, 140)
    elif cls in ("car", "truck", "bus"):
        w = rng.randint(80, 200)
        h = rng.randint(50, 120)
    elif cls == "motorcycle":
        w = rng.randint(50, 90)
        h = rng.randint(60, 100)
    else:  # traffic_light
        w = rng.randint(15, 30)
        h = rng.randint(40, 70)
    x1 = rng.randint(0, max(1, W - w - 1))
    y1 = rng.randint(0, max(1, H - h - 1))
    return x1, y1, x1 + w, y1 + h


def _to_yolo(x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float, float, float]:
    """Convert pixel box to normalised YOLO format (cx, cy, bw, bh)."""
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return cx, cy, bw, bh


def _draw_object(
    img: np.ndarray,
    cls: str,
    x1: int, y1: int, x2: int, y2: int,
) -> None:
    """Draw a simple colored rectangle + label onto the frame."""
    import cv2
    color = CLASS_COLORS.get(cls, (200, 200, 200))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), 1)
    cv2.putText(img, cls[:3], (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def _road_background(rng: random.Random) -> np.ndarray:
    """Create a simple road-scene background."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Sky
    img[:H // 3, :] = [rng.randint(140, 200), rng.randint(160, 220), rng.randint(80, 130)]
    # Road
    img[H // 3:, :] = [rng.randint(50, 80), rng.randint(50, 80), rng.randint(50, 80)]
    # Lane markings
    import cv2
    y_start = H // 3
    for x in range(0, W, 100):
        cv2.line(img, (x, y_start + (H - y_start) // 2), (x + 50, y_start + (H - y_start) // 2),
                 (200, 200, 80), 3)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Image generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_images(n: int, seed: int = 42) -> List[Path]:
    """Generate ``n`` synthetic images and their YOLO label files.

    Parameters
    ----------
    n : int
        Number of images to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[Path]
        Paths to the generated image files.
    """
    import cv2

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    paths: List[Path] = []
    classes = list(CLASS_NAMES.keys())

    for i in range(n):
        img = _road_background(rng)
        label_lines: List[str] = []
        n_objects = rng.randint(2, 6)

        for _ in range(n_objects):
            cls = rng.choice(classes)
            x1, y1, x2, y2 = _rand_box(cls, rng)
            _draw_object(img, cls, x1, y1, x2, y2)
            cx, cy, bw, bh = _to_yolo(x1, y1, x2, y2)
            label_lines.append(f"{CLASS_NAMES[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        img_path = IMG_DIR / f"synth_{i + 1:04d}.jpg"
        lbl_path = LBL_DIR / f"synth_{i + 1:04d}.txt"
        cv2.imwrite(str(img_path), img)
        lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        paths.append(img_path)

    print(f"Generated {n} synthetic images -> {IMG_DIR}")
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Video generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_video(n_frames: int = 150, fps: float = 15.0, seed: int = 99) -> Path:
    """Generate a short synthetic traffic video for pipeline testing.

    Parameters
    ----------
    n_frames : int
        Number of frames in the video.
    fps : float
        Frame rate.
    seed : int
        Random seed.

    Returns
    -------
    Path
        Path to the generated video file.
    """
    import cv2

    out_path = OUT_DIR / "sample_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    rng = random.Random(seed)
    classes = list(CLASS_NAMES.keys())

    # Animate a few objects across frames.
    n_objects = 5
    objects = []
    for _ in range(n_objects):
        cls = rng.choice(classes)
        x1, y1, x2, y2 = _rand_box(cls, rng)
        vx = rng.uniform(-3, 3)
        vy = rng.uniform(-1, 2)
        objects.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "vx": vx, "vy": vy})

    for _ in range(n_frames):
        img = _road_background(rng)
        for obj in objects:
            x1, y1, x2, y2 = int(obj["x1"]), int(obj["y1"]), int(obj["x2"]), int(obj["y2"])
            # Bounce off edges.
            if x1 <= 0 or x2 >= W:
                obj["vx"] *= -1
            if y1 <= H // 3 or y2 >= H:
                obj["vy"] *= -1
            obj["x1"] = max(0, obj["x1"] + obj["vx"])
            obj["y1"] = max(H // 3, obj["y1"] + obj["vy"])
            obj["x2"] = min(W, obj["x2"] + obj["vx"])
            obj["y2"] = min(H, obj["y2"] + obj["vy"])
            _draw_object(img, obj["cls"], x1, y1, x2, y2)

        writer.write(img)

    writer.release()
    print(f"Generated test video ({n_frames} frames @ {fps} FPS) -> {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# data.yaml for test_data
# ──────────────────────────────────────────────────────────────────────────────

def write_test_data_yaml() -> None:
    """Write datasets/test_data/data.yaml for use with evaluation mode."""
    yaml_path = OUT_DIR / "data.yaml"
    content = (
        "# Synthetic test dataset for pipeline validation.\n"
        "# Classes match the COCO IDs emitted by the base YOLO11n detector.\n"
        "path: .\n"
        "train: images/test\n"
        "val:   images/test\n"
        "test:  images/test\n"
        "\n"
        "nc: 6\n"
        "names:\n"
        "  0: person\n"
        "  2: car\n"
        "  3: motorcycle\n"
        "  5: bus\n"
        "  7: truck\n"
        "  9: traffic_light\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"Wrote {yaml_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic test images and video for the CCTV pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_images", type=int, default=30, help="Number of synthetic images.")
    parser.add_argument("--video_frames", type=int, default=150, help="Number of video frames.")
    parser.add_argument("--fps", type=float, default=15.0, help="Video FPS.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_video", action="store_true", help="Skip video generation.")
    args = parser.parse_args(argv)

    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: Required library missing: {exc}")
        print("Install: pip install opencv-python numpy")
        sys.exit(1)

    print(f"Generating {args.n_images} synthetic test images …")
    generate_images(args.n_images, seed=args.seed)

    if not args.no_video:
        print(f"Generating synthetic test video ({args.video_frames} frames) …")
        generate_video(args.video_frames, args.fps, seed=args.seed + 1)

    write_test_data_yaml()
    print("Done.")


if __name__ == "__main__":
    main()
