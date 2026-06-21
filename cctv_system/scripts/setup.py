#!/usr/bin/env python3
"""One-command environment setup for the CCTV violation pipeline.

Creates required directories, validates the Python environment,
pre-caches the base YOLO model, and generates synthetic test data
so the full pipeline can be exercised immediately without real camera footage.

Usage
-----
    python scripts/setup.py              # full setup
    python scripts/setup.py --no_synth  # skip synthetic data generation
    python scripts/setup.py --check     # environment check only
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# Ensure package root is importable when run as a script.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("setup")

# Directories that must exist.
REQUIRED_DIRS = [
    "models/helmet",
    "models/seatbelt",
    "models/plugins",
    "logs",
    "results",
    "datasets/helmet_data/images/train",
    "datasets/helmet_data/images/val",
    "datasets/helmet_data/images/test",
    "datasets/helmet_data/labels/train",
    "datasets/helmet_data/labels/val",
    "datasets/helmet_data/labels/test",
    "datasets/seatbelt_data/images/train",
    "datasets/seatbelt_data/images/val",
    "datasets/seatbelt_data/images/test",
    "datasets/seatbelt_data/labels/train",
    "datasets/seatbelt_data/labels/val",
    "datasets/seatbelt_data/labels/test",
    "datasets/test_data/images/test",
    "datasets/test_data/labels/test",
]


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — create directories
# ──────────────────────────────────────────────────────────────────────────────

def create_dirs() -> None:
    """Create all required directories."""
    logger.info("── Step 1: Creating directory structure ──────────────────────")
    for rel in REQUIRED_DIRS:
        d = PACKAGE_ROOT / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").touch(exist_ok=True)
    logger.info("  ✓ All directories ready.")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — validate Python environment
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("PIL", "Pillow"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
]

_OPTIONAL_PACKAGES = [
    ("paddleocr", "paddleocr"),
    ("easyocr", "easyocr"),
    ("folium", "folium"),
    ("gdown", "gdown"),
    ("roboflow", "roboflow"),
]


def check_environment() -> bool:
    """Check required and optional packages. Return True if all required are present."""
    logger.info("── Step 2: Environment check ─────────────────────────────────")
    missing_required: List[str] = []

    for module, pkg in _REQUIRED_PACKAGES:
        try:
            __import__(module)
            logger.info("  ✓  %-15s OK", pkg)
        except ImportError:
            logger.warning("  ✗  %-15s MISSING  (pip install %s)", pkg, pkg)
            missing_required.append(pkg)

    logger.info("")
    for module, pkg in _OPTIONAL_PACKAGES:
        try:
            __import__(module)
            logger.info("  ○  %-15s OK (optional)", pkg)
        except ImportError:
            logger.info("  -  %-15s not installed (optional — see README)", pkg)

    if missing_required:
        logger.error("  Required packages missing: %s", missing_required)
        logger.error("  Run: pip install -r requirements.txt")
        return False

    # Check CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        logger.info("")
        logger.info("  CUDA available: %s", cuda)
        if cuda:
            logger.info("  GPU: %s", torch.cuda.get_device_name(0))
    except Exception:
        pass

    logger.info("  ✓ Environment OK.")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — pre-cache base model
# ──────────────────────────────────────────────────────────────────────────────

def precache_base_model() -> None:
    """Download yolo11n.pt from Ultralytics if not already cached."""
    logger.info("── Step 3: Base model (yolo11n.pt) ──────────────────────────")
    dest = PACKAGE_ROOT / "models" / "yolo11n.pt"
    if dest.exists():
        logger.info("  Already present: %s — skipping.", dest)
        return
    try:
        from ultralytics import YOLO
        logger.info("  Downloading yolo11n.pt from Ultralytics …")
        model = YOLO("yolo11n.pt")   # triggers auto-download to Ultralytics cache
        # Try to find the cached file and copy it.
        candidates = [
            Path("yolo11n.pt"),
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 1_000_000:
                shutil.copy(str(c), str(dest))
                logger.info("  ✓ Saved to %s", dest)
                c.unlink(missing_ok=True)
                return
        logger.info("  ✓ yolo11n.pt ready in Ultralytics cache (will auto-download on first run).")
    except Exception as exc:  # noqa: BLE001
        logger.warning("  Could not pre-cache yolo11n.pt: %s", exc)
        logger.warning("  It will auto-download on first inference run.")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — generate synthetic test data
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data() -> None:
    """Delegate to generate_test_images.py."""
    logger.info("── Step 4: Generating synthetic test data ────────────────────")
    gen_script = PACKAGE_ROOT / "scripts" / "generate_test_images.py"
    if not gen_script.exists():
        logger.warning("  generate_test_images.py not found — skipping.")
        return
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(gen_script)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("  ✓ Synthetic test data generated.")
            for line in result.stdout.strip().splitlines():
                if line.strip():
                    logger.info("    %s", line)
        else:
            logger.warning("  Synthetic data generation had issues:")
            for line in (result.stderr or result.stdout or "").strip().splitlines():
                logger.warning("    %s", line)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  Could not generate synthetic data: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """Run full project setup."""
    parser = argparse.ArgumentParser(
        description="One-command setup for the CCTV violation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only check the environment, do not create dirs or download.",
    )
    parser.add_argument(
        "--no_synth", action="store_true",
        help="Skip synthetic test data generation.",
    )
    parser.add_argument(
        "--no_model", action="store_true",
        help="Skip pre-caching the base YOLO model.",
    )
    args = parser.parse_args(argv)

    logger.info("=" * 62)
    logger.info(" CCTV Traffic Violation Detection — Project Setup")
    logger.info(" Package root: %s", PACKAGE_ROOT)
    logger.info("=" * 62)
    logger.info("")

    if args.check:
        check_environment()
        return

    create_dirs()
    logger.info("")
    env_ok = check_environment()
    logger.info("")

    if not env_ok:
        logger.error("Fix missing packages before continuing.")
        sys.exit(1)

    if not args.no_model:
        precache_base_model()
        logger.info("")

    if not args.no_synth:
        generate_synthetic_data()
        logger.info("")

    logger.info("=" * 62)
    logger.info(" Setup complete!  Next steps:")
    logger.info("")
    logger.info("  1. Download plugin weights (optional but recommended):")
    logger.info("       python scripts/download_plugins.py")
    logger.info("")
    logger.info("  2. Download training datasets:")
    logger.info("       See DATASETS.md for step-by-step instructions.")
    logger.info("")
    logger.info("  3. Fine-tune helmet & seatbelt detectors:")
    logger.info("       python main.py --mode train --epochs 10")
    logger.info("")
    logger.info("  4. Run inference on a sample image:")
    logger.info("       python main.py --mode infer \\")
    logger.info("           --image datasets/test_data/images/test/synth_001.jpg \\")
    logger.info("           --output results/")
    logger.info("")
    logger.info("  5. Run the unit tests:")
    logger.info("       pytest tests/ -v")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
