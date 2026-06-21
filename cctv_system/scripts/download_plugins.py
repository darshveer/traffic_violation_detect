#!/usr/bin/env python3
"""Download pretrained plugin model weights for the CCTV violation pipeline.

This script fetches the following model weights into ``models/plugins/``:

=================  ====================================  ==================
Plugin             Source                                Status
=================  ====================================  ==================
helmet (base)      haiderzm/Helmet-Detection (GDrive)    auto (needs gdown)
red_light          MohammedHamza0 → yolov8m.pt (COCO)   auto (Ultralytics)
triple_rider       kashishparmar02 → Roboflow            manual (API key)
seatbelt           sankethsj → no public weights         N/A (fine-tune)
wrong_side         sriramcu → YOLOv4 Darknet             N/A (Linux only)
=================  ====================================  ==================

Usage
-----
    # Download all available plugins automatically:
    python scripts/download_plugins.py

    # Download only specific plugins:
    python scripts/download_plugins.py --plugins helmet red_light

    # With a Roboflow API key (enables triple_rider download):
    python scripts/download_plugins.py --roboflow_key YOUR_KEY_HERE

Attribution
-----------
- haiderzm/Helmet-Detection: https://github.com/haiderzm/Helmet-Detection
- MohammedHamza0/Traffic-Signal-Violation: https://github.com/MohammedHamza0/traffic-signal-violation-detection
- kashishparmar02/triple-rider-detection: https://github.com/kashishparmar02/triple-rider-detection
- sriramcu/yolov4_wrong_side_driving: https://github.com/sriramcu/yolov4_wrong_side_driving_detection
  (YOLOv4 Darknet — Linux-only; not downloaded on Windows)
- sankethsj/seatbelt-detection: https://github.com/sankethsj/seatbelt-detection
  (No public weight file — use fine-tuning instead; see DATASETS.md)
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
logger = logging.getLogger("download_plugins")

PLUGINS_DIR = PACKAGE_ROOT / "models" / "plugins"

# ──────────────────────────────────────────────────────────────────────────────
# Google Drive IDs / download URLs
# ──────────────────────────────────────────────────────────────────────────────
# haiderzm/Helmet-Detection — YOLOv5 best.pt
# Classes: 0=helmet  1=no-helmet (nc=2)
HELMET_GDRIVE_ID = "1M5MAgENH1y7DgzISWVKkx_QklMGfB3-l"
HELMET_OUT = PLUGINS_DIR / "helmet_haiderzm.pt"

# MohammedHamza0 uses standard yolov8m.pt (80-class COCO) — auto-downloads
REDLIGHT_ULTRALYTICS_NAME = "yolov8m.pt"
REDLIGHT_OUT = PLUGINS_DIR / "red_light_yolov8m.pt"


# ──────────────────────────────────────────────────────────────────────────────
# Individual download functions
# ──────────────────────────────────────────────────────────────────────────────

def download_helmet() -> Optional[Path]:
    """Download haiderzm/Helmet-Detection best.pt via gdown from Google Drive.

    Returns
    -------
    Path or None
        Path to the downloaded weight file, or ``None`` on failure.
    """
    logger.info("=== Helmet Detection (haiderzm/Helmet-Detection) ===")
    if HELMET_OUT.exists():
        logger.info("  Already exists: %s — skipping.", HELMET_OUT)
        return HELMET_OUT

    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import gdown  # type: ignore[import]
    except ImportError:
        logger.warning("  'gdown' not installed.  Run: pip install gdown")
        logger.warning("  Then re-run this script, or download manually:")
        logger.warning("  URL: https://drive.google.com/uc?id=%s", HELMET_GDRIVE_ID)
        logger.warning("  Save as: %s", HELMET_OUT)
        return None

    url = f"https://drive.google.com/uc?id={HELMET_GDRIVE_ID}"
    logger.info("  Downloading from Google Drive: %s", url)
    try:
        gdown.download(url, str(HELMET_OUT), quiet=False)
        if HELMET_OUT.exists() and HELMET_OUT.stat().st_size > 1_000_000:
            logger.info("  ✓ Saved: %s (%.1f MB)", HELMET_OUT, HELMET_OUT.stat().st_size / 1e6)
            return HELMET_OUT
        else:
            logger.error("  Download appears incomplete or failed.")
            HELMET_OUT.unlink(missing_ok=True)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.error("  Download failed: %s", exc)
        HELMET_OUT.unlink(missing_ok=True)
        return None


def download_red_light() -> Optional[Path]:
    """Download yolov8m.pt (standard COCO) for the red-light plugin slot.

    MohammedHamza0/Traffic-Signal-Violation uses the standard Ultralytics
    yolov8m model (80-class COCO). Vehicle + traffic-light classes are a
    subset — no custom training required. Ultralytics downloads it automatically
    on first use; this function just pre-caches it to models/plugins/.

    Returns
    -------
    Path or None
        Path to the downloaded weight file, or ``None`` on failure.
    """
    logger.info("=== Red-Light Plugin (yolov8m.pt — MohammedHamza0 approach) ===")
    if REDLIGHT_OUT.exists():
        logger.info("  Already exists: %s — skipping.", REDLIGHT_OUT)
        return REDLIGHT_OUT

    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError:
        logger.warning("  ultralytics not installed. Run: pip install ultralytics")
        return None

    logger.info("  Downloading yolov8m.pt from Ultralytics CDN …")
    try:
        model = YOLO(REDLIGHT_ULTRALYTICS_NAME)          # triggers auto-download
        # Ultralytics saves weights to the CWD by default; find & move it.
        downloaded = Path(REDLIGHT_ULTRALYTICS_NAME)
        if downloaded.exists():
            shutil.move(str(downloaded), str(REDLIGHT_OUT))
        else:
            # Already in Ultralytics cache — copy from there.
            import torch
            hub_dir = Path(torch.hub.get_dir()) / "ultralytics" / "assets"
            src = hub_dir / REDLIGHT_ULTRALYTICS_NAME
            if src.exists():
                shutil.copy(str(src), str(REDLIGHT_OUT))
            else:
                # Last resort: save using ultralytics export path.
                model.save(str(REDLIGHT_OUT))

        if REDLIGHT_OUT.exists() and REDLIGHT_OUT.stat().st_size > 1_000_000:
            logger.info("  ✓ Saved: %s (%.1f MB)", REDLIGHT_OUT, REDLIGHT_OUT.stat().st_size / 1e6)
            return REDLIGHT_OUT
        else:
            logger.error("  Could not locate downloaded yolov8m.pt.")
            return None
    except Exception as exc:  # noqa: BLE001
        logger.error("  Download failed: %s", exc)
        return None


def download_triple_rider(api_key: Optional[str]) -> Optional[Path]:
    """Download kashishparmar02/triple-rider-detection via Roboflow API.

    Parameters
    ----------
    api_key : str or None
        Roboflow API key. If ``None``, prints manual instructions and returns.

    Returns
    -------
    Path or None
        Path to the downloaded weight file, or ``None`` on failure.
    """
    out = PLUGINS_DIR / "triple_rider_kashish.pt"
    logger.info("=== Triple Rider (kashishparmar02/triple-rider-detection) ===")

    if out.exists():
        logger.info("  Already exists: %s — skipping.", out)
        return out

    if not api_key:
        logger.warning("  Roboflow API key not provided — cannot auto-download.")
        _print_triple_rider_manual()
        return None

    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow  # type: ignore[import]
    except ImportError:
        logger.warning("  'roboflow' package not installed. Run: pip install roboflow")
        _print_triple_rider_manual()
        return None

    try:
        logger.info("  Connecting to Roboflow workspace 'kashish', project '3riders' …")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("kashish").project("3riders")
        version = project.version(1)
        dataset = version.download("yolov8", location=str(PLUGINS_DIR / "3riders_data"))

        # The model weights are in the downloaded dataset's run folder,
        # or we can access via the model API.
        model_obj = version.model
        # Export / get the local pt path — Roboflow stores it in a cache folder.
        pt_candidates = list((PLUGINS_DIR / "3riders_data").rglob("*.pt"))
        if pt_candidates:
            best = sorted(pt_candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
            shutil.copy(str(best), str(out))
            logger.info("  ✓ Saved: %s", out)
            return out
        else:
            logger.warning("  .pt file not found in downloaded data; saved model metadata only.")
            logger.warning("  You may need to train the model via Roboflow or export manually.")
            _print_triple_rider_manual()
            return None
    except Exception as exc:  # noqa: BLE001
        logger.error("  Roboflow download failed: %s", exc)
        _print_triple_rider_manual()
        return None


def _print_triple_rider_manual() -> None:
    logger.info("")
    logger.info("  ── Manual steps for triple_rider weights ────────────────────")
    logger.info("  1. Visit https://universe.roboflow.com/kashish/3riders")
    logger.info("  2. Click 'Download Dataset' -> choose 'YOLOv8' format")
    logger.info("  3. To get the trained model weights (.pt), click 'Model'")
    logger.info("     then 'Deploy' -> 'Download Weights' (requires Roboflow account)")
    logger.info("  4. Save the downloaded .pt as:")
    logger.info("     models/plugins/triple_rider_kashish.pt")
    logger.info("  5. Then run: python scripts/update_pipeline_config.py")
    logger.info("  ─────────────────────────────────────────────────────────────")
    logger.info("")


def print_skipped_plugins() -> None:
    """Print explanations for plugins that cannot be auto-downloaded."""
    logger.info("")
    logger.info("=== Seatbelt Detection (sankethsj/seatbelt-detection) ===")
    logger.info("  ⚠  No public weight file exists for this repo.")
    logger.info("  The custom-trained best.pt is not hosted publicly.")
    logger.info("  -> Solution: Fine-tune our own seatbelt model.")
    logger.info("    See DATASETS.md for dataset download instructions.")
    logger.info("    Then run: python main.py --mode train --targets seatbelt")
    logger.info("")
    logger.info("=== Wrong-Side Driving (sriramcu/yolov4_wrong_side_driving) ===")
    logger.info("  ⚠  YOLOv4 Darknet weights — Linux-only, not usable on Windows.")
    logger.info("  The pipeline's built-in SORT-based wrong-side detector handles")
    logger.info("  this violation type without a separate model.")
    logger.info("  -> No action needed; built-in logic is active by default.")
    logger.info("")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline config update
# ──────────────────────────────────────────────────────────────────────────────

def update_pipeline_config(results: dict) -> None:
    """Update configs/pipeline.yaml to point to downloaded plugin weights.

    Parameters
    ----------
    results : dict
        Mapping of plugin name -> Path (or None if not downloaded).
    """
    import yaml

    cfg_path = PACKAGE_ROOT / "configs" / "pipeline.yaml"
    if not cfg_path.exists():
        logger.warning("pipeline.yaml not found at %s; skipping config update.", cfg_path)
        return

    with open(cfg_path, "r", encoding="utf-8") as fh:
        content = fh.read()
        cfg = yaml.safe_load(content)

    models_cfg = cfg.get("models", {})
    changed = False

    # helmet plugin (YOLOv5 from haiderzm — loaded via Ultralytics compat layer)
    if results.get("helmet") and results["helmet"].exists():
        rel = str(results["helmet"].relative_to(PACKAGE_ROOT))
        # Update the helmet_plugin slot (separate from fine-tuned helmet)
        models_cfg["helmet_plugin"] = rel.replace("\\", "/")
        logger.info("  config: helmet_plugin -> %s", rel)
        changed = True

    # red_light plugin (yolov8m COCO)
    if results.get("red_light") and results["red_light"].exists():
        rel = str(results["red_light"].relative_to(PACKAGE_ROOT))
        models_cfg["red_light"] = rel.replace("\\", "/")
        logger.info("  config: red_light -> %s", rel)
        changed = True

    # triple_rider plugin
    if results.get("triple_rider") and results["triple_rider"].exists():
        rel = str(results["triple_rider"].relative_to(PACKAGE_ROOT))
        models_cfg["triple_rider"] = rel.replace("\\", "/")
        logger.info("  config: triple_rider -> %s", rel)
        changed = True

    if changed:
        cfg["models"] = models_cfg
        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info("  ✓ Updated %s", cfg_path)
    else:
        logger.info("  No config changes needed.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """Download plugin weights and update pipeline.yaml."""
    parser = argparse.ArgumentParser(
        description="Download pretrained plugin model weights for the CCTV pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--plugins",
        nargs="*",
        choices=["helmet", "red_light", "triple_rider", "all"],
        default=["all"],
        help="Which plugins to download. Defaults to all auto-downloadable ones.",
    )
    parser.add_argument(
        "--roboflow_key",
        default=None,
        help="Roboflow API key (needed for triple_rider download).",
    )
    parser.add_argument(
        "--no_config_update",
        action="store_true",
        help="Skip updating configs/pipeline.yaml after download.",
    )
    args = parser.parse_args(argv)

    targets = set(args.plugins)
    if "all" in targets:
        targets = {"helmet", "red_light", "triple_rider"}

    logger.info("Download targets: %s", sorted(targets))
    logger.info("Output directory: %s", PLUGINS_DIR)
    logger.info("")

    results: dict = {}

    if "helmet" in targets:
        results["helmet"] = download_helmet()

    if "red_light" in targets:
        results["red_light"] = download_red_light()

    if "triple_rider" in targets:
        results["triple_rider"] = download_triple_rider(args.roboflow_key)

    print_skipped_plugins()

    # Summary
    logger.info("=" * 60)
    logger.info("Download summary:")
    for name, path in results.items():
        status = f"✓ {path}" if path else "✗ failed / skipped"
        logger.info("  %-15s %s", name, status)
    logger.info("")

    if not args.no_config_update:
        logger.info("Updating pipeline.yaml …")
        update_pipeline_config(results)

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. See DATASETS.md for dataset download instructions")
    logger.info("  2. python main.py --mode train --targets helmet seatbelt")
    logger.info("  3. python main.py --mode infer --image <path> --output results/")


if __name__ == "__main__":
    main()
