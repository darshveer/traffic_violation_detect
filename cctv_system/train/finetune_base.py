"""Shared fine-tuning logic for the helmet and seatbelt detectors.

Both detectors are YOLO11n models fine-tuned with the Ultralytics trainer,
starting from the COCO-pretrained ``yolo11n.pt`` for fast transfer learning.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Make the package root importable when run as a script.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pipelines.common import resolve_path, select_device, setup_logging  # noqa: E402

logger = setup_logging()


def _resolve_data_yaml(data_yaml: str) -> str:
    """Return a data.yaml whose ``path`` is an absolute directory.

    The committed dataset YAMLs use ``path: .`` for portability. Ultralytics
    needs an absolute root, so we rewrite ``path`` to the YAML's own directory
    into a temporary file and return its path.
    """
    src = resolve_path(data_yaml)
    assert src is not None
    if not src.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {src}")

    with open(src, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    declared = cfg.get("path", ".")
    root = (src.parent / declared).resolve() if not Path(declared).is_absolute() else Path(declared)
    cfg["path"] = str(root)

    tmp = Path(tempfile.gettempdir()) / f"_resolved_{src.parent.name}_{src.name}"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    logger.info("Resolved dataset root: %s", root)
    return str(tmp)


def finetune(
    task: str,
    train_cfg: Dict[str, Any],
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """Fine-tune a YOLO11n model for ``task`` ('helmet' or 'seatbelt').

    Parameters
    ----------
    task : str
        Either ``"helmet"`` or ``"seatbelt"``; selects the config section.
    train_cfg : dict
        Parsed ``train/config.yaml`` (contains ``common`` + per-task sections).
    epochs : int, optional
        Override the configured epoch count.
    batch : int, optional
        Override the configured batch size.
    device : str
        Device string (``auto``/``cuda``/``cpu``).

    Returns
    -------
    dict
        ``{"weights": <path>, "metrics": {precision, recall, f1, mAP50, mAP50_95}}``.

    Raises
    ------
    RuntimeError
        If Ultralytics is unavailable.
    KeyError
        If ``task`` has no config section.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required for fine-tuning") from exc

    common = train_cfg.get("common", {})
    task_cfg = train_cfg[task]
    aug = common.get("augment", {})
    dev = select_device(device)

    data_yaml = _resolve_data_yaml(task_cfg["data"])
    base_weights = str(resolve_path(common.get("base_weights", "models/yolo11n.pt")))
    n_epochs = int(epochs if epochs is not None else task_cfg.get("epochs", 10))
    n_batch = int(batch if batch is not None else task_cfg.get("batch", 16))

    logger.info(
        "Fine-tuning %s: base=%s epochs=%d batch=%d device=%s",
        task, base_weights, n_epochs, n_batch, dev,
    )

    model = YOLO(base_weights)
    results = model.train(
        data=data_yaml,
        epochs=n_epochs,
        batch=n_batch,
        imgsz=int(common.get("imgsz", 640)),
        device=dev,
        patience=int(common.get("patience", 3)),
        optimizer=common.get("optimizer", "SGD"),
        lr0=float(common.get("lr0", 0.001)),
        lrf=float(common.get("lrf", 0.01)),
        momentum=float(common.get("momentum", 0.937)),
        weight_decay=float(common.get("weight_decay", 0.0005)),
        warmup_epochs=float(common.get("warmup_epochs", 1.0)),
        cos_lr=bool(common.get("cos_lr", True)),
        seed=int(common.get("seed", 42)),
        workers=int(common.get("workers", 4)),
        project=task_cfg.get("project", f"runs/{task}"),
        name=task_cfg.get("name", "finetune"),
        exist_ok=True,
        # augmentation
        fliplr=float(aug.get("fliplr", 0.5)),
        flipud=float(aug.get("flipud", 0.0)),
        degrees=float(aug.get("degrees", 0.0)),
        translate=float(aug.get("translate", 0.1)),
        scale=float(aug.get("scale", 0.5)),
        hsv_h=float(aug.get("hsv_h", 0.015)),
        hsv_s=float(aug.get("hsv_s", 0.7)),
        hsv_v=float(aug.get("hsv_v", 0.4)),
        mosaic=float(aug.get("mosaic", 1.0)),
        mixup=float(aug.get("mixup", 0.0)),
    )

    # Locate best weights and copy to the configured output path.
    best = Path(results.save_dir) / "weights" / "best.pt"
    out_weights = resolve_path(task_cfg["out_weights"])
    assert out_weights is not None
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy(best, out_weights)
        logger.info("Saved fine-tuned %s model -> %s", task, out_weights)
    else:
        logger.warning("best.pt not found at %s; check training run.", best)

    metrics = _summarise_metrics(results)
    logger.info("%s fine-tune metrics: %s", task, metrics)
    return {"weights": str(out_weights), "metrics": metrics, "save_dir": str(results.save_dir)}


def _summarise_metrics(results: Any) -> Dict[str, float]:
    """Extract precision/recall/F1/mAP from an Ultralytics results object."""
    try:
        box = results.box  # results_dict also available via results.results_dict
        p, r = float(box.mp), float(box.mr)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
        }
    except Exception as exc:  # noqa: BLE001 - metrics are best-effort
        logger.warning("Could not summarise metrics: %s", exc)
        return {}
