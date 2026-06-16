"""Fine-tune the helmet detector (helmet_absent / helmet_present) on custom data.

Starts from ``yolo11n.pt`` and fine-tunes on ``datasets/helmet_data`` using the
hyperparameters in ``train/config.yaml``. Saves the best weights to
``models/helmet/helmet_finetuned.pt``.

Usage
-----
    python train/finetune_helmet.py --epochs 10 --batch_size 16
    python train/finetune_helmet.py --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pipelines.common import load_yaml, setup_logging  # noqa: E402

from finetune_base import finetune  # noqa: E402  (same-dir script import)

logger = setup_logging()


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI args and fine-tune the helmet model."""
    parser = argparse.ArgumentParser(description="Fine-tune the helmet detector.")
    parser.add_argument("--config", default="train/config.yaml", help="Training config YAML.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    args = parser.parse_args(argv)

    train_cfg = load_yaml(args.config)
    result = finetune(
        task="helmet",
        train_cfg=train_cfg,
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,
    )
    logger.info("Helmet fine-tuning complete: %s", result["weights"])


if __name__ == "__main__":
    main()
