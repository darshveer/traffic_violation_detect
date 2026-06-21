"""CLI entry point for the CCTV Traffic Violation Detection system.

Modes
-----
``train``
    Fine-tune the helmet and/or seatbelt detectors.
``infer``
    Run the full violation pipeline on an image or a video.
``eval``
    Evaluate a fine-tuned detector on a test split (mAP / PR / confusion matrix).

Examples
--------
    python main.py --mode train --epochs 10 --batch_size 16
    python main.py --mode infer --image sample.jpg --output results/
    python main.py --mode infer --video sample.mp4 --output results/ --skip_frames 5
    python main.py --mode eval --test_data datasets/helmet_data/data.yaml --output results/metrics/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure the package root is importable however the script is launched.
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pipelines.common import load_yaml, setup_logging  # noqa: E402

logger = setup_logging(log_file="logs/cctv.log")


# --------------------------------------------------------------------------- #
# Mode handlers
# --------------------------------------------------------------------------- #
def run_train(args: argparse.Namespace) -> None:
    """Fine-tune helmet and/or seatbelt detectors."""
    sys.path.insert(0, str(PACKAGE_ROOT / "train"))
    from finetune_base import finetune  # type: ignore

    train_cfg = load_yaml(args.config)
    targets = args.targets or ["helmet", "seatbelt"]
    for task in targets:
        logger.info("=== Fine-tuning: %s ===", task)
        try:
            result = finetune(
                task=task,
                train_cfg=train_cfg,
                epochs=args.epochs,
                batch=args.batch_size,
                device=args.device,
            )
            logger.info("[%s] done -> %s (%s)", task, result["weights"], result["metrics"])
        except Exception as exc:  # noqa: BLE001 - report per-task and continue
            logger.error("[%s] fine-tuning failed: %s", task, exc)


def run_infer(args: argparse.Namespace) -> None:
    """Run inference on an image or a video and write evidence reports."""
    from evidence.report_generator import generate_reports
    from pipelines.violation_detector import ViolationDetector

    if not args.image and not args.video:
        raise SystemExit("infer mode requires --image or --video")

    detector = ViolationDetector(config_path=args.config, device=args.device)
    out_dir = args.output or "results/"

    if args.image:
        result = detector.infer_image(args.image)
        h, w = result["frame"].shape[:2]
        # Persist annotated image.
        import cv2

        from evidence.annotate import annotate_frame

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        annotated = annotate_frame(result["frame"], result["detections"], result["timestamp"], result["violations"])
        cv2.imwrite(str(Path(out_dir) / "annotated.jpg"), annotated)

        record = {k: result[k] for k in ("frame_id", "timestamp", "summary", "violations", "counts")}
        paths = generate_reports([record], out_dir, frame_size=(w, h))
        logger.info("Image inference complete. Violations: %s", record["counts"])
        logger.info("Reports: %s", paths)

        if args.show:
            cv2.imshow("CCTV Traffic Violation Detection", annotated)
            logger.info("Showing annotated image. Press any key in the popup window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if args.video:
        out = detector.infer_video(
            args.video, out_dir, skip_frames=args.skip_frames, resume=not args.no_resume, show=args.show
        )
        w, h = int(out["video"]["width"]), int(out["video"]["height"])
        paths = generate_reports(out["records"], out_dir, frame_size=(w, h))
        logger.info("Video inference complete (%d frames).", len(out["records"]))
        logger.info("Reports: %s", paths)


def run_eval(args: argparse.Namespace) -> None:
    """Evaluate a fine-tuned detector on a test split."""
    from evidence.report_generator import evaluate

    if not args.test_data:
        raise SystemExit("eval mode requires --test_data (a dataset data.yaml)")
    model_path = args.weights or _default_eval_weights(args)
    out_dir = args.output or "results/metrics/"
    metrics = evaluate(model_path, args.test_data, out_dir, device=args.device)
    logger.info("Evaluation metrics: %s", metrics)


def _default_eval_weights(args: argparse.Namespace) -> str:
    """Infer which fine-tuned model to evaluate from the dataset path."""
    cfg = load_yaml(args.config)
    models = cfg.get("models", {})
    td = (args.test_data or "").lower()
    if "seatbelt" in td:
        return models.get("seatbelt", "models/seatbelt/seatbelt_finetuned.pt")
    return models.get("helmet", "models/helmet/helmet_finetuned.pt")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        description="CCTV Traffic Violation Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=["train", "infer", "eval"])
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline config (infer/eval).")
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")

    # infer
    parser.add_argument("--image", help="Path to an input image (infer mode).")
    parser.add_argument("--video", help="Path to an input video (infer mode).")
    parser.add_argument("--output", help="Output directory.")
    parser.add_argument("--skip_frames", type=int, default=None, help="Process every Nth frame.")
    parser.add_argument("--no_resume", action="store_true", help="Ignore any video checkpoint.")
    parser.add_argument("--show", action="store_true", help="Show live popup window during inference.")

    # train
    parser.add_argument(
        "--targets", nargs="*", choices=["helmet", "seatbelt"],
        help="Which models to fine-tune (default: both).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Fine-tune epochs override.")
    parser.add_argument("--batch_size", type=int, default=None, help="Fine-tune batch size override.")

    # train config (separate from pipeline config)
    parser.add_argument(
        "--train_config", dest="train_config", default="train/config.yaml",
        help="Training hyperparameter YAML (train mode).",
    )

    # eval
    parser.add_argument("--test_data", help="Dataset data.yaml for evaluation (eval mode).")
    parser.add_argument("--weights", help="Model weights to evaluate (eval mode).")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Dispatch to the requested mode."""
    parser = build_parser()
    args = parser.parse_args(argv)
    # In train mode, --config refers to the training YAML.
    if args.mode == "train":
        args.config = args.train_config

    logger.info("Mode: %s | device: %s", args.mode, args.device)
    if args.mode == "train":
        run_train(args)
    elif args.mode == "infer":
        run_infer(args)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
