# CCTV Traffic Violation Detection System

A production-grade pipeline that detects traffic violations from CCTV footage by
chaining an **Ultralytics YOLO11n** base detector with helmet/seatbelt
classifiers, self-contained logic for triple-riding / red-light / wrong-side
driving, and **PaddleOCR** license-plate reading — then produces court-ready
evidence (annotated video/frames, CSV/JSON reports, and a violation heatmap).

> **Design note.** Instead of training everything from scratch, this system uses
> a COCO-pretrained YOLO11n base and only *fine-tunes* the helmet and seatbelt
> detectors on small custom datasets. The remaining violation types are derived
> with self-contained logic on top of the base detections, and each can be
> upgraded with an external pretrained plug-in model when available.

---

## Violations detected

| Violation | How it's produced | Needs custom training? |
|---|---|---|
| `helmet_absent` / `helmet_present` | Fine-tuned YOLO11n on rider crops | ✅ fine-tune |
| `seatbelt_absent` / `seatbelt_present` | Fine-tuned YOLO11n on car crops | ✅ fine-tune |
| `triple_rider` | ≥3 persons on one motorcycle (base detector) | ❌ built-in |
| `red_light_violation` | Red light (HSV) + vehicle crossing stop-line ROI | ❌ built-in |
| `wrong_side_driving` | SORT tracking + motion vs. allowed direction | ❌ built-in |
| `license_plate` | PaddleOCR on vehicle crops | ❌ pretrained |

Each built-in detector has an **optional plug-in slot** (`triple_rider`,
`red_light`, `wrong_side` in `configs/pipeline.yaml`): drop in a `.pt` model and
it overrides the built-in logic; leave it `null` and the built-in logic runs.

---

## Requirements

- **OS / GPU:** built and tuned for **Windows + NVIDIA RTX 3060 (8GB VRAM, CUDA 12.x)**. Runs on CPU too (slower).
- **Python:** **3.11 or 3.12 recommended.** (3.13/3.14 not recommended — torch/paddle wheels lag.)

### Install

```bash
# 1) Create an environment (Python 3.11/3.12)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

# 2) Install the CUDA build of PyTorch FIRST (RTX 3060 -> cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) Install the rest
pip install -r requirements.txt

# 4) Sanity check
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

The base `yolo11n.pt` weights download automatically on first run.
For CUDA-accelerated OCR, replace `paddlepaddle` with `paddlepaddle-gpu`.

---

## Project layout

```
cctv_system/
├── configs/pipeline.yaml          # inference config: model paths, thresholds, ROIs
├── pipelines/
│   ├── common.py                  # paths, config loading, logging, device select
│   ├── model_loader.py            # ModelLoader: load/cache 6 models, CPU fallback
│   ├── violation_detector.py      # ViolationDetector: the unified pipeline
│   ├── inference_utils.py         # crops, batching, frame iteration, checkpoints
│   └── tracking.py                # SORT tracker (Kalman + Hungarian)
├── evidence/
│   ├── annotate.py                # draw boxes/labels/confidence/timestamp
│   ├── report_generator.py        # CSV + JSON + heatmap + summary + eval metrics
│   └── ocr_handler.py             # PaddleOCR (EasyOCR fallback) plate reader
├── train/
│   ├── config.yaml                # fine-tuning hyperparameters
│   ├── finetune_base.py           # shared fine-tune logic
│   ├── finetune_helmet.py         # helmet fine-tune entry point
│   └── finetune_seatbelt.py       # seatbelt fine-tune entry point
├── datasets/
│   ├── helmet_data/   (images/, labels/, data.yaml)  # 2 classes
│   ├── seatbelt_data/ (images/, labels/, data.yaml)  # 2 classes
│   └── test_data/
├── tests/                         # pytest unit tests
├── main.py                        # CLI: train | infer | eval
└── requirements.txt
```

---

## Usage

Run all commands from the `cctv_system/` directory.

### 1. Fine-tune helmet & seatbelt (custom data required)

```bash
python main.py --mode train --epochs 10 --batch_size 16 --device cuda:0
# or individually:
python train/finetune_helmet.py   --epochs 10 --batch_size 16
python train/finetune_seatbelt.py --epochs 10 --batch_size 16
```

Best weights are saved to `models/helmet/helmet_finetuned.pt` and
`models/seatbelt/seatbelt_finetuned.pt`.

### 2. Inference

```bash
# Single image
python main.py --mode infer --image sample.jpg --output results/

# Video (process every 5th frame; resumes from checkpoint if interrupted)
python main.py --mode infer --video sample.mp4 --output results/ --skip_frames 5
```

Outputs in `results/`: `annotated_video.mp4`, `annotated_frames/`,
`violations_report.csv`, `violations_metadata.json`, `summary_stats.json`,
`heatmap.html`.

### 3. Evaluate a fine-tuned detector

```bash
python main.py --mode eval --test_data datasets/helmet_data/data.yaml --output results/metrics/
```

Produces `metrics.json` (precision/recall/F1/mAP), `confusion_matrix.png`,
`precision_recall_curve.png`.

### 4. Regenerate reports from existing metadata

```bash
python evidence/report_generator.py --metadata results/violations_metadata.json --output results/
```

---

## Dataset format (YOLO)

```
datasets/helmet_data/
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt   # "class x_center y_center w h" (normalised)
└── data.yaml                       # path/train/val/test + nc=2 + names
```

The committed `data.yaml` uses `path: .` (its own directory); the fine-tune
scripts resolve it to an absolute path at runtime.

---

## Configuration

Tune `configs/pipeline.yaml` for your site:

- **`models`** — weight paths; set plug-in slots to `null` to use built-in logic.
- **`thresholds`** — `violation_conf` (default `0.6`) gates what gets reported.
- **`red_light.stop_line`** — normalised `[x1,y1,x2,y2]` stop-line band.
- **`wrong_side.allowed_direction`** — allowed travel vector (`+y` = downward).
- **`video.skip_frames`** — frame-skip factor for speed.

Fine-tuning hyperparameters live in `train/config.yaml`.

---

## Testing

```bash
pytest tests/          # logic + shape tests (no GPU required)
```

---

## Model attribution

This system integrates open-source models and ideas. Credit to:

| Component | Source |
|---|---|
| YOLO11 base detector & training | **Ultralytics** — https://github.com/ultralytics/ultralytics |
| Helmet detection (concept/data) | **haiderzm/Helmet-Detection** |
| Seatbelt detection (concept/data) | **sankethsj/seatbelt-detection** |
| Triple-rider detection (concept/data) | **kashishparmar02** |
| Red-light violation (concept) | **MohammedHamza0/Traffic-Signal-Violation** |
| Wrong-side driving (concept) | **sriramcu** (YOLOv4 + Kalman) |
| License-plate OCR | **PaddleOCR (PaddlePaddle)** — https://github.com/PaddlePaddle/PaddleOCR |
| SORT tracking algorithm | Bewley et al., *Simple Online and Realtime Tracking* |

> The helmet/seatbelt detectors here are fine-tuned from `yolo11n.pt` rather than
> the original YOLOv5 weights, for framework consistency. The other repositories
> are credited for their concepts/datasets and are supported as optional plug-in
> slots. Using and crediting open-source components this way is standard practice.

---

## Performance (reference, RTX 3060)

- mAP ≥ 0.70 on helmet/seatbelt after fine-tuning (data-dependent).
- ~15–30 FPS on GPU, ~2–5 FPS on CPU.
- < 4 GB VRAM at `batch=16`, `imgsz=640`.

## Limitations

- `wrong_side_driving` requires sequential frames (video), not single images.
- `red_light` uses an HSV heuristic + a configured stop-line ROI; tune per camera.
- License-plate OCR accuracy depends on resolution and plate visibility.
