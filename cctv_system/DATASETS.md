# Dataset & Model Download Guide

This guide covers every dataset and model weight you need, with step-by-step
manual instructions for each source.

---

## Quick Reference

| What | Source | Size | Priority |
|---|---|---|---|
| **Helmet dataset** | Roboflow Universe | ~600–2000 images | 🔴 Required for training |
| **Seatbelt dataset** | Roboflow Universe | ~800–1000 images | 🔴 Required for training |
| **haiderzm helmet weights** | Google Drive | ~14 MB | 🟡 Optional (warm-start) |
| **yolov8m.pt (red-light plugin)** | Ultralytics CDN | ~52 MB | 🟡 Optional plugin |
| **triple_rider weights** | Roboflow (API) | ~6 MB | 🟡 Optional plugin |
| **yolo11n.pt (base model)** | Auto-download | ~5.5 MB | ✅ Auto on first run |

---

## 1. Helmet Detection Dataset

**Recommended source:** [Roboflow Universe — Helmet Detection](https://universe.roboflow.com)

### Step-by-step

1. Visit **https://universe.roboflow.com**
2. Search for: `helmet detection motorcycle`
3. Recommended datasets (click to open):
   - **"Helmet and no-helmet rider detection"** by `nckh-2023` (~1,600 images)
   - **"Helmet Detection"** by `hard-hat-sample` (~3,600 images)
4. Click **"Download Dataset"**
5. Choose format: **YOLOv8** (or YOLOv11 — same format)
6. Choose split: **train/valid/test** (70/20/10)
7. Click **"show download code"** → select **"Zip archive"**
8. Download the ZIP file

### After downloading

Extract the ZIP into `datasets/helmet_data/`:

```
datasets/helmet_data/
├── images/
│   ├── train/   ← copy all train images here
│   ├── val/     ← copy all valid images here
│   └── test/    ← copy all test images here
└── labels/
    ├── train/   ← copy all train labels here
    ├── val/     ← copy all valid labels here
    └── test/    ← copy all test labels here
```

> ⚠️ **Important**: The Roboflow ZIP uses `valid/` not `val/`. Either rename
> the folder or change `data.yaml`'s `val:` entry to `images/valid`.

> ⚠️ **Class names**: Roboflow datasets may use `With Helmet` / `Without Helmet`
> or similar. You must update `datasets/helmet_data/data.yaml` to match:
> ```yaml
> nc: 2
> names:
>   0: helmet_absent      # ← whatever Roboflow calls "no helmet"
>   1: helmet_present     # ← whatever Roboflow calls "with helmet"
> ```
> Check the downloaded `data.yaml` to find the correct class order.

---

## 2. Seatbelt Detection Dataset

**Recommended source:** [Roboflow Universe — Seatbelt Detection](https://universe.roboflow.com)

### Step-by-step

1. Visit **https://universe.roboflow.com**
2. Search for: `seatbelt detection`
3. Recommended datasets:
   - **"Seat-Belt Detection"** by `2tech` (~870 images, well-annotated)
   - **"Seat Belt Detection"** by `dti` (~1,000 images)
   - **"seatbelt-detection"** by `seatbelttraining` (widely cited)
4. Click **"Download Dataset"** → **YOLOv8** format → **Zip archive**
5. Download the ZIP

### After downloading

Extract into `datasets/seatbelt_data/` (same structure as helmet above).

Class name mapping for `datasets/seatbelt_data/data.yaml`:
```yaml
nc: 2
names:
  0: seatbelt_absent    # class for no seatbelt
  1: seatbelt_present   # class for seatbelt worn
```

---

## 3. haiderzm/Helmet-Detection Pre-trained Weights (Optional)

**Source:** [haiderzm/Helmet-Detection](https://github.com/haiderzm/Helmet-Detection) — Google Drive

Using these weights as the *starting point* for fine-tuning (instead of raw COCO)
gives a better initialization since the model already knows what helmets look like.

### Automated download

```bash
# Install gdown first
pip install gdown

# Then run:
python scripts/download_plugins.py --plugins helmet
```

### Manual download

1. Visit: `https://drive.google.com/file/d/1M5MAgENH1y7DgzISWVKkx_QklMGfB3-l/view`
2. Click **"Download"** (top-right)
3. Save as `models/plugins/helmet_haiderzm.pt`

### Using the downloaded weights for fine-tuning

After downloading, edit `train/config.yaml`:
```yaml
common:
  # Change this line:
  base_weights: models/yolo11n.pt
  # To:
  # base_weights: models/plugins/helmet_haiderzm.pt   # warm-start from haiderzm
```

> ℹ️ Note: The haiderzm weights are YOLOv5 format. Ultralytics is backward-compatible
> and can fine-tune from these weights via the YOLO() API.

---

## 4. Red-Light Plugin Weights — yolov8m.pt (Optional)

**Source:** Ultralytics CDN (MohammedHamza0 approach uses standard COCO weights)

This is the **standard yolov8m.pt** model trained on 80-class COCO — no custom
training needed. The red-light logic in the pipeline uses this for vehicle and
traffic-light detection.

### Automated download

```bash
python scripts/download_plugins.py --plugins red_light
```

### Manual download

```bash
# Option 1: Let Ultralytics auto-download on first run (happens automatically)
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Option 2: Direct URL
# https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt
# Save as: models/plugins/red_light_yolov8m.pt
```

After downloading, update `configs/pipeline.yaml`:
```yaml
models:
  red_light: models/plugins/red_light_yolov8m.pt
```

---

## 5. Triple-Rider Plugin Weights (Optional — requires Roboflow account)

**Source:** [kashishparmar02/triple-rider-detection](https://github.com/kashishparmar02/triple-rider-detection) — Roboflow

### Via Roboflow API (recommended if you have a free account)

```bash
# Install roboflow
pip install roboflow

# Download with your API key
python scripts/download_plugins.py --roboflow_key YOUR_API_KEY_HERE
```

Get your free API key at: https://app.roboflow.com → Settings → API Keys

### Manual download from Roboflow Universe

1. Visit: https://universe.roboflow.com/kashish/3riders
2. Click **"Download"** → choose **YOLOv8** format
3. Log in with a free Roboflow account if prompted
4. If model weights are available for download:
   - Click **"Model"** tab → **"Deploy"** → **"Download Weights"**
   - Save as `models/plugins/triple_rider_kashish.pt`
5. After saving, update `configs/pipeline.yaml`:
   ```yaml
   models:
     triple_rider: models/plugins/triple_rider_kashish.pt
   ```

> ℹ️ If model weights cannot be downloaded from Roboflow, the pipeline's
> built-in person-counting heuristic (≥3 persons on a motorcycle) handles
> this violation type automatically.

---

## 6. Wrong-Side Driving Plugin (Not available on Windows)

**Source:** [sriramcu/yolov4_wrong_side_driving_detection](https://github.com/sriramcu/yolov4_wrong_side_driving_detection)

This plugin uses **YOLOv4 Darknet** weights (`.weights` format) and requires
compiling Darknet from source — **Linux only**.

**On Windows:** The pipeline's built-in SORT tracker + motion vector analysis
is used automatically. No additional action needed.

---

## 7. Verification

After downloading datasets, check the structure:

```bash
# Check that images exist
dir datasets\helmet_data\images\train
dir datasets\seatbelt_data\images\train

# Run setup to validate environment
python scripts/setup.py --check

# Run tests (no GPU required)
pytest tests/ -v

# Start fine-tuning
python main.py --mode train --epochs 10 --device cuda:0
```

---

## Directory Structure After Full Setup

```
cctv_system/
├── models/
│   ├── yolo11n.pt                    ← auto-downloaded
│   ├── helmet/
│   │   └── helmet_finetuned.pt       ← generated by: python main.py --mode train
│   ├── seatbelt/
│   │   └── seatbelt_finetuned.pt     ← generated by: python main.py --mode train
│   └── plugins/
│       ├── helmet_haiderzm.pt        ← from Google Drive (optional)
│       ├── red_light_yolov8m.pt      ← from Ultralytics (optional)
│       └── triple_rider_kashish.pt   ← from Roboflow (optional, API key needed)
├── datasets/
│   ├── helmet_data/
│   │   ├── images/{train,val,test}/  ← populate from Roboflow download
│   │   ├── labels/{train,val,test}/  ← populate from Roboflow download
│   │   └── data.yaml
│   ├── seatbelt_data/
│   │   └── [same structure]
│   └── test_data/
│       ├── images/test/              ← generated by: python scripts/generate_test_images.py
│       └── sample_test.mp4           ← generated by: python scripts/generate_test_images.py
└── logs/
    └── cctv.log                      ← generated at runtime
```

---

## Plugin Attribution

| Plugin | GitHub Repository | Paper/License |
|---|---|---|
| Helmet detection | [haiderzm/Helmet-Detection](https://github.com/haiderzm/Helmet-Detection) | YOLOv5 (GPL-3.0) |
| Seatbelt detection | [sankethsj/seatbelt-detection](https://github.com/sankethsj/seatbelt-detection) | No custom weights available |
| Triple-rider | [kashishparmar02/triple-rider-detection](https://github.com/kashishparmar02/triple-rider-detection) | YOLOv8 (AGPL-3.0) |
| Red-light violation | [MohammedHamza0/Traffic-Signal-Violation-Detection](https://github.com/MohammedHamza0/traffic-signal-violation-detection) | Standard COCO yolov8m |
| Wrong-side driving | [sriramcu/yolov4_wrong_side_driving_detection](https://github.com/sriramcu/yolov4_wrong_side_driving_detection) | YOLOv4 Darknet (Linux only) |
| Base model | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| License-plate OCR | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache 2.0 |
