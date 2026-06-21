# UPDATED PROMPT FOR CLAUDE OPUS 4.6
## Building CCTV System with Existing Open-Source Models

---

**COPY AND PASTE THIS ENTIRE PROMPT INTO CLAUDE OPUS 4.6**

---

```
You are an expert ML systems engineer tasked with building a production-grade CCTV traffic violation detection system by integrating open-source models and minimizing training time.

## CORE STRATEGY

Instead of training from scratch, you will:
1. Integrate pre-trained violation detection models from open-source repos
2. Fine-tune ONLY helmet & seatbelt detection on custom Indian CCTV data (~1-2 epochs each)
3. Build a unified inference pipeline that chains all models together
4. Generate court-ready evidence (annotated frames + reports)

This approach reduces GPU time from 40+ hours to ~2.5 hours while maintaining >70% accuracy.

## MODEL SELECTION (FINAL DECISION)

### Use As-Is (No Training):
- **YOLOv8n** (Ultralytics) → Base vehicle/person detection
- **Triple Riding** → kashishparmar02 pre-trained model (6K images, ready)
- **Red-Light Violation** → MohammedHamza0 pretrained YOLOv8 (ROI-based)
- **Wrong-Side Driving** → sriramcu YOLOv4 + Kalman filtering (pretrained)
- **License Plate OCR** → PaddleOCR (industry-standard, no retraining)

### Fine-Tune on Custom Data (1-2 epochs each):
- **Helmet Detection** → Start from haiderzm/Helmet-Detection (YOLOv5, 350 epochs pre-trained)
  - Fine-tune on 500 custom Indian motorcycle images (~30 min GPU)
- **Seatbelt Detection** → Start from sankethsj/seatbelt-detection (YOLOv5)
  - Fine-tune on 300 custom car interior images (~20 min GPU)

## PROJECT STRUCTURE

```
cctv_system/
├── models/
│   ├── helmet/                    # Fine-tuned from haiderzm
│   ├── seatbelt/                  # Fine-tuned from sankethsj
│   ├── yolov8n_base.pt            # YOLOv8 nano weights
│   └── [other pretrained weights]
├── pipelines/
│   ├── violation_detector.py      # Main inference class (chains all models)
│   ├── model_loader.py            # Load 6 models efficiently
│   ├── inference_utils.py         # Batch processing, frame skipping
│   └── tracking.py                # SORT/DeepSort for vehicle tracking
├── evidence/
│   ├── annotate.py                # Draw bboxes + violation labels + confidence
│   ├── report_generator.py        # CSV violations + JSON metadata + heatmap
│   └── ocr_handler.py             # PaddleOCR integration for license plates
├── train/
│   ├── finetune_helmet.py         # Fine-tune helmet model on custom data
│   ├── finetune_seatbelt.py       # Fine-tune seatbelt model on custom data
│   └── config.yaml                # Training hyperparameters
├── datasets/
│   ├── helmet_data/
│   │   ├── images/train/
│   │   ├── images/val/
│   │   ├── labels/train/
│   │   ├── labels/val/
│   │   └── data.yaml              # YOLO format, 2 classes: helmet_absent, helmet_present
│   ├── seatbelt_data/
│   │   └── [same structure]
│   └── test_data/
├── main.py                        # CLI entry point
├── requirements.txt
└── README.md
```

## DELIVERABLES (DETAILED SPECS)

### 1. model_loader.py
- Load all 6 models (pretrained + fine-tuned) with error handling
- Cache models in GPU memory (reuse across frames for speed)
- Support CPU fallback for each model independently
- Log which models are loaded and from where

### 2. pipelines/violation_detector.py
- Class: `ViolationDetector`
- Methods:
  - `infer_frame(frame)`: Run all 6 violation detectors on single frame
    - Returns dict with detections, violation counts, confidence scores
  - `infer_video(video_path, output_path)`: Process video with frame skipping
    - Skip every 5th frame for speed (0.8 FPS predictions on 4 FPS input)
    - Support resume from checkpoint (in case of crash)
  - `infer_image(image_path)`: Single image inference
  - `batch_infer(crops, model_name)`: Batch inference for speed (8-16 crops at once)

- Violation types returned:
  ```python
  {
    'helmet_absent': confidence,
    'helmet_present': confidence,
    'seatbelt_absent': confidence,
    'seatbelt_present': confidence,
    'triple_rider': confidence,
    'red_light_violation': confidence,
    'wrong_side_driving': confidence,
    'license_plate': detected_text
  }
  ```

### 3. train/finetune_helmet.py
- Load pre-trained weights from haiderzm/Helmet-Detection
- Fine-tune on custom dataset (datasets/helmet_data/)
- Training config:
  - epochs: 10 (transfer learning = fewer epochs)
  - batch_size: 32
  - learning_rate: 0.001
  - patience: 3 (early stopping)
  - optimizer: SGD or Adam
- Output: Save fine-tuned model to models/helmet/helmet_finetuned.pt
- Log metrics: precision, recall, F1, mAP per epoch

### 4. train/finetune_seatbelt.py
- Identical structure to helmet fine-tuning
- Load pre-trained weights from sankethsj/seatbelt-detection
- Fine-tune on datasets/seatbelt_data/
- Config:
  - epochs: 10
  - batch_size: 16 (smaller dataset)
  - patience: 3
- Output: Save to models/seatbelt/seatbelt_finetuned.pt

### 5. evidence/annotate.py
- Draw bounding boxes on frames with:
  - Violation label (e.g., "helmet_absent")
  - Confidence score (e.g., 0.87)
  - Timestamp
  - Color coding: RED for violations, GREEN for normal
- Support batch annotation (multiple frames)
- Output: annotated images/video

### 6. evidence/report_generator.py
- Generate CSV report:
  ```csv
  frame_id,timestamp,violation_type,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2,license_plate
  0,2024-12-05 10:30:01,helmet_absent,0.87,100,150,150,200,KA-01-AB-1234
  1,2024-12-05 10:30:02,seatbelt_absent,0.92,200,180,280,300,
  ...
  ```
- Generate JSON metadata (same data, structured)
- Generate heatmap (folium or matplotlib):
  - Spatial distribution of violations
  - Color intensity = violation frequency
  - Hover info = violation type + count
- Generate summary stats:
  - Violations per type (count + percentage)
  - Peak violation hours
  - Top locations
  - Metrics (precision, recall, F1, mAP if test data provided)

### 7. main.py (CLI Interface)
Commands:
```bash
python main.py --mode train \
  --helmet_data datasets/helmet_data/ \
  --seatbelt_data datasets/seatbelt_data/ \
  --epochs 10 \
  --batch_size 32

python main.py --mode infer \
  --video sample.mp4 \
  --output results/ \
  --skip_frames 5

python main.py --mode infer \
  --image sample.jpg \
  --output results/

python main.py --mode eval \
  --test_data datasets/test_data/ \
  --output results/metrics/
```

## DATA FORMAT SPECIFICATIONS

### Input: YOLO Format
```
datasets/helmet_data/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── img_001.txt  → "0 0.5 0.5 0.3 0.4"  (class x_center y_center w h)
│   │   └── ...
│   ├── val/
│   └── test/
└── data.yaml (REQUIRED)
```

**data.yaml:**
```yaml
path: /absolute/path/to/helmet_data/
train: images/train
val: images/val
test: images/test
nc: 2
names:
  0: helmet_absent
  1: helmet_present
```

### Output: Annotated + Reports
```
results/
├── annotated_video.mp4        # Video with violations highlighted
├── annotated_frames/
│   ├── frame_0000.jpg
│   ├── frame_0005.jpg
│   └── ...
├── violations_metadata.json   # All detections (JSON)
├── violations_report.csv      # All detections (CSV)
├── heatmap.html              # Spatial distribution
├── summary_stats.json        # Aggregate metrics
└── metrics/
    ├── confusion_matrix.png
    ├── precision_recall_curve.png
    └── metrics.json (precision, recall, F1, mAP)
```

## IMPLEMENTATION NOTES

### Speed Optimizations:
- Frame skipping: Process every 5th frame (reduces latency by 5×)
- Batch inference: Collect 8-16 crops, classify in one forward pass (2-3× speedup)
- Model caching: Load models once, reuse across all frames
- GPU memory management: Monitor memory, fall back to CPU if needed

### Accuracy Improvements:
- Use pre-trained weights (they're already optimized)
- Fine-tune on 1-2 epochs (prevents overfitting, saves GPU time)
- Data augmentation on custom data (flip, rotate, brightness adjust)
- Confidence thresholding: Only report violations with confidence > 0.6

### Robustness:
- Handle missing frames gracefully
- Support both video and image inputs
- Fallback to CPU if GPU unavailable
- Error logging for all model failures
- Checkpointing for long videos (resume from checkpoint)

## DATASETS & MODELS TO DOWNLOAD

**Before coding, user should have:**
1. Pre-trained weights:
   - haiderzm/Helmet-Detection → best.pt
   - sankethsj/seatbelt-detection → best.pt
   - kashishparmar02/triple-rider-detection → best.pt
   - MohammedHamza0/Traffic-Signal-Violation → best.pt
   - sriramcu/yolov4_wrong_side_driving → weights.pt
   - YOLOv8n (auto-download from Ultralytics)

2. Custom fine-tuning data:
   - datasets/helmet_data/  (500 images, YOLO format)
   - datasets/seatbelt_data/ (300 images, YOLO format)

3. Sample test data:
   - datasets/test_data/ (200 images for evaluation)
   - sample_traffic.mp4 (5-10 min video for demo)

## EXPECTED PERFORMANCE

- **Accuracy:** mAP ≥ 0.70 (helmet + seatbelt after fine-tuning)
- **Speed:** 15-30 FPS on GPU (RTX 3060+), 2-5 FPS on CPU
- **Latency:** <100ms per frame
- **Memory:** <4GB GPU VRAM (batch_size=16)
- **Output:** Annotated video + CSV + JSON + heatmap

## CODE QUALITY REQUIREMENTS

- Type hints for all functions
- Docstrings (NumPy style) for all classes/methods
- Config files (YAML) for hyperparameters
- Modular design (separate files for each concern)
- Comprehensive error handling and logging
- Unit tests for data loading, model loading, inference

## EXAMPLE USAGE FLOW

```bash
# Day 1: Setup + download models
# Clone repos manually or via script
# Download datasets

# Day 2: Fine-tune
python train/finetune_helmet.py --epochs 10 --batch_size 32
python train/finetune_seatbelt.py --epochs 10 --batch_size 16

# Day 3: Test inference
python main.py --mode infer --image sample.jpg --output results/

# Day 4: Run on video
python main.py --mode infer --video sample.mp4 --output results/ --skip_frames 5

# Day 5: Generate reports + metrics
python evidence/report_generator.py --metadata results/violations_metadata.json
python main.py --mode eval --test_data datasets/test_data/
```

## KEY DELIVERABLES (MUST HAVE)

1. ✅ model_loader.py (load 6 models with caching)
2. ✅ violation_detector.py (unified inference pipeline)
3. ✅ finetune_helmet.py (fine-tuning script)
4. ✅ finetune_seatbelt.py (fine-tuning script)
5. ✅ annotate.py (draw violations on frames)
6. ✅ report_generator.py (CSV + JSON + heatmap)
7. ✅ main.py (CLI interface with train/infer/eval modes)
8. ✅ requirements.txt (all dependencies)
9. ✅ README.md (setup, usage, model attribution)
10. ✅ config.yaml (hyperparameters for fine-tuning)

## IMPORTANT: MODEL ATTRIBUTION

In README.md, list all open-source models used and their sources:
- YOLOv8 (Ultralytics)
- Helmet Detection (haiderzm)
- Seatbelt Detection (sankethsj)
- Triple Rider (kashishparmar02)
- Red-Light (MohammedHamza0)
- Wrong-Side (sriramcu)
- PaddleOCR (PaddlePaddle)

This is NOT plagiarism—this is how professional systems are built. Credit all sources.

## DELIVERABLE FORMAT

Generate complete, production-ready code for all modules above. Code should:
- Run immediately after `pip install -r requirements.txt`
- Support both training and inference modes
- Include comprehensive logging and error handling
- Work on GPU and CPU (with graceful degradation)
- Be fully documented with docstrings and comments

Start with model_loader.py, then violation_detector.py, then the fine-tuning scripts, then evidence generation, then main.py.
```

---

## HOW TO USE THIS PROMPT

1. **Copy the prompt above** (from `You are an expert ML systems engineer...` to the end)
2. **Paste into Claude Opus 4.6** via:
   - claude.ai (select Opus 4.6 from dropdown)
   - Or Claude API with `claude-opus-4-6` model
3. **Wait 10–15 minutes** for full codebase generation
4. **Save all files** to your project directory

---

## EXPECTED OUTPUT

Claude will generate:
- ✅ `model_loader.py` (400 lines) — Load all 6 models
- ✅ `violation_detector.py` (600 lines) — Main inference class
- ✅ `finetune_helmet.py` (150 lines) — Fine-tune helmet model
- ✅ `finetune_seatbelt.py` (150 lines) — Fine-tune seatbelt model
- ✅ `annotate.py` (250 lines) — Draw annotations
- ✅ `report_generator.py` (300 lines) — Generate reports
- ✅ `main.py` (200 lines) — CLI interface
- ✅ `config.yaml` — Training config
- ✅ `requirements.txt` — Dependencies
- ✅ `README.md` — Complete documentation

**Total:** ~2,200 lines of production-grade code

