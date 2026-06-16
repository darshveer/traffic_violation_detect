# PROMPT FOR CLAUDE OPUS 4.6
## Build CCTV Traffic Violation Detection System

---

**COPY EVERYTHING BELOW AND PASTE INTO CLAUDE OPUS 4.6 (either via claude.ai Pro or API)**

---

```
You are an expert ML engineer tasked with building a production-ready CCTV traffic violation detection system. The system must:

1. **Detect traffic violations** from video/image frames using a two-stage pipeline
2. **Generate court-ready evidence** with annotated frames and metadata
3. **Run at 15+ FPS** on GPU and produce actionable reports
4. **Be deployable and tested** within 48 hours

## CONTEXT

- **Dataset:** 2,000–5,000 images (helmet, seatbelt violations; some from BDD100K, some manually curated from YouTube)
- **Deadline:** 48-hour hackathon (code must be production-ready, tested, and reproducible)
- **Target hardware:** NVIDIA GPU (RTX 3060+ preferred), CPU fallback supported
- **Deployment:** Jupyter notebook or Python script (no web UI required)
- **Evaluation:** mAP ≥0.65, F1 ≥0.70, latency <100ms per frame

## DELIVERABLES

Your code MUST include:

1. **data_loader.py**
   - Load images from `dataset/images/{train,val,test}`
   - Load labels from YOLO format (`dataset/labels/`)
   - Data augmentation pipeline (Albumentations)
   - Class balance checking and oversampling

2. **model_training.py**
   - YOLOv8n for detection (vehicle/person bboxes)
   - ResNet50 fine-tuning for violation classification
   - Training loop with validation, early stopping, checkpointing
   - Logging and loss tracking (save plots)

3. **inference.py**
   - Load trained detection + classification models
   - Process video frames or images
   - Batch inference for speed
   - Output: JSON metadata + annotated images
   - Frame skipping (every 5th frame) for real-time performance

4. **evidence_generator.py**
   - Draw bboxes + violation labels + confidence on frames
   - Generate violation reports (CSV: timestamp, location, violation_type, confidence)
   - Create heatmap of violations (folium or matplotlib)
   - Summary statistics: violations per hour, top violation types, spatial distribution

5. **metrics.py**
   - Compute precision, recall, F1, mAP on test set
   - Confusion matrix
   - Per-class metrics (helmet vs seatbelt etc.)
   - Latency benchmarks (FPS, memory usage)

6. **main.py**
   - Unified entry point: train or infer
   - Example: `python main.py --mode train --data dataset/`
   - Example: `python main.py --mode infer --video sample.mp4 --output results/`

## ARCHITECTURE DETAILS

### Detection Stage (YOLOv8n)
- Input: 640×640 RGB image
- Output: Bboxes of vehicles and riders (class: car, motorcycle, truck, pedestrian, rider)
- Model: YOLOv8n (nano) for speed; can switch to YOLOv8s if accuracy is poor
- Training: 50–100 epochs on 2K–5K images
- Expected accuracy: mAP50 ≥0.70

### Classification Stage (ResNet50)
- Input: Cropped region from detection (128×128 or 256×256)
- Output: Violation class (helmet_absent, helmet_present, seatbelt_absent, seatbelt_present, triple_rider, normal_riding)
- Model: ResNet50, pretrained on ImageNet, fine-tuned for 30–50 epochs
- Expected accuracy: F1 ≥0.75 per class

### Inference Pipeline
1. Load frame from video
2. Run YOLOv8n → get rider/driver bboxes
3. Crop regions around detected persons
4. Run ResNet50 classifier on crops
5. Store results: bbox coords, violation type, confidence score
6. Annotate and save frame

### Optimizations
- Frame skip: Process every 5th frame (4 FPS input → 0.8 FPS predictions, 30–40ms latency acceptable for CCTV)
- Batch inference: Collect 8–16 crops, classify in one forward pass
- Quantization: INT8 YOLOv8 for 30% speedup (optional)
- GPU memory: <4GB with batch_size=16

## DATA FORMAT SPECIFICATIONS

**Input data structure:**
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml (YOLO format)
└── README.md (class descriptions)
```

**YOLO label format (txt files):**
```
<class_id> <x_center> <y_center> <width> <height>
# Example: 0 0.5 0.5 0.3 0.4
```

**Classes (in data.yaml):**
```yaml
nc: 6
names:
  0: helmet_absent
  1: helmet_present
  2: seatbelt_absent
  3: seatbelt_present
  4: triple_rider
  5: normal_riding
```

**Output format (JSON metadata):**
```json
{
  "frame_id": 0,
  "timestamp": "2024-12-05T10:30:45.123Z",
  "detections": [
    {
      "bbox": [100, 150, 150, 200],
      "violation_type": "helmet_absent",
      "confidence": 0.87,
      "location": {"x": 125, "y": 175}
    }
  ],
  "frame_violations_count": 1
}
```

## TESTING & VALIDATION

1. **Unit tests for data loading** (check class balance, bbox validity)
2. **Test on held-out set** (compute mAP, F1, confusion matrix)
3. **Run on sample video** (5–10 min CCTV footage or YouTube video)
4. **Benchmark performance** (FPS, latency, GPU memory)
5. **Visual inspection** (manually check 20 annotated frames for false positives)

## DEPENDENCIES

Required libraries (generate requirements.txt):
```
torch>=2.0
torchvision>=0.15
ultralytics>=8.0  # YOLOv8
timm>=0.9.0  # ResNet50 from timm
albumentations>=1.3
opencv-python>=4.8
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
folium>=0.14
scikit-learn>=1.3
tqdm>=4.66
```

## IMPORTANT CONSTRAINTS

1. **No external APIs** (no cloud services; everything runs locally)
2. **No pretrained violation classifiers** (must train your own ResNet50, but can use YOLO8n detection weights)
3. **Reproducible** (set random seeds, save/load checkpoints, log everything)
4. **Tested** (must run without errors on a fresh GPU with your dataset)
5. **Well-documented** (docstrings for all functions, config file for hyperparameters)

## EXPECTED OUTCOMES (48H TARGET)

- **Accuracy:** mAP ≥0.65 (detection) + F1 ≥0.70 (classification)
- **Speed:** 15–30 FPS on GPU (RTX 3060 or better)
- **Latency:** <100ms per frame
- **Output:** 
  - Annotated video/images with violations highlighted
  - CSV report of all violations
  - Spatial heatmap of violations
  - Summary stats (count by type, count by hour, etc.)
  - Metrics JSON (precision, recall, F1, confusion matrix)

## CODE STYLE

- Use type hints for all functions
- Add docstrings (NumPy style)
- Use config files (JSON or YAML) for hyperparameters
- Organize code into separate modules (not one monolithic script)
- Add error handling and logging
- Use relative paths or environment variables for dataset paths

## EXAMPLE USAGE

```bash
# Training
python main.py --mode train --data dataset/ --epochs 50 --batch_size 32 --lr 0.001

# Inference on video
python main.py --mode infer --video sample.mp4 --weights yolov8n.pt --classifier_weights resnet50.pt --output results/

# Inference on image
python main.py --mode infer --image sample.jpg --output results/

# Evaluation
python main.py --mode eval --data dataset/ --weights yolov8n.pt --classifier_weights resnet50.pt
```

## ADDITIONAL NOTES

- This is for a hackathon, so polish matters: clean code, clear logging, reproducible results
- You may assume access to a GPU; but provide CPU fallback
- If dataset is small (<1K images), use aggressive augmentation and pretrained weights
- If dataset is large (>5K images), train from scratch or use transfer learning
- Submit a Jupyter notebook OR Python scripts + requirements.txt + README

Now, **generate the complete, production-ready code** for all modules above. Start with data_loader.py, then model_training.py, then inference.py, then evidence_generator.py, and finally main.py. Include a config.json for hyperparameters and a comprehensive README.md.

Make the code modular, well-tested, and ready to run on day 1 of the hackathon.
```

---

## HOW TO USE THIS PROMPT

1. **Copy the prompt above** (from `You are an expert ML engineer...` to the end)
2. **Paste into Claude Opus 4.6:**
   - Via **claude.ai**: Paste into a new chat, use Claude 3.5 Sonnet (or pay for Opus if available)
   - Via **Claude API**: Use `claude-opus-4-6` model with your API key
3. **Let it generate the full codebase** (will output 3,000–5,000 lines of production code)
4. **Save outputs to files**:
   ```bash
   mkdir cctv_system
   cd cctv_system
   # Save Claude's output files here
   ```
5. **Run it**:
   ```bash
   pip install -r requirements.txt
   python main.py --mode train --data dataset/ --epochs 50
   ```

---

## WHAT TO EXPECT

**Claude Opus will generate:**
- ✅ `data_loader.py` (500 lines) — data pipeline with augmentation
- ✅ `model_training.py` (800 lines) — training loop with validation
- ✅ `inference.py` (600 lines) — fast batch inference
- ✅ `evidence_generator.py` (400 lines) — annotated output + reports
- ✅ `metrics.py` (300 lines) — evaluation metrics
- ✅ `main.py` (200 lines) — CLI entry point
- ✅ `config.json` — hyperparameters
- ✅ `requirements.txt` — dependencies
- ✅ `README.md` — full documentation

**Total:** ~3,200 lines of **tested, production-grade code**

---

## TIMELINE ESTIMATE

| Phase | Duration | Task |
|-------|----------|------|
| **Days 1–2** | ~7 hours | Data sourcing, labeling, curation |
| **Day 3** | ~4 hours | Claude generates code, install deps, validate on small subset |
| **Days 4–5** | ~12 hours | Train models (YOLOv8 + ResNet50), tune hyperparameters |
| **Days 6–7** | ~8 hours | Test on full dataset, optimize inference, generate reports, demo prep |
| **Buffer** | ~10 hours | Debugging, fine-tuning, edge case handling |

**Total: ~48 hours** ✓

