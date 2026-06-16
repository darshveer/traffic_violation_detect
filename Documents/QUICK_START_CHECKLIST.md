# QUICK START CHECKLIST
## Your Next 48 Hours (Dec 5 Midnight → Dec 7 Midnight)

---

## ⏰ HOUR-BY-HOUR BREAKDOWN

### **DAYS 1–2: Data Prep (Hours 0–7, parallel tasks)**

#### Hour 0–1: Setup
- [ ] Create project directory: `mkdir -p cctv_system && cd cctv_system`
- [ ] Create subdirectories:
  ```bash
  mkdir -p dataset/{images/{train,val,test},labels/{train,val,test}}
  mkdir -p models results logs
  ```
- [ ] Read the Dataset Sourcing Guide (30 min)
- [ ] Decide on dataset source (BDD100K? YouTube? Hybrid?)

#### Hour 1–3: Download Base Data (Run in parallel on 2 terminals)

**Terminal 1: Download BDD100K sample**
```bash
# Visit https://bdd-data.berkeley.edu/
# Register, accept terms
# Download: train_images.zip (~1-2 GB for sample)
# Extract to: dataset/images/train_bdd/
unzip train_images.zip -d dataset/images/train_bdd/

# This gives you ~1K real-world driving images
```

**Terminal 2: Download Cityscapes (parallel)**
```bash
# Visit https://www.cityscapes-dataset.org/
# Register, download: leftImg8bit_trainvaltest.zip (~5-10 GB)
# Extract to: dataset/images/train_city/
unzip leftImg8bit_trainvaltest.zip -d dataset/images/train_city/
```

**Expected outcome:** 2K–5K images ready to label

#### Hour 3–5: Manual Labeling (Parallel with downloads)

**Option A: CVAT Online (Recommended, fastest)**
1. Go to https://app.cvat.ai (free, no signup if you just want to try)
2. Create project with 6 classes:
   - `helmet_absent`
   - `helmet_present`
   - `seatbelt_absent`
   - `seatbelt_present`
   - `triple_rider`
   - `normal_riding`
3. Upload 200–300 images from YouTube (next step)
4. Label for 2–3 hours → export as YOLO format

**Option B: YouTube Curation (Skip CVAT, go raw)**
```bash
# Install yt-dlp
pip install yt-dlp ffmpeg-python

# Download traffic/motorcycle video
yt-dlp -f best "https://www.youtube.com/watch?v=TRAFFIC_VIDEO_ID" -o "raw_video.mp4"

# Extract frames at 1 FPS
ffmpeg -i raw_video.mp4 -vf fps=1 dataset/images/raw/frame_%04d.jpg

# Now manually label using CVAT (faster than frame extraction)
```

**Search terms for videos:**
- "India traffic violations CCTV"
- "Motorcycle helmet safety footage"
- "Traffic enforcement footage"
- "Dashcam traffic violations"
- "City traffic intersection" (for illegal parking)

#### Hour 5–7: Organize Data

```bash
# Move labeled images to train/val/test split (80/10/10)
python -c "
import os, shutil, random
from pathlib import Path

images = list(Path('dataset/images/labeled').glob('*.jpg'))
random.shuffle(images)

split = int(len(images) * 0.8), int(len(images) * 0.9)
for i, img in enumerate(images):
    if i < split[0]:
        dst = 'dataset/images/train/'
    elif i < split[1]:
        dst = 'dataset/images/val/'
    else:
        dst = 'dataset/images/test/'
    shutil.copy(str(img), dst)
"

# Verify structure
tree dataset/ -L 3
```

**Output by Hour 7:**
```
dataset/
├── images/
│   ├── train/  (1600 images)
│   ├── val/    (200 images)
│   └── test/   (200 images)
├── labels/
│   ├── train/  (1600 txt files, YOLO format)
│   ├── val/    (200 txt files)
│   └── test/   (200 txt files)
└── data.yaml
```

---

### **DAY 2: Generate Code (Hours 8–24)**

#### Hour 8–9: Get Claude to Generate Code

**Step 1:** Copy the prompt from `PROMPT_FOR_OPUS.md`

**Step 2:** Paste into Claude Opus 4.6:
- **Option A (Fast):** Use Claude.ai Pro (paid) → select "Claude Opus 4.6"
  - https://claude.ai
  - Paste prompt → Hit "Generate" → Wait 5–10 min
- **Option B (Slower):** Use Claude API
  ```python
  import anthropic
  
  client = anthropic.Anthropic(api_key="YOUR_API_KEY")
  
  with open("PROMPT_FOR_OPUS.md") as f:
      prompt = f.read()
  
  message = client.messages.create(
      model="claude-opus-4-6",
      max_tokens=8000,
      messages=[{"role": "user", "content": prompt}]
  )
  
  print(message.content[0].text)
  ```

**Step 3:** Claude generates 6 files. Save them:
```bash
# Create files from Claude's output
touch data_loader.py
touch model_training.py
touch inference.py
touch evidence_generator.py
touch metrics.py
touch main.py
touch config.json
touch requirements.txt
touch README.md

# Copy Claude's output into each file
```

**Expected time:** 10–15 minutes

#### Hour 9–10: Install Dependencies
```bash
pip install -r requirements.txt
# Takes 5–10 min (PyTorch install is slow)

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Hour 10–12: Test Data Pipeline
```bash
# Quick smoke test
python data_loader.py --test

# Expected output:
# - Loaded 1600 train images
# - Loaded 200 val images
# - Loaded 200 test images
# - Class balance: helmet_absent: 320, helmet_present: 280, ...
```

#### Hour 12–15: Start Training (Can run in background)
```bash
# Terminal 1: Start training (will take 6–8 hours)
python main.py --mode train \
  --data dataset/ \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001 \
  --device cuda:0

# Monitors:
# - Training loss (should decrease)
# - Validation F1 (should increase)
# - Checkpoints saved every 5 epochs
```

**While training runs (Hours 15–24):**
- Work on inference.py optimizations (batch processing, frame skip)
- Prepare sample video for testing
- Write evidence_generator.py logic
- Create test suite

#### Hour 24: Check Training Progress
```bash
# Check logs
tail -f logs/training.log

# Expected metrics by Hour 24:
# - mAP (detection): 0.55–0.65
# - F1 (classification): 0.65–0.75
# - Training loss: Decreasing
```

---

### **DAY 3: Testing & Optimization (Hours 24–40)**

#### Hour 24–28: Complete Training
```bash
# Let training finish
# By Hour 28, you should have:
# - Trained YOLOv8n weights → models/yolov8n.pt
# - Trained ResNet50 weights → models/resnet50.pt
# - Metrics → logs/metrics.json
```

#### Hour 28–32: Inference Testing
```bash
# Test on sample video (5 min CCTV footage)
python main.py --mode infer \
  --video sample_traffic.mp4 \
  --weights models/yolov8n.pt \
  --classifier_weights models/resnet50.pt \
  --output results/

# Expected output:
# results/
# ├── annotated_video.mp4 (violations highlighted)
# ├── violations_metadata.json
# ├── violations_report.csv
# └── heatmap.html
```

#### Hour 32–36: Evaluation & Metrics
```bash
# Run full evaluation on test set
python main.py --mode eval \
  --data dataset/ \
  --weights models/yolov8n.pt \
  --classifier_weights models/resnet50.pt

# Generates:
# results/metrics.json
# results/confusion_matrix.png
# results/per_class_metrics.csv
```

**Expected metrics:**
```json
{
  "detection_mAP": 0.68,
  "classification_F1": 0.72,
  "precision": 0.75,
  "recall": 0.70,
  "latency_ms": 45,
  "fps": 22
}
```

#### Hour 36–40: Report Generation
```bash
# Generate final report
python evidence_generator.py \
  --violations results/violations_metadata.json \
  --output results/

# Creates:
# results/violation_summary.csv
# results/violations_by_hour.csv
# results/heatmap.html
# results/statistics.json
```

**Example output:**
```
VIOLATION SUMMARY
=================
Total violations detected: 847
- Helmet absent: 423 (50%)
- Seatbelt absent: 312 (37%)
- Triple riding: 112 (13%)

Top 10 violation locations (by frequency):
1. Intersection A: 156 violations
2. Intersection B: 142 violations
...

Peak violation hours:
8:00-9:00 AM: 234 violations
5:00-6:00 PM: 198 violations
...
```

---

### **DAY 4: Presentation & Demo (Hours 40–48)**

#### Hour 40–42: Create Demo Notebook
```bash
# Create Jupyter notebook for live demo
jupyter notebook demo.ipynb
```

**Notebook contents:**
1. Load sample image
2. Run detection
3. Show annotated output
4. Display metrics
5. Generate heatmap

```python
# Example cell:
from inference import Detector

detector = Detector(
    yolo_weights='models/yolov8n.pt',
    classifier_weights='models/resnet50.pt'
)

image = cv2.imread('sample.jpg')
results = detector.infer(image)

# Show annotated image
annotated = detector.visualize(image, results)
cv2.imshow('Violations Detected', annotated)
cv2.waitKey(0)

# Print metrics
print(f"Violations: {len(results)}")
for vio in results:
    print(f"  - {vio['type']}: confidence {vio['conf']:.2f}")
```

#### Hour 42–45: Documentation
```bash
# Create presentation slides (Markdown or PowerPoint)
# - Problem statement
# - Architecture diagram
# - Dataset statistics
# - Results (metrics, confusion matrix)
# - Sample outputs (annotated frames, heatmap)
# - Deployment plan
```

#### Hour 45–48: Final Testing & Packaging
```bash
# Final smoke test
python main.py --mode infer \
  --image test_image.jpg \
  --output final_demo/

# Create tarball for submission
tar -czf cctv_system.tar.gz \
  --exclude='dataset' \
  cctv_system/

# File checklist for submission:
# - ✓ All .py files
# - ✓ requirements.txt
# - ✓ config.json
# - ✓ README.md
# - ✓ Model weights (if <200 MB each)
# - ✓ Sample output (annotated image + JSON)
# - ✓ Demo notebook
# - ✓ Presentation slides
```

---

## 📋 FINAL CHECKLIST

- [ ] Data sourced & labeled (2K–5K images)
- [ ] Claude-generated code working locally
- [ ] Dependencies installed (torch, ultralytics, etc.)
- [ ] Training completed (YOLOv8n + ResNet50)
- [ ] Inference tested on sample video
- [ ] Metrics computed (mAP ≥0.65, F1 ≥0.70)
- [ ] Evidence reports generated
- [ ] Demo notebook created
- [ ] Presentation slides ready
- [ ] Code packaged & documented
- [ ] Final testing passed
- [ ] Submission ready

---

## 🎯 SUCCESS METRICS (FOR JUDGING)

**Must have:**
- ✅ Working detection + classification pipeline
- ✅ Real-time inference (>10 FPS)
- ✅ Annotated output with violations highlighted
- ✅ Evidence reports (CSV + JSON)
- ✅ Metrics showing 60%+ accuracy

**Nice to have:**
- 🎁 Heatmap of violations
- 🎁 Interactive demo
- 🎁 Docker container
- 🎁 Comparison with baseline (YOLOv8 alone)

---

## 🚨 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce batch_size to 8 or 16, enable gradient accumulation |
| Low accuracy (F1 < 0.60) | More training data (augment), more epochs (100+), lower learning rate |
| Slow inference (<5 FPS) | Skip frames (every 5th), reduce input size (480p), use INT8 quantization |
| Data loading fails | Check dataset/data.yaml paths, verify YOLO label format |
| Training stalls | Check learning rate, try warmup, reduce batch size |

---

## 📚 REFERENCE DOCS

- Dataset Sourcing: See `DATASET_SOURCING_GUIDE.md`
- Opus Prompt: See `PROMPT_FOR_OPUS.md`
- YOLOv8 Docs: https://docs.ultralytics.com
- ResNet50 Fine-tuning: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- CVAT Labeling: https://docs.cvat.ai/docs/

---

## 🚀 YOU'RE READY TO START

**Next action:** 
1. Download the Dataset Sourcing Guide
2. Start downloading BDD100K OR curating from YouTube
3. Get Claude to generate code (in parallel)
4. Start training
5. Win the hackathon! 🏆

