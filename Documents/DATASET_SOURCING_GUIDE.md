# CCTV Traffic Violation Detection System
## Complete Dataset Sourcing & Setup Guide

---

## PART 1: WHERE TO SOURCE DATASETS

### **Option 1A: BDD100K (FASTEST + EASIEST)**
**What it is:** 100K+ diverse driving images with bboxes, semantic segmentation, labels  
**Why use it:** Vehicle/person detection already annotated; road scenarios  
**Time to download:** 30 min (partial dataset)  
**Link:** https://bdd-data.berkeley.edu/

**How to use:**
```bash
# Register, download train_images.zip (~50GB for full; ~5GB for sample)
# Extract and use for vehicle detection training

# Sample images include:
# - Daytime/nighttime (street lights, poor visibility)
# - Rain, fog, shadows
# - Multiple vehicles, pedestrians, motorcycles
# - Urban intersections (your use case!)
```

**What you get for free:**
- 100K images with vehicle bboxes
- Object categories: car, truck, bike, pedestrian, rider
- Daytime + nighttime coverage
- Reasonable diversity (cities, highways, parking lots)

**Limitation:** No helmet/seatbelt labels (you'll add those)

---

### **Option 1B: Cityscapes (ALTERNATIVE IF BDD IS SLOW)**
**What it is:** High-quality urban street scene images (50 cities, 5K finely annotated)  
**Why use it:** Detailed segmentation, good for fine details (clothing, equipment)  
**Time to download:** 20 min  
**Link:** https://www.cityscapes-dataset.org/

**How to use:**
```bash
# Register, download leftImg8bit_trainvaltest.zip (~11GB)
# Use for semantic segmentation to isolate upper bodies (helmet detection)
```

---

### **Option 2: CARLA Simulator (SYNTHETIC GENERATION)**
**What it is:** Open-source autonomous driving simulator  
**Why use it:** Generate helmet violations with ground-truth labels; control conditions  
**Time to setup:** 1–2 hours (one-time)  
**Link:** https://carla.org/

**How to use:**
```python
# Install CARLA
pip install carla

# Generate synthetic scenes:
# - Place rider with/without helmet in various poses
# - Vary lighting, weather, camera angles
# - Export frames + bounding boxes automatically

# Example: Generate 1K helmet/no-helmet pairs in 30 min
```

**Advantage:** Unlimited labeled data, control over scenarios  
**Limitation:** Sim-to-real gap (but fine-tuning bridges it)

---

### **Option 3: YouTube CCTV FOOTAGE (MANUAL CURATION)**
**Best sources:**
1. **City traffic cameras** (search: "traffic camera live India"/"CCTV footage traffic")
2. **Dashcam compilations** (YouTube channels: "best dashcam", "motorcycle riding")
3. **News footage** (traffic incidents, rallies, events)

**How to extract data:**
```bash
# Use youtube-dl to download
pip install yt-dlp

# Download and extract frames
yt-dlp -f best https://youtube.com/watch?v=VIDEO_ID -o "video.mp4"

# Extract frames at 1 FPS (this gives ~3600 frames from 1 hour video)
ffmpeg -i video.mp4 -vf fps=1 frame_%04d.jpg

# Manually label violations using CVAT or Label Studio (see below)
```

**Effort:** 2–3 hours for 500 labeled violations  
**Result:** Highly realistic, domain-specific data

---

### **Option 4: OpenImages (FOR MISC OBJECTS)**
**What it is:** 9M+ images with open-vocab annotations  
**Why use it:** Fallback for specific object classes (helmets, seatbelts)  
**Link:** https://storage.googleapis.com/openimages/web/index.html

```bash
# Download images containing "helmet" or "motorcycle"
# Use downloader tool: https://github.com/openimages/dataset

pip install open-images-downloader
openimagesdownload --classes Helmet --maxlinks 1000
```

---

## PART 2: QUICK SETUP (BY PRIORITY)

### **Week 1: Data Prep (Days 1–3)**

**Priority Tier 1 (START HERE):**
1. Download BDD100K sample (~1K images) → Extract vehicles
2. Download Cityscapes leftImg8bit (~500 images) → Extract upper bodies
3. Total effort: 1 hour download + 30 min preprocessing

**Priority Tier 2 (DAYS 2–3):**
1. Curate 500 images from YouTube (helmet/no-helmet, seatbelt/no-seatbelt)
2. Label using CVAT (free, open-source)
3. Augment with Albumentations (flip, rotate, brightness adjust)

**Priority Tier 3 (STRETCH, if time permits):**
1. Generate 1K synthetic images from CARLA
2. Fine-tune YOLOv8 on your dataset

---

## PART 3: LABELING WORKFLOW

### **Tool: CVAT (Recommended)**
Free, open-source annotation platform.

```bash
# Online version (no setup):
# https://app.cvat.ai

# Or self-hosted:
docker run -p 8080:8080 cvat/cvat:latest
# Then visit http://localhost:8080
```

**Labeling scheme for violations:**

| Violation Type | Class | Indicators |
|---|---|---|
| No helmet | `helmet_absent` | Bare head, hair visible, no headgear |
| Has helmet | `helmet_present` | Helmet on head (color doesn't matter) |
| Seatbelt off | `seatbelt_absent` | Driver/passenger torso visible, no belt |
| Seatbelt on | `seatbelt_present` | Clear belt across torso |
| Triple riding | `triple_rider` | 3+ people on 2-wheeler |
| Normal riding | `normal_riding` | 1–2 people, safe |

**Estimated time to label 500 images:** 3–4 hours at ~30 sec/image

---

## PART 4: DATASET STRUCTURE FOR TRAINING

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/  (YOLO format)
│   ├── train/
│   │   ├── img_001.txt  → "0 0.5 0.5 0.3 0.4"  (class x_center y_center w h)
│   │   └── ...
│   ├── val/
│   └── test/
└── data.yaml
```

**data.yaml (for YOLO training):**
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 6  # number of classes
names:
  0: helmet_absent
  1: helmet_present
  2: seatbelt_absent
  3: seatbelt_present
  4: triple_rider
  5: normal_riding
```

---

## PART 5: QUICK STATS TO TARGET

**Realistic for 48h hackathon:**
- Training images: 2,000–5,000
- Test images: 500
- Violation samples per class: 200–300 (minimum)
- mAP target: 0.60–0.75 (detection) + 0.70–0.85 (classification)
- FPS on GPU: 15–30 FPS @ 640p

---

## PART 6: COMMAND CHEAT SHEET

### Extract frames from video:
```bash
ffmpeg -i video.mp4 -vf fps=1 frame_%04d.jpg
```

### Augment dataset (Python):
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.GaussNoise(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
```

### Check dataset balance:
```python
import os
from collections import Counter

labels = []
for fname in os.listdir('labels/train/'):
    with open(f'labels/train/{fname}') as f:
        for line in f:
            labels.append(int(line.split()[0]))

print(Counter(labels))
```

### Download YOLOv8:
```bash
pip install ultralytics opencv-python
```

---

## PART 7: IF YOU'RE SHORT ON TIME

**Minimum viable dataset (start with this):**
1. **Use pretrained YOLOv8n on raw footage** (no retraining)
   - Detects people, vehicles, bikes out-of-the-box
   - Extract crops of riders/drivers
   - Fine-tune a small ResNet50 on just helmet/seatbelt (2K images)
   - Takes 3–4 hours total

2. **Or: Use existing violation detection models**
   - Search Hugging Face for "helmet detection" or "seatbelt detection"
   - Many researchers publish pretrained weights
   - Fine-tune on your 500 images (30 min to 2 hours)

---

## PART 8: SUMMARY CHECKLIST

**Before you start coding (Do in parallel):**
- [ ] Download BDD100K OR Cityscapes (~1 hour)
- [ ] Curate 500 images from YouTube (~2 hours)
- [ ] Label violations in CVAT (~3 hours)
- [ ] Organize into YOLO format (~30 min)
- [ ] Test data pipeline locally (~30 min)

**Total prep time: ~7 hours (spread over Days 1–2)**

Once data is ready, training + inference pipeline takes ~20 hours (Days 3–6).

---

## REFERENCE LINKS

| Resource | Purpose | Link |
|---|---|---|
| BDD100K | Detection base dataset | https://bdd-data.berkeley.edu/ |
| Cityscapes | Segmentation dataset | https://www.cityscapes-dataset.org/ |
| CARLA | Synthetic generation | https://carla.org/ |
| CVAT | Annotation tool | https://app.cvat.ai |
| YOLOv8 | Detection | https://github.com/ultralytics/ultralytics |
| Albumentations | Data augmentation | https://albumentations.ai/ |
| Label Studio | Annotation (alternative) | https://labelstud.io/ |

