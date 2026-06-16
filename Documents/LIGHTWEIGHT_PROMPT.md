# LIGHTWEIGHT PROMPT (FAST VERSION)
## For quick prototyping if time is tight

Use this if:
- You want working code ASAP (ignore optimization)
- Your dataset is ready but time is running out
- You want to see results in <2 hours, not perfect results

---

## COPY-PASTE THIS INTO CLAUDE OPUS

```
Build a complete CCTV traffic violation detection system in Python. Use:
- YOLOv8n for vehicle/person detection (pretrained)
- ResNet50 for violation classification (fine-tuned on custom data)
- Inference on video/images with annotated output

Deliverables:
1. train.py: Fine-tune ResNet50 on violation dataset
2. infer.py: Run detection + classification on frames
3. generate_report.py: Create annotated images + CSV report
4. main.py: CLI wrapper
5. requirements.txt

Dataset structure:
  dataset/
  ├── images/train/ (1000+ images)
  ├── images/val/   (200 images)
  ├── labels/train/ (YOLO format)
  ├── labels/val/
  └── data.yaml (classes: helmet_absent, helmet_present, seatbelt_absent, seatbelt_present)

Key requirements:
- Use ultralytics YOLOv8 (pretrained detection)
- Fine-tune ResNet50 from timm (violation classification)
- Process frames at 10+ FPS on GPU
- Output: annotated images + JSON metadata + CSV report
- Handle both video and image inputs
- Include data augmentation and validation metrics

Constraints:
- No external APIs, everything runs locally
- CPU fallback (slower)
- Code should work out-of-the-box with provided dataset
- Include error handling and logging

Example usage:
  python train.py --data dataset/ --epochs 30 --batch 32
  python infer.py --video sample.mp4 --output results/
  python generate_report.py --metadata results/metadata.json

Generate production-ready code now. Start with train.py.
```

---

## EXPECTED OUTPUTS (30 min later)

Claude will give you:
1. **train.py** (~200 lines) - ResNet50 fine-tuning
2. **infer.py** (~300 lines) - Batch inference on frames
3. **generate_report.py** (~200 lines) - Annotated output
4. **main.py** (~100 lines) - CLI integration
5. **requirements.txt** - Dependencies

Then immediately:
```bash
pip install -r requirements.txt
python train.py --data dataset/ --epochs 30
python infer.py --video sample.mp4 --output results/
```

---

## IF EVEN THAT IS TOO SLOW

Use this ultra-minimal version (no training, just inference):

```
Create a Python script that:
1. Loads YOLOv8n (pretrained)
2. Loads a pretrained ResNet50 from Hugging Face for violation detection
3. Processes video frames
4. Outputs annotated images + JSON

Code should be <500 lines, run in <5 min to first output.

Dataset: just images/test/ with labels/test/ (no training needed)
Usage: python detect.py --video sample.mp4 --output results/
```

This skips training entirely and assumes you find a pretrained violation classifier online.

```

