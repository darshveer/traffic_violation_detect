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
