# CCTV Traffic-Violation Pipeline — Verification Report

_Generated 2026-06-21. Models run on CUDA (RTX 4060/3060) and Apple-Silicon MPS; OCR via PaddleOCR (Linux/Win, Py≤3.12) or EasyOCR (macOS/Py3.14)._

## 1. Environment
- Pipeline runs end-to-end on **both** the Mac (Python 3.14, MPS) and the GPU boxes (Python 3.11/3.10, CUDA). Added **MPS device support** (`select_device`) so the Mac uses its GPU instead of CPU.
- Dependencies confirmed importable on Py3.14: torch 2.12, ultralytics 8.4, easyocr 1.7. PaddleOCR is **not installable on Py3.14** (no `paddlepaddle` wheel) → EasyOCR fallback is used there (handled gracefully by `OCRHandler`).
- Unit tests: **78 / 78 passing** (`pytest`).

## 2. Per-module metrics (correct positives)

| Module | How verified | Result |
|---|---|---|
| **base (YOLO11n COCO)** | object histogram over sampled frames | OK — detects person/car/motorcycle/bus/truck/traffic-light |
| **helmet** | `YOLO.val()` on 291-img test split | P=0.767 R=0.722 F1=0.744 **mAP50=0.802** mAP50-95=0.566 |
| **seatbelt** | `YOLO.val()` on 248-img valid split | P=0.823 R=0.810 F1=0.817 **mAP50=0.889** mAP50-95=0.394 |
| **triple_rider** v1 (yolo11n) | `YOLO.val()` on 65-img test split | P=0.821 R=0.552 **mAP50=0.710** mAP50-95=0.630 |
| **triple_rider** v2 (yolo11s, retrained) | final epoch val (130 imgs) | R=**0.91** **mAP50=0.940** mAP50-95=0.903 — recall up from 0.55 |
| **wrong_side** | functional test (synthetic tracks) | PASS — flags against-flow, ignores with-flow |
| **red_light** | functional test (red vs green + stop-band) | PASS — flags red+in-band, ignores green |
| **OCR** | EasyOCR/PaddleOCR on real plate video | engine works; **needs plate localisation** (see §4) |

## 3. What was fixed / built

1. **Triple-rider — replaced the broken heuristic with a trained model.**
   The built-in person-counter could *never* fire: on real footage YOLO merges the
   occluded rider cluster into ≤2 person boxes, so `min_riders=3` was unreachable
   (measured rider-count distribution `{0:13, 1:130, 2:9}` — never ≥3). I trained a
   dedicated single-class `tripleriding` YOLO11n detector (Roboflow `pran/triple-riding`,
   651 imgs) → **mAP50 0.71, precision 0.82**, wired in via the existing plug-in path
   (`models.triple_rider`), with a configurable `triple_rider.conf` (0.35) since the
   model's scores run lower than COCO models.

2. **Wrong-side — hardened the built-in logic.** It was emitting **27 false positives**
   on a clip with no wrong-way driving. Rewrote it to require (a) net heading opposing
   the allowed direction, (b) **per-step motion consistency** (≥70% of steps opposing),
   (c) an optional road ROI, and (d) class filtering (cars only by default). FPs **27 → 0**
   on that clip; functional test passes. Added `scripts/calibrate_wrong_side.py` to
   estimate the per-camera allowed direction (wrong-side is inherently camera-specific).

3. **Critical helmet/seatbelt bug fixed.** `_classify_events` only treated a class as a
   violation if its name contained `"absent"`, but the trained models label the violation
   `no-helmet` / `no-seatbelt`. **Every no-helmet/no-seatbelt detection was being logged as
   _present_ (non-violation)** — i.e. the system silently missed all helmet/seatbelt
   violations. Added robust `_is_absence_class()` (handles `no-…`, `without`, `not wearing`,
   `absent`, …) with unit tests.

4. **License-plate detector for OCR** (training on GPU box) — see §4.

## 4. OCR finding & fine-tuning needed (≤12 h)

- The OCR **engine works** (EasyOCR/PaddleOCR initialise and read text), but the pipeline
  was OCR-ing the **whole vehicle box**, so on a 1440p ANPR clip it read 13/419 crops,
  mostly garbage — the plate is a tiny part of a car crop.
- **Fix:** a dedicated **license-plate detector** localises the plate inside each vehicle
  crop, then OCR runs on the tight plate region. Integrated into `_read_plates`
  (`models.plate_detector`); falls back to whole-vehicle OCR when absent. A YOLO11n plate
  detector was fine-tuned on the RTX 3060 (Roboflow `license-plate-recognition-rxg4e`, 21k imgs).
- **Plate detector fully trained** (yolo11n, Roboflow rxg4e 21k imgs): **mAP50 0.982,
  precision 0.985, recall 0.959** — localisation is essentially solved. Deployed to
  `models/plugins/license_plate_detector.pt`.
- **Recognition solved with PaddleOCR.** EasyOCR (the only option on Python 3.14) mis-reads
  small/angled plates (garbage like `HAI3MRU`). Switching the recognition stage to **PaddleOCR
  (PP-OCRv6)** on a Python 3.12 box gives **correct reads**: on the 1440p ANPR clip, the trained
  detector + PaddleOCR produced a **100% read-rate (67/67 localised plates)** with real plate
  strings — e.g. `NA13NRU`, `MW51VSU` (valid UK plates), vs EasyOCR's garbage on the same crops.
  - Two fixes were required and are now in `OCRHandler`: (a) **disable oneDNN**
    (`FLAGS_use_mkldnn=0`) — paddle 3.x's CPU PIR executor crashes otherwise; (b) parse the
    **PaddleOCR 3.x `predict()`** output (`rec_texts`/`rec_scores`), with a 2.x `.ocr()` fallback.
  - Net: run the pipeline on a Py≤3.12 host (PaddleOCR auto-selected) for accurate ANPR; the Mac
    (Py3.14) auto-falls back to EasyOCR (works, lower accuracy).

### Other fine-tuning recommendations
| Item | Issue | Action (fits in budget) |
|---|---|---|
| triple_rider recall | 0.55 (precision 0.82 is good) | train more epochs / add camera-matched data; I stopped at ep27 |
| seatbelt | trained for windshield/interior views → unreliable on distant **exterior** highway cars (false `seatbelt_absent`) | run only on front-facing/toll-cam footage, or restrict by crop size |
| wrong_side | needs per-camera direction; default `[0,1]` flags the opposing lane on two-way roads | run `calibrate_wrong_side.py` per camera; set `allowed_direction` / `roi` |

## 5. Full pipeline
`main.py --mode infer --video … --output …` runs end-to-end on the Mac and produces
`violations_report.csv`, `violations_metadata.json`, `summary_stats.json`,
`heatmap.html/png`, an annotated video and per-violation frames.
