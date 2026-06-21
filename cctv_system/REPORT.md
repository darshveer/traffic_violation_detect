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
| **triple_rider** (new) | `YOLO.val()` on 65-img test split | P=0.821 R=0.552 F1=0.661 **mAP50=0.710** mAP50-95=0.630 |
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
- **Measured improvement** on the 1440p ANPR clip (EasyOCR on the Mac): plate localisation
  lifted the read-rate from **3.1% (garbage) → 17.5%** with a **1-epoch** detector, and
  consistent real plates began to appear across frames (e.g. `NAIJNRU`, `GXIS0GJ`). This climbs
  further as the detector trains to convergence and with **PaddleOCR** (better than EasyOCR for
  plates, available on the Py≤3.12 boxes). The detector continues training; pull the final
  `best.pt` from the box into `models/plugins/license_plate_detector.pt`.

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
