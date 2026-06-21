# Traffic Violation Detection — Web App

A self-contained, hostable web UI for the CCTV traffic-violation pipeline.
Analyze an **uploaded video** or a **YouTube URL** and get timestamped
violations with seekable evidence snapshots.

Detects: **no-helmet · no-seatbelt · triple-riding · red-light jump ·
wrong-side driving**, plus **license-plate OCR**.

## Run

```bash
./run.sh                 # → http://localhost:8000
PORT=9000 ./run.sh       # custom port
```

`run.sh` reuses the existing project venv if present, otherwise creates a local
`venv/` and installs `requirements.txt`. To host on another machine, copy this
folder and run the same command (use a Python ≤3.12 host to get PaddleOCR).

Manual / production:
```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000        # add --workers 1
```

## Features
- **Upload** (drag-drop) or **YouTube URL** (auto-downloaded ≤720p via yt-dlp).
- Live **progress** bar; jobs run in the background.
- **Seekable timeline** with colour-coded markers per violation.
- **Evidence gallery**: annotated snapshot + type + timestamp + confidence +
  plate for every detection; click to jump the video to that moment.
- **Summary** counts, per-type filters, and **plates read**.
- Speed/accuracy slider (frame-skip).

## How it works
`app.py` (FastAPI) accepts the source, runs `analyzer.analyze_video()` which
drives `pipelines.violation_detector.ViolationDetector` over the frames,
clusters detections into time-stamped clips, and saves annotated snapshots.
The bundled `models/` hold the trained weights (base YOLO11n, helmet, seatbelt,
triple-rider, red-light, license-plate detector).

## OCR note
- **Python ≤ 3.12** → PaddleOCR (PP-OCRv6) is used — accurate plate reads
  (e.g. `NA13NRU`). oneDNN is auto-disabled (`FLAGS_use_mkldnn=0`).
- **Python 3.13/3.14** → EasyOCR fallback (detects + reads, lower accuracy).

The pipeline auto-selects GPU (CUDA), Apple-Silicon MPS, or CPU.
