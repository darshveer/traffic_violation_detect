---
title: Traffic Violation Detection
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

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

## Deploying to your own server / website
The model weights in `models/` are **not** committed to the main git repo (too
large — GitHub push timed out at ~100 MB). They live on disk in this folder, so
deploy by **copying the whole `final_app/` folder** (rsync/scp/zip) to your host,
then `./run.sh`. Put it behind nginx/Caddy as a reverse proxy to `127.0.0.1:8000`
and add HTTPS for a public site.

## Deploy to Hugging Face Spaces (Docker)
This folder is Space-ready: the YAML header sets `sdk: docker`, the `Dockerfile`
installs ffmpeg + CPU torch + PaddleOCR and serves on **port 7860**. On Spaces'
Python 3.10 the more accurate **PaddleOCR** engine is used automatically.

The weights must be uploaded to the Space (they're large → use Git LFS):
```bash
# from inside final_app/  (after creating a Docker Space named <user>/<space>)
git init && git lfs install && git lfs track "*.pt"
huggingface-cli login
git remote add hf https://huggingface.co/spaces/<user>/<space>
git add -A && git commit -m "deploy" && git push hf main
```
Or upload `models/` via the Space's web UI. The container caches go to `/tmp`
and runtime data to `/code` (made writable in the Dockerfile).

## OCR note
- **Python ≤ 3.12** → PaddleOCR (PP-OCRv6) is used — accurate plate reads
  (e.g. `NA13NRU`). oneDNN is auto-disabled (`FLAGS_use_mkldnn=0`).
- **Python 3.13/3.14** → EasyOCR fallback (detects + reads, lower accuracy).

The pipeline auto-selects GPU (CUDA), Apple-Silicon MPS, or CPU.
