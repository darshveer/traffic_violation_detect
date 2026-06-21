"""Traffic Violation Detection — web app backend (FastAPI).

Run:
    python app.py                       # then open http://localhost:8000
    uvicorn app:app --host 0.0.0.0 --port 8000   # hostable

Features
- Analyze an uploaded video OR a YouTube URL (downloaded via yt-dlp).
- Runs the full ViolationDetector pipeline (helmet/seatbelt/triple-rider/
  red-light/wrong-side + license-plate OCR).
- Returns timestamped violation "clips" with evidence snapshots; the UI shows a
  seekable timeline and an evidence gallery.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make the bundled pipeline importable + fix macOS SSL for model/yt downloads.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from analyzer import analyze_video

# Runtime data dir (override with DATA_DIR; e.g. /tmp on read-only hosts like HF Spaces).
DATA = Path(os.environ.get("DATA_DIR", str(ROOT)))
UPLOADS = DATA / "uploads"
RESULTS = DATA / "results"
JOBS = DATA / "jobs"
for d in (UPLOADS, RESULTS, JOBS):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Traffic Violation Detection")

# ---- lazy singletons (load heavy model once) --------------------------------
_detector = None
_detector_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)   # serialize GPU jobs
JOBS_STATE: dict = {}


def get_detector():
    global _detector
    with _detector_lock:
        if _detector is None:
            from pipelines.violation_detector import ViolationDetector
            _detector = ViolationDetector(config_path=str(ROOT / "configs" / "pipeline.yaml"))
        return _detector


def _set(job_id, **kw):
    JOBS_STATE.setdefault(job_id, {}).update(kw)


def _save_job(job_id):
    (JOBS / f"{job_id}.json").write_text(json.dumps(JOBS_STATE[job_id], default=str))


# ---- background pipeline ----------------------------------------------------
def _download_youtube(url: str, dest: Path) -> Path:
    import yt_dlp
    opts = {
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        "outtmpl": str(dest.with_suffix(".%(ext)s")),
        "noplaylist": True, "quiet": True, "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():  # fall back to any produced file
        cands = list(dest.parent.glob(dest.stem + ".*"))
        path = cands[0] if cands else path
    return path


def _run_job(job_id: str, src_kind: str, src: str, skip: int):
    out_dir = RESULTS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if src_kind == "youtube":
            _set(job_id, status="downloading", message="Downloading video from YouTube…")
            video = _download_youtube(src, UPLOADS / job_id)
        else:
            video = Path(src)
        _set(job_id, status="analyzing", message="Loading models…", video=str(video))

        det = get_detector()
        _set(job_id, message="Analyzing video…",
             ocr_backend=getattr(det.ocr, "backend", "off"), device=det.device)

        def progress(done, total):
            _set(job_id, progress=round(100.0 * done / max(1, total), 1))

        result = analyze_video(str(video), str(out_dir), det, skip_frames=skip, progress_cb=progress)
        result["video_url"] = f"/api/jobs/{job_id}/video"
        if result.get("annotated"):
            result["annotated_url"] = f"/api/jobs/{job_id}/annotated"
        result["csv_url"] = f"/api/jobs/{job_id}/report.csv"
        result["pdf_url"] = f"/api/jobs/{job_id}/report.pdf"
        _set(job_id, status="done", progress=100.0, message="Done", result=result)
    except Exception as exc:  # noqa: BLE001
        import traceback
        _set(job_id, status="error", message=f"{type(exc).__name__}: {exc}",
             trace=traceback.format_exc())
    finally:
        _save_job(job_id)


# ---- API --------------------------------------------------------------------
@app.post("/api/analyze")
async def analyze(youtube_url: str = Form(None), skip_frames: int = Form(8),
                  file: UploadFile = File(None)):
    job_id = uuid.uuid4().hex[:12]
    skip = max(1, min(int(skip_frames), 30))
    if file is not None and file.filename:
        dest = UPLOADS / f"{job_id}_{Path(file.filename).name}"
        with open(dest, "wb") as fh:
            while chunk := await file.read(1 << 20):
                fh.write(chunk)
        _set(job_id, status="queued", progress=0.0, source=file.filename, kind="upload")
        _pool.submit(_run_job, job_id, "upload", str(dest), skip)
    elif youtube_url:
        _set(job_id, status="queued", progress=0.0, source=youtube_url, kind="youtube")
        _pool.submit(_run_job, job_id, "youtube", youtube_url, skip)
    else:
        raise HTTPException(400, "Provide a video file or a YouTube URL.")
    _save_job(job_id)
    return {"job_id": job_id}


def _get_state(job_id: str) -> dict:
    st = JOBS_STATE.get(job_id)
    if st is None:
        p = JOBS / f"{job_id}.json"
        if p.exists():
            st = json.loads(p.read_text())
        else:
            raise HTTPException(404, "Job not found")
    return st


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    return JSONResponse(_get_state(job_id))


@app.get("/api/jobs/{job_id}/annotated")
async def job_annotated(job_id: str):
    p = RESULTS / job_id / "annotated.mp4"
    if not p.exists():
        raise HTTPException(404, "Annotated video not found")
    return FileResponse(str(p))


@app.get("/api/jobs/{job_id}/report.csv")
async def job_csv(job_id: str):
    result = (_get_state(job_id) or {}).get("result")
    if not result:
        raise HTTPException(404, "No result yet")
    from report_export import to_csv
    return Response(to_csv(result), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="violations_{job_id}.csv"'})


@app.get("/api/jobs/{job_id}/report.pdf")
async def job_pdf(job_id: str):
    st = _get_state(job_id) or {}
    result = st.get("result")
    if not result:
        raise HTTPException(404, "No result yet")
    from report_export import build_pdf
    pdf = RESULTS / job_id / "report.pdf"
    if not pdf.exists():
        build_pdf(result, str(RESULTS / job_id), str(pdf), source=st.get("source", ""))
    return FileResponse(str(pdf), media_type="application/pdf",
                        filename=f"violations_{job_id}.pdf")


def _range_file(path: Path, request_range: str | None):
    return FileResponse(str(path))  # Starlette FileResponse supports Range


@app.get("/api/jobs/{job_id}/video")
async def job_video(job_id: str):
    st = JOBS_STATE.get(job_id) or {}
    vid = st.get("video")
    if not vid or not Path(vid).exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(vid)


@app.get("/api/jobs/{job_id}/frames/{name}")
async def job_frame(job_id: str, name: str):
    p = RESULTS / job_id / "frames" / Path(name).name
    if not p.exists():
        raise HTTPException(404, "Frame not found")
    return FileResponse(str(p))


@app.get("/", response_class=HTMLResponse)
async def index():
    return (ROOT / "static" / "index.html").read_text()


app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
