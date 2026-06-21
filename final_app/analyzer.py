"""Video analysis core for the web app.

Wraps the ViolationDetector pipeline to process a video, returning timestamped
violation events (clustered into clips) plus annotated evidence snapshots, with
a progress callback for the UI.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from pipelines.violation_detector import ViolationDetector
from pipelines.inference_utils import iter_video_frames, video_metadata
from evidence.annotate import annotate_frame

# Human-friendly labels + colours (hex) per violation type.
VIOLATION_META = {
    "helmet_absent":       {"label": "No Helmet",        "color": "#ef4444"},
    "seatbelt_absent":     {"label": "No Seatbelt",      "color": "#f97316"},
    "triple_rider":        {"label": "Triple Riding",    "color": "#a855f7"},
    "red_light_violation": {"label": "Red-Light Jump",   "color": "#dc2626"},
    "wrong_side_driving":  {"label": "Wrong-Side Driving","color": "#0ea5e9"},
}


def _fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _encode_h264(raw: Path, out: Path) -> bool:
    """Transcode the raw annotated video to browser-playable H.264 via ffmpeg.

    Falls back to the raw mp4v file if ffmpeg is unavailable. Returns True if an
    annotated video exists at ``out``.
    """
    ff = shutil.which("ffmpeg")
    if ff and raw.exists():
        try:
            subprocess.run(
                [ff, "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", "-loglevel", "error", str(out)],
                check=True,
            )
            raw.unlink(missing_ok=True)
            return True
        except Exception:
            pass
    if raw.exists():  # no ffmpeg: serve the raw file (may not play in all browsers)
        raw.replace(out)
        return True
    return False


def analyze_video(
    video_path: str,
    out_dir: str,
    detector: ViolationDetector,
    skip_frames: int = 8,
    cluster_gap: float = 1.5,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """Process ``video_path`` and return events + metadata.

    Events are clustered per violation type: consecutive detections of the same
    type within ``cluster_gap`` seconds are merged into one evidence clip with a
    start/end time, peak confidence, plate and a snapshot image.
    """
    meta = video_metadata(video_path)
    fps = meta["fps"] or 30.0
    total_kept = max(1, meta["frame_count"] // max(1, skip_frames))

    frames_dir = Path(out_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Annotated video: kept frames written at fps/skip -> SAME duration as the
    # source, so the UI timeline stays aligned. Encoded H.264 below for browsers.
    out_fps = max(1.0, fps / max(1, skip_frames))
    raw_path = Path(out_dir) / "annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (meta["width"], meta["height"]))

    open_clusters: Dict[str, Dict] = {}
    events: List[Dict] = []
    counts: Dict[str, int] = {}
    plates: Dict[str, int] = {}
    processed = 0

    def _close(ctype: str):
        if ctype in open_clusters:
            events.append(open_clusters.pop(ctype))

    for idx, frame, ts_sec in iter_video_frames(video_path, skip_frames, 0):
        result = detector.infer_frame(frame, frame_id=idx, timestamp=_fmt(ts_sec), use_tracking=True)
        active = [v for v in result["violations"] if v.get("is_violation", True)]

        # plate tally (from summary)
        plate = result["summary"].get("license_plate") or ""
        if plate:
            plates[plate] = plates.get(plate, 0) + 1

        # one annotated frame per kept frame -> annotated video + reused snapshots
        ann = annotate_frame(frame, result["detections"], _fmt(ts_sec), active)
        writer.write(ann)

        seen_types = set()
        for v in active:
            vtype = v["type"]
            seen_types.add(vtype)
            counts[vtype] = counts.get(vtype, 0) + 1
            cl = open_clusters.get(vtype)
            if cl and (ts_sec - cl["_last"]) <= cluster_gap:
                cl["end"] = ts_sec
                cl["_last"] = ts_sec
                cl["count"] += 1
                if v["confidence"] > cl["peak_conf"]:
                    cl["peak_conf"] = v["confidence"]
                if v.get("plate") and not cl.get("plate"):
                    cl["plate"] = v["plate"]
            else:
                _close(vtype)
                snap = f"frames/{vtype}_{idx:06d}.jpg"
                cv2.imwrite(str(Path(out_dir) / snap), ann)
                open_clusters[vtype] = {
                    "type": vtype,
                    "label": VIOLATION_META.get(vtype, {}).get("label", vtype),
                    "color": VIOLATION_META.get(vtype, {}).get("color", "#64748b"),
                    "start": ts_sec, "end": ts_sec, "_last": ts_sec,
                    "peak_conf": v["confidence"], "count": 1,
                    "plate": v.get("plate") or plate or "",
                    "snapshot": snap,
                }
        # close clusters whose type wasn't seen and that are stale
        for ctype in list(open_clusters):
            if ctype not in seen_types and (ts_sec - open_clusters[ctype]["_last"]) > cluster_gap:
                _close(ctype)

        processed += 1
        if progress_cb and processed % 3 == 0:
            progress_cb(processed, total_kept)

    for ctype in list(open_clusters):
        _close(ctype)
    writer.release()
    annotated_ok = _encode_h264(raw_path, Path(out_dir) / "annotated.mp4")
    events.sort(key=lambda e: e["start"])

    # finalize event display fields
    for e in events:
        e["start_str"] = _fmt(e["start"])
        e["end_str"] = _fmt(e["end"])
        e["peak_conf"] = round(e["peak_conf"], 3)
        e.pop("_last", None)

    if progress_cb:
        progress_cb(total_kept, total_kept)

    top_plates = sorted(plates.items(), key=lambda x: -x[1])[:10]
    return {
        "annotated": bool(annotated_ok),
        "meta": {
            "fps": round(fps, 2),
            "width": meta["width"], "height": meta["height"],
            "frame_count": meta["frame_count"],
            "duration": round(meta["frame_count"] / fps, 1),
            "skip_frames": skip_frames,
        },
        "summary": {
            "total_events": len(events),
            "by_type": {
                t: {"label": VIOLATION_META.get(t, {}).get("label", t),
                    "color": VIOLATION_META.get(t, {}).get("color", "#64748b"),
                    "events": sum(1 for e in events if e["type"] == t),
                    "frames": counts.get(t, 0)}
                for t in sorted(counts)
            },
            "top_plates": [{"plate": p, "count": c} for p, c in top_plates],
            "ocr_backend": getattr(detector.ocr, "backend", "off"),
            "device": detector.device,
        },
        "events": events,
    }
