"""Generate court-ready evidence reports from inference records.

Produces, for a set of per-frame inference records:

- ``violations_report.csv`` : one row per violation event
- ``violations_metadata.json`` : full structured dump
- ``summary_stats.json`` : per-type counts/percentages, peak hours, hot locations
- ``heatmap.html`` (+ ``heatmap.png``) : spatial distribution of violations

The ``evaluate`` helper runs Ultralytics validation on a fine-tuned model to
emit a confusion matrix, PR curve and ``metrics.json`` (precision/recall/F1/mAP).
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cctv")

CSV_HEADER = [
    "frame_id",
    "timestamp",
    "violation_type",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "license_plate",
]


# --------------------------------------------------------------------------- #
# Flattening records -> rows
# --------------------------------------------------------------------------- #
def records_to_rows(records: List[Dict]) -> List[Dict]:
    """Flatten per-frame records into one row per violation event.

    Parameters
    ----------
    records : list of dict
        Per-frame results (each with ``frame_id``, ``timestamp``, ``violations``,
        and optionally a top-level ``summary.license_plate``).

    Returns
    -------
    list of dict
        Rows keyed by :data:`CSV_HEADER`.
    """
    rows: List[Dict] = []
    for rec in records:
        plate = rec.get("summary", {}).get("license_plate", "")
        for v in rec.get("violations", []):
            if not v.get("is_violation", True):
                continue  # skip "*_present" observations
            box = v.get("box", [0, 0, 0, 0])
            rows.append(
                {
                    "frame_id": rec.get("frame_id", 0),
                    "timestamp": rec.get("timestamp", ""),
                    "violation_type": v.get("type", ""),
                    "confidence": round(float(v.get("confidence", 0.0)), 4),
                    "bbox_x1": int(box[0]),
                    "bbox_y1": int(box[1]),
                    "bbox_x2": int(box[2]),
                    "bbox_y2": int(box[3]),
                    "license_plate": v.get("plate") or plate or "",
                }
            )
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    """Write violation rows to a CSV file with the standard header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote CSV report: %s (%d rows).", path, len(rows))


def write_json(data: Any, path: Path) -> None:
    """Write any JSON-serializable object to ``path`` (pretty-printed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Wrote JSON: %s", path)


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
def _extract_hour(timestamp: str) -> Optional[int]:
    """Best-effort hour extraction from an ISO or ``HH:MM:SS`` timestamp."""
    if not timestamp:
        return None
    ts = timestamp.strip()
    try:
        if "T" in ts:  # ISO datetime
            return int(ts.split("T")[1][:2])
        if ":" in ts:  # HH:MM:SS elapsed
            head = ts.split(",")[-1].strip() if "day" in ts else ts
            return int(head.split(":")[0]) % 24
    except (ValueError, IndexError):
        return None
    return None


def compute_summary(rows: List[Dict], grid: int = 6) -> Dict[str, Any]:
    """Compute aggregate statistics over violation rows.

    Parameters
    ----------
    rows : list of dict
        Flattened violation rows (see :func:`records_to_rows`).
    grid : int
        Number of cells per axis when bucketing locations.

    Returns
    -------
    dict
        ``by_type``, ``peak_hours``, ``top_locations`` and ``total``.
    """
    total = len(rows)
    type_counts = Counter(r["violation_type"] for r in rows)
    by_type = {
        t: {"count": c, "percentage": round(100.0 * c / total, 2) if total else 0.0}
        for t, c in type_counts.most_common()
    }

    hour_counts: Counter = Counter()
    for r in rows:
        hr = _extract_hour(str(r.get("timestamp", "")))
        if hr is not None:
            hour_counts[hr] += 1
    peak_hours = [
        {"hour": h, "count": c} for h, c in sorted(hour_counts.items(), key=lambda x: -x[1])[:5]
    ]

    loc_counts: defaultdict = defaultdict(int)
    for r in rows:
        cx = (r["bbox_x1"] + r["bbox_x2"]) / 2.0
        cy = (r["bbox_y1"] + r["bbox_y2"]) / 2.0
        # Bucket assumes 1920x1080-ish; uses fractional cell of max observed.
        loc_counts[(int(cx // max(1, 1920 // grid)), int(cy // max(1, 1080 // grid)))] += 1
    top_locations = [
        {"cell": f"{gx},{gy}", "count": c}
        for (gx, gy), c in sorted(loc_counts.items(), key=lambda x: -x[1])[:10]
    ]

    return {
        "total": total,
        "by_type": by_type,
        "peak_hours": peak_hours,
        "top_locations": top_locations,
    }


# --------------------------------------------------------------------------- #
# Heatmap
# --------------------------------------------------------------------------- #
def generate_heatmap(
    rows: List[Dict],
    out_html: Path,
    frame_size: tuple[int, int] = (1920, 1080),
    bins: int = 48,
) -> None:
    """Render an interactive spatial heatmap of violation centres.

    Tries Plotly first (hover info: violation type, count, timestamp).
    Falls back to a matplotlib PNG embedded in HTML if Plotly is absent.

    Parameters
    ----------
    rows : list of dict
        Violation rows (from :func:`records_to_rows`).
    out_html : Path
        Output HTML path; a PNG is written alongside for the matplotlib fallback.
    frame_size : tuple[int, int]
        ``(width, height)`` used to scale the plane.
    bins : int
        Histogram resolution per axis.
    """
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # ── Try Plotly (interactive, hover info) ─────────────────────────────────
    try:
        import plotly.graph_objects as go  # type: ignore[import]
        _generate_heatmap_plotly(rows, out_html, frame_size, bins)
        return
    except ImportError:
        logger.debug("plotly not installed; falling back to matplotlib heatmap.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Plotly heatmap failed (%s); falling back to matplotlib.", exc)

    # ── Fallback: matplotlib PNG-in-HTML ─────────────────────────────────────
    _generate_heatmap_matplotlib(rows, out_html, frame_size, bins)


def _generate_heatmap_plotly(
    rows: List[Dict],
    out_html: Path,
    frame_size: tuple[int, int],
    bins: int,
) -> None:
    """Interactive Plotly density heatmap with per-cell hover tooltips.

    Each cell in the histogram grid shows:
    - Total violation count
    - Breakdown by violation type (e.g. helmet_absent×3, triple_rider×1)
    - First/last timestamp in the cell
    """
    import plotly.graph_objects as go  # type: ignore[import]
    import numpy as np

    w, h = frame_size
    xs = np.array([(r["bbox_x1"] + r["bbox_x2"]) / 2.0 for r in rows], dtype=np.float64)
    ys = np.array([(r["bbox_y1"] + r["bbox_y2"]) / 2.0 for r in rows], dtype=np.float64)

    # Build per-cell metadata for hover text.
    cell_w = w / bins
    cell_h = h / bins
    from collections import defaultdict, Counter
    cell_data: dict = defaultdict(lambda: {"types": [], "timestamps": []})
    for r in rows:
        cx = (r["bbox_x1"] + r["bbox_x2"]) / 2.0
        cy = (r["bbox_y1"] + r["bbox_y2"]) / 2.0
        gi = min(int(cx / cell_w), bins - 1)
        gj = min(int(cy / cell_h), bins - 1)
        cell_data[(gi, gj)]["types"].append(r.get("violation_type", "unknown"))
        cell_data[(gi, gj)]["timestamps"].append(str(r.get("timestamp", "")))

    if len(xs) == 0:
        # Empty — write a placeholder page.
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Violation Heatmap</title></head><body>"
            "<h2>Violation Spatial Heatmap</h2><p>No violations detected.</p>"
            "</body></html>"
        )
        out_html.write_text(html, encoding="utf-8")
        logger.info("Wrote empty heatmap: %s", out_html)
        return

    # 2-D histogram: count per cell.
    counts, xedges, yedges = np.histogram2d(xs, ys, bins=bins,
                                             range=[[0, w], [0, h]])
    counts = counts.T  # transpose so rows=y, cols=x (image convention)

    # Build hover text grid.
    hover = np.full(counts.shape, "", dtype=object)
    for (gi, gj), cdata in cell_data.items():
        if gi >= bins or gj >= bins:
            continue
        type_counts = Counter(cdata["types"])
        breakdown = "<br>".join(f"  {t}: {c}" for t, c in type_counts.most_common())
        ts_sample = cdata["timestamps"][0] if cdata["timestamps"] else ""
        hover[gj, gi] = (
            f"Cell ({gi},{gj})<br>"
            f"Total: {int(counts[gj, gi])}<br>"
            f"{breakdown}<br>"
            f"First seen: {ts_sample}"
        )

    fig = go.Figure(
        data=go.Heatmap(
            z=counts,
            x=xedges[:-1],
            y=yedges[:-1],
            colorscale="Inferno",
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorbar={"title": "Violation count"},
        )
    )
    fig.update_layout(
        title=f"Violation Spatial Density — {len(rows)} violations",
        xaxis_title="x (px)",
        yaxis_title="y (px)",
        yaxis_autorange="reversed",   # image origin top-left
        template="plotly_dark",
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    logger.info("Wrote interactive heatmap (Plotly): %s", out_html)


def _generate_heatmap_matplotlib(
    rows: List[Dict],
    out_html: Path,
    frame_size: tuple[int, int],
    bins: int,
) -> None:
    """Fallback: static matplotlib 2-D histogram embedded as PNG in HTML."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        logger.warning("matplotlib/numpy unavailable; skipping heatmap (%s).", exc)
        return

    png_path = out_html.with_suffix(".png")
    xs = [(r["bbox_x1"] + r["bbox_x2"]) / 2.0 for r in rows]
    ys = [(r["bbox_y1"] + r["bbox_y2"]) / 2.0 for r in rows]
    w, h = frame_size

    fig, ax = plt.subplots(figsize=(10, 6))
    if xs:
        _, _, _, im = ax.hist2d(xs, ys, bins=bins, range=[[0, w], [0, h]], cmap="inferno")
        fig.colorbar(im, ax=ax, label="violation count")
    else:
        ax.text(0.5, 0.5, "No violations", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Violation spatial density (image plane)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Violation Heatmap</title></head><body>"
        "<h2>Violation Spatial Heatmap</h2>"
        f"<p>Total violations: {len(rows)}</p>"
        f"<img src='data:image/png;base64,{encoded}' style='max-width:100%'/>"
        "</body></html>"
    )
    out_html.write_text(html, encoding="utf-8")
    logger.info("Wrote heatmap (matplotlib fallback): %s", out_html)



# --------------------------------------------------------------------------- #
# Top-level report orchestration
# --------------------------------------------------------------------------- #
def generate_reports(
    records: List[Dict],
    output_dir: str,
    frame_size: tuple[int, int] = (1920, 1080),
) -> Dict[str, str]:
    """Generate CSV, JSON, summary and heatmap from inference records.

    Parameters
    ----------
    records : list of dict
        Per-frame inference results.
    output_dir : str
        Destination directory.
    frame_size : tuple[int, int]
        Frame ``(width, height)`` for heatmap scaling.

    Returns
    -------
    dict
        Mapping of artefact name -> written path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = records_to_rows(records)

    csv_path = out / "violations_report.csv"
    meta_path = out / "violations_metadata.json"
    summary_path = out / "summary_stats.json"
    heatmap_path = out / "heatmap.html"

    write_csv(rows, csv_path)
    write_json({"records": records, "violations": rows}, meta_path)
    write_json(compute_summary(rows), summary_path)
    generate_heatmap(rows, heatmap_path, frame_size=frame_size)

    return {
        "csv": str(csv_path),
        "metadata": str(meta_path),
        "summary": str(summary_path),
        "heatmap": str(heatmap_path),
    }


# --------------------------------------------------------------------------- #
# Evaluation (mAP / PR-curve / confusion matrix) via Ultralytics
# --------------------------------------------------------------------------- #
def evaluate(model_path: str, data_yaml: str, output_dir: str, device: str = "auto") -> Dict[str, Any]:
    """Validate a fine-tuned detector and emit standard metric artefacts.

    Runs Ultralytics ``model.val()`` which writes a confusion matrix and PR
    curve, then copies the key plots into ``output_dir`` and saves a
    ``metrics.json`` with precision/recall/F1/mAP.

    Parameters
    ----------
    model_path : str
        Path to a fine-tuned ``.pt`` model.
    data_yaml : str
        Dataset YAML (must define a ``test`` or ``val`` split).
    output_dir : str
        Destination directory for metrics artefacts.
    device : str
        Device string (``auto``/``cuda``/``cpu``).

    Returns
    -------
    dict
        The metrics dictionary (also written to ``metrics.json``).
    """
    import shutil

    from pipelines.common import select_device

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = select_device(device)

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required for evaluation") from exc

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split="test", device=dev, plots=True)

    p = float(metrics.box.mp)
    r = float(metrics.box.mr)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    out_metrics = {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "per_class_map50": [round(float(x), 4) for x in getattr(metrics.box, "maps", [])],
    }
    write_json(out_metrics, out / "metrics.json")

    # Copy Ultralytics-generated plots into our metrics dir.
    save_dir = Path(getattr(metrics, "save_dir", "")) if getattr(metrics, "save_dir", "") else None
    if save_dir and save_dir.exists():
        for name, dst in (
            ("confusion_matrix.png", "confusion_matrix.png"),
            ("PR_curve.png", "precision_recall_curve.png"),
        ):
            src = save_dir / name
            if src.exists():
                shutil.copy(src, out / dst)
    logger.info("Evaluation metrics: %s", out_metrics)
    return out_metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_records_from_metadata(path: str) -> List[Dict]:
    """Load records from a metadata JSON (``{"records": [...]}``) or a JSONL file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized metadata format: {path}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: regenerate reports from an existing metadata file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Generate violation evidence reports.")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON/JSONL of records.")
    parser.add_argument("--output", default="results/", help="Output directory.")
    parser.add_argument("--width", type=int, default=1920, help="Frame width for heatmap scaling.")
    parser.add_argument("--height", type=int, default=1080, help="Frame height for heatmap scaling.")
    args = parser.parse_args(argv)

    records = _load_records_from_metadata(args.metadata)
    paths = generate_reports(records, args.output, frame_size=(args.width, args.height))
    logger.info("Reports written: %s", paths)


if __name__ == "__main__":
    main()
