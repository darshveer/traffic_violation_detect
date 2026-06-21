"""CSV and PDF export of violation results for the web app."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Dict


def to_csv(result: Dict) -> str:
    """Return violation events as CSV text."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["#", "violation", "type", "start", "end", "peak_confidence",
                "frames", "plate"])
    for i, e in enumerate(result.get("events", []), 1):
        w.writerow([i, e["label"], e["type"], e["start_str"], e["end_str"],
                    f'{e["peak_conf"]:.3f}', e["count"], e.get("plate", "")])
    return buf.getvalue()


def build_pdf(result: Dict, out_dir: str, pdf_path: str, source: str = "") -> str:
    """Render a PDF report with summary + per-violation image, time & details."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image, HRFlowable)

    out_dir = Path(out_dir)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, spaceAfter=2)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=9, textColor=colors.grey)
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=10, leading=14)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm)
    story = []
    meta, summ = result.get("meta", {}), result.get("summary", {})
    story.append(Paragraph("Traffic Violation Report", h1))
    story.append(Paragraph(
        f"Source: {source or 'video'} &nbsp;|&nbsp; Duration: {meta.get('duration','?')}s "
        f"&nbsp;|&nbsp; {meta.get('width')}×{meta.get('height')} @ {meta.get('fps')}fps "
        f"&nbsp;|&nbsp; Device: {summ.get('device')} &nbsp;|&nbsp; OCR: {summ.get('ocr_backend')}", small))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", color=colors.lightgrey))
    story.append(Spacer(1, 8))

    # summary table
    by = summ.get("by_type", {})
    rows = [["Violation", "Events", "Frames"]] + [
        [o["label"], str(o["events"]), str(o["frames"])] for o in by.values()]
    rows.append(["TOTAL", str(summ.get("total_events", 0)), ""])
    t = Table(rows, colWidths=[80 * mm, 30 * mm, 30 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3658")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#eef1f8")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f7f8fc")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    if summ.get("top_plates"):
        plates = ", ".join(f'{p["plate"]} (×{p["count"]})' for p in summ["top_plates"])
        story.append(Paragraph(f"<b>Plates read:</b> {plates}", cell))
        story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Detections</b>", styles["Heading2"]))
    story.append(Spacer(1, 4))

    # per-event rows: [snapshot image | details]
    for i, e in enumerate(result.get("events", []), 1):
        img_path = out_dir / e["snapshot"]
        img_flow = ""
        if img_path.exists():
            try:
                iw, ih = ImageReader(str(img_path)).getSize()
                w = 62 * mm
                img_flow = Image(str(img_path), width=w, height=w * ih / iw)
            except Exception:
                img_flow = ""
        details = Paragraph(
            f'<b>{i}. {e["label"]}</b><br/>'
            f'Time: {e["start_str"]}' + (f' – {e["end_str"]}' if e["end_str"] != e["start_str"] else "") +
            f'<br/>Confidence: {e["peak_conf"]*100:.0f}%<br/>'
            f'Frames: {e["count"]}' + (f'<br/>Plate: <b>{e["plate"]}</b>' if e.get("plate") else ""),
            cell)
        row = Table([[img_flow, details]], colWidths=[66 * mm, 96 * mm])
        row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#e3e6ef")),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(row)

    if not result.get("events"):
        story.append(Paragraph("No violations detected.", cell))

    doc.build(story)
    return str(pdf_path)
