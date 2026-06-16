"""Unit tests for report generation (flattening, summary, CSV, heatmap)."""

import csv
from pathlib import Path

from evidence.report_generator import (
    CSV_HEADER,
    compute_summary,
    generate_reports,
    records_to_rows,
    write_csv,
)


def _sample_records():
    return [
        {
            "frame_id": 0,
            "timestamp": "2024-12-05T10:30:01",
            "summary": {"license_plate": "KA01AB1234"},
            "violations": [
                {"type": "helmet_absent", "confidence": 0.87, "box": [100, 150, 150, 200],
                 "plate": "KA01AB1234", "is_violation": True},
                {"type": "helmet_present", "confidence": 0.9, "box": [10, 10, 20, 20],
                 "is_violation": False},  # should be skipped
            ],
        },
        {
            "frame_id": 5,
            "timestamp": "2024-12-05T11:30:02",
            "summary": {"license_plate": ""},
            "violations": [
                {"type": "seatbelt_absent", "confidence": 0.92, "box": [200, 180, 280, 300],
                 "is_violation": True},
            ],
        },
    ]


def test_records_to_rows_skips_present_observations():
    rows = records_to_rows(_sample_records())
    assert len(rows) == 2
    types = {r["violation_type"] for r in rows}
    assert types == {"helmet_absent", "seatbelt_absent"}


def test_records_to_rows_carries_plate():
    rows = records_to_rows(_sample_records())
    helmet_row = next(r for r in rows if r["violation_type"] == "helmet_absent")
    assert helmet_row["license_plate"] == "KA01AB1234"
    assert helmet_row["bbox_x1"] == 100 and helmet_row["bbox_y2"] == 200


def test_compute_summary_counts_and_percentages():
    rows = records_to_rows(_sample_records())
    summary = compute_summary(rows)
    assert summary["total"] == 2
    assert summary["by_type"]["helmet_absent"]["count"] == 1
    assert summary["by_type"]["helmet_absent"]["percentage"] == 50.0


def test_compute_summary_peak_hours():
    rows = records_to_rows(_sample_records())
    summary = compute_summary(rows)
    hours = {h["hour"] for h in summary["peak_hours"]}
    assert hours == {10, 11}


def test_write_csv_roundtrip(tmp_path):
    rows = records_to_rows(_sample_records())
    out = tmp_path / "report.csv"
    write_csv(rows, out)
    with open(out, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CSV_HEADER
        loaded = list(reader)
    assert len(loaded) == 2


def test_generate_reports_writes_all_artifacts(tmp_path):
    paths = generate_reports(_sample_records(), str(tmp_path), frame_size=(640, 480))
    for key in ("csv", "metadata", "summary", "heatmap"):
        assert Path(paths[key]).exists(), f"missing {key}"
    # heatmap PNG is produced alongside the HTML
    assert (tmp_path / "heatmap.png").exists()


def test_generate_reports_handles_empty_records(tmp_path):
    paths = generate_reports([], str(tmp_path), frame_size=(640, 480))
    assert Path(paths["csv"]).exists()
    assert Path(paths["heatmap"]).exists()
