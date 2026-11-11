"""PDF, CSV, and JSON report generation for change-detection results."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_csv_report(result_data: dict[str, Any]) -> bytes:
    """Serialize quantitative prediction & geospatial summary metrics into CSV bytes."""
    geo = result_data.get("geospatial", {})
    rows = [
        ("result_id", result_data.get("result_id")),
        ("model_name", result_data.get("model_name")),
        ("training_status", result_data.get("training_status")),
        ("threshold", result_data.get("threshold")),
        ("changed_pixels", result_data.get("changed_pixels")),
        ("total_valid_pixels", result_data.get("total_valid_pixels")),
        ("masked_invalid_pixels", result_data.get("masked_invalid_pixels")),
        ("change_percentage", result_data.get("change_percentage")),
        ("inference_time_sec", result_data.get("inference_time_sec")),
        ("is_georeferenced", geo.get("is_georeferenced")),
        ("crs", geo.get("crs")),
        ("area_calculation_method", geo.get("area_calculation_method")),
        ("changed_area_m2", geo.get("changed_area_m2")),
        ("changed_area_ha", geo.get("changed_area_ha")),
        ("changed_area_km2", geo.get("changed_area_km2")),
        ("num_detected_regions", len(result_data.get("regions", []))),
    ]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["metric", "value"])
    for k, v in rows:
        writer.writerow([k, "" if v is None else v])
    return buf.getvalue().encode("utf-8")


def generate_pdf_report(
    result_data: dict[str, Any],
    result_dir: Path,
    output_pdf_path: Path,
) -> Path:
    """Create a structured PDF analysis report using ReportLab."""
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    story.append(
        Paragraph("Vision Transformers for Satellite Land-Use Monitoring — Analysis Report", styles["Title"])
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            f"<b>Result ID:</b> {result_data.get('result_id')}<br/>"
            f"<b>Model:</b> {result_data.get('model_name')} "
            f"(Status: {result_data.get('training_status')})<br/>"
            f"<b>Confidence Threshold:</b> {result_data.get('threshold')}<br/>"
            f"<b>Inference Duration:</b> {result_data.get('inference_time_sec')} s",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    geo = result_data.get("geospatial", {})
    table_data = [
        ["Metric", "Value"],
        ["Changed Pixels", f"{result_data.get('changed_pixels', 0):,}"],
        ["Total Valid Pixels", f"{result_data.get('total_valid_pixels', 0):,}"],
        ["Masked / Cloud Pixels", f"{result_data.get('masked_invalid_pixels', 0):,}"],
        ["Change Percentage (%)", f"{result_data.get('change_percentage', 0.0):.4f}%"],
        ["Georeferenced", str(geo.get("is_georeferenced", False))],
        ["CRS", str(geo.get("crs") or "N/A (image-relative)")],
        [
            "Changed Area (ha)",
            f"{geo['changed_area_ha']:.4f} ha" if geo.get("changed_area_ha") is not None else "N/A",
        ],
        ["Detected Regions", str(len(result_data.get("regions", [])))],
    ]
    tbl = Table(table_data, colWidths=[2.4 * inch, 3.6 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 0.2 * inch))

    overlay_path = result_dir / "overlay.png"
    mask_path = result_dir / "binary_mask.png"
    for caption, img_p in (("Change Detection Overlay", overlay_path), ("Binary Change Mask", mask_path)):
        if img_p.is_file():
            story.append(Paragraph(f"<b>{caption}</b>", styles["Heading3"]))
            story.append(RLImage(str(img_p), width=4.6 * inch, height=3.2 * inch, kind="proportional"))
            story.append(Spacer(1, 0.12 * inch))

    story.append(
        Paragraph(
            "<i>Disclaimer: Binary change masks identify radiometric/structural land-surface "
            "changes and do not alone classify specific causal categories (e.g., deforestation vs. "
            "seasonal agriculture) without labeled multi-class ground truth.</i>",
            styles["Italic"],
        )
    )
    doc.build(story)
    return output_pdf_path

