"""PDF report generation for Pothole & Road Damage Reporter using ReportLab."""

import json
import logging
import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# Primary & semantic color palette
PRIMARY: HexColor = HexColor('#1a56db')
DANGER: HexColor = HexColor('#dc2626')
WARNING: HexColor = HexColor('#f59e0b')
SUCCESS: HexColor = HexColor('#16a34a')
GRAY: HexColor = HexColor('#6b7280')
LIGHT_GRAY: HexColor = HexColor('#f3f4f6')


def _severity_color(label: Optional[str]) -> HexColor:
    """
    Map severity label string to corresponding theme color.

    Args:
        label: Severity label ('High', 'Medium', 'Low').

    Returns:
        HexColor object for styling.
    """
    if not label:
        return SUCCESS
    norm_label = str(label).strip().capitalize()
    if norm_label == 'High':
        return DANGER
    elif norm_label == 'Medium':
        return WARNING
    return SUCCESS


def generate_report(
    job: Dict[str, Any],
    findings: List[Dict[str, Any]],
    image_dir: str,
    output_path: str
) -> str:
    """
    Generate a comprehensive PDF report for a road damage analysis job.

    Args:
        job: Job dictionary containing id, location_text, video_key,
             created_at, and optional summary_json.
        findings: List of finding dictionaries (typically confirmed and approved).
        image_dir: Local filesystem directory containing finding images.
        output_path: Target filesystem path where the PDF will be saved.

    Returns:
        Absolute or relative path to the generated PDF.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=PRIMARY,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=GRAY,
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=PRIMARY,
        spaceBefore=20,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=8
    )
    small_style = ParagraphStyle(
        'ReportSmall',
        parent=styles['Normal'],
        fontSize=8,
        textColor=GRAY,
        leading=10
    )
    finding_title_style = ParagraphStyle(
        'FindingTitle',
        parent=styles['Heading3'],
        fontSize=13,
        spaceBefore=4,
        spaceAfter=6
    )

    elements: List[Any] = []

    # --- PAGE 1: Title and Summary ---
    elements.append(Paragraph("Road Damage Assessment Report", title_style))
    elements.append(Paragraph("Automated Analysis by Pothole & Road Damage Reporter", subtitle_style))

    # Addressee
    elements.append(Paragraph("<b>To:</b> Municipal Roads Department", body_style))
    elements.append(Paragraph("<b>Attention:</b> _______________________ (Officer Name)", body_style))
    elements.append(Spacer(1, 12))

    # Report metadata
    created_raw = job.get('created_at', datetime.now(timezone.utc).isoformat())
    if isinstance(created_raw, datetime):
        created_str = created_raw.strftime('%Y-%m-%d')
    else:
        created_str = str(created_raw)[:10] if len(str(created_raw)) >= 10 else str(created_raw)

    location = job.get('location_text', 'Not specified')
    video_key = str(job.get('video_key', 'Unknown'))
    video_name = os.path.basename(video_key) if video_key else 'Unknown'

    meta_data = [
        ['Report Date', datetime.now().strftime('%B %d, %Y at %H:%M')],
        ['Analysis Date', created_str],
        ['Location', location],
        ['Video File', video_name],
        ['Job ID', str(job.get('id', 'N/A'))],
    ]

    meta_table = Table(meta_data, colWidths=[1.8 * inch, 4.5 * inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), GRAY),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 20))

    # Summary table
    elements.append(Paragraph("Summary of Findings", heading_style))

    # Count by type and severity
    type_counts: Dict[str, int] = {'pothole': 0, 'crack': 0}
    severity_counts: Dict[str, int] = {'High': 0, 'Medium': 0, 'Low': 0}
    for f in findings:
        f_type = str(f.get('type', 'unknown')).lower()
        type_counts[f_type] = type_counts.get(f_type, 0) + 1
        sev = str(f.get('severity_label', 'Low')).capitalize()
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    pothole_high = sum(1 for f in findings if str(f.get('type', '')).lower() == 'pothole' and str(f.get('severity_label', '')).capitalize() == 'High')
    pothole_med = sum(1 for f in findings if str(f.get('type', '')).lower() == 'pothole' and str(f.get('severity_label', '')).capitalize() == 'Medium')
    pothole_low = sum(1 for f in findings if str(f.get('type', '')).lower() == 'pothole' and str(f.get('severity_label', '')).capitalize() == 'Low')

    crack_high = sum(1 for f in findings if str(f.get('type', '')).lower() == 'crack' and str(f.get('severity_label', '')).capitalize() == 'High')
    crack_med = sum(1 for f in findings if str(f.get('type', '')).lower() == 'crack' and str(f.get('severity_label', '')).capitalize() == 'Medium')
    crack_low = sum(1 for f in findings if str(f.get('type', '')).lower() == 'crack' and str(f.get('severity_label', '')).capitalize() == 'Low')

    summary_data = [
        ['Category', 'Count', 'Severity Breakdown'],
        ['Potholes', str(type_counts.get('pothole', 0)),
         f"High: {pothole_high}, Medium: {pothole_med}, Low: {pothole_low}"],
        ['Cracks', str(type_counts.get('crack', 0)),
         f"High: {crack_high}, Medium: {crack_med}, Low: {crack_low}"],
        ['Total', str(len(findings)),
         f"High: {severity_counts.get('High', 0)}, Medium: {severity_counts.get('Medium', 0)}, Low: {severity_counts.get('Low', 0)}"]
    ]

    summary_table = Table(summary_data, colWidths=[1.5 * inch, 1.0 * inch, 3.8 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, -1), (-1, -1), LIGHT_GRAY),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, GRAY),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 15))

    # Method note
    elements.append(Paragraph("Analysis Method", heading_style))
    elements.append(Paragraph(
        "This report was generated using automated computer vision analysis with OpenCV. "
        "The system processes dashcam video footage to detect road surface defects including "
        "potholes and cracks. Detection uses adaptive thresholding, contour analysis, and "
        "edge detection. Multi-frame tracking with motion-aware linking confirms persistent "
        "defects and filters transient noise. An agent loop performs re-inspection of "
        "medium-confidence detections at 3× zoom with enhanced processing. Findings marked "
        "as 'needs review' were verified by a human operator.",
        body_style
    ))

    # --- FINDING PAGES ---
    if findings:
        elements.append(PageBreak())
        elements.append(Paragraph("Detailed Findings", heading_style))

        for idx, finding in enumerate(findings, 1):
            elements.append(Spacer(1, 10))
            elements.append(HRFlowable(width="100%", color=LIGHT_GRAY))
            elements.append(Spacer(1, 8))

            # Finding header
            sev_label = str(finding.get('severity_label', 'Low')).capitalize()
            det_type = str(finding.get('type', 'unknown')).capitalize()
            status = str(finding.get('status', 'unknown'))
            verification = 'Auto-confirmed' if status == 'confirmed' else 'Human-approved'

            color_obj = _severity_color(sev_label)
            color_hex = color_obj.hexval() if hasattr(color_obj, 'hexval') else '#16a34a'

            elements.append(Paragraph(
                f"Finding #{idx}: {det_type} — Severity: "
                f"<font color='{color_hex}'><b>{sev_label}</b></font>",
                finding_title_style
            ))

            # Finding details table
            conf_val = float(finding.get('confidence', 0.0))
            ts_val = float(finding.get('timestamp_s', 0.0))
            sev_score = float(finding.get('severity_score', 0.0))

            details = [
                ['Severity Score', f"{sev_score:.1f} / 100"],
                ['Confidence', f"{conf_val:.1%}"],
                ['Timestamp', f"{ts_val:.1f}s"],
                ['Frame Index', str(finding.get('frame_index', 'N/A'))],
                ['Verification', verification],
            ]

            # Try to resolve finding image
            image_key = str(finding.get('image_key') or finding.get('image_path') or '')
            image_path: Optional[str] = None
            if image_key:
                candidates = [
                    image_key,
                    os.path.join(image_dir, image_key),
                    os.path.join(image_dir, os.path.basename(image_key))
                ]
                for cand in candidates:
                    if cand and os.path.isfile(cand):
                        image_path = cand
                        break

            if image_path and os.path.exists(image_path):
                detail_table = Table(details, colWidths=[1.3 * inch, 1.4 * inch])
                detail_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('TEXTCOLOR', (0, 0), (0, -1), GRAY),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                try:
                    img = RLImage(image_path, width=4.1 * inch, height=2.8 * inch, kind='proportional')
                    layout_table = Table(
                        [[detail_table, img]],
                        colWidths=[2.8 * inch, 4.2 * inch]
                    )
                    layout_table.setStyle(TableStyle([
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ]))
                    elements.append(layout_table)
                except Exception as e:
                    logger.warning(f"Could not include image for finding {idx}: {e}")
                    full_detail_table = Table(details, colWidths=[1.5 * inch, 2.5 * inch])
                    full_detail_table.setStyle(TableStyle([
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('TEXTCOLOR', (0, 0), (0, -1), GRAY),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ]))
                    elements.append(full_detail_table)
            else:
                full_detail_table = Table(details, colWidths=[1.5 * inch, 2.5 * inch])
                full_detail_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('TEXTCOLOR', (0, 0), (0, -1), GRAY),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                elements.append(full_detail_table)

            # Measurements
            measurements = finding.get('measurements_json', '{}')
            if isinstance(measurements, str):
                try:
                    measurements = json.loads(measurements)
                except json.JSONDecodeError:
                    measurements = {}
            elif not isinstance(measurements, dict):
                measurements = {}

            if measurements:
                meas_parts = []
                for k, v in measurements.items():
                    if isinstance(v, float):
                        meas_parts.append(f"{k}: {v:.2f}")
                    else:
                        meas_parts.append(f"{k}: {v}")
                meas_text = ", ".join(meas_parts)
                elements.append(Paragraph(
                    f"<i>Measurements: {meas_text}</i>", small_style
                ))

    # --- LAST SECTION: Limitations and Disclaimer ---
    elements.append(PageBreak())
    elements.append(Paragraph("Limitations & Disclaimer", heading_style))

    limitations = [
        "Detection accuracy depends on video quality, lighting conditions, and camera angle.",
        "The system is optimized for forward-facing dashcam footage of asphalt roads.",
        "Night footage, heavy rain, or snow conditions may significantly reduce accuracy.",
        "Detected measurements are in pixels and do not represent physical dimensions without calibration.",
        "The system cannot assess structural depth or subsurface damage.",
        "False positives may occur from shadows, stains, manholes, or road markings.",
        "Very small defects (< 0.3% of the visible road area) may not be detected.",
    ]

    for lim in limitations:
        elements.append(Paragraph(f"• {lim}", body_style))

    elements.append(Spacer(1, 20))

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=body_style,
        fontSize=9,
        textColor=GRAY,
        borderWidth=1,
        borderColor=GRAY,
        borderPadding=10,
        backColor=LIGHT_GRAY
    )
    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by an automated computer vision system "
        "and is intended for advisory purposes only. All findings should be independently "
        "verified through on-site inspection by qualified road maintenance personnel before "
        "any repair decisions or budget allocations are made. The system developers and "
        "operators accept no liability for decisions made solely on the basis of this report.",
        disclaimer_style
    ))

    # Build PDF
    doc.build(elements)
    logger.info(f"Report generated: {output_path}")
    return output_path
