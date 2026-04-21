"""
PDF Report Generator
Generates professional executive PDF reports using ReportLab.
Fallback: returns HTML report if ReportLab is unavailable.
"""

import io
from datetime import datetime
from typing import Dict, Any, List

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


# ─── Colors ───────────────────────────────────────────────────────────────────
ACCENT   = HexColor("#6366f1") if HAS_REPORTLAB else None
SUCCESS  = HexColor("#10b981") if HAS_REPORTLAB else None
DANGER   = HexColor("#ef4444") if HAS_REPORTLAB else None
WARNING  = HexColor("#f59e0b") if HAS_REPORTLAB else None
DARK     = HexColor("#1e2135") if HAS_REPORTLAB else None
LIGHT_BG = HexColor("#f8f9ff") if HAS_REPORTLAB else None
GRAY     = HexColor("#64748b") if HAS_REPORTLAB else None


def generate_pdf_report(data: Dict[str, Any]) -> bytes:
    """
    Generate a PDF executive report.
    Returns raw PDF bytes (stream directly to HTTP response).
    Falls back to generating a summary bytes string if ReportLab unavailable.
    """
    if not HAS_REPORTLAB:
        return _fallback_text_report(data).encode("utf-8")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Custom Styles ─────────────────────────────────────────────────────────
    title_style = ParagraphStyle("Title2", parent=styles["Normal"],
        fontSize=22, textColor=ACCENT, spaceAfter=4, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle("Sub", parent=styles["Normal"],
        fontSize=11, textColor=GRAY, spaceAfter=20)
    h2_style = ParagraphStyle("H2", parent=styles["Normal"],
        fontSize=14, textColor=DARK, spaceAfter=8, fontName="Helvetica-Bold",
        spaceBefore=16)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=10, leading=16, textColor=HexColor("#374151"))
    kpi_label = ParagraphStyle("KpiLabel", parent=styles["Normal"],
        fontSize=9, textColor=GRAY)
    kpi_value = ParagraphStyle("KpiValue", parent=styles["Normal"],
        fontSize=16, textColor=ACCENT, fontName="Helvetica-Bold")

    now = datetime.utcnow().strftime("%B %Y")
    summary = data.get("executive_summary", {})
    recommendations = data.get("recommendations", [])
    risks = data.get("risks", [])

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("SalesAI — Executive Report", title_style))
    story.append(Paragraph(f"Period: {now} &nbsp;|&nbsp; Generated: {datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=16))

    # ── KPI Summary ───────────────────────────────────────────────────────────
    story.append(Paragraph("Key Performance Indicators", h2_style))

    kpi_data = [
        [
            _kpi_cell("Total Revenue", f"₹{summary.get('total_revenue',0)/100000:.1f}L", kpi_label, kpi_value),
            _kpi_cell("Total Orders", f"{summary.get('total_orders',0):,}", kpi_label, kpi_value),
            _kpi_cell("Profit Margin", f"{summary.get('profit_margin',0):.1f}%", kpi_label, kpi_value),
            _kpi_cell("Sentiment Score", f"{summary.get('sentiment_score',0):.1f}", kpi_label, kpi_value),
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHT_BG),
        ("BOX",        (0,0), (-1,-1), 0.5, HexColor("#e2e8f0")),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, HexColor("#e2e8f0")),
        ("TOPPADDING", (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 16))

    # ── Top Performer ─────────────────────────────────────────────────────────
    story.append(Paragraph("Business Highlights", h2_style))
    story.append(Paragraph(
        f"<b>Top Product:</b> {summary.get('top_product','—')} &nbsp;|&nbsp; "
        f"<b>Top Region:</b> {summary.get('top_region','—')}",
        body_style
    ))
    story.append(Spacer(1, 12))

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(Paragraph("AI Recommendations", h2_style))
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", body_style))
    story.append(Spacer(1, 12))

    # ── Risks ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Risk Register", h2_style))
    risk_rows = [["#", "Risk", "Action"]]
    for i, risk in enumerate(risks, 1):
        risk_rows.append([str(i), risk, "Investigate & resolve"])
    risk_table = Table(risk_rows, colWidths=[1*cm, 11*cm, 4.5*cm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), DARK),
        ("TEXTCOLOR",   (0,0), (-1,0), white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, LIGHT_BG]),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("GRID",        (0,0), (-1,-1), 0.25, HexColor("#e2e8f0")),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 20))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#e2e8f0")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated by SalesAI — AI-Powered Sales Prediction System | Confidential",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=GRAY, alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def _kpi_cell(label: str, value: str, label_style, value_style) -> list:
    return [Paragraph(label, label_style), Paragraph(value, value_style)]


def _fallback_text_report(data: Dict) -> str:
    summary = data.get("executive_summary", {})
    lines = [
        "=" * 60,
        "SalesAI — Executive Report",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 60,
        "",
        "KEY PERFORMANCE INDICATORS",
        f"  Revenue:        ₹{summary.get('total_revenue',0):,.0f}",
        f"  Orders:         {summary.get('total_orders',0):,}",
        f"  Profit Margin:  {summary.get('profit_margin',0):.1f}%",
        f"  Sentiment Score:{summary.get('sentiment_score',0):.1f}",
        f"  Top Product:    {summary.get('top_product','—')}",
        f"  Top Region:     {summary.get('top_region','—')}",
        "",
        "AI RECOMMENDATIONS",
    ]
    for i, r in enumerate(data.get("recommendations", []), 1):
        lines.append(f"  {i}. {r}")
    lines.extend(["", "RISKS"])
    for i, r in enumerate(data.get("risks", []), 1):
        lines.append(f"  {i}. {r}")
    lines.append("=" * 60)
    return "\n".join(lines)
