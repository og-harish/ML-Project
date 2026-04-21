"""
Reports Router — PDF, Excel, and JSON report generation.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, Response
from routers.auth import get_current_user
from services.pdf_generator import generate_pdf_report
import io
from datetime import datetime

router = APIRouter()

DEMO_REPORT_DATA = {
    "period": "January 2024",
    "generated_at": datetime.utcnow().isoformat(),
    "executive_summary": {
        "total_revenue":    4280000,
        "total_orders":     8432,
        "profit_margin":    28.6,
        "top_product":      "Laptop Pro X1",
        "top_region":       "Mumbai",
        "sentiment_score":  74.2,
    },
    "recommendations": [
        "Restock Laptop Pro X1 immediately — only 8 units remaining (0.5 days of stock)",
        "Investigate Smart Watch S3 refund spike (18%) — likely quality defect in Batch #2024-01",
        "Increase ad spend in Chennai — fastest growing market at +23% MoM",
        "Launch weekend flash sale — historically drives +28% revenue lift",
        "Reduce price on USB-C Hub to clear overstock (320 units idle)",
    ],
    "risks": [
        "Stock-out risk for Laptop Pro X1 and Wireless Earbuds within 24 hours",
        "Smart Watch S3 negative sentiment rising to 42%",
        "Delhi region revenue anomaly requires investigation",
    ],
}


@router.get("/summary")
async def summary_report(current_user=Depends(get_current_user)):
    return DEMO_REPORT_DATA


@router.get("/download/pdf")
async def download_pdf(current_user=Depends(get_current_user)):
    pdf_bytes = generate_pdf_report(DEMO_REPORT_DATA)
    filename = f"salesai_report_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/download/excel")
async def download_excel(current_user=Depends(get_current_user)):
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        wb = openpyxl.Workbook()
        accent_fill = PatternFill("solid", fgColor="6366F1")
        header_font = Font(bold=True, color="FFFFFF", size=11)

        def style_headers(ws, row, cols):
            for col in range(1, cols+1):
                cell = ws.cell(row=row, column=col)
                cell.fill = accent_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center")

        ws1 = wb.active
        ws1.title = "KPI Summary"
        ws1.column_dimensions["A"].width = 28
        ws1.column_dimensions["B"].width = 20
        ws1.column_dimensions["C"].width = 16
        ws1.append(["Metric", "Value", "MoM Change"])
        style_headers(ws1, 1, 3)
        for row in [
            ("Total Revenue (INR)", "42,80,000", "+12.4%"),
            ("Total Orders", "8,432", "+8.2%"),
            ("Avg Order Value", "507", "+3.8%"),
            ("Profit Margin", "28.6%", "+2.3%"),
            ("Sentiment Score", "74.2", "+5.1 pts"),
        ]:
            ws1.append(row)

        ws2 = wb.create_sheet("Products")
        ws2.append(["Product", "Revenue", "Orders", "Sentiment"])
        style_headers(ws2, 1, 4)
        for row in [
            ("Laptop Pro X1", 2840000, 142, "82%"),
            ("Wireless Earbuds", 1560000, 780, "91%"),
            ("Smart Watch S3", 1340000, 268, "58%"),
            ("USB-C Hub", 890000, 1112, "76%"),
        ]:
            ws2.append(row)

        ws3 = wb.create_sheet("Recommendations")
        ws3.column_dimensions["A"].width = 8
        ws3.column_dimensions["B"].width = 60
        ws3.append(["#", "Recommendation"])
        style_headers(ws3, 1, 2)
        for i, rec in enumerate(DEMO_REPORT_DATA["recommendations"], 1):
            ws3.append([i, rec])

        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        filename = f"salesai_{datetime.utcnow().strftime('%Y%m%d')}.xlsx"
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except ImportError:
        return {"error": "openpyxl not installed", "install": "pip install openpyxl"}
