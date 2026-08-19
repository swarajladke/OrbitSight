"""Module to generate Evaluation_Metrics.xlsx report using openpyxl."""

from pathlib import Path
from typing import Any, Dict, List, Union
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side


def write_metrics_xlsx(
    metrics_rows: List[Dict[str, Any]],
    overall_row: Dict[str, Any],
    out_path: Union[str, Path],
) -> None:
    """Write Evaluation_Metrics.xlsx using openpyxl.

    Sheet layout:
    - Row 1 header: Sequence, Split, Sensor, TP, FP, FN, Precision, Recall, F1, AP@IoU0.5
    - One row per sequence (all 21)
    - Final row labelled OVERALL with aggregate Precision, Recall, F1 and mAP@IoU0.5
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Evaluation_Metrics"

    headers = [
        "Sequence",
        "Split",
        "Sensor",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1",
        "AP@IoU0.5",
    ]
    ws.append(headers)

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")
    right_align = Alignment(horizontal="right", vertical="center")

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align

    for row_data in metrics_rows:
        row_vals = [
            str(row_data.get("sequence", "")),
            str(row_data.get("split", "")),
            str(row_data.get("sensor", "")),
            int(row_data.get("tp", 0)),
            int(row_data.get("fp", 0)),
            int(row_data.get("fn", 0)),
            float(row_data.get("precision", 0.0)),
            float(row_data.get("recall", 0.0)),
            float(row_data.get("f1", 0.0)),
            float(row_data.get("ap", 0.0)),
        ]
        ws.append(row_vals)

        curr_row = ws.max_row
        for c_idx in range(1, len(headers) + 1):
            c = ws.cell(row=curr_row, column=c_idx)
            c.border = thin_border
            if c_idx == 1:
                c.alignment = left_align
            elif c_idx in (2, 3):
                c.alignment = center_align
            elif c_idx in (4, 5, 6):
                c.alignment = right_align
                c.number_format = "#,##0"
            else:
                c.alignment = right_align
                c.number_format = "0.000000"

    overall_vals = [
        "OVERALL",
        str(overall_row.get("split", "all")),
        "ALL",
        int(overall_row.get("tp", 0)),
        int(overall_row.get("fp", 0)),
        int(overall_row.get("fn", 0)),
        float(overall_row.get("precision", 0.0)),
        float(overall_row.get("recall", 0.0)),
        float(overall_row.get("f1", 0.0)),
        float(overall_row.get("mAP", overall_row.get("ap", 0.0))),
    ]
    ws.append(overall_vals)
    curr_row = ws.max_row

    overall_font = Font(name="Calibri", size=11, bold=True)
    overall_fill = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")

    for c_idx in range(1, len(headers) + 1):
        c = ws.cell(row=curr_row, column=c_idx)
        c.font = overall_font
        c.fill = overall_fill
        c.border = thin_border
        if c_idx == 1:
            c.alignment = left_align
        elif c_idx in (2, 3):
            c.alignment = center_align
        elif c_idx in (4, 5, 6):
            c.alignment = right_align
            c.number_format = "#,##0"
        else:
            c.alignment = right_align
            c.number_format = "0.000000"

    # Auto-fit column widths
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = openpyxl.utils.get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    wb.save(out_path)
    print(f"Metrics XLSX written successfully to: {out_path}", flush=True)
