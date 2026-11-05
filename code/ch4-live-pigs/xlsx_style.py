#!/usr/bin/env python3
"""
Readable Excel tables and embedded figures for the Live Pigs analysis outputs
(erg_cohort.py, master_analysis.py).

write_table(): header names without underscores, values ROUNDED to the precision they
are shown with (no 15-digit floats), a number format per column chosen from its name,
frozen header + filter, banded rows, sensible widths, an optional note above.
add_image(): embed a PNG (scaled to a width in pixels) at a cell.
"""
import re
from pathlib import Path

from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HDR_FILL = PatternFill("solid", fgColor="1F4E79")
BAND_FILL = PatternFill("solid", fgColor="F2F6FA")
FLAG_FILL = PatternFill("solid", fgColor="FCE4D6")
WHITE_BOLD = Font(color="FFFFFF", bold=True)
TITLE = Font(bold=True, size=13, color="1F4E79")
NOTE = Font(italic=True, color="595959")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(bottom=THIN)


def pretty(name):
    """'Test_minus_Control' -> 'Test minus Control', 'b_t_ms' -> 'b t ms'."""
    s = str(name).replace("_", " ")
    s = re.sub(r"\buV\b", "µV", s)
    s = re.sub(r"\bdegC\b|\bdeg C\b", "°C", s)
    s = re.sub(r"\bCI95\b", "95% CI", s)
    return s


def fmt_for(name):
    """(number_format, decimals) chosen from the column name."""
    n = str(name).lower()
    if re.search(r"(^|_|\s)p(_|\s|$)", n) or n.startswith("p_"):
        return "0.000", 3
    if "fraction" in n:
        return "0%", 2
    if re.search(r"(^|_)(n|count|pigs?|session_no|duration_min|repeat|sweeps_(kept|total)|n_pooled|"
                 r"n_animals|n readings)($|_)", n) or n in ("pig", "n"):
        return "0", 0
    if re.search(r"(uv|amp|_sd$|limit|change_sd)", n) and "temp" not in n:
        return "0.0", 1
    return "0.0", 1


def _round(v, d):
    if isinstance(v, float):
        if v != v:                      # NaN
            return None
        return round(v, d) if d else int(round(v))
    return v


def write_table(wb, title, rows, note=None, flag_col=None, start_row=1):
    """A sheet with a title, an optional note and the rows as a formatted table.
    Rows whose `flag_col` value is truthy are shaded. Returns the worksheet."""
    ws = wb.create_sheet(title[:31])
    ws.sheet_view.showGridLines = False
    r = start_row
    ws.cell(r, 1, title).font = TITLE
    r += 1
    if note:
        ws.cell(r, 1, note).font = NOTE
        r += 1
    r += 1
    if not rows:
        ws.cell(r, 1, "(no rows)").font = NOTE
        return ws
    keys = list(dict.fromkeys(k for row in rows for k in row))
    fmts = {k: fmt_for(k) for k in keys}
    hdr = r
    for j, k in enumerate(keys, 1):
        c = ws.cell(hdr, j, pretty(k))
        c.font, c.fill = WHITE_BOLD, HDR_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[hdr].height = 32
    widths = {k: max(8, min(40, len(pretty(k)) // 2 + 4)) for k in keys}
    for i, row in enumerate(rows, hdr + 1):
        flagged = bool(flag_col and row.get(flag_col))
        for j, k in enumerate(keys, 1):
            f, d = fmts[k]
            v = _round(row.get(k), d)
            c = ws.cell(i, j, v)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                c.number_format = f
                c.alignment = Alignment(horizontal="right")
            else:
                c.alignment = Alignment(wrap_text=isinstance(v, str) and len(v) > 40, vertical="top")
            c.border = BORDER
            if flagged:
                c.fill = FLAG_FILL
            elif (i - hdr) % 2 == 0:
                c.fill = BAND_FILL
            if isinstance(v, str):
                widths[k] = max(widths[k], min(60, len(v) + 2))
            elif v is not None:
                widths[k] = max(widths[k], len(str(v)) + 2)
    for j, k in enumerate(keys, 1):
        ws.column_dimensions[get_column_letter(j)].width = widths[k]
    ws.freeze_panes = ws.cell(hdr + 1, 1)
    ws.auto_filter.ref = f"A{hdr}:{get_column_letter(len(keys))}{hdr + len(rows)}"
    return ws


def add_image(ws, png, cell, width_px=900):
    """Embed a PNG at `cell`, scaled to width_px. Returns rows it spans (approx.)."""
    png = Path(png)
    if not png.exists():
        return 0
    img = XLImage(str(png))
    s = width_px / img.width
    img.width, img.height = int(img.width * s), int(img.height * s)
    ws.add_image(img, cell)
    return int(img.height / 20) + 2


def text_block(ws, r, c, lines, bold_first=False):
    """Write lines of text down a column; returns the next free row."""
    for i, ln in enumerate(lines):
        cell = ws.cell(r + i, c, ln)
        if bold_first and i == 0:
            cell.font = TITLE
    return r + len(lines)
