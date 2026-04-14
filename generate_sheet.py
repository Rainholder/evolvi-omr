"""
generate_sheet.py — EVOLVI OMR answer-sheet PDF generator.

Entry point:
    pdf_bytes, sheet_coords = build_sheet("EV-ATR-JUN26")

sheet_coords (for OMR processing):
    {"1": {"A": [cx_px, cy_px], "B": [...], "C": [...]}, ..., "90": {...}}

    Pixel coordinates at 300 DPI with top-left origin.
    Matches a 2550 × 3300 px image of the printed Letter page.

Layout (Letter 612 × 792 pt):
    ┌────────────────────────────────┐
    │ ▪  TL marker          TR ▪    │  ← page-corner markers (OMR ref)
    │    Header / title / code       │
    │    NOMBRE / APELLIDO fields    │
    │  ▪ ┌──────────────────┐ ▪     │  ← celular-zone markers
    │    │  CELULAR grid    │       │
    │  ▪ └──────────────────┘ ▪     │
    │    RESPUESTAS  (6 cols × 15)  │
    │    …                          │
    │         EVOLVI                │
    │ ▪  BL marker          BR ▪    │
    └────────────────────────────────┘
"""

import io
import json
import logging
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas as rl_canvas

log = logging.getLogger("omr.sheet")

# ---------------------------------------------------------------------------
# Exam code → display title
# ---------------------------------------------------------------------------
EXAM_TITLES: dict[str, str] = {
    "ATR": "ÁREAS TRANSVERSALES",
    "BIO": "BIOLOGÍA",
    "FIS": "FÍSICA",
    "QUI": "QUÍMICA",
    "MAT": "MATEMÁTICAS",
    "LEN": "LENGUAJE Y COMUNICACIÓN",
    "HIS": "HISTORIA Y GEOGRAFÍA",
    "GEO": "GEOGRAFÍA",
    "CDS": "CIENCIAS DE LA SALUD",
    "RAZ": "RAZONAMIENTO VERBAL",
    "ING": "INGLÉS",
    "INF": "INFORMÁTICA",
    "ECO": "ECONOMÍA",
    "SOC": "CIENCIAS SOCIALES",
    "FIL": "FILOSOFÍA",
}


def parse_exam_code(raw: str) -> dict:
    """
    Decompose an exam code into its parts.

    "EV-ATR-JUN26" → {code, area: "ATR", title: "ÁREAS TRANSVERSALES", period: "JUN26"}
    "BIO"          → {code, area: "BIO", title: "BIOLOGÍA", period: ""}
    """
    code  = raw.strip().upper()
    parts = code.split("-")

    if len(parts) >= 3:
        area, period = parts[1], parts[2]
    elif len(parts) == 2:
        area, period = parts[1], ""
    else:
        area, period = parts[0], ""

    return {
        "code":   code,
        "area":   area,
        "title":  EXAM_TITLES.get(area, area),
        "period": period,
    }


# ---------------------------------------------------------------------------
# Page constants
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = LETTER     # 612 × 792 pt  (Letter)

# 300 DPI export → Letter = 2550 × 3300 px
DPI   = 300
PT_PX = DPI / 72.0          # ≈ 4.1667 px/pt

# ── Corner / zone markers ────────────────────────────────────────────────────
MARKER    = 23   # pt  (≈ 8.1 mm — spec calls for 8 mm)
MARK_EDGE = 12   # pt from page edge to near side of page-corner markers

# ── Respuestas bubble geometry ────────────────────────────────────────────────
N_COLS    = 6
N_ROWS    = 15
RESP_ML   = 36   # pt left margin for the answer grid
QCOL_W    = 90   # pt per question column  (6 × 90 = 540 = 612 − 72 exactly)
R_RESP    = 10   # pt bubble radius  (diameter 20 pt ≈ 7 mm)
OPT_STEP  = 29   # pt horizontal center-to-center A→B and B→C  (≈ 10.2 mm)
QROW_STEP = 24   # pt vertical center-to-center between questions  (≈ 8.5 mm)
#   NUM_W = space for question-number label = QCOL_W − 2×OPT_STEP − 2×R_RESP
NUM_W     = QCOL_W - 2 * OPT_STEP - 2 * R_RESP   # = 90 − 58 − 20 = 12 pt

# ── Celular (phone-number) bubble geometry ────────────────────────────────────
N_CEL_COLS   = 10    # digit positions (0–9)
N_CEL_ROWS   = 10    # digit values (0–9)
R_CEL        = 5     # pt bubble radius  (diameter 10 pt ≈ 3.5 mm)
CEL_COL_STEP = 20    # pt horizontal center-to-center
CEL_ROW_STEP = 15    # pt vertical center-to-center
#   Centre first column horizontally on the page
CEL_CX0 = (PAGE_W - (N_CEL_COLS - 1) * CEL_COL_STEP) / 2   # = 216 pt

# ── Vertical layout  (all values: pt measured from TOP of page) ───────────────
# Page-corner markers occupy 12–35 pt from top.
Y_HEADER    = 40     # "EXÁMENES…" baseline
Y_TITLE     = 57     # exam title baseline
Y_CODE      = 72     # exam code baseline
Y_NAME      = 88     # NOMBRE(S) field box top
Y_APELL     = 110    # APELLIDO(S) field box top   (bottom at 124 pt from top)
Y_CEL_LBL   = 132    # "CELULAR" section label baseline
Y_CEL_HDR   = 148    # digit-position header row baseline
Y_CEL_ROW0  = 162    # first celular bubble-row centre  (digit 0)
#   Celular grid bottom edge = 162 + 9×15 + 5 = 302 pt from top
#   Celular zone markers: top=306, bottom=329 pt from top
Y_RESP_LBL  = 340    # "RESPUESTAS" section label baseline  (11 pt gap)
Y_RESP_HDR  = 356    # A / B / C column-header baseline
Y_RESP_ROW0 = 372    # first question-row bubble centres   (Q1, Q16, Q31 …)
#   Last row centre  = 372 + 14×24 = 708 pt from top
#   Last bubble bottom = 718 pt → footer at 748 pt → markers at 757 pt  ✓


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def _y(y_from_top: float) -> float:
    """Top-anchored y → ReportLab bottom-anchored y."""
    return PAGE_H - y_from_top


def _pt2px(x_pt: float, y_from_top: float) -> list[int]:
    """
    PDF layout point → image pixel coordinates (top-left origin, 300 DPI).
    Returned as a two-element list so it serialises cleanly to JSON.
    """
    return [round(x_pt * PT_PX), round(y_from_top * PT_PX)]


# ---------------------------------------------------------------------------
# Geometry: exact bubble centres in PDF points
# ---------------------------------------------------------------------------
def _resp_centers(col: int, row: int) -> dict[str, tuple[float, float]]:
    """
    Return {option: (x_pt, y_from_top_pt)} for one question cell.

    col : 0–5   question column
    row : 0–14  row within that column
    """
    a_cx = RESP_ML + col * QCOL_W + NUM_W + R_RESP
    cy   = Y_RESP_ROW0 + row * QROW_STEP
    return {
        "A": (a_cx,              cy),
        "B": (a_cx + OPT_STEP,   cy),
        "C": (a_cx + 2*OPT_STEP, cy),
    }


def _cel_center(col: int, row: int) -> tuple[float, float]:
    """Return (x_pt, y_from_top_pt) for one celular grid cell."""
    return (
        CEL_CX0 + col * CEL_COL_STEP,
        Y_CEL_ROW0 + row * CEL_ROW_STEP,
    )


# ---------------------------------------------------------------------------
# Compute SHEET_COORDS (pixel coordinates at 300 DPI)
# ---------------------------------------------------------------------------
def compute_sheet_coords() -> dict:
    """
    Build the SHEET_COORDS dict for the RESPUESTAS section.

    Returns:
        {
          "1":  {"A": [cx_px, cy_px], "B": [...], "C": [...]},
          …
          "90": {…}
        }

    Questions are numbered column-by-column:
        col 0 → Q1–Q15,  col 1 → Q16–Q30,  …  col 5 → Q76–Q90.
    """
    coords: dict = {}
    for col in range(N_COLS):
        for row in range(N_ROWS):
            q = col * N_ROWS + row + 1
            centers = _resp_centers(col, row)
            coords[str(q)] = {
                opt: _pt2px(cx, cy) for opt, (cx, cy) in centers.items()
            }
    return coords


# ---------------------------------------------------------------------------
# Low-level drawing helpers
# ---------------------------------------------------------------------------
def _draw_solid_marker(c, x_left: float, y_top_from_top: float) -> None:
    """
    Draw a solid black MARKER × MARKER pt square.

    x_left           : left edge of the square in PDF points from left
    y_top_from_top   : top edge measured from the top of the page
    """
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)
    c.rect(
        x_left,
        _y(y_top_from_top + MARKER),   # ReportLab rect origin = bottom-left
        MARKER,
        MARKER,
        fill=1,
        stroke=0,
    )


def _draw_bubble(
    c,
    cx_pt: float,
    y_from_top: float,
    r: float,
    label: str = "",
    label_gray: float = 0.72,
    border_width: float = 0.5,
) -> None:
    """
    Draw an empty circle bubble with a faint gray border.

    Spec: diameter 14 mm, border 0.5 pt gray (0.6, 0.6, 0.6), no fill,
          A/B/C label inside in light gray.
    (Implemented at proportional scale for Letter paper.)
    """
    cy_pdf = _y(y_from_top)
    c.setStrokeColor(colors.Color(0.6, 0.6, 0.6))
    c.setLineWidth(border_width)
    c.setFillColor(colors.white)
    c.circle(cx_pt, cy_pdf, r, fill=1, stroke=1)

    if label:
        font_sz = max(3.5, r * 0.80)
        c.setFont("Helvetica", font_sz)
        c.setFillColor(colors.Color(label_gray, label_gray, label_gray))
        # Vertically center the label (approximate: descend 38 % of font size)
        c.drawCentredString(cx_pt, cy_pdf - font_sz * 0.38, label)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_sheet(exam_code: str) -> tuple[bytes, dict]:
    """
    Generate the EVOLVI answer-sheet PDF.

    Returns:
        pdf_bytes    : raw PDF content (suitable for a Flask send_file response)
        sheet_coords : SHEET_COORDS dict  {"1": {"A": [x,y], …}, …}
    """
    info = parse_exam_code(exam_code)
    log.info("Building sheet for code=%s area=%s title=%s", info["code"], info["area"], info["title"])

    buf = io.BytesIO()
    c   = rl_canvas.Canvas(buf, pagesize=LETTER)

    center_x = PAGE_W / 2

    # =========================================================================
    # 1. PAGE-CORNER MARKERS
    #    Four solid black squares placed near the page corners.
    #    find_corner_markers() in app.py uses these for warpPerspective.
    # =========================================================================
    for mx, my_from_top in [
        (MARK_EDGE,               MARK_EDGE),                        # TL
        (PAGE_W - MARK_EDGE - MARKER, MARK_EDGE),                    # TR
        (MARK_EDGE,               PAGE_H - MARK_EDGE - MARKER),      # BL
        (PAGE_W - MARK_EDGE - MARKER, PAGE_H - MARK_EDGE - MARKER),  # BR
    ]:
        _draw_solid_marker(c, mx, my_from_top)

    # =========================================================================
    # 2. HEADER
    # =========================================================================
    c.setFillColor(colors.black)

    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(center_x, _y(Y_HEADER), "EXÁMENES DE SIMULACIÓN EVOLVI 2026")

    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(center_x, _y(Y_TITLE), info["title"])

    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(center_x, _y(Y_CODE), info["code"])

    # Thin decorative line below code
    c.setStrokeColor(colors.Color(0.6, 0.6, 0.6))
    c.setLineWidth(0.4)
    c.line(RESP_ML, _y(Y_CODE + 10), PAGE_W - RESP_ML, _y(Y_CODE + 10))

    # =========================================================================
    # 3. NOMBRE / APELLIDO FIELDS
    # =========================================================================
    field_h   = 14    # pt height of each input box
    label_w   = 58    # pt width of the label before the box
    field_x   = RESP_ML
    field_box_w = PAGE_W - 2 * RESP_ML - label_w

    for label, y_top in [("NOMBRE(S):", Y_NAME), ("APELLIDO(S):", Y_APELL)]:
        baseline = y_top + field_h - 4   # text baseline within the row
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(colors.black)
        c.drawString(field_x, _y(baseline), label)

        c.setFillColor(colors.white)
        c.setStrokeColor(colors.Color(0.3, 0.3, 0.3))
        c.setLineWidth(0.5)
        c.rect(
            field_x + label_w,
            _y(y_top + field_h),       # bottom of box in PDF coords
            field_box_w,
            field_h,
            fill=1,
            stroke=1,
        )

    # =========================================================================
    # 4. CELULAR GRID  — 10 digit positions × 10 digit values (0–9)
    # =========================================================================
    # --- Section label ---
    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(colors.black)
    c.drawString(RESP_ML, _y(Y_CEL_LBL), "CELULAR  (marque un dígito por columna):")

    # --- Column-position headers (1 … 10 = digit position of the phone number) ---
    c.setFont("Helvetica-Bold", 5)
    c.setFillColor(colors.Color(0.40, 0.40, 0.40))
    for col in range(N_CEL_COLS):
        cx_col, _ = _cel_center(col, 0)
        c.drawCentredString(cx_col, _y(Y_CEL_HDR), str(col + 1))

    # --- Celular bubbles (row = digit value 0-9) ---
    for row in range(N_CEL_ROWS):
        digit = str(row)
        cx_first, cy_first = _cel_center(0, row)

        # Row label to the left of the first column
        c.setFont("Helvetica", 5)
        c.setFillColor(colors.Color(0.40, 0.40, 0.40))
        c.drawRightString(cx_first - R_CEL - 3, _y(cy_first) - 2, digit)

        for col in range(N_CEL_COLS):
            cx_col, cy_col = _cel_center(col, row)
            _draw_bubble(c, cx_col, cy_col, R_CEL, label=digit, label_gray=0.70)

    # --- Celular zone corner markers ---
    cel_grid_left  = CEL_CX0 - R_CEL
    cel_grid_right = CEL_CX0 + (N_CEL_COLS - 1) * CEL_COL_STEP + R_CEL
    cel_grid_top   = Y_CEL_ROW0 - R_CEL        # y from top, top edge of top-row bubbles
    cel_grid_bot   = Y_CEL_ROW0 + (N_CEL_ROWS - 1) * CEL_ROW_STEP + R_CEL

    CEL_GAP = 4   # pt between grid edge and marker
    for mx, my_from_top in [
        (cel_grid_left  - CEL_GAP - MARKER, cel_grid_top - CEL_GAP),          # TL
        (cel_grid_right + CEL_GAP,          cel_grid_top - CEL_GAP),           # TR
        (cel_grid_left  - CEL_GAP - MARKER, cel_grid_bot + CEL_GAP),           # BL
        (cel_grid_right + CEL_GAP,          cel_grid_bot + CEL_GAP),           # BR
    ]:
        _draw_solid_marker(c, mx, my_from_top)

    # =========================================================================
    # 5. RESPUESTAS GRID — 90 questions × 3 options (A / B / C)
    # =========================================================================
    # --- Section label ---
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawString(RESP_ML, _y(Y_RESP_LBL), "RESPUESTAS")

    c.setStrokeColor(colors.Color(0.5, 0.5, 0.5))
    c.setLineWidth(0.4)
    c.line(RESP_ML, _y(Y_RESP_LBL + 4), PAGE_W - RESP_ML, _y(Y_RESP_LBL + 4))

    # --- A / B / C column headers ---
    c.setFont("Helvetica-Bold", 5.5)
    c.setFillColor(colors.Color(0.35, 0.35, 0.35))
    for col in range(N_COLS):
        for opt, (cx, _) in _resp_centers(col, 0).items():
            c.drawCentredString(cx, _y(Y_RESP_HDR), opt)

    # --- Bubbles + question numbers ---
    for col in range(N_COLS):
        for row in range(N_ROWS):
            q_num   = col * N_ROWS + row + 1
            centers = _resp_centers(col, row)
            a_cx, cy = centers["A"]

            # Question number to the left of bubble A
            c.setFont("Helvetica", 5)
            c.setFillColor(colors.black)
            c.drawRightString(a_cx - R_RESP - 2, _y(cy) - 2, str(q_num))

            # Three bubbles
            for opt, (cx, bcy) in centers.items():
                _draw_bubble(c, cx, bcy, R_RESP, label=opt)

    # =========================================================================
    # 6. FOOTER — EVOLVI logo
    # =========================================================================
    # Footer sits between last bubble (bottom ≈ 718 pt from top) and
    # the bottom page-corner markers (start ≈ 757 pt from top).
    footer_pdf_y = MARK_EDGE + MARKER + 14   # ≈ 49 pt from PDF bottom = 743 pt from top

    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.Color(0.13, 0.31, 0.65))   # EVOLVI blue
    c.drawCentredString(center_x, footer_pdf_y + 6, "EVOLVI")

    c.setFont("Helvetica", 5)
    c.setFillColor(colors.Color(0.50, 0.50, 0.50))
    c.drawCentredString(
        center_x,
        footer_pdf_y - 1,
        "Plataforma de Simulación Académica  ·  evolvi.pe",
    )

    # =========================================================================
    # 7. Finalise PDF
    # =========================================================================
    c.showPage()
    c.save()

    sheet_coords = compute_sheet_coords()
    log.info(
        "Sheet built: code=%s  bubbles=%d  pdf_bytes=%d",
        info["code"], len(sheet_coords) * 3, len(buf.getvalue()),
    )
    return buf.getvalue(), sheet_coords


# ---------------------------------------------------------------------------
# CLI helper (python generate_sheet.py EV-ATR-JUN26 → writes sheet.pdf)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    code = sys.argv[1] if len(sys.argv) > 1 else "EV-ATR-JUN26"
    pdf_bytes, coords = build_sheet(code)

    out_pdf   = Path("sheet.pdf")
    out_json  = Path("sheet_coords.json")

    out_pdf.write_bytes(pdf_bytes)
    out_json.write_text(json.dumps(coords, indent=2))

    print(f"PDF  → {out_pdf}  ({len(pdf_bytes):,} bytes)")
    print(f"JSON → {out_json}  ({len(coords)} questions, {len(coords)*3} bubbles)")
