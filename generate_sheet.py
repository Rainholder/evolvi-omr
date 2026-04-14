"""
generate_sheet.py — EVOLVI OMR answer-sheet PDF generator.

Entry point:
    pdf_bytes, sheet_coords = build_sheet("EV-ATR-JUN26")

sheet_coords:
    {
      "celular": {
        "0": {"0": [x,y], "1": [x,y], …, "9": [x,y]},   # digit-position → digit-value → [cx_px, cy_px]
        …
        "9": {…}
      },
      "respuestas": {
        "1":  {"A": [cx_px, cy_px], "B": [cx_px, cy_px], "C": [cx_px, cy_px]},
        …
        "90": {…}
      }
    }

    Pixel coordinates at 300 DPI, top-left origin → matches a 2550×3300 px image.

Page layout (Letter 612×792 pt, 20 pt margins all sides):

    ┌─────────────────────────────────────────┐
    │ LEFT COL (20–330)  │ RIGHT COL (345–592)│
    │  Exam title 14pt   │  ▪ CELULAR ▪       │
    │  Exam code 11pt    │  [ ][ ]…[ ] ←sq   │
    │  NOMBRE rect       │  ○ ○ ○ ○ ○ …      │
    │  APELLIDOS rect    │  ○ ○ ○ ○ ○ …      │
    │  INSTRUCCIONES     │  (10×10 grid)      │
    │                    │  ▪           ▪     │
    ├────── separator line ───────────────────┤
    │  ▪ RESPUESTAS               ▪           │
    │  6 cols × 15 rows of A B C bubbles      │
    │  ▪                          ▪           │
    │  EV-ATR-JUN26              EVOLVI       │
    └─────────────────────────────────────────┘
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
    Decompose an exam code into its components.

    "EV-ATR-JUN26" → {code, area: "ATR", title: "ÁREAS TRANSVERSALES", period: "JUN26"}
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
# Page geometry
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = LETTER      # 612 × 792 pt
MARGIN = 20                  # pt, all four sides

# Export resolution
DPI   = 300
PT_PX = DPI / 72.0           # ≈ 4.1667 px / pt   →   Letter = 2550 × 3300 px

# ---------------------------------------------------------------------------
# Corner / zone markers — solid black 8 × 8 pt squares
# ---------------------------------------------------------------------------
MARKER = 8      # pt side length (≈ 2.8 mm; spec says "8x8pt")

# ---------------------------------------------------------------------------
# Two-column split (top section)
# ---------------------------------------------------------------------------
LEFT_X     = 20    # left  col: x 20–330  (width 310 pt)
LEFT_X_END = 330
RIGHT_X    = 345   # right col: x 345–592 (width 247 pt)
RIGHT_X_END = 592

# ---------------------------------------------------------------------------
# CELULAR layout (right column)
# ---------------------------------------------------------------------------
N_CEL_COLS  = 10       # digit positions
N_CEL_ROWS  = 10       # digit values 0–9

CEL_SQ_SIZE = 22       # pt — each digit-input square (width = height)
# 10 squares × 22 pt = 220 pt, centred in 247 pt right column
_CEL_SQ_OFFSET = (RIGHT_X_END - RIGHT_X - N_CEL_COLS * CEL_SQ_SIZE) / 2.0  # = 13.5 pt
CEL_SQ_X0 = RIGHT_X + _CEL_SQ_OFFSET          # = 358.5 pt — left edge of first square

R_CEL        = 8    # pt radius (diameter 16 pt — as spec)
CEL_CX0      = CEL_SQ_X0 + CEL_SQ_SIZE / 2.0  # = 369.5 pt — centre of first bubble column
CEL_COL_STEP = CEL_SQ_SIZE                     # = 22 pt — horizontal centre-to-centre
CEL_ROW_STEP = 20                              # pt — vertical centre-to-centre

# Vertical positions (y measured from TOP of page)
Y_CEL_MARKER_TOP = MARGIN                      # = 20 pt — TL/TR celular markers
Y_CEL_LABEL      = 33    # "CELULAR" text baseline
Y_CEL_SQ_TOP     = 38    # digit-input squares: top edge
Y_CEL_SQ_BOT     = Y_CEL_SQ_TOP + CEL_SQ_SIZE  # = 60 pt
Y_CEL_ROW0       = 70    # first bubble-row centre (digit 0)
#   Last row centre   : Y_CEL_ROW0 + 9×20 = 248 pt
#   Last bubble bottom: 248 + 8    = 256 pt
Y_CEL_MARKER_BOT = 260   # BL/BR celular markers (span 260–268 pt from top)

# Celular zone marker x positions (at the outer edges of the right column)
CEL_MARKER_L = RIGHT_X                         # = 345 pt
CEL_MARKER_R = RIGHT_X_END - MARKER            # = 584 pt

# ---------------------------------------------------------------------------
# LEFT COLUMN vertical positions
# ---------------------------------------------------------------------------
Y_TITLE     = 34    # exam title baseline (14 pt bold)
Y_CODE      = 51    # exam code baseline (11 pt)
Y_NOMBRE_LBL = 65   # "NOMBRE(S):" label baseline (8 pt bold)
Y_NOMBRE_TOP = 68   # NOMBRE rect top edge
Y_NOMBRE_BOT = 96   # = 68 + 28 pt
Y_APELL_LBL  = 108  # "APELLIDO(S):" label baseline
Y_APELL_TOP  = 111  # APELLIDOS rect top edge
Y_APELL_BOT  = 139  # = 111 + 28 pt
Y_INST_LBL   = 153  # "INSTRUCCIONES:" label baseline
Y_INST_TEXT  = 163  # first instruction text-line baseline

INSTRUCTIONS = (
    "Rellena solamente un alvéolo por respuesta, asegúrate que el alvéolo "
    "está completamente lleno, si tuviste algún error y cambiaste de respuesta "
    "asegúrate de borrar muy bien el alvéolo, esto puede afectar tu resultado "
    "final. Escribe el celular que tengas registrado en tu plataforma Evolvi y "
    "asegúrate que los alvéolos estén correctos, debes tener un alvéolo por "
    "columna y tiene que coincidir con el número que hayas escrito en la parte "
    "superior de la columna."
)

# ---------------------------------------------------------------------------
# Separator & RESPUESTAS section
# ---------------------------------------------------------------------------
Y_SEPARATOR       = 290  # horizontal separator line (must stay clear)
Y_RESP_MARKER_TOP = 290  # TL/TR respuestas markers (span 290–298 pt from top)
Y_RESP_LABEL      = 265  # "RESPUESTAS" label baseline — drawn at x=30 to clear TL marker
Y_RESP_HDRS       = 310  # A / B / C column-header baseline
Y_RESP_ROW0       = 320  # first question-row bubble centre

N_COLS    = 6
N_ROWS    = 15
R_RESP    = 8     # pt radius (diameter 16 pt — as spec)
OPT_STEP  = 22    # pt horizontal centre-to-centre A→B and B→C (as spec)
QROW_STEP = 22    # pt vertical centre-to-centre between questions

#   Last row centre   : Y_RESP_ROW0 + 14 × 22 = 628 pt from top
#   Last bubble bottom: 628 + 8 = 636 pt from top
Y_RESP_MARKER_BOT = 644  # BL/BR respuestas markers (span 644–652 pt from top)
Y_FOOTER          = 790  # footer text baseline (2 pt above page bottom)

# Respuestas column geometry
# Markers at x=20 (TL) right edge=28 pt, and x=584 (TR) left edge.
# Bubble content starts at 34 pt (6 pt gap after TL marker) and ends ≤ 584 pt.
_RESP_CONTENT_X0  = 34    # left of column-0 number label
_RESP_COL_PITCH   = 94    # pt between column starts (6 cols × 94 = 564 pt)
_RESP_NUM_W       = 15    # pt reserved for question-number label

#   A_cx(col) = _RESP_CONTENT_X0 + col × _RESP_COL_PITCH + _RESP_NUM_W + R_RESP
#             = 34 + col×94 + 15 + 8  =  57 + col×94
#   B_cx(col) = A_cx + OPT_STEP       =  79 + col×94
#   C_cx(col) = B_cx + OPT_STEP       = 101 + col×94
#
#   Verification col 5:  C_cx = 101 + 470 = 571  right_edge = 579 < 584 (TR marker) ✓


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def _y(y_from_top: float) -> float:
    """Top-anchored y → ReportLab bottom-anchored y."""
    return PAGE_H - y_from_top


def _pt2px(x_pt: float, y_from_top: float) -> list[int]:
    """PDF point (bottom-left origin) → image pixel (top-left origin, 300 DPI)."""
    return [round(x_pt * PT_PX), round(y_from_top * PT_PX)]


# ---------------------------------------------------------------------------
# Exact bubble centres (PDF points)
# ---------------------------------------------------------------------------
def _resp_cx(col: int, opt_idx: int) -> float:
    """x-centre of option opt_idx (0=A,1=B,2=C) in column col."""
    return 57.0 + col * _RESP_COL_PITCH + opt_idx * OPT_STEP


def _resp_cy(row: int) -> float:
    """y_from_top centre of question row."""
    return Y_RESP_ROW0 + row * QROW_STEP


def _cel_cx(col: int) -> float:
    """x-centre of celular digit-position column."""
    return CEL_CX0 + col * CEL_COL_STEP


def _cel_cy(row: int) -> float:
    """y_from_top centre of celular digit-value row."""
    return Y_CEL_ROW0 + row * CEL_ROW_STEP


# ---------------------------------------------------------------------------
# Compute SHEET_COORDS dict
# ---------------------------------------------------------------------------
def compute_sheet_coords() -> dict:
    """
    Build the complete SHEET_COORDS dict in pixel space (300 DPI, top-left origin).

    Returns:
        {
          "celular": {
            "0": {"0": [x,y], "1": [x,y], …, "9": [x,y]},   # position → digit
            …
            "9": {…}
          },
          "respuestas": {
            "1":  {"A": [x,y], "B": [x,y], "C": [x,y]},
            …
            "90": {…}
          }
        }
    """
    opts = ["A", "B", "C"]

    respuestas: dict = {}
    for col in range(N_COLS):
        for row in range(N_ROWS):
            q   = col * N_ROWS + row + 1
            cy  = _resp_cy(row)
            respuestas[str(q)] = {
                opts[i]: _pt2px(_resp_cx(col, i), cy) for i in range(3)
            }

    celular: dict = {}
    for pos in range(N_CEL_COLS):        # digit position 0–9
        row_map: dict = {}
        cx = _cel_cx(pos)
        for dig in range(N_CEL_ROWS):    # digit value 0–9
            row_map[str(dig)] = _pt2px(cx, _cel_cy(dig))
        celular[str(pos)] = row_map

    return {"celular": celular, "respuestas": respuestas}


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _marker(c, x_left: float, y_top: float) -> None:
    """Draw a solid black MARKER × MARKER pt square. y_top is from page top."""
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)
    c.rect(x_left, _y(y_top + MARKER), MARKER, MARKER, fill=1, stroke=0)


def _bubble(c, cx: float, y_from_top: float, r: float,
            label: str = "", font_size: float = 7.0) -> None:
    """
    Draw an empty bubble (circle) per spec:
    border 0.5 pt gray (0.65, 0.65, 0.65), no fill, label centred in light gray.
    """
    cy_pdf = _y(y_from_top)
    c.setStrokeColor(colors.Color(0.65, 0.65, 0.65))
    c.setLineWidth(0.5)
    c.setFillColor(colors.white)
    c.circle(cx, cy_pdf, r, fill=1, stroke=1)
    if label:
        c.setFont("Helvetica", font_size)
        c.setFillColor(colors.Color(0.70, 0.70, 0.70))
        c.drawCentredString(cx, cy_pdf - font_size * 0.36, label)


def _count_lines(c, text: str, max_width: float, font: str, size: float) -> int:
    """Return the number of wrapped lines without drawing anything."""
    words  = text.split()
    count  = 0
    w_used = 0.0
    for word in words:
        w = c.stringWidth(word + " ", font, size)
        if w_used + w > max_width and w_used > 0:
            count  += 1
            w_used  = w
        else:
            w_used += w
    if w_used > 0:
        count += 1
    return count


def _wrap_text(c, text: str, x: float, y_from_top: float,
               max_width: float, font: str, size: float,
               line_height: float) -> float:
    """
    Draw word-wrapped text. Returns y_from_top of the line AFTER the last line.
    """
    words  = text.split()
    lines: list[str] = []
    buf:   list[str] = []
    w_used = 0.0

    for word in words:
        w = c.stringWidth(word + " ", font, size)
        if w_used + w > max_width and buf:
            lines.append(" ".join(buf))
            buf    = [word]
            w_used = w
        else:
            buf.append(word)
            w_used += w
    if buf:
        lines.append(" ".join(buf))

    c.setFont(font, size)
    for i, line in enumerate(lines):
        c.drawString(x, _y(y_from_top + i * line_height), line)

    return y_from_top + len(lines) * line_height


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_sheet(exam_code: str) -> tuple[bytes, dict]:
    """
    Generate the EVOLVI answer-sheet PDF.

    Returns:
        pdf_bytes    : raw PDF bytes (send directly as application/pdf)
        sheet_coords : SHEET_COORDS dict for OMR bubble detection
    """
    info = parse_exam_code(exam_code)
    log.info("Building sheet: code=%s  title=%s", info["code"], info["title"])

    buf = io.BytesIO()
    c   = rl_canvas.Canvas(buf, pagesize=LETTER)

    # =========================================================================
    # RIGHT COLUMN — CELULAR SECTION
    # =========================================================================

    # ── Celular zone markers (TL / TR / BL / BR) ─────────────────────────────
    for mx, my in [
        (CEL_MARKER_L, Y_CEL_MARKER_TOP),   # TL
        (CEL_MARKER_R, Y_CEL_MARKER_TOP),   # TR
        (CEL_MARKER_L, Y_CEL_MARKER_BOT),   # BL
        (CEL_MARKER_R, Y_CEL_MARKER_BOT),   # BR
    ]:
        _marker(c, mx, my)

    # ── "CELULAR" label ────────────────────────────────────────────────────────
    cel_center_x = (RIGHT_X + RIGHT_X_END) / 2   # = 468.5 pt
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.black)
    c.drawCentredString(cel_center_x, _y(Y_CEL_LABEL), "CELULAR")

    # ── Digit-input squares ──────────────────────────────────────────────────
    # Left edge derived from the same bubble-centre formula so squares sit
    # exactly above the corresponding bubble column.
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.setFillColor(colors.white)
    for i in range(N_CEL_COLS):
        sx = _cel_cx(i) - CEL_SQ_SIZE / 2.0
        c.rect(sx, _y(Y_CEL_SQ_BOT), CEL_SQ_SIZE, CEL_SQ_SIZE, fill=1, stroke=1)

    # ── Celular bubble grid ──────────────────────────────────────────────────
    for col in range(N_CEL_COLS):
        cx = _cel_cx(col)
        for row in range(N_CEL_ROWS):
            _bubble(c, cx, _cel_cy(row), R_CEL, label=str(row), font_size=7)

    # =========================================================================
    # LEFT COLUMN — EXAM INFO + FIELDS + INSTRUCTIONS
    # =========================================================================
    lx = LEFT_X   # left edge of content
    lw = LEFT_X_END - LEFT_X   # = 310 pt

    # ── Exam title ─────────────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.black)
    c.drawString(lx, _y(Y_TITLE), info["title"])

    # ── Exam code ──────────────────────────────────────────────────────────────
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.Color(0.40, 0.40, 0.40))
    c.drawString(lx, _y(Y_CODE), info["code"])

    # ── NOMBRE field ───────────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawString(lx, _y(Y_NOMBRE_LBL), "NOMBRE(S)")

    c.setFillColor(colors.white)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(lx, _y(Y_NOMBRE_BOT), lw, Y_NOMBRE_BOT - Y_NOMBRE_TOP, fill=1, stroke=1)

    # ── APELLIDO field ─────────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawString(lx, _y(Y_APELL_LBL), "APELLIDO(S)")

    c.setFillColor(colors.white)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(lx, _y(Y_APELL_BOT), lw, Y_APELL_BOT - Y_APELL_TOP, fill=1, stroke=1)

    # ── INSTRUCCIONES ──────────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawString(lx, _y(Y_INST_LBL), "INSTRUCCIONES:")

    _INST_FONT = "Helvetica"
    _INST_LINE_H = 9.0
    _LEFT_COL_BOTTOM = 275   # max y the left column may reach (pt from top)
    inst_size = 6.5
    n_inst_lines = _count_lines(c, INSTRUCTIONS, lw, _INST_FONT, inst_size)
    if Y_INST_TEXT + n_inst_lines * _INST_LINE_H > _LEFT_COL_BOTTOM:
        inst_size    = 6.0
        n_inst_lines = _count_lines(c, INSTRUCTIONS, lw, _INST_FONT, inst_size)

    c.setFillColor(colors.Color(0.15, 0.15, 0.15))
    _wrap_text(c, INSTRUCTIONS, lx, Y_INST_TEXT, lw, _INST_FONT, inst_size, _INST_LINE_H)

    # =========================================================================
    # SEPARATOR LINE
    # =========================================================================
    c.setStrokeColor(colors.Color(0.55, 0.55, 0.55))
    c.setLineWidth(0.5)
    c.line(LEFT_X, _y(Y_SEPARATOR), RIGHT_X_END, _y(Y_SEPARATOR))

    # =========================================================================
    # RESPUESTAS SECTION
    # =========================================================================

    # ── Respuestas zone markers (TL / TR / BL / BR) ───────────────────────────
    resp_marker_l = LEFT_X               # = 20 pt
    resp_marker_r = RIGHT_X_END - MARKER # = 584 pt
    for mx, my in [
        (resp_marker_l, Y_RESP_MARKER_TOP),   # TL
        (resp_marker_r, Y_RESP_MARKER_TOP),   # TR
        (resp_marker_l, Y_RESP_MARKER_BOT),   # BL
        (resp_marker_r, Y_RESP_MARKER_BOT),   # BR
    ]:
        _marker(c, mx, my)

    # ── Section label ──────────────────────────────────────────────────────────
    # Start at x=30 (LEFT_X + MARKER + 2) to clear the TL corner marker (x 20–28).
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.black)
    c.drawString(LEFT_X + MARKER + 2, _y(Y_RESP_LABEL), "RESPUESTAS")

    # ── A / B / C column headers ───────────────────────────────────────────────
    opts = ["A", "B", "C"]
    c.setFont("Helvetica-Bold", 6)
    c.setFillColor(colors.Color(0.35, 0.35, 0.35))
    for col in range(N_COLS):
        for i, opt in enumerate(opts):
            c.drawCentredString(_resp_cx(col, i), _y(Y_RESP_HDRS), opt)

    # ── Bubbles + question numbers ─────────────────────────────────────────────
    for col in range(N_COLS):
        for row in range(N_ROWS):
            q_num = col * N_ROWS + row + 1
            cy    = _resp_cy(row)
            a_cx  = _resp_cx(col, 0)

            # Question number — right-aligned, just left of bubble A
            c.setFont("Helvetica-Bold", 5.5)
            c.setFillColor(colors.black)
            c.drawRightString(a_cx - R_RESP - 2, _y(cy) - 2, str(q_num))

            # Bubbles A, B, C
            for i, opt in enumerate(opts):
                _bubble(c, _resp_cx(col, i), cy, R_RESP, label=opt, font_size=6)

    # =========================================================================
    # FOOTER
    # =========================================================================
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.Color(0.40, 0.40, 0.40))
    c.drawString(LEFT_X, _y(Y_FOOTER), info["code"])
    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(colors.Color(0.13, 0.31, 0.65))
    c.drawRightString(RIGHT_X_END, _y(Y_FOOTER), "EVOLVI")

    # =========================================================================
    c.showPage()
    c.save()

    sheet_coords = compute_sheet_coords()

    # ── Layout verification (printed once per build call) ─────────────────
    _last_cel_row_y   = Y_CEL_ROW0 + (N_CEL_ROWS - 1) * CEL_ROW_STEP
    _last_resp_row_y  = Y_RESP_ROW0 + (N_ROWS - 1) * QROW_STEP
    print("=== Layout Y positions (pt from page top) ===")
    print(f"  Celular TL/TR markers : {Y_CEL_MARKER_TOP} – {Y_CEL_MARKER_TOP + MARKER}")
    print(f"  Celular label         : {Y_CEL_LABEL}")
    print(f"  Digit squares         : {Y_CEL_SQ_TOP} – {Y_CEL_SQ_BOT}")
    print(f"  Celular row 0 centre  : {Y_CEL_ROW0}")
    print(f"  Celular row 9 centre  : {_last_cel_row_y}")
    print(f"  Celular BL/BR markers : {Y_CEL_MARKER_BOT} – {Y_CEL_MARKER_BOT + MARKER}")
    print(f"  Top section ends at   : {Y_CEL_MARKER_BOT + MARKER}  (limit 280)")
    print(f"  Instructions text end : {Y_INST_TEXT + n_inst_lines * _INST_LINE_H:.0f}"
          f"  (font {inst_size}pt, {n_inst_lines} lines, limit 275)")
    print(f"  Separator             : {Y_SEPARATOR}")
    print(f"  Resp TL/TR markers    : {Y_RESP_MARKER_TOP} – {Y_RESP_MARKER_TOP + MARKER}")
    print(f"  RESPUESTAS label      : {Y_RESP_LABEL}")
    print(f"  A/B/C headers         : {Y_RESP_HDRS}")
    print(f"  First resp row centre : {Y_RESP_ROW0}")
    print(f"  Last  resp row centre : {_last_resp_row_y}")
    print(f"  Last  resp row bottom : {_last_resp_row_y + R_RESP}")
    print(f"  Resp BL/BR markers    : {Y_RESP_MARKER_BOT} – {Y_RESP_MARKER_BOT + MARKER}")
    print(f"  Footer baseline       : {Y_FOOTER}  (page bottom 792)")
    print(f"  Celular sq X0         : {CEL_SQ_X0:.1f}  (first sq centre {CEL_CX0:.1f})")
    print(f"  Margin check X        : left={LEFT_X}  right_end={RIGHT_X_END}  (limits 20–592)")
    print("=============================================")

    log.info(
        "Sheet built: code=%s  resp_bubbles=%d  cel_bubbles=%d  pdf=%d bytes",
        info["code"],
        len(sheet_coords["respuestas"]) * 3,
        N_CEL_COLS * N_CEL_ROWS,
        len(buf.getvalue()),
    )
    return buf.getvalue(), sheet_coords


# ---------------------------------------------------------------------------
# CLI: python generate_sheet.py EV-ATR-JUN26
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    code = sys.argv[1] if len(sys.argv) > 1 else "EV-ATR-JUN26"
    pdf_bytes, coords = build_sheet(code)

    out_pdf  = Path("sheet.pdf")
    out_json = Path("sheet_coords.json")

    out_pdf.write_bytes(pdf_bytes)
    out_json.write_text(json.dumps(coords, indent=2, ensure_ascii=False))

    n_resp = len(coords["respuestas"])
    n_cel  = sum(len(v) for v in coords["celular"].values())
    print(f"PDF  -> {out_pdf}  ({len(pdf_bytes):,} bytes)")
    print(f"JSON -> {out_json}  ({n_resp} preguntas, {n_resp*3} resp bubbles, {n_cel} cel bubbles)")
