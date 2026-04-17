"""
generate_sheet.py — EVOLVI OMR answer-sheet PDF generator.

Entry point:
    pdf_bytes, sheet_coords = build_sheet("EV-ATR-JUN26")

Page layout (Letter 612×792 pt, 35 pt margins):

    ┌─────────────────────────────────────────┐
    │■(TL,16)  ▪(edge_top,10)       ■(TR,16) │
    │ LEFT COL (35–330) │ RIGHT COL (345–577) │
    │  Exam title 14pt  │  ▫ CELULAR ▫        │
    │  Exam code        │  [ ][ ]…[ ]         │
    │  NOMBRE rect      │  ○ ○ ○ ○ ○ …       │
    │  APELLIDOS rect   │  (10×10 grid)       │
    │  INSTRUCCIONES    │  ▫           ▫      │
    ├────── separator line ───────────────────┤
    │  ▫ RESPUESTAS                  ▫        │
    │  6 cols × 15 rows of A B C bubbles     │
    │■(BL,16)  ▪(edge_btm,10)       ■(BR,16) │
    │  EV-ATR-JUN26                  EVOLVI   │
    └─────────────────────────────────────────┘

Markers:
  ■ LARGE  (16×16 pt) — 4 page corners
  ▪ MEDIUM (10×10 pt) — 4 edge centres
  ▫ SMALL  ( 8× 8 pt) — 4 zone corners (celular TL/TR + resp TL/TR)

Coordinates in sheet_coords are pixels at 300 DPI (top-left origin),
matching a 2550×3300 px image.  Conversion: px = pt × (300/72).
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
    "EV-ATR-JUN26" -> {code, area: "ATR", title: "ÁREAS TRANSVERSALES", period: "JUN26"}
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
PAGE_W, PAGE_H = LETTER      # 612 x 792 pt
MARGIN = 35                  # pt — all four sides

# Export resolution
DPI   = 300
PT_PX = DPI / 72.0           # ~4.1667 px/pt  ->  Letter = 2550 x 3300 px

# ---------------------------------------------------------------------------
# Marker sizes (pt)
# ---------------------------------------------------------------------------
MARKER_LARGE  = 16   # 4 page-corner markers
MARKER_MEDIUM = 10   # 4 edge-centre markers
MARKER_SMALL  =  8   # 4 zone markers (kept for backward compat as MARKER)
MARKER        = MARKER_SMALL   # alias used in drawing helpers

# ---------------------------------------------------------------------------
# Marker top-left positions (pt from page top-left, y measured from TOP)
# ---------------------------------------------------------------------------

# Large page-corner markers (16x16 pt)
_CORNER_TL_X, _CORNER_TL_Y = 35,  35
_CORNER_TR_X, _CORNER_TR_Y = 560, 35
_CORNER_BL_X, _CORNER_BL_Y = 35,  640
_CORNER_BR_X, _CORNER_BR_Y = 560, 640

# Medium edge-centre markers (10x10 pt)
_EDGE_TOP_X,    _EDGE_TOP_Y    = 306, 35    # horizontally centred: 306+5 = 311 ~ 612/2
_EDGE_BOTTOM_X, _EDGE_BOTTOM_Y = 306, 640
_EDGE_LEFT_X,   _EDGE_LEFT_Y   = 35,  320   # vertically centred: 396+5 = 401 ~ 792/2
_EDGE_RIGHT_X,  _EDGE_RIGHT_Y  = 560, 320

# ---------------------------------------------------------------------------
# Two-column split
# ---------------------------------------------------------------------------
LEFT_X      = 60     # = MARGIN
LEFT_X_END  = 330    # left col end
RIGHT_X     = 345    # right col start (15 pt gap)
RIGHT_X_END = 550    # right col end = 612 - MARGIN (was 592)

# ---------------------------------------------------------------------------
# CELULAR layout (right column, width 232 pt)
# ---------------------------------------------------------------------------
N_CEL_COLS  = 10
N_CEL_ROWS  = 10
CEL_SQ_SIZE = 22     # pt side of each digit-input square

_RIGHT_COL_W    = RIGHT_X_END - RIGHT_X                          # = 232 pt
_CEL_SQ_OFFSET  = (_RIGHT_COL_W - N_CEL_COLS * CEL_SQ_SIZE) / 2.0  # = 6 pt
CEL_SQ_X0       = RIGHT_X + _CEL_SQ_OFFSET                      # = 351 pt (left edge of sq 0)

R_CEL        = 7     # pt radius  (diameter 14 pt, was 16)
CEL_CX0      = CEL_SQ_X0 + CEL_SQ_SIZE / 2.0   # = 362 pt (centre of col 0)
CEL_COL_STEP = CEL_SQ_SIZE                       # = 22 pt
CEL_ROW_STEP = 20                                # pt  (vertical centre-to-centre)

# Celular zone small marker x positions
CEL_MARKER_L = RIGHT_X                           # = 345 pt
CEL_MARKER_R = RIGHT_X_END - MARKER_SMALL        # = 569 pt  (right edge = 577)

# Respuestas zone small marker x positions
RESP_MARKER_L = LEFT_X                           # = 35 pt
RESP_MARKER_R = RIGHT_X_END - MARKER_SMALL       # = 569 pt

# ---------------------------------------------------------------------------
# Vertical positions (pt from page TOP)
# ---------------------------------------------------------------------------

# ── Celular section ──────────────────────────────────────────────────────────
Y_CEL_MARKER_TOP = 35    # = MARGIN  (same y as large TL/TR corners; different x)
Y_CEL_LABEL      = 60    # "CELULAR" text baseline
Y_CEL_SQ_TOP     = 65    # digit-input squares top edge
Y_CEL_SQ_BOT     = 85    # = Y_CEL_SQ_TOP + CEL_SQ_SIZE
Y_CEL_ROW0       = 95    # first bubble-row centre  (digit 0)
#   Last row centre   : Y_CEL_ROW0 + 9 x 20 = 272 pt
#   Last bubble bottom: 272 + R_CEL = 279 pt
Y_CEL_MARKER_BOT = 282   # bottom small zone markers  (bottom edge = 290)

# ── Left column ──────────────────────────────────────────────────────────────
Y_TITLE      = 70    # exam title baseline (14 pt bold)
Y_CODE       = 87    # exam code baseline  (11 pt)
Y_NOMBRE_LBL = 101    # "NOMBRE(S):" label baseline
Y_NOMBRE_TOP = 105    # NOMBRE rect top edge
Y_NOMBRE_BOT = 133   # = Y_NOMBRE_TOP + 28 pt
Y_APELL_LBL  = 145   # "APELLIDO(S):" label baseline
Y_APELL_TOP  = 148   # APELLIDOS rect top edge
Y_APELL_BOT  = 176   # = Y_APELL_TOP + 28 pt
Y_INST_LBL   = 190   # "INSTRUCCIONES:" label baseline
Y_INST_TEXT  = 200   # first instruction text-line baseline

_LEFT_COL_BOTTOM = 285   # max y the left column may reach before separator

# ── Separator & Respuestas section ───────────────────────────────────────────
Y_RESP_LABEL     = 282   # "RESPUESTAS" label baseline (left col, above separator)
Y_SEPARATOR      = 293   # horizontal separator line
Y_RESP_MARKER_TOP = 298  # small zone markers top  (bottom edge = 306)
Y_RESP_HDRS      = 316   # A / B / C column-header baseline
Y_RESP_ROW0      = 326   # first question-row bubble centre

N_COLS    = 6
N_ROWS    = 15
R_RESP    = 6.5   # pt radius  (diameter 13 pt, was 16)
OPT_STEP  = 18    # pt horizontal centre-to-centre A->B and B->C
QROW_STEP = 21    # pt vertical centre-to-centre between questions

#   Last row centre   : Y_RESP_ROW0 + 14 x 22 = 634 pt
#   Last bubble bottom: 634 + R_RESP = 640.5 pt
Y_RESP_MARKER_BOT = 649  # bottom small zone markers  (bottom edge = 657 < 757 ✓)
Y_FOOTER          = 778  # footer text baseline (below BL/BR markers at 757-773)

# Respuestas column geometry
# Content starts 2 pt after the right edge of the RESP_MARKER_L square.
_RESP_CONTENT_X0 = LEFT_X + MARKER_SMALL + 2   # = 45 pt
_RESP_COL_PITCH  = 79    # pt between column starts  (6 x 87 = 522 pt; fits in 45-657)
_RESP_NUM_W      = 14    # pt reserved for question-number label

#   A_cx(col) = _RESP_CONTENT_X0 + _RESP_NUM_W + R_RESP + col x _RESP_COL_PITCH
#             = 45 + 13 + 6.5 + col x 87  =  64.5 + col x 87
#   B_cx(col) = A_cx + OPT_STEP           =  84.5 + col x 87
#   C_cx(col) = B_cx + OPT_STEP           = 104.5 + col x 87
#
#   Verification col 5:  C_cx = 104.5 + 435 = 539.5
#                        C right edge = 539.5 + 6.5 = 546 < 569 (resp_TR marker) ✓

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
# Coordinate helpers
# ---------------------------------------------------------------------------
def _y(y_from_top: float) -> float:
    """Top-anchored y -> ReportLab bottom-anchored y."""
    return PAGE_H - y_from_top


def _pt2px(x_pt: float, y_from_top: float) -> list[int]:
    """PDF point (bottom-left origin) -> pixel (top-left origin, 300 DPI)."""
    return [round(x_pt * PT_PX), round(y_from_top * PT_PX)]


def _marker_center_px(x_left: float, y_top: float, size: float) -> dict:
    """Return pixel-space centre of a marker square plus metadata."""
    half = size / 2.0
    return {
        "x": round((x_left + half) * PT_PX),
        "y": round((y_top  + half) * PT_PX),
    }


# ---------------------------------------------------------------------------
# Exact bubble centres (PDF points)
# ---------------------------------------------------------------------------
def _resp_cx(col: int, opt_idx: int) -> float:
    """x-centre of option opt_idx (0=A, 1=B, 2=C) in column col."""
    return _RESP_CONTENT_X0 + _RESP_NUM_W + R_RESP + col * _RESP_COL_PITCH + opt_idx * OPT_STEP


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
          "celular":    {"0": {"0":[x,y], …, "9":[x,y]}, …, "9":{…}},
          "respuestas": {"1": {"A":[x,y], "B":[x,y], "C":[x,y]}, …, "90":{…}},
          "markers": {
            "corner_TL":  {"x": px, "y": px, "size": "large",  "pt": 16},
            "corner_TR":  {…},
            "corner_BL":  {…},
            "corner_BR":  {…},
            "edge_top":   {"x": px, "y": px, "size": "medium", "pt": 10},
            "edge_bottom":{…},
            "edge_left":  {…},
            "edge_right": {…},
            "celular_TL": {"x": px, "y": px, "size": "small",  "pt": 8},
            "celular_TR": {…},
            "resp_TL":    {…},
            "resp_TR":    {…},
          }
        }
    """
    opts = ["A", "B", "C"]

    respuestas: dict = {}
    for col in range(N_COLS):
        for row in range(N_ROWS):
            q  = col * N_ROWS + row + 1
            cy = _resp_cy(row)
            respuestas[str(q)] = {
                opts[i]: _pt2px(_resp_cx(col, i), cy) for i in range(3)
            }

    celular: dict = {}
    for pos in range(N_CEL_COLS):
        cx      = _cel_cx(pos)
        row_map = {str(dig): _pt2px(cx, _cel_cy(dig)) for dig in range(N_CEL_ROWS)}
        celular[str(pos)] = row_map

    # ── Marker centres in pixel space ────────────────────────────────────────
    def _cm(x, y, size, label):
        return {**_marker_center_px(x, y, size), "size": label, "pt": size}

    markers = {
        # Large page-corner markers (16x16 pt)
        "corner_TL":   _cm(_CORNER_TL_X, _CORNER_TL_Y, MARKER_LARGE,  "large"),
        "corner_TR":   _cm(_CORNER_TR_X, _CORNER_TR_Y, MARKER_LARGE,  "large"),
        "corner_BL":   _cm(_CORNER_BL_X, _CORNER_BL_Y, MARKER_LARGE,  "large"),
        "corner_BR":   _cm(_CORNER_BR_X, _CORNER_BR_Y, MARKER_LARGE,  "large"),
        # Medium edge-centre markers (10x10 pt)
        "edge_top":    _cm(_EDGE_TOP_X,    _EDGE_TOP_Y,    MARKER_MEDIUM, "medium"),
        "edge_bottom": _cm(_EDGE_BOTTOM_X, _EDGE_BOTTOM_Y, MARKER_MEDIUM, "medium"),
        "edge_left":   _cm(_EDGE_LEFT_X,   _EDGE_LEFT_Y,   MARKER_MEDIUM, "medium"),
        "edge_right":  _cm(_EDGE_RIGHT_X,  _EDGE_RIGHT_Y,  MARKER_MEDIUM, "medium"),
        # Small zone-corner markers (8x8 pt)
        "celular_TL":  _cm(CEL_MARKER_L,  Y_CEL_MARKER_TOP,  MARKER_SMALL, "small"),
        "celular_TR":  _cm(CEL_MARKER_R,  Y_CEL_MARKER_TOP,  MARKER_SMALL, "small"),
        "resp_TL":     _cm(RESP_MARKER_L, Y_RESP_MARKER_TOP, MARKER_SMALL, "small"),
        "resp_TR":     _cm(RESP_MARKER_R, Y_RESP_MARKER_TOP, MARKER_SMALL, "small"),
    }

    return {"celular": celular, "respuestas": respuestas, "markers": markers}


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _marker(c, x_left: float, y_top: float, size: float = MARKER_SMALL) -> None:
    """Draw a solid black square marker. y_top is from page top."""
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)
    c.rect(x_left, _y(y_top + size), size, size, fill=1, stroke=0)


def _bubble(c, cx: float, y_from_top: float, r: float,
            label: str = "", font_size: float = 7.0) -> None:
    """Draw an empty bubble (circle): gray border, white fill, light gray label."""
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
    """Return the number of wrapped lines without drawing."""
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
    """Draw word-wrapped text. Returns y_from_top after the last line."""
    words = text.split()
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
        pdf_bytes    : raw PDF bytes
        sheet_coords : SHEET_COORDS dict for OMR bubble detection
    """
    info = parse_exam_code(exam_code)
    log.info("Building sheet: code=%s  title=%s", info["code"], info["title"])

    buf = io.BytesIO()
    c   = rl_canvas.Canvas(buf, pagesize=LETTER)

    # =========================================================================
    # MARKERS — 12 total
    # =========================================================================

    # ── Large page-corner markers (16x16 pt) ─────────────────────────────────
    for mx, my in [
        (_CORNER_TL_X, _CORNER_TL_Y),
        (_CORNER_TR_X, _CORNER_TR_Y),
        (_CORNER_BL_X, _CORNER_BL_Y),
        (_CORNER_BR_X, _CORNER_BR_Y),
    ]:
        _marker(c, mx, my, MARKER_LARGE)

    # ── Medium edge-centre markers (10x10 pt) ────────────────────────────────
    for mx, my in [
        (_EDGE_TOP_X,    _EDGE_TOP_Y),
        (_EDGE_BOTTOM_X, _EDGE_BOTTOM_Y),
        (_EDGE_LEFT_X,   _EDGE_LEFT_Y),
        (_EDGE_RIGHT_X,  _EDGE_RIGHT_Y),
    ]:
        _marker(c, mx, my, MARKER_MEDIUM)

    # ── Small celular zone markers (TL + TR only) ────────────────────────────
    for mx, my in [
        (CEL_MARKER_L, Y_CEL_MARKER_TOP),   # celular_TL
        (CEL_MARKER_R, Y_CEL_MARKER_TOP),   # celular_TR
    ]:
        _marker(c, mx, my, MARKER_SMALL)

    # ── Small respuestas zone markers (TL + TR only) ─────────────────────────
    for mx, my in [
        (RESP_MARKER_L, Y_RESP_MARKER_TOP),  # resp_TL
        (RESP_MARKER_R, Y_RESP_MARKER_TOP),  # resp_TR
    ]:
        _marker(c, mx, my, MARKER_SMALL)

    # =========================================================================
    # RIGHT COLUMN — CELULAR SECTION
    # =========================================================================

    # ── "CELULAR" label ───────────────────────────────────────────────────────
    cel_center_x = (RIGHT_X + RIGHT_X_END) / 2.0   # = 461 pt
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.black)
    c.drawCentredString(cel_center_x, _y(Y_CEL_LABEL), "CELULAR")

    # ── Digit-input squares ──────────────────────────────────────────────────
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
    lx = LEFT_X
    lw = LEFT_X_END - LEFT_X   # = 295 pt

    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.black)
    c.drawString(lx, _y(Y_TITLE), info["title"])

    c.setFont("Helvetica", 11)
    c.setFillColor(colors.Color(0.40, 0.40, 0.40))
    c.drawString(lx, _y(Y_CODE), info["code"])

    # NOMBRE field
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawString(lx, _y(Y_NOMBRE_LBL), "NOMBRE(S)")
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(lx, _y(Y_NOMBRE_BOT), lw, Y_NOMBRE_BOT - Y_NOMBRE_TOP, fill=1, stroke=1)

    # APELLIDOS field
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawString(lx, _y(Y_APELL_LBL), "APELLIDO(S)")
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(lx, _y(Y_APELL_BOT), lw, Y_APELL_BOT - Y_APELL_TOP, fill=1, stroke=1)

    # INSTRUCCIONES
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawString(lx, _y(Y_INST_LBL), "INSTRUCCIONES:")

    _INST_FONT   = "Helvetica"
    _INST_LINE_H = 9.0
    inst_size    = 6.5
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

    # ── Section label ─────────────────────────────────────────────────────────
    # Draw just right of resp_TL marker (x = RESP_MARKER_L + MARKER_SMALL + 2)
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.black)
    c.drawString(RESP_MARKER_L + MARKER_SMALL + 2, _y(Y_RESP_LABEL), "RESPUESTAS")

    # ── A / B / C column headers ──────────────────────────────────────────────
    opts = ["A", "B", "C"]
    c.setFont("Helvetica-Bold", 6)
    c.setFillColor(colors.Color(0.35, 0.35, 0.35))
    for col in range(N_COLS):
        for i, opt in enumerate(opts):
            c.drawCentredString(_resp_cx(col, i), _y(Y_RESP_HDRS), opt)

    # ── Bubbles + question numbers ────────────────────────────────────────────
    for col in range(N_COLS):
        for row in range(N_ROWS):
            q_num = col * N_ROWS + row + 1
            cy    = _resp_cy(row)
            a_cx  = _resp_cx(col, 0)

            # Question number right-aligned just left of bubble A
            c.setFont("Helvetica-Bold", 5.5)
            c.setFillColor(colors.black)
            c.drawRightString(a_cx - R_RESP - 2, _y(cy) - 2, str(q_num))

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

    # ── Layout verification ───────────────────────────────────────────────────
    _last_cel_row_y  = Y_CEL_ROW0  + (N_CEL_ROWS - 1) * CEL_ROW_STEP
    _last_resp_row_y = Y_RESP_ROW0 + (N_ROWS - 1)     * QROW_STEP
    _inst_end_y      = Y_INST_TEXT + n_inst_lines * _INST_LINE_H

    print("=" * 62)
    print("  EVOLVI sheet layout — all positions in pt from page top")
    print("=" * 62)
    print(f"  Page              : {PAGE_W:.0f} x {PAGE_H:.0f} pt  "
          f"({round(PAGE_W*PT_PX)} x {round(PAGE_H*PT_PX)} px @ 300 dpi)")
    print(f"  Margin            : {MARGIN} pt")
    print()
    print("  MARKERS (12 total):")
    print(f"  Large (16x16)     : TL({_CORNER_TL_X},{_CORNER_TL_Y})  "
          f"TR({_CORNER_TR_X},{_CORNER_TR_Y})  "
          f"BL({_CORNER_BL_X},{_CORNER_BL_Y})  "
          f"BR({_CORNER_BR_X},{_CORNER_BR_Y})")
    print(f"  Medium (10x10)    : top({_EDGE_TOP_X},{_EDGE_TOP_Y})  "
          f"btm({_EDGE_BOTTOM_X},{_EDGE_BOTTOM_Y})  "
          f"left({_EDGE_LEFT_X},{_EDGE_LEFT_Y})  "
          f"right({_EDGE_RIGHT_X},{_EDGE_RIGHT_Y})")
    print(f"  Small celular (8) : TL({CEL_MARKER_L},{Y_CEL_MARKER_TOP})  "
          f"TR({CEL_MARKER_R},{Y_CEL_MARKER_TOP})")
    print(f"  Small resp    (8) : TL({RESP_MARKER_L},{Y_RESP_MARKER_TOP})  "
          f"TR({RESP_MARKER_R},{Y_RESP_MARKER_TOP})")
    print()
    print("  CELULAR SECTION (right col):")
    print(f"  Right col x       : {RIGHT_X} – {RIGHT_X_END} ({_RIGHT_COL_W:.0f} pt wide)")
    print(f"  Bubble diameter   : {R_CEL*2:.0f} pt  radius={R_CEL}")
    print(f"  CEL_SQ_X0         : {CEL_SQ_X0:.1f}  CEL_CX0={CEL_CX0:.1f}")
    print(f"  Small mkr TL/TR y : {Y_CEL_MARKER_TOP} – {Y_CEL_MARKER_TOP + MARKER_SMALL}")
    print(f"  Label y           : {Y_CEL_LABEL}")
    print(f"  Digit squares y   : {Y_CEL_SQ_TOP} – {Y_CEL_SQ_BOT}")
    print(f"  Bubble row 0 y    : {Y_CEL_ROW0}")
    print(f"  Bubble row 9 y    : {_last_cel_row_y}  "
          f"bottom={_last_cel_row_y + R_CEL:.1f}")
    print(f"  Small mkr BL/BR y : — (none; large corner markers cover bottom)")
    print()
    print("  LEFT COLUMN:")
    print(f"  Left col x        : {LEFT_X} – {LEFT_X_END} ({LEFT_X_END-LEFT_X} pt wide)")
    print(f"  Title y           : {Y_TITLE}")
    print(f"  Code y            : {Y_CODE}")
    print(f"  NOMBRE rect y     : {Y_NOMBRE_TOP} – {Y_NOMBRE_BOT}")
    print(f"  APELLIDO rect y   : {Y_APELL_TOP} – {Y_APELL_BOT}")
    print(f"  Inst text start y : {Y_INST_TEXT}  "
          f"(font {inst_size}pt, {n_inst_lines} lines)")
    print(f"  Inst text end y   : {_inst_end_y:.0f}  (limit {_LEFT_COL_BOTTOM})")
    overlap_warn = " *** OVERLAP ***" if _inst_end_y > _LEFT_COL_BOTTOM else " OK"
    print(f"  Left col / limit  :{overlap_warn}")
    print()
    print("  RESPUESTAS SECTION:")
    print(f"  Separator y       : {Y_SEPARATOR}")
    print(f"  Small mkr TL/TR y : {Y_RESP_MARKER_TOP} – {Y_RESP_MARKER_TOP + MARKER_SMALL}")
    print(f"  Bubble diameter   : {R_RESP*2:.0f} pt  radius={R_RESP}  OPT_STEP={OPT_STEP}")
    print(f"  A/B/C headers y   : {Y_RESP_HDRS}")
    print(f"  Bubble row 0 y    : {Y_RESP_ROW0}")
    print(f"  Bubble row 14 y   : {_last_resp_row_y}  "
          f"bottom={_last_resp_row_y + R_RESP:.1f}")
    print(f"  Content x range   : A_col0={_resp_cx(0,0):.1f} – "
          f"C_col5={_resp_cx(5,2):.1f}  "
          f"(right edge {_resp_cx(5,2)+R_RESP:.1f})")
    print(f"  Resp TR marker x  : {RESP_MARKER_R}  "
          f"gap={RESP_MARKER_R - (_resp_cx(5,2)+R_RESP):.1f} pt")
    print()
    print("  BOTTOM BOUNDARY CHECK:")
    print(f"  Resp content ends : {_last_resp_row_y + R_RESP:.1f} pt")
    print(f"  BL/BR corner mkrs : {_CORNER_BL_Y} – {_CORNER_BL_Y + MARKER_LARGE}  "
          f"gap={(float(_CORNER_BL_Y) - (_last_resp_row_y + R_RESP)):.1f} pt")
    bottom_ok = _last_resp_row_y + R_RESP < _CORNER_BL_Y
    print(f"  Resp / BL marker  : {'OK' if bottom_ok else '*** OVERLAP ***'}")
    print(f"  Footer y          : {Y_FOOTER}  "
          f"(below BL bottom {_CORNER_BL_Y + MARKER_LARGE})")
    print("=" * 62)

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
    n_mkr  = len(coords["markers"])
    print(f"PDF  -> {out_pdf}  ({len(pdf_bytes):,} bytes)")
    print(f"JSON -> {out_json}  "
          f"({n_resp} preguntas, {n_resp*3} resp bubbles, "
          f"{n_cel} cel bubbles, {n_mkr} markers)")
    print()
    print("Markers in JSON:")
    for name, m in coords["markers"].items():
        print(f"  {name:15s}  x={m['x']:4d}  y={m['y']:4d}  "
              f"size={m['size']:6s}  pt={m['pt']}")
