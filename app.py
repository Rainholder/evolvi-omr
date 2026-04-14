"""
OMR (Optical Mark Recognition) microservice for answer sheet processing.

Sheet layout (known, fixed):
  - 6 columns x 15 rows = 90 questions
  - 3 bubbles per question: A (left), B (center), C (right)
  - 4 solid black square corner markers delimiting the answer grid
  - Bubble diameter: ~14 mm | Center-to-center spacing: 18 mm
"""

import base64
import logging
import sys
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("omr")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Sheet constants  (all values are in the *warped* coordinate system after
# perspective correction; calibrated to a 1 200 x 900 px output canvas)
# ---------------------------------------------------------------------------
WARP_W = 1200          # px — width  of the rectified grid area
WARP_H = 900           # px — height of the rectified grid area

N_COLS = 6             # groups of questions side by side
N_ROWS = 15            # questions per column
N_OPTIONS = 3          # A, B, C

# Spacing (px) in the warped canvas – derived from 18 mm @ ~50 px/mm
COL_STEP = WARP_W / N_COLS          # 200 px between column centres
ROW_STEP = WARP_H / N_ROWS          # 60  px between row centres

# First bubble centre (top-left bubble = Q1-A)
FIRST_X = COL_STEP / 2             # 100 px
FIRST_Y = ROW_STEP / 2             # 30  px

# Bubble geometry
BUBBLE_RADIUS = 18                  # px (≈ 14 mm scaled)
OPTION_STEP = 40                    # px between A/B/C centres in same row

# Decision threshold: fraction of bubble area that must be dark to be "filled"
FILL_THRESHOLD = 0.18               # 18 % of the bubble circle


# ---------------------------------------------------------------------------
# Helper: decode base64 → OpenCV image
# ---------------------------------------------------------------------------
def decode_image(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded image (with or without data-URI prefix)."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — check base64 payload")
    log.debug("Decoded image shape: %s", img.shape)
    return img


# ---------------------------------------------------------------------------
# Helper: find the 4 corner markers
# ---------------------------------------------------------------------------
def find_corner_markers(binary: np.ndarray) -> np.ndarray:
    """
    Locate the 4 solid black square corner markers.

    Strategy:
      1. Find all external contours on the *inverted* binary (markers are
         dark, so they appear as white blobs on an inverted image).
      2. Filter by:
         - Roughly square aspect ratio (0.7 – 1.4)
         - Area within a plausible range for ~8 mm squares
         - High solidity (> 0.85) — they are solid squares, not rings
      3. Return the 4 corners ordered: TL, TR, BR, BL.

    Returns an (4, 2) float32 array of corner centres.
    """
    h, w = binary.shape

    # Markers are black → they are 0 in the binary image.
    # Invert so they become white foreground.
    inverted = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    log.debug("Total contours found: %d", len(contours))

    # Plausible area range: markers are ~8 mm; at typical scan DPI the marker
    # covers 0.3–3 % of the image area.
    img_area = h * w
    min_area = img_area * 0.0008
    max_area = img_area * 0.03

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.80:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if not (0.6 < aspect < 1.7):
            continue

        cx = x + bw / 2
        cy = y + bh / 2
        candidates.append((cx, cy, area))
        log.debug("Marker candidate: centre=(%.1f, %.1f) area=%.0f solidity=%.2f aspect=%.2f",
                  cx, cy, area, solidity, aspect)

    if len(candidates) < 4:
        raise RuntimeError(
            f"Expected 4 corner markers, found only {len(candidates)} candidates. "
            "Check image quality and that markers are visible."
        )

    # If more than 4 candidates, keep the 4 most extreme (one per quadrant)
    centres = np.array([(c[0], c[1]) for c in candidates], dtype=np.float32)
    ordered = _order_four_corners(centres, w, h)
    log.info("Corner markers (TL, TR, BR, BL): %s", ordered.tolist())
    return ordered


def _order_four_corners(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Given N candidate points, pick the best one for each quadrant (TL/TR/BR/BL)
    and return them in that order.
    """
    mid_x, mid_y = w / 2, h / 2

    def quadrant_score(pt, prefer_left: bool, prefer_top: bool):
        x, y = pt
        dx = (mid_x - x) if prefer_left else (x - mid_x)
        dy = (mid_y - y) if prefer_top  else (y - mid_y)
        return dx + dy   # larger = more extreme in that quadrant

    quads = [
        (True,  True),   # TL
        (False, True),   # TR
        (False, False),  # BR
        (True,  False),  # BL
    ]

    result = []
    used = set()
    for prefer_left, prefer_top in quads:
        best_idx = max(
            (i for i in range(len(pts)) if i not in used),
            key=lambda i: quadrant_score(pts[i], prefer_left, prefer_top),
        )
        result.append(pts[best_idx])
        used.add(best_idx)

    return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper: perspective correction
# ---------------------------------------------------------------------------
def warp_perspective(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Rectify the answer grid using the 4 detected corner markers.
    The output is always WARP_W x WARP_H pixels.
    """
    dst = np.array([
        [0,      0     ],
        [WARP_W, 0     ],
        [WARP_W, WARP_H],
        [0,      WARP_H],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (WARP_W, WARP_H))
    log.debug("Warped image shape: %s", warped.shape)
    return warped


# ---------------------------------------------------------------------------
# Helper: bubble reading
# ---------------------------------------------------------------------------
def bubble_dark_fraction(binary_warped: np.ndarray, cx: int, cy: int, r: int) -> float:
    """
    Return the fraction of pixels inside the circle (cx, cy, r) that are dark
    (value == 0 in the binary image where dark = filled bubble).
    """
    h, w = binary_warped.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    total_px = cv2.countNonZero(mask)
    if total_px == 0:
        return 0.0
    # In the binary: 0 = dark (ink/filled), 255 = white (paper)
    # We want dark pixels → where binary_warped == 0 AND mask == 255
    dark_px = int(np.sum((binary_warped == 0) & (mask == 255)))
    return dark_px / total_px


def read_bubbles(binary_warped: np.ndarray, total_preguntas: int) -> dict:
    """
    Scan every bubble position and return a dict {question_number: answer_letter}.
    answer_letter is "A", "B", "C" or None.
    """
    results = {}
    filled_count = 0
    dark_fractions_log = []

    for q in range(1, total_preguntas + 1):
        # Map question number → (col_index, row_index)
        col_idx = (q - 1) // N_ROWS          # 0-based column
        row_idx = (q - 1) % N_ROWS           # 0-based row within column

        # Centre of the *first* option (A) for this question
        base_x = int(FIRST_X + col_idx * COL_STEP)
        base_y = int(FIRST_Y + row_idx * ROW_STEP)

        fractions = []
        for opt in range(N_OPTIONS):
            cx = base_x + opt * OPTION_STEP
            cy = base_y
            frac = bubble_dark_fraction(binary_warped, cx, cy, BUBBLE_RADIUS)
            fractions.append(frac)

        best_opt = int(np.argmax(fractions))
        best_frac = fractions[best_opt]

        dark_fractions_log.append((q, fractions))

        if best_frac >= FILL_THRESHOLD:
            answer = chr(ord("A") + best_opt)
            results[str(q)] = answer
            filled_count += 1
        else:
            results[str(q)] = None

    log.debug("Sample dark fractions (first 10): %s", dark_fractions_log[:10])
    log.info("Filled bubbles: %d / %d", filled_count, total_preguntas)
    return results


# ---------------------------------------------------------------------------
# Core OMR pipeline
# ---------------------------------------------------------------------------
def process_omr(img: np.ndarray, total_preguntas: int) -> dict:
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log.debug("Grayscale OK")

    # 2. Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Otsu binarisation — separates dark ink from white paper
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    log.debug("Otsu threshold OK")

    # 4. Find corner markers on the binary image
    corners = find_corner_markers(binary)

    # 5. Perspective correction
    warped_color = warp_perspective(img, corners)
    warped_gray  = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    _, warped_bin = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    log.debug("Perspective warp OK")

    # 6. Read bubbles
    respuestas = read_bubbles(warped_bin, total_preguntas)

    # 7. Confidence: fraction of questions with a detected answer
    total_detectadas = sum(1 for v in respuestas.values() if v is not None)
    confianza = round(total_detectadas / total_preguntas, 4) if total_preguntas else 0.0

    return {
        "ok": True,
        "respuestas": respuestas,
        "total_detectadas": total_detectadas,
        "confianza": confianza,
    }


# ---------------------------------------------------------------------------
# Flask endpoint
# ---------------------------------------------------------------------------
@app.route("/procesar", methods=["POST"])
def procesar():
    log.info("POST /procesar received")

    data = request.get_json(silent=True)
    if not data:
        log.warning("Empty or non-JSON body")
        return jsonify({"ok": False, "error": "Body must be JSON"}), 400

    # Validate required fields
    if "imagen_base64" not in data:
        return jsonify({"ok": False, "error": "Missing field: imagen_base64"}), 400

    total_preguntas = data.get("total_preguntas", 90)
    if not isinstance(total_preguntas, int) or not (1 <= total_preguntas <= 300):
        return jsonify({"ok": False, "error": "total_preguntas must be an integer 1–300"}), 400

    try:
        img = decode_image(data["imagen_base64"])
    except Exception as exc:
        log.error("Image decode failed: %s", exc)
        return jsonify({"ok": False, "error": f"Image decode error: {exc}"}), 422

    try:
        result = process_omr(img, total_preguntas)
        log.info("OMR OK — detectadas=%d confianza=%.4f", result["total_detectadas"], result["confianza"])
        return jsonify(result), 200
    except RuntimeError as exc:
        log.error("OMR pipeline error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 422
    except Exception as exc:
        log.error("Unexpected error:\n%s", traceback.format_exc())
        return jsonify({"ok": False, "error": "Internal server error", "detail": str(exc)}), 500


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "evolvi-omr"}), 200


# ---------------------------------------------------------------------------
# Entry point (dev only — production uses gunicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
