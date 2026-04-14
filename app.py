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

# Fixed threshold values tried in order.  50 isolates only very dark ink;
# 80 and 100 are fallbacks for lower-contrast or compressed images.
_MARKER_THRESHOLDS = (50, 80, 100)

# Absolute pixel area bounds for the solid black squares (~8 mm).
# Wide enough to work across phone photos (72 dpi-equivalent) up to
# flatbed scans (300 dpi).
_MARKER_AREA_MIN = 200
_MARKER_AREA_MAX = 5000


def _filter_marker_candidates(gray: np.ndarray, thresh_val: int) -> tuple[list, int]:
    """
    Apply a fixed BINARY_INV threshold and return filtered marker candidates.

    BINARY_INV turns pixels darker than thresh_val into white (255) and
    everything else to black — isolating the solid black squares as white blobs.

    Returns:
        candidates  : list of (cx, cy) float tuples that pass all shape filters
        total_cnts  : total raw contour count before filtering (for debug logs)
    """
    _, binary_inv = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_cnts = len(contours)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (_MARKER_AREA_MIN <= area <= _MARKER_AREA_MAX):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if not (0.5 < aspect < 2.0):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.85:
            continue

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        candidates.append((cx, cy))
        log.debug(
            "  thresh=%d candidate: centre=(%.1f, %.1f) area=%.0f aspect=%.2f solidity=%.2f",
            thresh_val, cx, cy, area, aspect, solidity,
        )

    return candidates, total_cnts


def _pick_nearest_to_corners(candidates: list, w: int, h: int) -> np.ndarray:
    """
    Given a list of (cx, cy) candidates, assign one point to each of the
    4 image corners (TL, TR, BR, BL) by nearest Euclidean distance.

    Each candidate is used at most once.  Returns an (4, 2) float32 array
    ordered TL → TR → BR → BL, matching the warpPerspective destination.
    """
    image_corners = [
        (0.0, 0.0),   # TL
        (w,   0.0),   # TR
        (w,   h  ),   # BR
        (0.0, h  ),   # BL
    ]
    pts = list(candidates)
    result = []
    used = set()

    for ic_x, ic_y in image_corners:
        best_idx = min(
            (i for i in range(len(pts)) if i not in used),
            key=lambda i: (pts[i][0] - ic_x) ** 2 + (pts[i][1] - ic_y) ** 2,
        )
        result.append(pts[best_idx])
        used.add(best_idx)

    return np.array(result, dtype=np.float32)


def find_corner_markers(gray: np.ndarray) -> np.ndarray:
    """
    Locate the 4 solid black square corner markers using progressive thresholds.

    Tries threshold values 50 → 80 → 100 in order, stopping as soon as at
    least 4 valid candidates are found.  From those candidates the 4 points
    closest to the image corners (TL/TR/BR/BL) are selected.

    Args:
        gray: single-channel (grayscale) image — no pre-thresholding required.

    Returns:
        (4, 2) float32 array ordered TL, TR, BR, BL.

    Raises:
        RuntimeError if no threshold produces ≥ 4 candidates.
    """
    h, w = gray.shape
    last_candidates: list = []

    for thresh_val in _MARKER_THRESHOLDS:
        candidates, total_cnts = _filter_marker_candidates(gray, thresh_val)
        log.info(
            "Marker search thresh=%d: %d raw contours → %d candidates",
            thresh_val, total_cnts, len(candidates),
        )

        if len(candidates) >= 4:
            ordered = _pick_nearest_to_corners(candidates, w, h)
            log.info(
                "Corner markers found at thresh=%d (TL, TR, BR, BL): %s",
                thresh_val, ordered.tolist(),
            )
            return ordered

        last_candidates = candidates  # keep for error message

    raise RuntimeError(
        f"Could not find 4 corner markers after trying thresholds {_MARKER_THRESHOLDS}. "
        f"Last attempt found {len(last_candidates)} candidate(s). "
        "Check that the 4 black corner squares are fully visible and not cropped."
    )


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

    # 2. Find corner markers directly on the grayscale image.
    #    The new detector applies its own fixed-threshold binarisation
    #    internally, so we do NOT pre-threshold here.
    corners = find_corner_markers(gray)

    # 3. Gaussian blur + Otsu binarisation — used only for bubble reading
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    log.debug("Otsu threshold OK (bubble reading)")

    # 4. Perspective correction
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
# Debug endpoint — contour counts per threshold, no full OMR
# ---------------------------------------------------------------------------
@app.route("/debug", methods=["GET", "POST"])
def debug():
    """
    Inspect how many contours and marker candidates are found at each
    threshold without running the full OMR pipeline.

    Accepts JSON body (works with both GET and POST):
        { "imagen_base64": "<base64 string>" }

    Returns per-threshold breakdown plus basic image info:
        {
          "image_shape": [h, w],
          "thresholds": {
            "50":  {"raw_contours": 312, "after_area": 8, "after_aspect": 6, "after_solidity": 4},
            "80":  {...},
            "100": {...}
          }
        }
    """
    log.info("%s /debug received", request.method)

    data = request.get_json(silent=True) or {}
    if "imagen_base64" not in data:
        return jsonify({"ok": False, "error": "Missing field: imagen_base64"}), 400

    try:
        img = decode_image(data["imagen_base64"])
    except Exception as exc:
        log.error("Debug image decode failed: %s", exc)
        return jsonify({"ok": False, "error": f"Image decode error: {exc}"}), 422

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    threshold_results = {}

    for thresh_val in _MARKER_THRESHOLDS:
        _, binary_inv = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw = len(contours)

        after_area = after_aspect = after_solidity = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (_MARKER_AREA_MIN <= area <= _MARKER_AREA_MAX):
                continue
            after_area += 1

            bx, by, bw2, bh2 = cv2.boundingRect(cnt)
            aspect = bw2 / bh2 if bh2 > 0 else 0
            if not (0.5 < aspect < 2.0):
                continue
            after_aspect += 1

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            if (area / hull_area) < 0.85:
                continue
            after_solidity += 1

        threshold_results[str(thresh_val)] = {
            "raw_contours": raw,
            "after_area_filter": after_area,
            "after_aspect_filter": after_aspect,
            "after_solidity_filter": after_solidity,
        }
        log.info(
            "Debug thresh=%d: raw=%d area=%d aspect=%d solidity=%d",
            thresh_val, raw, after_area, after_aspect, after_solidity,
        )

    return jsonify({
        "ok": True,
        "image_shape": [h, w],
        "marker_area_range": [_MARKER_AREA_MIN, _MARKER_AREA_MAX],
        "thresholds": threshold_results,
    }), 200


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
