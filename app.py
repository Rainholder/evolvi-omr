"""
OMR (Optical Mark Recognition) microservice — template-matching edition.

Architecture:
  1. POST /setup-template  — send a blank template image; HoughCircles detects
                             all 270 bubble positions ONCE and stores them in
                             memory.  Must be called before /procesar.
  2. POST /procesar        — send a student photo; the service aligns it to the
                             template via corner markers + warpPerspective, then
                             reads each bubble at the stored coordinates.

Sheet layout (fixed):
  - 6 question-columns × 15 rows = 90 questions
  - 3 bubbles per question (A, B, C) left-to-right
  - 4 solid black square corner markers at the grid corners
"""

import base64
import io
import json
import logging
import os
import sys
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

from generate_sheet import build_sheet, compute_sheet_coords, parse_exam_code, EXAM_TITLES

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
# Known sheet layout constants
# ---------------------------------------------------------------------------
N_COLS    = 6    # question-columns side by side
N_ROWS    = 15   # questions per column
N_OPTIONS = 3    # A, B, C
TOTAL_BUBBLES = N_COLS * N_ROWS * N_OPTIONS   # 270

# Minimum dark-pixel fraction to declare a bubble "filled".
FILL_THRESHOLD = 0.18

# ---------------------------------------------------------------------------
# Global template state
#   TEMPLATE_CIRCLES  : dict  {"1": {"A": [cx,cy,r], ...}, ..., "90": {...}}
#   TEMPLATE_DIMS     : tuple (width_px, height_px) of the template image
# Both are None until /setup-template is successfully called.
# ---------------------------------------------------------------------------
TEMPLATE_CIRCLES: dict | None = None
TEMPLATE_DIMS:    tuple | None = None

# Optional on-disk persistence — survives server restarts on Render.com.
_STATE_FILE = "template_state.json"


def _save_template_state() -> None:
    """Write current template state to disk so it survives restarts."""
    try:
        with open(_STATE_FILE, "w") as f:
            json.dump({"dims": list(TEMPLATE_DIMS), "circles": TEMPLATE_CIRCLES}, f)
        log.info("Template state saved to %s", _STATE_FILE)
    except Exception as exc:
        log.warning("Could not save template state: %s", exc)


def _load_template_state() -> None:
    """Load template state from disk if available."""
    global TEMPLATE_CIRCLES, TEMPLATE_DIMS
    if not os.path.exists(_STATE_FILE):
        return
    try:
        with open(_STATE_FILE) as f:
            data = json.load(f)
        TEMPLATE_DIMS    = tuple(data["dims"])
        TEMPLATE_CIRCLES = data["circles"]
        log.info(
            "Template state loaded from %s — %d questions, dims=%s",
            _STATE_FILE, len(TEMPLATE_CIRCLES), TEMPLATE_DIMS,
        )
    except Exception as exc:
        log.warning("Could not load template state from disk: %s", exc)


# Load on startup
_load_template_state()


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
# Corner marker detection
# ---------------------------------------------------------------------------
_MARKER_THRESHOLDS = (50, 80, 100)
_MARKER_AREA_MIN   = 200
_MARKER_AREA_MAX   = 5000


def _filter_marker_candidates(gray: np.ndarray, thresh_val: int) -> tuple[list, int]:
    """
    Apply THRESH_BINARY_INV at a fixed level and return shape-filtered candidates.

    Returns:
        candidates : list of (cx, cy) float tuples
        total_cnts : raw contour count before filtering (for /debug logs)
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

        candidates.append((x + bw / 2.0, y + bh / 2.0))
        log.debug(
            "  thresh=%d marker cand: (%.0f, %.0f) area=%.0f aspect=%.2f solid=%.2f",
            thresh_val, x + bw / 2.0, y + bh / 2.0, area, aspect, solidity,
        )

    return candidates, total_cnts


def _pick_nearest_to_corners(candidates: list, w: int, h: int) -> np.ndarray:
    """
    Assign one candidate to each image corner (TL, TR, BR, BL) by nearest
    Euclidean distance.  Each candidate is used at most once.

    Returns (4, 2) float32 ordered TL → TR → BR → BL.
    """
    image_corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    pts, used, result = list(candidates), set(), []
    for ic_x, ic_y in image_corners:
        best = min(
            (i for i in range(len(pts)) if i not in used),
            key=lambda i: (pts[i][0] - ic_x) ** 2 + (pts[i][1] - ic_y) ** 2,
        )
        result.append(pts[best])
        used.add(best)
    return np.array(result, dtype=np.float32)


def find_corner_markers(gray: np.ndarray) -> np.ndarray:
    """
    Locate 4 solid black square corner markers using thresholds 50 → 80 → 100.
    Stops at the first threshold that yields ≥ 4 candidates.

    Returns (4, 2) float32 ordered TL, TR, BR, BL.
    Raises RuntimeError if all thresholds fail.
    """
    h, w = gray.shape
    last: list = []
    for thresh_val in _MARKER_THRESHOLDS:
        candidates, total_cnts = _filter_marker_candidates(gray, thresh_val)
        log.info(
            "Marker search thresh=%d: %d raw contours → %d candidates",
            thresh_val, total_cnts, len(candidates),
        )
        if len(candidates) >= 4:
            ordered = _pick_nearest_to_corners(candidates, w, h)
            log.info("Markers found thresh=%d → %s", thresh_val, ordered.tolist())
            return ordered
        last = candidates

    raise RuntimeError(
        f"Could not find 4 corner markers (tried thresholds {_MARKER_THRESHOLDS}). "
        f"Last attempt: {len(last)} candidate(s). "
        "Ensure the 4 black corner squares are fully visible and unobstructed."
    )


# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------
def warp_to_template(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Rectify the answer grid to exactly the template canvas size (TEMPLATE_DIMS).
    Requires TEMPLATE_DIMS to be set.
    """
    tw, th = TEMPLATE_DIMS
    dst = np.array(
        [[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (tw, th))
    log.debug("Warped to template size (%d × %d)", tw, th)
    return warped


# ---------------------------------------------------------------------------
# Template circle detection
# ---------------------------------------------------------------------------
def _detect_template_circles(gray: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Run HoughCircles on the blank template's grayscale image.

    Parameters are derived from the image dimensions, assuming Letter paper
    (215.9 mm wide).  Tries progressively lower accumulator thresholds until
    the detected count is in the range [260, 280] (= 270 ± 10).

    Returns a list of (cx, cy, r) integer tuples.
    Raises RuntimeError if no parameter set yields a plausible count.
    """
    h, w = gray.shape
    px_per_mm  = w / 215.9                  # assumes Letter paper width
    r_est      = int(7.0  * px_per_mm)      # 14 mm diameter → 7 mm radius
    min_dist   = int(14.0 * px_per_mm)      # tighter than smallest option gap
    min_r      = max(4, int(r_est * 0.60))
    max_r      = int(r_est * 1.40)

    log.info(
        "HoughCircles init: image=%dx%d px_per_mm=%.1f r_est=%d minDist=%d r=[%d,%d]",
        w, h, px_per_mm, r_est, min_dist, min_r, max_r,
    )

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    best_circles = None
    best_delta   = float("inf")

    # param2 = accumulator threshold; lower → more (potentially false) circles.
    for param2 in (50, 40, 30, 25, 20, 15):
        raw = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=50,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        n = 0 if raw is None else raw.shape[1]
        log.info("HoughCircles param2=%d → %d circles (target %d)", param2, n, TOTAL_BUBBLES)

        if raw is not None:
            delta = abs(n - TOTAL_BUBBLES)
            if delta < best_delta:
                best_delta   = delta
                best_circles = raw
            if 260 <= n <= 280:
                log.info("Accepted param2=%d with %d circles", param2, n)
                break

    if best_circles is None:
        raise RuntimeError("HoughCircles found no circles on the template image.")

    n = best_circles.shape[1]
    if not (260 <= n <= 280):
        log.warning(
            "Best attempt found %d circles (expected ~270). "
            "Detection may be inaccurate — check template image quality.",
            n,
        )

    return [(int(c[0]), int(c[1]), int(c[2])) for c in best_circles[0]]


def _sort_circles_to_questions(circles: list) -> dict:
    """
    Organise raw (cx, cy, r) circles into a question-keyed dict.

    Layout assumption:
      - 18 x-columns  (6 question-columns × 3 option-columns A/B/C)
      - 15 rows per x-column (one circle per question per option)
      - Questions numbered column-by-column: Q1–15 in col 0, Q16–30 in col 1, …

    Strategy:
      1. Sort circles by x.
      2. Locate 17 largest x-gaps → split into 18 x-columns.
      3. Sort each x-column by y.
      4. Group x-columns in consecutive triples → A (left), B (mid), C (right).
      5. Build question dict.

    Returns:
        {
          "1":  {"A": [cx, cy, r], "B": [cx, cy, r], "C": [cx, cy, r]},
          ...
          "90": {...}
        }
    """
    n_xcols = N_COLS * N_OPTIONS   # 18

    # ---- Step 1: sort by x ------------------------------------------------
    sorted_c = sorted(circles, key=lambda c: c[0])
    xs       = np.array([c[0] for c in sorted_c], dtype=float)

    # ---- Step 2: find 17 largest gaps → 18 x-columns ----------------------
    diffs         = np.diff(xs)
    split_indices = np.sort(np.argsort(diffs)[::-1][:n_xcols - 1]) + 1
    x_col_idx_groups = np.split(np.arange(len(sorted_c)), split_indices)
    x_cols = [[sorted_c[i] for i in grp] for grp in x_col_idx_groups]

    log.info(
        "Circle sorting: %d circles → %d x-columns (expected %d), "
        "sizes: %s",
        len(circles), len(x_cols), n_xcols,
        [len(g) for g in x_cols],
    )

    # ---- Step 3: sort each x-column by y ----------------------------------
    x_cols = [sorted(col, key=lambda c: c[1]) for col in x_cols]

    # ---- Step 4-5: group into question dict --------------------------------
    options  = ["A", "B", "C"]
    q_map    = {}
    warnings = []

    for qcol in range(N_COLS):                        # 0-5
        triple = x_cols[qcol * N_OPTIONS : qcol * N_OPTIONS + N_OPTIONS]

        for row in range(N_ROWS):                     # 0-14
            q_num = str(qcol * N_ROWS + row + 1)
            entry = {}
            for opt_i, opt in enumerate(options):
                if opt_i < len(triple) and row < len(triple[opt_i]):
                    c = triple[opt_i][row]
                    entry[opt] = [c[0], c[1], c[2]]
                else:
                    entry[opt] = None
                    warnings.append(f"Q{q_num}/{opt} missing in template circles")
            q_map[q_num] = entry

    if warnings:
        log.warning("Incomplete circles detected:\n  %s", "\n  ".join(warnings[:10]))

    log.info("Sorted %d circles into %d questions", len(circles), len(q_map))
    return q_map


# ---------------------------------------------------------------------------
# Bubble reading (uses template coordinates)
# ---------------------------------------------------------------------------
def _bubble_dark_fraction(binary: np.ndarray, cx: int, cy: int, r: int) -> float:
    """
    Fraction of pixels inside circle (cx, cy, r) that are dark (value == 0).
    """
    img_h, img_w = binary.shape
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    total_px = cv2.countNonZero(mask)
    if total_px == 0:
        return 0.0
    dark_px = int(np.sum((binary == 0) & (mask == 255)))
    return dark_px / total_px


def read_bubbles(binary_warped: np.ndarray, circles_map: dict, total_preguntas: int) -> dict:
    """
    Read answers using the template circle positions.

    Args:
        binary_warped   : binarised, perspective-corrected student image
        circles_map     : from TEMPLATE_CIRCLES
        total_preguntas : how many questions to read (≤ 90)

    Returns:
        {"1": "A", "2": None, "3": "C", …}
    """
    results      = {}
    filled_count = 0
    sample_log   = []

    for q in range(1, total_preguntas + 1):
        q_str = str(q)
        bubbles = circles_map.get(q_str)
        if not bubbles:
            results[q_str] = None
            continue

        fractions = {}
        for opt, coords in bubbles.items():
            if coords is None:
                fractions[opt] = 0.0
                continue
            cx, cy, r = int(coords[0]), int(coords[1]), int(coords[2])
            fractions[opt] = _bubble_dark_fraction(binary_warped, cx, cy, r)

        best_opt  = max(fractions, key=fractions.get)
        best_frac = fractions[best_opt]

        if q <= 5:
            sample_log.append(f"  Q{q}: {fractions} → {best_opt}={best_frac:.3f}")

        if best_frac >= FILL_THRESHOLD:
            results[q_str] = best_opt
            filled_count  += 1
        else:
            results[q_str] = None

    log.debug("Sample dark fractions:\n%s", "\n".join(sample_log))
    log.info("Filled bubbles: %d / %d", filled_count, total_preguntas)
    return results


# ---------------------------------------------------------------------------
# Core OMR pipeline
# ---------------------------------------------------------------------------
def process_omr(img: np.ndarray, total_preguntas: int) -> dict:
    if TEMPLATE_CIRCLES is None:
        raise RuntimeError(
            "No template loaded. Call POST /setup-template with the blank "
            "template image before processing student sheets."
        )

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Detect corner markers
    corners = find_corner_markers(gray)

    # 3. Warp to template dimensions
    warped_color = warp_to_template(img, corners)

    # 4. Binarise warped image for bubble reading (Otsu on blurred gray)
    warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    blurred     = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    _, warped_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    log.debug("Otsu threshold applied to warped image")

    # 5. Read bubbles at template positions
    respuestas = read_bubbles(warped_bin, TEMPLATE_CIRCLES, total_preguntas)

    total_detectadas = sum(1 for v in respuestas.values() if v is not None)
    confianza = round(total_detectadas / total_preguntas, 4) if total_preguntas else 0.0

    return {
        "ok": True,
        "respuestas": respuestas,
        "total_detectadas": total_detectadas,
        "confianza": confianza,
    }


# ---------------------------------------------------------------------------
# Flask endpoints
# ---------------------------------------------------------------------------

@app.route("/setup-template", methods=["POST"])
def setup_template():
    """
    Load a blank template image, detect all 270 bubble positions with
    HoughCircles, and store them for use by /procesar.

    Request body (JSON):
        {
          "imagen_base64": "<base64 image of the blank answer sheet>"
        }

    Response:
        {
          "ok": true,
          "total_circles_detected": 270,
          "template_dims": [2550, 3300],
          "questions_mapped": 90,
          "circles": {
            "1":  {"A": [cx, cy, r], "B": [...], "C": [...]},
            ...
            "90": {...}
          }
        }
    """
    global TEMPLATE_CIRCLES, TEMPLATE_DIMS

    log.info("POST /setup-template received")

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False, "error": "Body must be JSON"}), 400
    if "imagen_base64" not in data:
        return jsonify({"ok": False, "error": "Missing field: imagen_base64"}), 400

    try:
        img = decode_image(data["imagen_base64"])
    except Exception as exc:
        log.error("Template decode failed: %s", exc)
        return jsonify({"ok": False, "error": f"Image decode error: {exc}"}), 422

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        circles = _detect_template_circles(gray)
    except RuntimeError as exc:
        log.error("HoughCircles failed: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 422
    except Exception as exc:
        log.error("Unexpected error in circle detection:\n%s", traceback.format_exc())
        return jsonify({"ok": False, "error": "Internal error", "detail": str(exc)}), 500

    try:
        q_map = _sort_circles_to_questions(circles)
    except Exception as exc:
        log.error("Circle sorting failed:\n%s", traceback.format_exc())
        return jsonify({"ok": False, "error": "Circle sorting error", "detail": str(exc)}), 500

    # Commit to global state
    TEMPLATE_CIRCLES = q_map
    TEMPLATE_DIMS    = (w, h)
    _save_template_state()

    log.info(
        "Template loaded: %d circles → %d questions, dims=(%d, %d)",
        len(circles), len(q_map), w, h,
    )

    return jsonify({
        "ok":                    True,
        "total_circles_detected": len(circles),
        "template_dims":         [w, h],
        "questions_mapped":      len(q_map),
        "circles":               q_map,
    }), 200


@app.route("/procesar", methods=["POST"])
def procesar():
    """
    Process a student answer sheet.

    Request body (JSON):
        {
          "imagen_base64":  "<base64 student photo>",
          "total_preguntas": 90          (optional, default 90)
        }

    Response:
        {
          "ok": true,
          "respuestas": {"1": "A", "2": null, "3": "C", ...},
          "total_detectadas": 87,
          "confianza": 0.9667
        }
    """
    log.info("POST /procesar received")

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False, "error": "Body must be JSON"}), 400
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
        log.info(
            "OMR OK — detectadas=%d confianza=%.4f",
            result["total_detectadas"], result["confianza"],
        )
        return jsonify(result), 200
    except RuntimeError as exc:
        log.error("OMR pipeline error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 422
    except Exception as exc:
        log.error("Unexpected error:\n%s", traceback.format_exc())
        return jsonify({"ok": False, "error": "Internal server error", "detail": str(exc)}), 500


@app.route("/debug", methods=["GET", "POST"])
def debug():
    """
    Diagnostic endpoint — returns per-threshold contour counts and HoughCircles
    counts without running the full OMR pipeline.

    Request body (JSON):
        { "imagen_base64": "<base64 image>" }

    Response:
        {
          "ok": true,
          "image_shape": [h, w],
          "template_loaded": true,
          "marker_thresholds": {
            "50":  {"raw_contours": 18, "after_area": 5, "after_aspect": 4, "after_solidity": 4},
            ...
          },
          "hough_attempts": {
            "50": 270, "40": 270, ...
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
        return jsonify({"ok": False, "error": f"Image decode error: {exc}"}), 422

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- Marker detection stats ---
    marker_stats = {}
    for thresh_val in _MARKER_THRESHOLDS:
        _, binary_inv = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours, _   = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw = len(contours)
        aa = aas = sol = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (_MARKER_AREA_MIN <= area <= _MARKER_AREA_MAX):
                continue
            aa += 1
            bx, by, bw2, bh2 = cv2.boundingRect(cnt)
            aspect = bw2 / bh2 if bh2 > 0 else 0
            if not (0.5 < aspect < 2.0):
                continue
            aas += 1
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area and (area / hull_area) >= 0.85:
                sol += 1
        marker_stats[str(thresh_val)] = {
            "raw_contours": raw,
            "after_area_filter":     aa,
            "after_aspect_filter":   aas,
            "after_solidity_filter": sol,
        }
        log.info("Debug markers thresh=%d: raw=%d area=%d aspect=%d solid=%d",
                 thresh_val, raw, aa, aas, sol)

    # --- HoughCircles counts ---
    px_per_mm = w / 215.9
    r_est     = int(7.0  * px_per_mm)
    min_dist  = int(14.0 * px_per_mm)
    min_r     = max(4, int(r_est * 0.60))
    max_r     = int(r_est * 1.40)
    blurred   = cv2.GaussianBlur(gray, (9, 9), 2)
    hough_counts = {}
    for p2 in (50, 40, 30, 25, 20, 15):
        raw = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1,
            minDist=min_dist, param1=50, param2=p2,
            minRadius=min_r, maxRadius=max_r,
        )
        hough_counts[str(p2)] = 0 if raw is None else int(raw.shape[1])
    log.info("Debug HoughCircles counts: %s", hough_counts)

    return jsonify({
        "ok":              True,
        "image_shape":     [h, w],
        "template_loaded": TEMPLATE_CIRCLES is not None,
        "template_dims":   list(TEMPLATE_DIMS) if TEMPLATE_DIMS else None,
        "marker_area_range": [_MARKER_AREA_MIN, _MARKER_AREA_MAX],
        "marker_thresholds": marker_stats,
        "hough_attempts":  hough_counts,
    }), 200


@app.route("/sheet/<exam_code>", methods=["GET"])
def get_sheet(exam_code: str):
    """
    Generate and return the answer-sheet PDF for the given exam code.

    Also saves the bubble coordinates to sheet_coords.json so /procesar
    can use them without calling /setup-template on a template photo.

    Path param:
        exam_code : e.g. EV-ATR-JUN26  or  BIO  or  FIS

    Response: PDF file (Content-Disposition: attachment)

    Side-effect: writes sheet_coords.json to the working directory.
    """
    log.info("GET /sheet/%s", exam_code)

    try:
        pdf_bytes, sheet_coords = build_sheet(exam_code)
    except Exception as exc:
        log.error("Sheet generation failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"ok": False, "error": str(exc)}), 500

    # Persist coords so the OMR pipeline can load them directly
    coords_path = "sheet_coords.json"
    try:
        with open(coords_path, "w") as f:
            json.dump({"exam_code": exam_code.upper(), "coords": sheet_coords}, f)
        log.info("Saved %d question coords to %s", len(sheet_coords), coords_path)
    except Exception as exc:
        log.warning("Could not save sheet_coords.json: %s", exc)

    safe_code = exam_code.upper().replace("/", "-").replace(" ", "_")
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"hoja_{safe_code}.pdf",
    )


@app.route("/coords/<exam_code>", methods=["GET"])
def get_coords(exam_code: str):
    """
    Return the SHEET_COORDS JSON for the given exam code.

    Generates the coordinates on the fly (same geometry as /sheet/<exam_code>).
    Does NOT generate the full PDF — fast, stateless.

    Response:
        {
          "ok": true,
          "exam_code": "EV-ATR-JUN26",
          "title": "ÁREAS TRANSVERSALES",
          "total_questions": 90,
          "bubble_radius_px": 42,
          "coords": {
            "1":  {"A": [242, 1550], "B": [363, 1550], "C": [484, 1550]},
            ...
            "90": {"A": [...], "B": [...], "C": [...]}
          }
        }
    """
    log.info("GET /coords/%s", exam_code)

    try:
        info   = parse_exam_code(exam_code)
        coords = compute_sheet_coords()
    except Exception as exc:
        log.error("Coords computation failed: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500

    # Bubble radius in pixels at 300 DPI (R_RESP = 10 pt)
    from generate_sheet import R_RESP, PT_PX
    r_px = round(R_RESP * PT_PX)

    return jsonify({
        "ok":             True,
        "exam_code":      info["code"],
        "title":          info["title"],
        "total_questions": len(coords),
        "bubble_radius_px": r_px,
        "dpi":            300,
        "image_size_px":  [2550, 3300],
        "coords":         coords,
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "service":         "evolvi-omr",
        "template_loaded": TEMPLATE_CIRCLES is not None,
        "template_dims":   list(TEMPLATE_DIMS) if TEMPLATE_DIMS else None,
        "questions_ready": len(TEMPLATE_CIRCLES) if TEMPLATE_CIRCLES else 0,
    }), 200


# ---------------------------------------------------------------------------
# Entry point (dev only — production uses gunicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
