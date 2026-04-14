"""
OMR (Optical Mark Recognition) microservice — template-matching edition.

Coordinate sources (in priority order):
  1. POST /setup-template  — send a blank template image; HoughCircles detects
                             all 270 bubble positions and stores them.
                             Most accurate; required only once per deployment.
  2. sheet_coords.json     — coordinates computed from the PDF geometry at
                             300 DPI.  Loaded automatically on startup if
                             template_state.json is absent.
  3. {EXAM_CODE}_coords.json — per-exam variant; loaded on demand via the
                               exam_code field in /procesar.

POST /procesar accepts an optional exam_code field.  If provided (and
/setup-template has not been called), it loads the matching coords file.

Sheet layout (fixed):
  - 6 question-columns × 15 rows = 90 questions
  - 3 bubbles per question (A, B, C) left-to-right
  - 4 solid black square corner markers used for warpPerspective alignment
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
# Global OMR state
#   TEMPLATE_CIRCLES : dict  {"1": {"A": [cx,cy,r], ...}, ..., "90": {...}}
#   TEMPLATE_DIMS    : tuple (width_px, height_px) of the coordinate space
#   COORDS_SOURCE    : str   how the state was populated
# ---------------------------------------------------------------------------
TEMPLATE_CIRCLES:     dict | None = None
TEMPLATE_DIMS:        tuple | None = None
COORDS_SOURCE:        str          = "none"   # "setup_template" | "sheet_coords" | "none"
LAST_PERSPECTIVE_MODE: str         = "none"   # "markers" | "orb" | "edge_detection" | "resize_only"

# On-disk state files
_STATE_FILE        = "template_state.json"   # HoughCircles state (highest accuracy)
_SHEET_COORDS_FILE = "sheet_coords.json"     # PDF-geometry coords (auto-loaded fallback)

# ORB feature matching template image
_TEMPLATE_IMAGE_FILE = "template_image.png"
_TEMPLATE_IMAGE:    np.ndarray | None = None  # rendered reference image for ORB alignment
_last_orb_viz:      np.ndarray | None = None  # last drawMatches output for /debug-visual
_COORDS_EXAM_CODE:  str | None        = None  # exam_code from the last loaded coords file

# Standard Letter page at 300 DPI — matches generate_sheet coordinate space
_LETTER_W_PX = 2550
_LETTER_H_PX = 3300


def _sheet_respuestas_to_circles(respuestas: dict) -> dict:
    """
    Convert the respuestas section of a sheet_coords file to TEMPLATE_CIRCLES format.

    Input : {"1": {"A": [cx_px, cy_px], ...}, ..., "90": {...}}
    Output: {"1": {"A": [cx_px, cy_px, r_px], ...}, ...}

    The radius is taken from generate_sheet constants (R_RESP in PDF points
    converted to pixels at 300 DPI).
    """
    from generate_sheet import R_RESP, PT_PX
    r_px = round(R_RESP * PT_PX)
    return {
        q: {opt: [c[0], c[1], r_px] for opt, c in opts.items()}
        for q, opts in respuestas.items()
    }


def _save_template_state() -> None:
    """Write current template state to disk so it survives restarts."""
    try:
        with open(_STATE_FILE, "w") as f:
            json.dump({"dims": list(TEMPLATE_DIMS), "circles": TEMPLATE_CIRCLES}, f)
        log.info("Template state saved to %s", _STATE_FILE)
    except Exception as exc:
        log.warning("Could not save template state: %s", exc)


def _load_template_state() -> bool:
    """
    Load HoughCircles template state from template_state.json.
    Returns True on success, False if the file is absent or unreadable.
    """
    global TEMPLATE_CIRCLES, TEMPLATE_DIMS, COORDS_SOURCE
    if not os.path.exists(_STATE_FILE):
        return False
    try:
        with open(_STATE_FILE) as f:
            data = json.load(f)
        TEMPLATE_DIMS    = tuple(data["dims"])
        TEMPLATE_CIRCLES = data["circles"]
        COORDS_SOURCE    = "setup_template"
        log.info(
            "Template state loaded from %s — %d questions, dims=%s",
            _STATE_FILE, len(TEMPLATE_CIRCLES), TEMPLATE_DIMS,
        )
        return True
    except Exception as exc:
        log.warning("Could not load template state from disk: %s", exc)
        return False


def _load_sheet_coords_file(path: str) -> bool:
    """
    Load bubble coordinates from a sheet_coords JSON file.

    Expected JSON structure:
        {"exam_code": "EV-ATR-JUN26", "coords": {"celular": {...}, "respuestas": {...}}}

    On success: populates TEMPLATE_CIRCLES / TEMPLATE_DIMS / COORDS_SOURCE /
    _COORDS_EXAM_CODE, returns True.
    On failure: logs a warning, leaves globals unchanged, returns False.

    NOTE: does NOT render the template image — caller is responsible for
    calling _render_template_image(_COORDS_EXAM_CODE) when appropriate.
    """
    global TEMPLATE_CIRCLES, TEMPLATE_DIMS, COORDS_SOURCE, _COORDS_EXAM_CODE
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Support both wrapped {"exam_code":…, "coords":{…}} and bare {…} formats
        coords     = data["coords"] if "coords" in data else data
        respuestas = coords["respuestas"]
        circles    = _sheet_respuestas_to_circles(respuestas)
        TEMPLATE_CIRCLES  = circles
        TEMPLATE_DIMS     = (_LETTER_W_PX, _LETTER_H_PX)
        COORDS_SOURCE     = "sheet_coords"
        _COORDS_EXAM_CODE = data.get("exam_code")   # may be None for bare-format files
        r_sample = list(circles.values())[0]["A"][2]
        log.info(
            "Sheet coords loaded from %s — %d questions, r=%d px, dims=%s, exam_code=%s",
            path, len(circles), r_sample, TEMPLATE_DIMS, _COORDS_EXAM_CODE,
        )
        return True
    except Exception as exc:
        log.warning("Could not load sheet coords from %s: %s", path, exc)
        return False


def _load_template_image_file() -> bool:
    """
    Load a pre-rendered reference image from template_image.png into _TEMPLATE_IMAGE.
    Returns True on success, False if the file is absent or unreadable.
    """
    global _TEMPLATE_IMAGE
    if not os.path.exists(_TEMPLATE_IMAGE_FILE):
        return False
    try:
        img = cv2.imread(_TEMPLATE_IMAGE_FILE)
        if img is None:
            log.warning("cv2.imread returned None for %s", _TEMPLATE_IMAGE_FILE)
            return False
        _TEMPLATE_IMAGE = img
        log.info(
            "ORB template image loaded from %s — shape=%s",
            _TEMPLATE_IMAGE_FILE, img.shape,
        )
        return True
    except Exception as exc:
        log.warning("Could not load template image from %s: %s", _TEMPLATE_IMAGE_FILE, exc)
        return False


def _render_template_image(exam_code: str) -> bool:
    """
    Render the answer-sheet PDF for exam_code to a PNG at 150 DPI, save it as
    template_image.png, and load it into _TEMPLATE_IMAGE.

    Requires pdf2image (poppler) to be installed.
    Returns True on success, False on any error (non-fatal).
    """
    global _TEMPLATE_IMAGE
    try:
        from pdf2image import convert_from_bytes  # noqa: PLC0415
        pdf_bytes, _ = build_sheet(exam_code)
        pages = convert_from_bytes(pdf_bytes, dpi=150)
        if not pages:
            log.warning("pdf2image returned no pages for exam_code=%s", exam_code)
            return False
        img_np = np.array(pages[0].convert("RGB"))
        bgr    = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(_TEMPLATE_IMAGE_FILE, bgr)
        _TEMPLATE_IMAGE = bgr
        log.info(
            "ORB template image rendered for %s — shape=%s, saved to %s",
            exam_code, bgr.shape, _TEMPLATE_IMAGE_FILE,
        )
        return True
    except Exception as exc:
        log.warning("Could not render template image for %s: %s", exam_code, exc)
        return False


def _bootstrap() -> None:
    """
    Populate OMR state at server startup.

    Priority:
      1. template_state.json  (HoughCircles — highest positional accuracy)
      2. sheet_coords.json    (PDF-geometry — no template photo required)

    After loading coords the template image is always re-rendered from the PDF
    in memory (not read from disk) so it is available for ORB alignment even
    after an ephemeral-filesystem redeploy on Render.
    """
    if _load_template_state():
        # HoughCircles state loaded; try to render template image if we know
        # the exam_code from a previously saved sheet_coords.json.
        _try_render_template_from_coords_file()
        return
    if _load_sheet_coords_file(_SHEET_COORDS_FILE):
        # Coords loaded — immediately render the template image in memory.
        if _COORDS_EXAM_CODE:
            _render_template_image(_COORDS_EXAM_CODE)
        else:
            log.warning(
                "sheet_coords.json has no exam_code — ORB template image unavailable. "
                "Re-generate the sheet via GET /sheet/<exam_code>."
            )
        return
    log.info(
        "No coords loaded at startup. Call GET /sheet/<exam_code> to generate "
        "sheet_coords.json and bootstrap the service."
    )


def _try_render_template_from_coords_file() -> None:
    """
    Read exam_code from sheet_coords.json (if present) and render the template
    image for ORB alignment.  Called after HoughCircles state is loaded so ORB
    is available as a fallback even when /setup-template coordinates are in use.
    """
    if not os.path.exists(_SHEET_COORDS_FILE):
        return
    try:
        with open(_SHEET_COORDS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        exam_code = data.get("exam_code")
        if exam_code:
            _render_template_image(exam_code)
    except Exception as exc:
        log.warning("Could not read exam_code from %s for ORB render: %s", _SHEET_COORDS_FILE, exc)


_bootstrap()


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


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points as TL, TR, BR, BL.

    Uses the standard sum/diff trick:
      TL = min(x+y)   BR = max(x+y)
      TR = min(y-x)   BL = max(y-x)
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    d    = pts[:, 1] - pts[:, 0]
    rect[0] = pts[np.argmin(s)]   # TL
    rect[1] = pts[np.argmin(d)]   # TR
    rect[2] = pts[np.argmax(s)]   # BR
    rect[3] = pts[np.argmax(d)]   # BL
    return rect


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
# Fallback: paper-edge detection via Canny + contour approximation
# ---------------------------------------------------------------------------
def _find_page_corners_by_edges(gray: np.ndarray) -> np.ndarray | None:
    """
    Locate the paper boundary using Canny edge detection.

    Strategy:
      1. Blur → Canny(50,150) → dilate to close gaps.
      2. Find external contours; sort by area descending.
      3. For each large contour try approxPolyDP with several epsilon factors
         until we get exactly 4 vertices.
      4. Order them TL/TR/BR/BL with _order_corners().

    Returns (4, 2) float32 or None if no valid quadrilateral is found.
    """
    h, w    = gray.shape
    min_area = h * w * 0.20   # paper must cover ≥ 20 % of the frame

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    # Dilate to close small gaps along sheet edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check only the 10 largest contours to keep it fast
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break   # sorted — remainder will also be too small

        peri = cv2.arcLength(cnt, True)
        for eps in (0.02, 0.015, 0.03, 0.04):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                pts     = approx.reshape(4, 2).astype(np.float32)
                ordered = _order_corners(pts)
                log.info(
                    "Edge detection: page found — area=%.0f eps=%.3f corners=%s",
                    area, eps, ordered.tolist(),
                )
                return ordered

    log.warning(
        "Edge detection: no 4-vertex contour ≥ %.0f px² found (checked %d contours)",
        min_area, len(contours),
    )
    return None


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


def _warp_by_resize(img: np.ndarray) -> np.ndarray:
    """
    Last-resort alignment: scale the image directly to TEMPLATE_DIMS.
    No perspective correction — assumes the photo is already roughly frontal.
    """
    tw, th  = TEMPLATE_DIMS
    resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    log.info("Fallback resize to (%d x %d) — no perspective correction", tw, th)
    return resized


def _warp_by_orb(
    img: np.ndarray,
) -> tuple["np.ndarray | None", "np.ndarray | None"]:
    """
    Align a student photo to the template reference using ORB feature matching.

    Steps:
      1. Resize both images to MAX_DIM (1100 px on the long side) for speed.
      2. Detect ORB keypoints + descriptors on both.
      3. BFMatcher (NORM_HAMMING) + knnMatch, Lowe ratio test 0.75.
      4. findHomography (RANSAC) on good matches.
      5. Scale the small-image homography up to full TEMPLATE_DIMS:
           H_full = S_tpl_inv @ H_small @ S_img
      6. warpPerspective(img, H_full, TEMPLATE_DIMS).

    Returns:
        (warped_color, student_corners_in_original)
        Both are None on failure (too few keypoints/matches/inliers).
    """
    global _last_orb_viz
    _last_orb_viz = None

    if _TEMPLATE_IMAGE is None:
        log.warning("ORB: no template image — skipping")
        return None, None

    tw, th = TEMPLATE_DIMS
    MAX_DIM = 1100

    # ── Scale both images down for feature matching ──────────────────────────
    img_h, img_w = img.shape[:2]
    img_scale    = MAX_DIM / max(img_h, img_w)
    small_img    = cv2.resize(img,
                              (int(img_w * img_scale), int(img_h * img_scale)),
                              interpolation=cv2.INTER_AREA)

    tpl_h, tpl_w = _TEMPLATE_IMAGE.shape[:2]
    tpl_scale    = MAX_DIM / max(tpl_h, tpl_w)
    small_tpl    = cv2.resize(_TEMPLATE_IMAGE,
                              (int(tpl_w * tpl_scale), int(tpl_h * tpl_scale)),
                              interpolation=cv2.INTER_AREA)

    # ── ORB keypoints ─────────────────────────────────────────────────────────
    orb      = cv2.ORB_create(nfeatures=2000)
    gray_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    gray_tpl = cv2.cvtColor(small_tpl, cv2.COLOR_BGR2GRAY)
    kp_img, des_img = orb.detectAndCompute(gray_img, None)
    kp_tpl, des_tpl = orb.detectAndCompute(gray_tpl, None)

    n_img = len(kp_img) if kp_img else 0
    n_tpl = len(kp_tpl) if kp_tpl else 0
    log.info("ORB keypoints: student=%d  template=%d", n_img, n_tpl)

    if des_img is None or des_tpl is None or n_img < 10 or n_tpl < 10:
        log.warning("ORB: insufficient keypoints (student=%d, template=%d)", n_img, n_tpl)
        return None, None

    # ── BFMatcher + ratio test ────────────────────────────────────────────────
    bf         = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_pairs  = bf.knnMatch(des_img, des_tpl, k=2)
    good       = [m for m, n in raw_pairs if m.distance < 0.75 * n.distance]
    log.info("ORB matches: %d raw → %d good (ratio 0.75)", len(raw_pairs), len(good))

    if len(good) < 10:
        log.warning("ORB: too few good matches (%d < 10)", len(good))
        return None, None

    # ── Homography in small-image space ──────────────────────────────────────
    src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_tpl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_small, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H_small is None:
        log.warning("ORB: findHomography returned None")
        return None, None

    inliers = int(mask.sum()) if mask is not None else 0
    log.info("ORB: homography inliers=%d / %d matches", inliers, len(good))

    if inliers < 8:
        log.warning("ORB: too few inliers (%d < 8)", inliers)
        return None, None

    # ── Scale homography to full resolution ──────────────────────────────────
    # H_full maps original-student-coords → original-template-coords
    # H_full = S_tpl_inv  @  H_small  @  S_img
    S_img = np.array([[img_scale, 0,         0],
                      [0,         img_scale, 0],
                      [0,         0,         1]], dtype=np.float64)
    S_tpl_inv = np.array([[1.0 / tpl_scale, 0,               0],
                          [0,               1.0 / tpl_scale, 0],
                          [0,               0,               1]], dtype=np.float64)
    H_full = S_tpl_inv @ H_small.astype(np.float64) @ S_img

    # ── Warp student image to template canvas ────────────────────────────────
    warped = cv2.warpPerspective(img, H_full, (tw, th))

    # ── Save match visualisation for /debug-visual ───────────────────────────
    inlier_matches = [m for m, keep in zip(good, mask.ravel()) if keep] if mask is not None else good
    _last_orb_viz = cv2.drawMatches(
        small_img, kp_img,
        small_tpl, kp_tpl,
        inlier_matches[:60], None,
        matchColor=(0, 255, 0),
        singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # ── Compute student-image corners (for debug overlay) ───────────────────
    try:
        H_inv      = np.linalg.inv(H_full)
        tpl_corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
        stu_corners = cv2.perspectiveTransform(tpl_corners, H_inv).reshape(4, 2)
        return warped, stu_corners
    except Exception as exc:
        log.warning("ORB: could not compute student corners: %s", exc)
        return warped, None


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
# Perspective correction pipeline (shared by /procesar and /debug-visual)
# ---------------------------------------------------------------------------
def _run_perspective_correction(
    img: np.ndarray, gray: np.ndarray
) -> tuple[np.ndarray, "np.ndarray | None", str]:
    """
    Try the 4-stage perspective correction pipeline.

    Returns:
        warped_color : BGR image aligned to TEMPLATE_DIMS
        corners      : (4, 2) float32 source corners in the ORIGINAL image space,
                       or None when resize_only was used
        mode         : "markers" | "orb" | "edge_detection" | "resize_only"
    """
    # Stage 1: solid black corner markers
    try:
        corners = find_corner_markers(gray)
        warped  = warp_to_template(img, corners)
        log.info("Perspective: corner markers -> warpPerspective")
        return warped, corners, "markers"
    except RuntimeError as exc:
        log.warning("Corner markers failed (%s) — trying ORB", exc)

    # Stage 2: ORB feature matching against rendered PDF template
    if _TEMPLATE_IMAGE is not None:
        warped_orb, stu_corners = _warp_by_orb(img)
        if warped_orb is not None:
            log.info("Perspective: ORB feature matching")
            return warped_orb, stu_corners, "orb"
        log.warning("ORB matching failed — trying Canny edge detection")
    else:
        log.warning("No template image for ORB — trying Canny edge detection")

    # Stage 3: Canny paper-edge detection
    corners = _find_page_corners_by_edges(gray)
    if corners is not None:
        warped = warp_to_template(img, corners)
        log.info("Perspective: edge detection -> warpPerspective")
        return warped, corners, "edge_detection"
    log.warning("Edge detection also failed — using resize fallback")

    # Stage 4: simple resize
    return _warp_by_resize(img), None, "resize_only"


# ---------------------------------------------------------------------------
# Core OMR pipeline
# ---------------------------------------------------------------------------
def process_omr(img: np.ndarray, total_preguntas: int) -> dict:
    """
    Full OMR pipeline with a 3-stage perspective correction fallback.
    The active stage is recorded in LAST_PERSPECTIVE_MODE.
    """
    global LAST_PERSPECTIVE_MODE

    if TEMPLATE_CIRCLES is None:
        raise RuntimeError(
            "No coords loaded. Call POST /setup-template or GET /sheet/<exam_code> first."
        )

    gray                              = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    warped_color, _corners, mode      = _run_perspective_correction(img, gray)
    LAST_PERSPECTIVE_MODE             = mode

    warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    blurred     = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    _, warped_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    respuestas       = read_bubbles(warped_bin, TEMPLATE_CIRCLES, total_preguntas)
    total_detectadas = sum(1 for v in respuestas.values() if v is not None)
    confianza        = round(total_detectadas / total_preguntas, 4) if total_preguntas else 0.0

    return {
        "ok":               True,
        "respuestas":       respuestas,
        "total_detectadas": total_detectadas,
        "confianza":        confianza,
        "perspective_mode": mode,
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
    global TEMPLATE_CIRCLES, TEMPLATE_DIMS, COORDS_SOURCE

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
    COORDS_SOURCE    = "setup_template"
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
          "imagen_base64":   "<base64 student photo>",
          "total_preguntas":  90,            (optional, default 90)
          "exam_code":       "EV-ATR-JUN26"  (optional)
        }

    When exam_code is provided and the current source is NOT setup_template,
    the service tries to load {EXAM_CODE}_coords.json first, then falls back
    to sheet_coords.json.  If /setup-template was called its HoughCircles state
    is always used regardless of exam_code.

    Response:
        {
          "ok": true,
          "respuestas": {"1": "A", "2": null, "3": "C", ...},
          "total_detectadas": 87,
          "confianza": 0.9667,
          "coords_source": "sheet_coords"
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

    # Load per-exam coords on demand (only when not using HoughCircles state).
    exam_code = data.get("exam_code")
    if exam_code and COORDS_SOURCE != "setup_template":
        code_safe = exam_code.upper().replace("/", "-").replace(" ", "_")
        for candidate in (f"{code_safe}_coords.json", _SHEET_COORDS_FILE):
            if _load_sheet_coords_file(candidate):
                log.info("Loaded coords for exam %s from %s", exam_code, candidate)
                break
        else:
            log.warning(
                "No coords file found for exam_code=%s; using current state (%s)",
                exam_code, COORDS_SOURCE,
            )

    try:
        img = decode_image(data["imagen_base64"])
    except Exception as exc:
        log.error("Image decode failed: %s", exc)
        return jsonify({"ok": False, "error": f"Image decode error: {exc}"}), 422

    try:
        result = process_omr(img, total_preguntas)
        result["coords_source"] = COORDS_SOURCE
        log.info(
            "OMR OK — detectadas=%d confianza=%.4f source=%s",
            result["total_detectadas"], result["confianza"], COORDS_SOURCE,
        )
        return jsonify(result), 200
    except RuntimeError as exc:
        log.error("OMR pipeline error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 422
    except Exception as exc:
        log.error("Unexpected error:\n%s", traceback.format_exc())
        return jsonify({"ok": False, "error": "Internal server error", "detail": str(exc)}), 500


@app.route("/debug-visual", methods=["POST"])
def debug_visual():
    """
    Visual debugger — runs the full OMR pipeline and returns annotated images
    showing exactly where the algorithm is searching for bubbles.

    Request body (JSON):
        {
          "imagen_base64":   "<base64 photo>",
          "exam_code":       "EV-ATR-JUN26",   (optional)
          "total_preguntas":  90               (optional, default 90)
        }

    Response:
        {
          "ok": true,
          "perspective_mode": "markers",
          "total_detectadas": 45,
          "confianza": 0.50,
          "imagen_procesada_base64": "data:image/jpeg;base64,…",
              // Warped/aligned image with:
              //   BLUE  filled circles  — 4 warp-anchor corner points
              //   RED   hollow circles  — every bubble lookup position
              //   GREEN filled circles  — bubbles detected as marked
          "imagen_perspectiva_base64": "data:image/jpeg;base64,…"
              // Original photo with blue polygon showing the detected
              // paper boundary (absent when mode is resize_only)
        }
    """
    log.info("POST /debug-visual received")

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False, "error": "Body must be JSON"}), 400
    if "imagen_base64" not in data:
        return jsonify({"ok": False, "error": "Missing field: imagen_base64"}), 400

    total_preguntas = data.get("total_preguntas", 90)
    if not isinstance(total_preguntas, int) or not (1 <= total_preguntas <= 300):
        return jsonify({"ok": False, "error": "total_preguntas must be an integer 1-300"}), 400

    # Load per-exam coords if needed (same logic as /procesar)
    exam_code = data.get("exam_code")
    if exam_code and COORDS_SOURCE != "setup_template":
        code_safe = exam_code.upper().replace("/", "-").replace(" ", "_")
        for candidate in (f"{code_safe}_coords.json", _SHEET_COORDS_FILE):
            if _load_sheet_coords_file(candidate):
                break

    if TEMPLATE_CIRCLES is None:
        return jsonify({
            "ok": False,
            "error": "No coords loaded. Call GET /sheet/<exam_code> or POST /setup-template first.",
        }), 422

    try:
        img = decode_image(data["imagen_base64"])
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Image decode error: {exc}"}), 422

    try:
        # ── Run the perspective + OMR pipeline ────────────────────────────────
        gray                         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        warped_color, corners, mode  = _run_perspective_correction(img, gray)
        global LAST_PERSPECTIVE_MODE
        LAST_PERSPECTIVE_MODE        = mode

        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
        blurred     = cv2.GaussianBlur(warped_gray, (5, 5), 0)
        _, warped_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        respuestas = read_bubbles(warped_bin, TEMPLATE_CIRCLES, total_preguntas)

        total_det  = sum(1 for v in respuestas.values() if v is not None)
        confianza  = round(total_det / total_preguntas, 4) if total_preguntas else 0.0

        # ── Build output scale (cap long side at 1200 px for fast transfer) ──
        tw, th    = TEMPLATE_DIMS
        MAX_PX    = 1200
        sc        = min(1.0, MAX_PX / max(tw, th))
        vis_w     = int(tw * sc)
        vis_h     = int(th * sc)
        r_min     = 3

        # ── Annotate the PROCESSED (warped) image ────────────────────────────
        vis = cv2.resize(warped_color, (vis_w, vis_h), interpolation=cv2.INTER_AREA)

        # Blue filled circles at the 4 warp-anchor corners of the output canvas
        corner_r = max(r_min, int(18 * sc))
        for cx, cy in [
            (corner_r,        corner_r),
            (vis_w - corner_r, corner_r),
            (vis_w - corner_r, vis_h - corner_r),
            (corner_r,        vis_h - corner_r),
        ]:
            cv2.circle(vis, (cx, cy), corner_r,     (255, 80, 0), -1)
            cv2.circle(vis, (cx, cy), corner_r + 2, (255, 255, 255), 2)

        # Batch-project all bubble positions at the output scale
        # (direct scale: warped coords → display coords)
        for q_str, opts in TEMPLATE_CIRCLES.items():
            answer = respuestas.get(q_str)
            for opt, coords in opts.items():
                cx = int(coords[0] * sc)
                cy = int(coords[1] * sc)
                r  = max(r_min, int(coords[2] * sc))
                if answer == opt:
                    cv2.circle(vis, (cx, cy), r,     (30, 210, 30), -1)   # green filled
                    cv2.circle(vis, (cx, cy), r + 1, (10, 140, 10), 1)
                else:
                    cv2.circle(vis, (cx, cy), r, (30, 30, 220), 1)        # red hollow

        # Mode + stats label
        label = f"{mode} | {total_det}/{total_preguntas} detectados"
        cv2.putText(vis, label, (10, vis_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
        b64_proc = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

        # ── Perspective image: ORB match viz OR original with corner polygon ───
        b64_persp = None

        if mode == "orb" and _last_orb_viz is not None:
            # Show ORB keypoint matches between student photo and template
            orb_h, orb_w = _last_orb_viz.shape[:2]
            orb_sc = min(1.0, MAX_PX / max(orb_w, orb_h))
            vis_orb = cv2.resize(_last_orb_viz,
                                 (int(orb_w * orb_sc), int(orb_h * orb_sc)),
                                 interpolation=cv2.INTER_AREA)
            cv2.putText(vis_orb, f"ORB matches | {mode}",
                        (10, vis_orb.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
            _, buf2 = cv2.imencode(".jpg", vis_orb, [cv2.IMWRITE_JPEG_QUALITY, 82])
            b64_persp = "data:image/jpeg;base64," + base64.b64encode(buf2).decode("ascii")

        elif corners is not None:
            # Show original photo with blue polygon marking the detected paper boundary
            orig_sc  = min(1.0, MAX_PX / max(img.shape[1], img.shape[0]))
            vis_orig = cv2.resize(img.copy(),
                                  (int(img.shape[1] * orig_sc), int(img.shape[0] * orig_sc)),
                                  interpolation=cv2.INTER_AREA)
            sc_corners = (corners * orig_sc).astype(np.int32)

            cv2.polylines(vis_orig, [sc_corners.reshape(-1, 1, 2)], True, (255, 80, 0), 2)
            for i, pt in enumerate(sc_corners):
                cv2.circle(vis_orig, tuple(pt), 10, (255, 80, 0), -1)
                cv2.circle(vis_orig, tuple(pt), 12, (255, 255, 255), 2)
                label_corner = ["TL", "TR", "BR", "BL"][i]
                cv2.putText(vis_orig, label_corner,
                            (int(pt[0]) + 14, int(pt[1]) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(vis_orig, label_corner,
                            (int(pt[0]) + 14, int(pt[1]) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 0), 1, cv2.LINE_AA)

            cv2.putText(vis_orig, mode,
                        (10, vis_orig.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

            _, buf2 = cv2.imencode(".jpg", vis_orig, [cv2.IMWRITE_JPEG_QUALITY, 82])
            b64_persp = "data:image/jpeg;base64," + base64.b64encode(buf2).decode("ascii")

        result = {
            "ok":                      True,
            "perspective_mode":        mode,
            "total_detectadas":        total_det,
            "confianza":               confianza,
            "imagen_procesada_base64": b64_proc,
        }
        if b64_persp:
            result["imagen_perspectiva_base64"] = b64_persp

        return jsonify(result), 200

    except Exception as exc:
        log.error("debug-visual error:\n%s", traceback.format_exc())
        return jsonify({"ok": False, "error": str(exc)}), 500


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

    safe_code   = exam_code.upper().replace("/", "-").replace(" ", "_")
    per_exam    = f"{safe_code}_coords.json"
    coords_data = {"exam_code": exam_code.upper(), "coords": sheet_coords}

    # Persist coords — per-exam file and generic fallback
    for coords_path in (per_exam, _SHEET_COORDS_FILE):
        try:
            with open(coords_path, "w", encoding="utf-8") as f:
                json.dump(coords_data, f)
        except Exception as exc:
            log.warning("Could not save %s: %s", coords_path, exc)

    log.info(
        "Saved coords (%d resp + %d cel) to %s and %s",
        len(sheet_coords["respuestas"]), len(sheet_coords["celular"]),
        per_exam, _SHEET_COORDS_FILE,
    )

    # Auto-load into memory so /procesar works immediately after /sheet.
    # _load_sheet_coords_file also sets _COORDS_EXAM_CODE.
    if COORDS_SOURCE != "setup_template":
        _load_sheet_coords_file(per_exam)

    # Always render the template image fresh (ephemeral FS on Render means
    # we cannot rely on a PNG surviving across redeploys — keep it in memory).
    global _COORDS_EXAM_CODE
    _COORDS_EXAM_CODE = safe_code
    _render_template_image(safe_code)

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
          "bubble_radius_px": 33,
          "coords": {
            "celular": {
              "0": {"0": [1538, 284], ..., "9": [1538, 1033]},
              ...
              "9": {...}
            },
            "respuestas": {
              "1":  {"A": [238, 1313], "B": [330, 1313], "C": [421, 1313]},
              ...
              "90": {...}
            }
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
        "total_questions": len(coords["respuestas"]),
        "bubble_radius_px": r_px,
        "dpi":            300,
        "image_size_px":  [2550, 3300],
        "coords":         coords,
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":                "ok",
        "service":               "evolvi-omr",
        "template_loaded":       TEMPLATE_CIRCLES is not None,
        "source":                COORDS_SOURCE,
        "exam_code":             _COORDS_EXAM_CODE,
        "questions_ready":       len(TEMPLATE_CIRCLES) if TEMPLATE_CIRCLES else 0,
        "template_image_loaded": _TEMPLATE_IMAGE is not None,
        "perspective_mode":      LAST_PERSPECTIVE_MODE,
    }), 200


# ---------------------------------------------------------------------------
# Entry point (dev only — production uses gunicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
