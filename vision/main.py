"""
BoardSnap Vision Service  –  FastAPI
=====================================
Owner : Muavia   |   Event : Banana Hacks 2026
Stack : OpenCV region detection  ·  Microsoft TrOCR  ·  optional Grounding DINO

Endpoints
---------
  GET  /health
  POST /vision/clean    – perspective rectification + glare suppression
  POST /vision/detect   – local Grounding DINO whiteboard region detector
  POST /vision/ocr      – TrOCR handwriting → typed text per region
  POST /vision/graph    – spatial arrow↔box graph → Mermaid diagram

Environment variables
---------------------
  DETECTOR_PROVIDER   (default: "opencv"; optional "grounding-dino" or "roboflow")
  GROUNDING_DINO_MODEL_ID (default: "IDEA-Research/grounding-dino-tiny")
  ROBOFLOW_API_KEY    (required only for the optional Roboflow provider)

Run
---
  pip install -r requirements.txt
  uvicorn main:app --port 8000 --reload
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import time
import importlib.util
import inspect
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("boardsnap.vision")

# Load Vision-local settings without overriding variables already set by the shell.
load_dotenv(Path(__file__).with_name(".env"), override=False)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="BoardSnap Vision Service", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── env ───────────────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "whiteboard-ctvgi/1")
ROBOFLOW_WORKFLOW_URL = os.environ.get("ROBOFLOW_WORKFLOW_URL", "").strip()
DETECTOR_PROVIDER = os.environ.get("DETECTOR_PROVIDER", "opencv").strip().lower()
GROUNDING_DINO_MODEL_ID = os.environ.get(
    "GROUNDING_DINO_MODEL_ID",
    "IDEA-Research/grounding-dino-tiny",
)
GROUNDING_DINO_THRESHOLD = float(os.environ.get("GROUNDING_DINO_THRESHOLD", "0.25"))
GROUNDING_DINO_TEXT_THRESHOLD = float(os.environ.get("GROUNDING_DINO_TEXT_THRESHOLD", "0.2"))
GROUNDING_DINO_IMAGE_SIZE = int(os.environ.get("GROUNDING_DINO_IMAGE_SIZE", "640"))
GROUNDING_DINO_CPU_THREADS = int(os.environ.get("GROUNDING_DINO_CPU_THREADS", "1"))
OPENCV_TEXT_CONFIDENCE = 0.7
OPENCV_SHAPE_CONFIDENCE = 0.75
OPENCV_ARROW_CONFIDENCE = 0.65
ALLOW_DEMO_DETECTOR_FALLBACK = os.environ.get("ALLOW_DEMO_DETECTOR_FALLBACK", "").lower() == "true"
TROCR_MODEL_ID = os.environ.get("TROCR_MODEL_ID", "microsoft/trocr-base-handwritten")

# ── lazy local Grounding DINO loader ─────────────────────────────────────────
_grounding_dino_processor = None
_grounding_dino_model = None
_grounding_dino_device = None
_grounding_dino_lock = Lock()


def _get_grounding_dino():
    global _grounding_dino_processor, _grounding_dino_model, _grounding_dino_device
    if _grounding_dino_processor is None:
        with _grounding_dino_lock:
            if _grounding_dino_processor is None:
                log.info("Loading local Grounding DINO model (%s) …", GROUNDING_DINO_MODEL_ID)
                import torch
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    torch.set_num_threads(max(1, GROUNDING_DINO_CPU_THREADS))
                processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
                model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    GROUNDING_DINO_MODEL_ID
                ).to(device)
                model.eval()
                _grounding_dino_processor = processor
                _grounding_dino_model = model
                _grounding_dino_device = device
                log.info("Grounding DINO loaded on %s.", device)
    return _grounding_dino_processor, _grounding_dino_model, _grounding_dino_device


# ── lazy TrOCR loader (loaded once on first call) ─────────────────────────────
_trocr_processor = None
_trocr_model = None

def _get_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        log.info("Loading TrOCR model (%s) …", TROCR_MODEL_ID)
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
        _trocr_model.eval()
        log.info("TrOCR loaded.")
    return _trocr_processor, _trocr_model


# ═════════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═════════════════════════════════════════════════════════════════════════════

class RegionItem(BaseModel):
    id: str
    type: str          # text | box | circle | arrow | table
    box: List[float]   # [x%, y%, w%, h%] – normalised 0–100
    label: Optional[str] = None
    confidence: float


class CleanResponse(BaseModel):
    status: str
    quality: float
    isQualityPass: bool
    cleanImageB64: str          # base64 PNG of the cleaned board
    metrics: Dict[str, str]


class DetectResponse(BaseModel):
    regions: List[RegionItem]


class OCRResponse(BaseModel):
    regions: List[Dict[str, Any]]


class GraphResponse(BaseModel):
    nodes: List[Dict[str, str]]
    edges: List[Dict[str, str]]
    mermaid: str


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS – IMAGE CLEANING (OpenCV)
# ═════════════════════════════════════════════════════════════════════════════

def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _cv_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp perspective to a top-down rectangle view."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def _find_board_corners(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Try to find a quadrilateral board contour.
    Returns 4 corner points or None if not found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    return None


def _suppress_glare(img: np.ndarray) -> np.ndarray:
    """
    Suppress specular highlights (glare) using inpainting on bright regions.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Mask pixels brighter than 240 (likely glare)
    _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    # Dilate slightly to cover glare halos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)
    # Inpaint the masked region
    inpainted = cv2.inpaint(img, glare_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return inpainted


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE-based local contrast enhancement for whiteboard text."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _laplacian_sharpness(img: np.ndarray) -> float:
    """Return Laplacian variance as a sharpness score (higher = sharper)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clean_whiteboard(raw_bytes: bytes) -> Tuple[np.ndarray, dict]:
    """
    Full cleaning pipeline:
      1. Perspective rectification (board corner detection + warp)
      2. Glare suppression (inpainting bright spots)
      3. Contrast enhancement (CLAHE)
      4. Quality scoring (Laplacian sharpness)
    Returns (cleaned_image_bgr, metrics_dict).
    """
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    h, w = img.shape[:2]
    metrics: dict = {}
    perspective_applied = False

    # 1. Perspective rectification
    corners = _find_board_corners(img)
    if corners is not None:
        try:
            img = _four_point_transform(img, corners)
            perspective_applied = True
            # Estimate skew angle (angle of top edge)
            tl, tr = corners[np.argmin(corners.sum(axis=1))], corners[np.argmin(np.diff(corners, axis=1))]
            angle_deg = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
            metrics["perspectiveCorrection"] = f"Rectified ({angle_deg:+.1f}° skew)"
        except Exception:
            metrics["perspectiveCorrection"] = "Skipped (warp failed)"
    else:
        metrics["perspectiveCorrection"] = "Not needed (board fills frame)"

    # 2. Glare suppression
    before_mean = float(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0])
    img = _suppress_glare(img)
    after_mean = float(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0])
    glare_delta = abs(before_mean - after_mean)
    metrics["glareSuppressed"] = (
        f"High (Δ{glare_delta:.1f} px brightness)" if glare_delta > 5 else "Low (minimal glare)"
    )

    # 3. Contrast boost
    before_std = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())
    img = _enhance_contrast(img)
    after_std = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())
    contrast_pct = ((after_std - before_std) / max(before_std, 1)) * 100
    metrics["contrastBoost"] = f"{contrast_pct:+.0f}% (CLAHE)"

    # 4. Quality score
    sharpness = _laplacian_sharpness(img)
    # Normalise: 100+ variance → quality 1.0, below 20 → fail
    quality = min(1.0, sharpness / 150.0)
    metrics["textConfidence"] = f"{quality * 100:.1f}%"

    return img, metrics, quality


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS – LOCAL GROUNDING DINO AND OPTIONAL ROBOFLOW INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def _point_to_box_distance(point: Tuple[float, float], box: List[float]) -> float:
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def _pixel_region(
    region_id: str,
    region_type: str,
    box: List[float],
    image_width: int,
    image_height: int,
    label: str,
    confidence: float,
) -> RegionItem:
    x1, y1, x2, y2 = box
    return RegionItem(
        id=region_id,
        type=region_type,
        box=[
            round(x1 / image_width * 100, 2),
            round(y1 / image_height * 100, 2),
            round((x2 - x1) / image_width * 100, 2),
            round((y2 - y1) / image_height * 100, 2),
        ],
        label=label,
        confidence=confidence,
    )


def _opencv_detect(image_b64: str) -> List[RegionItem]:
    """Find diagram shapes, connectors, and text lines without a remote model."""
    if image_b64.startswith("data:") and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray.shape
    minimum_dimension = min(image_width, image_height)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    shape_candidates: List[Tuple[float, str, List[float]]] = []
    for contour in contours:
        contour_area = abs(cv2.contourArea(contour))
        perimeter = cv2.arcLength(contour, True)
        if contour_area < image_width * image_height * 0.002 or perimeter <= 0:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        area_ratio = width * height / (image_width * image_height)
        if (
            area_ratio > 0.7
            or width < max(20, round(image_width * 0.035))
            or height < max(20, round(image_height * 0.035))
        ):
            continue

        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) >= 6:
            circularity = 4 * math.pi * contour_area / (perimeter * perimeter)
            if circularity >= 0.7:
                shape_type = "circle"
            else:
                continue
        elif len(polygon) == 4 and cv2.isContourConvex(polygon):
            center_x = x + width / 2
            center_y = y + height / 2
            near_side_midpoint = sum(
                min(
                    abs(float(point[0][0]) - center_x) / max(width, 1),
                    abs(float(point[0][1]) - center_y) / max(height, 1),
                ) < 0.16
                for point in polygon
            )
            shape_type = "diamond" if near_side_midpoint >= 3 else "box"
        else:
            continue

        box = [float(x), float(y), float(x + width), float(y + height)]
        confidence = OPENCV_SHAPE_CONFIDENCE
        if any(
            _box_iou(box, previous_box) > 0.65
            for _, _, previous_box in shape_candidates
        ):
            continue
        shape_candidates.append((confidence, shape_type, box))

    shape_candidates.sort(key=lambda item: item[0], reverse=True)
    shape_regions = [
        _pixel_region(
            f"s{index + 1}",
            shape_type,
            box,
            image_width,
            image_height,
            f"OpenCV {shape_type}",
            confidence,
        )
        for index, (confidence, shape_type, box) in enumerate(shape_candidates)
    ]
    shape_boxes = [
        [
            region.box[0] * image_width / 100,
            region.box[1] * image_height / 100,
            (region.box[0] + region.box[2]) * image_width / 100,
            (region.box[1] + region.box[3]) * image_height / 100,
        ]
        for region in shape_regions
    ]

    edges = cv2.Canny(gray, 50, 150)
    line_segments = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(20, round(minimum_dimension * 0.08)),
        minLineLength=max(24, round(minimum_dimension * 0.08)),
        maxLineGap=max(5, round(minimum_dimension * 0.025)),
    )
    connector_candidates: Dict[Tuple[int, int], Tuple[float, List[float]]] = {}
    if line_segments is not None and len(shape_boxes) >= 2:
        endpoint_tolerance = max(10.0, minimum_dimension * 0.04)
        padding = max(3.0, minimum_dimension * 0.01)
        for segment in np.asarray(line_segments).reshape(-1, 4):
            x1, y1, x2, y2 = (float(value) for value in segment)
            start = (x1, y1)
            end = (x2, y2)
            length = math.dist(start, end)
            if length < max(24, minimum_dimension * 0.08):
                continue
            start_matches = [
                (index, _point_to_box_distance(start, box))
                for index, box in enumerate(shape_boxes)
            ]
            end_matches = [
                (index, _point_to_box_distance(end, box))
                for index, box in enumerate(shape_boxes)
            ]
            start_matches = [item for item in start_matches if item[1] <= endpoint_tolerance]
            end_matches = [item for item in end_matches if item[1] <= endpoint_tolerance]
            connections = [
                (start_index, end_index)
                for start_index, _ in start_matches
                for end_index, _ in end_matches
                if start_index != end_index
            ]
            if not connections:
                continue
            connection = tuple(sorted(connections[0]))

            arrow_box = [
                max(0.0, min(x1, x2) - padding),
                max(0.0, min(y1, y2) - padding),
                min(float(image_width), max(x1, x2) + padding),
                min(float(image_height), max(y1, y2) + padding),
            ]
            previous = connector_candidates.get(connection)
            if previous is None or length > previous[0]:
                connector_candidates[connection] = (length, arrow_box)

    arrow_boxes = [candidate[1] for candidate in connector_candidates.values()]

    arrow_regions = [
        _pixel_region(
            f"a{index + 1}",
            "arrow",
            box,
            image_width,
            image_height,
            "Connector",
            OPENCV_ARROW_CONFIDENCE,
        )
        for index, box in enumerate(arrow_boxes)
    ]

    text_ink = ink.copy()
    for box in shape_boxes + arrow_boxes:
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(
            text_ink,
            (max(0, x1 - 2), max(0, y1 - 2)),
            (min(image_width - 1, x2 + 2), min(image_height - 1, y2 + 2)),
            0,
            thickness=-1,
        )

    row_counts = np.count_nonzero(text_ink, axis=1)
    active_rows = np.flatnonzero(row_counts >= max(2, round(image_width * 0.005)))
    max_row_gap = max(1, min(4, image_height // 100))
    line_groups: List[Tuple[int, int]] = []
    if active_rows.size:
        start = previous = int(active_rows[0])
        for row_value in active_rows[1:]:
            row = int(row_value)
            if row - previous > max_row_gap + 1:
                line_groups.append((start, previous))
                start = row
            previous = row
        line_groups.append((start, previous))

    text_regions: List[RegionItem] = []
    min_line_height = max(6, round(image_height * 0.012))
    for top, bottom in line_groups:
        if bottom - top + 1 < min_line_height:
            continue
        columns = np.flatnonzero(np.any(text_ink[top:bottom + 1], axis=0))
        if not columns.size:
            continue
        pad_x = max(2, round(image_width * 0.005))
        pad_y = max(2, round(image_height * 0.005))
        line_box = [
            float(max(0, int(columns[0]) - pad_x)),
            float(max(0, top - pad_y)),
            float(min(image_width, int(columns[-1]) + 1 + pad_x)),
            float(min(image_height, bottom + 1 + pad_y)),
        ]
        if any(
            _intersection_over_smaller_box(line_box, shape_box) > 0.8
            for shape_box in shape_boxes
        ):
            continue
        center = ((line_box[0] + line_box[2]) / 2, (line_box[1] + line_box[3]) / 2)
        if any(
            box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]
            for box in shape_boxes
        ):
            continue
        if any(_box_iou(line_box, box) > 0.6 for box in arrow_boxes):
            continue
        text_regions.append(
            _pixel_region(
                f"t{len(text_regions) + 1}",
                "text",
                line_box,
                image_width,
                image_height,
                "Detected text line",
                OPENCV_TEXT_CONFIDENCE,
            )
        )

    regions = shape_regions + arrow_regions + text_regions
    regions.sort(key=lambda region: (region.box[1], region.box[0], region.type))
    for index, region in enumerate(regions, start=1):
        region.id = f"r{index}"
    log.info(
        "OpenCV detected %s shapes, %s connectors, and %s text lines",
        len(shape_regions),
        len(arrow_regions),
        len(text_regions),
    )
    return regions


GROUNDING_DINO_PROMPT = (
    "handwritten text. flowchart box. flowchart arrow. "
    "flowchart diamond. flowchart circle."
)


def _grounding_region_type(label: str) -> Optional[str]:
    normalized = label.lower()
    matches = [
        (normalized.find(word), region_type)
        for region_type, words in (
            ("arrow", ("arrow", "connector")),
            ("diamond", ("diamond",)),
            ("circle", ("circle", "oval")),
            ("table", ("table",)),
            ("text", ("text", "handwriting", "note")),
            ("box", ("box", "rectangle", "square")),
        )
        for word in words
        if normalized.find(word) >= 0
    ]
    return min(matches)[1] if matches else None


def _box_iou(left: List[float], right: List[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _intersection_over_smaller_box(left: List[float], right: List[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller_area = min(
        max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1]),
        max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1]),
    )
    return intersection / smaller_area if smaller_area else 0.0


def _grounding_dino_detect(image_b64: str) -> List[RegionItem]:
    """Detect whiteboard content with the local, zero-shot Grounding DINO model."""
    import torch

    if image_b64.startswith("data:") and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    processor, model, device = _get_grounding_dino()
    inputs = processor(
        images=image,
        text=GROUNDING_DINO_PROMPT,
        return_tensors="pt",
        size={
            "shortest_edge": GROUNDING_DINO_IMAGE_SIZE,
            "longest_edge": int(GROUNDING_DINO_IMAGE_SIZE * 1.6),
        },
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    post_process = processor.post_process_grounded_object_detection
    parameters = inspect.signature(post_process).parameters
    threshold_args = {
        "text_threshold": GROUNDING_DINO_TEXT_THRESHOLD,
        "target_sizes": [(image.height, image.width)],
    }
    if "threshold" in parameters:
        threshold_args["threshold"] = GROUNDING_DINO_THRESHOLD
    else:
        threshold_args["box_threshold"] = GROUNDING_DINO_THRESHOLD

    detections = post_process(
        outputs,
        inputs.input_ids,
        **threshold_args,
    )[0]
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections.get("text_labels")
    if labels is None:
        labels = detections["labels"]
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    candidates = []
    for box, score, label in zip(boxes, scores, labels):
        region_type = _grounding_region_type(str(label))
        if region_type is None:
            continue

        x1, y1, x2, y2 = (float(value) for value in box)
        x1 = min(float(image.width), max(0.0, x1))
        y1 = min(float(image.height), max(0.0, y1))
        x2 = min(float(image.width), max(x1, x2))
        y2 = min(float(image.height), max(y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        area_ratio = (x2 - x1) * (y2 - y1) / (image.width * image.height)
        if region_type != "text" and area_ratio > 0.85:
            continue
        candidates.append((float(score), str(label), region_type, [x1, y1, x2, y2]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    regions: List[RegionItem] = []
    accepted: List[Tuple[str, List[float]]] = []
    for score, label, region_type, box in candidates:
        if any(
            _box_iou(box, accepted_box) > 0.65
            or (
                region_type != accepted_type
                and region_type != "text"
                and accepted_type != "text"
                and _intersection_over_smaller_box(box, accepted_box) > 0.7
            )
            for accepted_type, accepted_box in accepted
        ):
            continue
        accepted.append((region_type, box))
        x1, y1, x2, y2 = box
        regions.append(
            RegionItem(
                id=f"r{len(regions) + 1}",
                type=region_type,
                box=[
                    round(x1 / image.width * 100, 2),
                    round(y1 / image.height * 100, 2),
                    round((x2 - x1) / image.width * 100, 2),
                    round((y2 - y1) / image.height * 100, 2),
                ],
                label=label,
                confidence=round(score, 3),
            )
        )

    log.info("Grounding DINO detected %s regions", len(regions))
    return regions


def _roboflow_detect(image_b64: str, model_id: str, api_key: str) -> List[RegionItem]:
    """Call the configured hosted Roboflow Workflow."""
    import requests

    if not ROBOFLOW_WORKFLOW_URL:
        raise ValueError("ROBOFLOW_WORKFLOW_URL is missing from vision/.env")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY is missing from vision/.env")

    # Remove an optional image data-URL prefix, if present.
    if image_b64.startswith("data:") and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    payload = {
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_b64,
            },
            "classes": "box, arrow, diamond, whiteboard, handwriting",
        }
    }

    resp = requests.post(
        ROBOFLOW_WORKFLOW_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )

    if not resp.ok:
        log.error(
            "Roboflow workflow HTTP %s: %s",
            resp.status_code,
            resp.text[:1000],
        )

    resp.raise_for_status()
    data = resp.json()
    log.info("Roboflow workflow response received for %s", model_id)
    # Workflow results may nest predictions inside "predictions".
    outputs = data.get("outputs", [])

    if outputs and isinstance(outputs[0], dict):
        result = outputs[0].get("predictions", {})
    else:
        result = data.get("predictions", data)

    if isinstance(result, dict):
        img_info = result.get("image", {})
        predictions = result.get("predictions", [])
    else:
        img_info = data.get("image", {})
        predictions = result

    if not isinstance(predictions, list):
        raise ValueError("Roboflow workflow response is missing a predictions list")
    log.info("Roboflow workflow prediction count: %s", len(predictions))

    img_w = img_info.get("width", 1) or 1
    img_h = img_info.get("height", 1) or 1
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Roboflow workflow returned invalid image dimensions")

    type_map = {
        "whiteboard": "box",
        "box": "box",
        "handwriting": "text",
        "text": "text",
        "arrow": "arrow",
        "diamond": "diamond",
    }

    regions: List[RegionItem] = []

    for i, pred in enumerate(predictions):
        confidence = float(pred.get("confidence", 0))
        class_name = pred.get("class", "text").lower()

        cx = float(pred["x"])
        cy = float(pred["y"])
        pw = float(pred["width"])
        ph = float(pred["height"])

        x1 = min(100.0, max(0.0, (cx - pw / 2) / img_w * 100))
        y1 = min(100.0, max(0.0, (cy - ph / 2) / img_h * 100))
        x2 = min(100.0, max(x1, (cx + pw / 2) / img_w * 100))
        y2 = min(100.0, max(y1, (cy + ph / 2) / img_h * 100))
        if x2 <= x1 or y2 <= y1:
            continue

        regions.append(
            RegionItem(
                id=f"r{i + 1}",
                type=type_map.get(class_name, "text"),
                box=[
                    round(x1, 2),
                    round(y1, 2),
                    round(x2 - x1, 2),
                    round(y2 - y1, 2),
                ],
                label=class_name.title(),
                confidence=round(confidence, 3),
            )
        )

    log.info("Roboflow workflow detected %s regions", len(regions))
    return regions


def _fallback_regions(reason: str) -> List[RegionItem]:
    """Return hardcoded demo regions when hosted detection is unavailable."""
    log.warning("%s – using fallback demo regions.", reason)
    return [
        RegionItem(id="r1", type="text",  box=[5,  5,  40, 10], label="Distributed Event Pipeline", confidence=0.96),
        RegionItem(id="r2", type="box",   box=[5,  25, 22, 14], label="API Gateway",                confidence=0.94),
        RegionItem(id="r3", type="arrow", box=[28, 30, 14,  5], label="Arrow",                      confidence=0.91),
        RegionItem(id="r4", type="box",   box=[43, 23, 22, 18], label="Kafka / Redis",              confidence=0.95),
        RegionItem(id="r5", type="arrow", box=[66, 30, 14,  5], label="Arrow",                      confidence=0.89),
        RegionItem(id="r6", type="box",   box=[81, 21, 18, 22], label="Worker Pool",                confidence=0.92),
        RegionItem(id="r7", type="arrow", box=[54, 42,  5, 18], label="Arrow",                      confidence=0.87),
        RegionItem(id="r8", type="box",   box=[43, 61, 22, 16], label="Dead Letter Queue",          confidence=0.93),
        RegionItem(id="r9", type="text",  box=[5,  52, 33, 36], label="Key Requirements",           confidence=0.95),
    ]


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS – TrOCR
# ═════════════════════════════════════════════════════════════════════════════

def _trocr_read(crop: Image.Image) -> str:
    """Run TrOCR on a cropped PIL image, return the transcribed text."""
    import torch

    processor, model = _get_trocr()
    pixel_values = processor(images=crop.convert("RGB"), return_tensors="pt").pixel_values
    with torch.inference_mode():
        generated_ids = model.generate(pixel_values, max_new_tokens=128)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def _split_text_lines(crop: Image.Image) -> List[Image.Image]:
    """Split a text block into line crops for TrOCR's line-oriented recognizer."""
    gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if gray.size == 0:
        return [crop]

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    height, width = ink.shape
    row_counts = np.count_nonzero(ink, axis=1)
    active_rows = np.flatnonzero(row_counts >= max(2, round(width * 0.005)))
    if active_rows.size == 0:
        return [crop]

    max_row_gap = max(1, min(4, height // 100))
    groups: List[Tuple[int, int]] = []
    start = previous = int(active_rows[0])
    for row_value in active_rows[1:]:
        row = int(row_value)
        if row - previous > max_row_gap + 1:
            groups.append((start, previous))
            start = row
        previous = row
    groups.append((start, previous))

    min_line_height = max(3, round(height * 0.02))
    line_crops: List[Image.Image] = []
    pad_x = max(2, round(width * 0.01))
    pad_y = max(2, round(height * 0.01))
    for top, bottom in groups:
        if bottom - top + 1 < min_line_height:
            continue
        columns = np.flatnonzero(np.any(ink[top:bottom + 1], axis=0))
        if columns.size == 0:
            continue
        left = max(0, int(columns[0]) - pad_x)
        right = min(width, int(columns[-1]) + 1 + pad_x)
        line_top = max(0, top - pad_y)
        line_bottom = min(height, bottom + 1 + pad_y)
        line_crops.append(crop.crop((left, line_top, right, line_bottom)))

    return line_crops if len(line_crops) > 1 else [crop]


def _ocr_regions(
    regions: List[Dict[str, Any]],
    image_b64: str,
) -> List[Dict[str, Any]]:
    """
    For each text/box region, crop the area from the image and run TrOCR.
    """
    # Decode image
    img_bytes = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_w, img_h = pil_img.size

    for r in regions:
        if r.get("type") not in ("text", "box", "circle", "diamond", "table", "parallelogram"):
            continue
        if not isinstance(r.get("box"), (list, tuple)) or len(r["box"]) != 4:
            raise ValueError(f"Region {r.get('id', '<unknown>')} has an invalid box")
        bx, by, bw, bh = r["box"]
        if not all(math.isfinite(float(value)) for value in (bx, by, bw, bh)):
            raise ValueError(f"Region {r.get('id', '<unknown>')} has non-finite box coordinates")
        if bw <= 0 or bh <= 0:
            continue
        x1 = max(0, min(img_w, int(bx / 100 * img_w)))
        y1 = max(0, min(img_h, int(by / 100 * img_h)))
        x2 = max(x1, min(img_w, int((bx + bw) / 100 * img_w)))
        y2 = max(y1, min(img_h, int((by + bh) / 100 * img_h)))
        pw = x2 - x1
        ph = y2 - y1
        if pw < 8 or ph < 8:
            continue
        crop = pil_img.crop((x1, y1, x2, y2))
        line_texts = []
        for line_crop in _split_text_lines(crop):
            line_text = _trocr_read(line_crop)
            if line_text:
                line_texts.append(line_text)
        text = "\n".join(line_texts)
        if text:
            r["text"] = text
            r["label"] = text  # update label with real OCR result
        else:
            r["text"] = ""

    return regions


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS – SPATIAL GRAPH BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _box_center(box: List[float]) -> Tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _build_graph(regions: List[Dict[str, Any]]) -> Tuple[list, list, str]:
    """
    Connect arrows to the nearest box/text on each side.
    Returns (nodes, edges, mermaid_string).
    """
    shapes = [
        r for r in regions
        if r["type"] in ("box", "text", "circle", "table", "parallelogram", "diamond")
    ]
    arrows = [r for r in regions if r["type"] == "arrow"]

    # Assign clean IDs to shapes
    node_id_map: Dict[str, str] = {}
    nodes = []
    for i, s in enumerate(shapes):
        nid = f"N{i}"
        node_id_map[s["id"]] = nid
        label = s.get("text") or s.get("label") or f"Node {i}"
        shape_type = s["type"]
        nodes.append({"id": nid, "label": label, "shape": shape_type})

    edges = []
    for arrow in arrows:
        acx, acy = _box_center(arrow["box"])

        if len(shapes) < 2:
            continue

        # Find the two closest shapes to the arrow
        dists = sorted(shapes, key=lambda s: _distance(_box_center(s["box"]), (acx, acy)))
        src = dists[0]
        dst = dists[1] if len(dists) > 1 else None

        if dst:
            edge_label = arrow.get("text") or arrow.get("label") or ""
            edges.append({
                "from": node_id_map[src["id"]],
                "to": node_id_map[dst["id"]],
                "label": edge_label,
            })

    # Build Mermaid
    mermaid_lines = ["graph LR"]
    for n in nodes:
        label = n["label"].replace('"', "'").replace("\r\n", "\n").replace("\r", "\n")
        label = label.replace("\n", "<br/>")
        shape = n["shape"]
        if shape == "diamond":
            mermaid_lines.append(f'    {n["id"]}{{"{label}"}}')
        elif shape == "circle":
            mermaid_lines.append(f'    {n["id"]}(("{label}"))')
        elif shape == "parallelogram":
            mermaid_lines.append(f'    {n["id"]}[/"{label}"/]')
        else:
            mermaid_lines.append(f'    {n["id"]}["{label}"]')

    for e in edges:
        lbl = e["label"]
        if lbl:
            mermaid_lines.append(f'    {e["from"]} -->|"{lbl}"| {e["to"]}')
        else:
            mermaid_lines.append(f'    {e["from"]} --> {e["to"]}')

    mermaid = "\n".join(mermaid_lines)
    return nodes, edges, mermaid


# ═════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    if DETECTOR_PROVIDER == "opencv":
        detector_model = "opencv-contours-hough-lines"
        detector_configured = True
        detector_loaded = True
    elif DETECTOR_PROVIDER == "grounding-dino":
        detector_model = GROUNDING_DINO_MODEL_ID
        detector_configured = bool(
            importlib.util.find_spec("torch")
            and importlib.util.find_spec("transformers")
        )
        detector_loaded = _grounding_dino_model is not None
    elif DETECTOR_PROVIDER == "roboflow":
        detector_model = ROBOFLOW_MODEL_ID
        detector_configured = bool(ROBOFLOW_API_KEY and ROBOFLOW_WORKFLOW_URL)
        detector_loaded = detector_configured
    else:
        detector_model = ""
        detector_configured = False
        detector_loaded = False

    return {
        "status": "ok",
        "service": "BoardSnap Vision & ML Service",
        "detector": {
            "provider": DETECTOR_PROVIDER,
            "model": detector_model,
            "configured": detector_configured,
            "model_loaded": detector_loaded,
            "fallback_enabled": ALLOW_DEMO_DETECTOR_FALLBACK,
        },
        "ocr": {
            "model": TROCR_MODEL_ID,
            "dependencies_installed": bool(
                importlib.util.find_spec("torch")
                and importlib.util.find_spec("transformers")
            ),
            "model_loaded": _trocr_model is not None,
        },
        "trocr_loaded": _trocr_processor is not None,
    }


# ── /vision/clean ─────────────────────────────────────────────────────────────
@app.post("/vision/clean")
async def clean_image(file: UploadFile = File(...)):
    """
    1. Detect whiteboard corners  →  perspective warp  →  glare inpainting
    2. CLAHE contrast enhancement
    3. Laplacian sharpness quality gate (threshold 0.40)
    Returns base64-encoded cleaned PNG + quality metrics.
    """
    t0 = time.time()
    raw = await file.read()

    try:
        cleaned_cv, metrics, quality = clean_whiteboard(raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {exc}")

    is_pass = quality >= 0.40
    b64 = _cv_to_b64(cleaned_cv)

    metrics["processingMs"] = f"{(time.time() - t0)*1000:.0f} ms"

    return {
        "status": "success",
        "quality": round(quality, 4),
        "isQualityPass": is_pass,
        "cleanImageB64": b64,
        "metrics": metrics,
    }


# ── /vision/detect ────────────────────────────────────────────────────────────
@app.post("/vision/detect")
async def detect_regions(
    file: UploadFile = File(None),
    clean_image_b64: str = Form(None),
):
    """
    Accepts either:
      - multipart file upload  (raw or cleaned image)
      - clean_image_b64        (base64 string from /vision/clean response)

    Uses OpenCV by default, or Grounding DINO/Roboflow when explicitly selected.
    Demo regions are only returned when ALLOW_DEMO_DETECTOR_FALLBACK=true.
    """
    if file is not None:
        raw = await file.read()
        image_b64 = base64.b64encode(raw).decode()
    elif clean_image_b64:
        image_b64 = clean_image_b64
    else:
        raise HTTPException(status_code=422, detail="Provide file or clean_image_b64")

    if DETECTOR_PROVIDER == "opencv":
        try:
            regions = _opencv_detect(image_b64)
        except Exception as exc:
            log.exception("OpenCV region detection failed")
            raise HTTPException(
                status_code=422,
                detail="OpenCV could not detect regions in the submitted image.",
            ) from exc
        return {"regions": [region.model_dump() for region in regions], "source": "opencv"}

    if DETECTOR_PROVIDER == "grounding-dino":
        try:
            regions = _grounding_dino_detect(image_b64)
        except Exception as exc:
            log.exception("Local Grounding DINO inference failed")
            raise HTTPException(
                status_code=503,
                detail="Local detector failed to load or process the image. Check model access and Vision logs.",
            ) from exc
        return {"regions": [r.model_dump() for r in regions], "source": "grounding-dino"}

    if DETECTOR_PROVIDER != "roboflow":
        raise HTTPException(
            status_code=503,
            detail="Unsupported detector provider. Set DETECTOR_PROVIDER to opencv, grounding-dino, or roboflow.",
        )

    if not ROBOFLOW_API_KEY or not ROBOFLOW_WORKFLOW_URL:
        if ALLOW_DEMO_DETECTOR_FALLBACK:
            regions = _fallback_regions("Roboflow detector is not configured")
            return {"regions": [r.model_dump() for r in regions], "source": "demo-fallback"}
        raise HTTPException(
            status_code=503,
            detail="Detector is not configured. Set ROBOFLOW_API_KEY and ROBOFLOW_WORKFLOW_URL in vision/.env.",
        )

    try:
        regions = _roboflow_detect(image_b64, ROBOFLOW_MODEL_ID, ROBOFLOW_API_KEY)
    except Exception as exc:
        if ALLOW_DEMO_DETECTOR_FALLBACK:
            regions = _fallback_regions(f"Roboflow request failed ({type(exc).__name__})")
            return {"regions": [r.model_dump() for r in regions], "source": "demo-fallback"}
        log.exception("Roboflow workflow inference failed")
        raise HTTPException(status_code=502, detail="Roboflow detector inference failed.") from exc

    return {"regions": [r.model_dump() for r in regions], "source": "roboflow"}


# ── /vision/ocr ───────────────────────────────────────────────────────────────
@app.post("/vision/ocr")
async def read_text_regions(
    regions_json: str = Form(...),
    clean_image_b64: str = Form(...),
):
    """
    For every text/box region, crop the cleaned image and run TrOCR.
    Returns updated regions with 'text' field populated.
    TrOCR is lazy-loaded on first call (~5 s download on first run, cached after).
    """
    try:
        regions: List[Dict[str, Any]] = json.loads(regions_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="regions_json must be valid JSON")

    if not clean_image_b64:
        raise HTTPException(status_code=422, detail="clean_image_b64 is required for OCR")

    if not importlib.util.find_spec("torch") or not importlib.util.find_spec("transformers"):
        raise HTTPException(
            status_code=503,
            detail="OCR dependencies are not installed. Install vision/requirements.txt.",
        )

    try:
        regions = _ocr_regions(regions, clean_image_b64)
    except Exception as exc:
        log.exception("TrOCR failed to process the submitted regions")
        raise HTTPException(
            status_code=503,
            detail="Handwriting OCR failed to load or process the image.",
        ) from exc

    return {"regions": regions, "model": TROCR_MODEL_ID}


# ── /vision/graph ─────────────────────────────────────────────────────────────
@app.post("/vision/graph")
async def build_diagram_graph(regions_json: str = Form(...)):
    """
    Spatial graph builder:
    - Assign node IDs to all box/text/shape regions
    - Connect each arrow to its two nearest shapes (source → target)
    - Generate valid Mermaid LR graph markdown
    """
    try:
        regions: List[Dict[str, Any]] = json.loads(regions_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="regions_json must be valid JSON")

    nodes, edges, mermaid = _build_graph(regions)

    # If graph is empty (e.g. no shapes detected), return a sensible default
    if not nodes:
        mermaid = (
            'graph LR\n'
            '    A["Whiteboard content detected"]\n'
            '    A --> B["Review regions and re-run"]\n'
        )

    return {"nodes": nodes, "edges": edges, "mermaid": mermaid}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
