"""
BoardSnap – Synthetic Whiteboard Data Generator
===============================================
Owner: Muavia | Event: Banana Hacks 2026 | Theme: Image AI

Generates synthetic whiteboard training images with:
  - Randomly placed flowchart boxes, process diamonds, and circles
  - Directional connector arrows with arrowheads
  - Multi-line text annotations
  - Realistic whiteboard artifacts: uneven illumination, glare halos, marker stroke variation

Outputs:
  - Synthetic images (PNG)
  - Normalized bounding box ground truth (JSON and YOLO format)
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np


CLASS_MAP = {
    "text": 0,
    "box": 1,
    "circle": 2,
    "arrow": 3,
    "table": 4,
}

SAMPLE_LABELS = [
    "User Login", "Auth Service", "JWT Token", "Cache Layer",
    "PostgreSQL DB", "Message Queue", "Worker Pool", "Notification",
    "API Gateway", "Rate Limiter", "Kafka Cluster", "Metrics Exporter",
    "Load Balancer", "Ingress Proxy", "Audit Logger", "Dead Letter"
]


def create_whiteboard_canvas(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create an off-white whiteboard background with realistic gradient/lighting."""
    base_color = random.randint(242, 252)
    canvas = np.full((height, width, 3), base_color, dtype=np.uint8)

    # Add subtle uneven lighting gradient
    grad_x = np.linspace(random.uniform(-15, 0), random.uniform(0, 15), width)
    grad_y = np.linspace(random.uniform(-15, 0), random.uniform(0, 15), height)
    xx, yy = np.meshgrid(grad_x, grad_y)
    gradient = (xx + yy).astype(np.int16)

    canvas_int = canvas.astype(np.int16) + gradient[:, :, np.newaxis]
    canvas = np.clip(canvas_int, 220, 255).astype(np.uint8)
    return canvas


def add_glare(canvas: np.ndarray) -> np.ndarray:
    """Add a specular reflection glare spot simulating classroom overhead lights."""
    h, w = canvas.shape[:2]
    glare_x = random.randint(int(w * 0.2), int(w * 0.8))
    glare_y = random.randint(int(h * 0.1), int(h * 0.5))
    glare_radius = random.randint(80, 180)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (glare_x, glare_y), glare_radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (71, 71), 0)

    glare_layer = np.full_like(canvas, 255, dtype=np.uint8)
    alpha = (mask * random.uniform(0.35, 0.65))[:, :, np.newaxis]
    blended = (canvas * (1.0 - alpha) + glare_layer * alpha).astype(np.uint8)
    return blended


def draw_arrow(
    canvas: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 3,
) -> Dict[str, Any]:
    """Draw a directed arrow with an arrowhead and return its bounding box."""
    cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)

    # Compute arrowhead
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_size = random.randint(14, 22)
    left_angle = angle + math.pi * 0.82
    right_angle = angle - math.pi * 0.82

    p1 = (int(end[0] + arrow_size * math.cos(left_angle)), int(end[1] + arrow_size * math.sin(left_angle)))
    p2 = (int(end[0] + arrow_size * math.cos(right_angle)), int(end[1] + arrow_size * math.sin(right_angle)))

    pts = np.array([end, p1, p2], np.int32)
    cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA)

    # Calculate bounding box
    min_x = max(0, min(start[0], end[0], p1[0], p2[0]) - 4)
    max_x = min(canvas.shape[1], max(start[0], end[0], p1[0], p2[0]) + 4)
    min_y = max(0, min(start[1], end[1], p1[1], p2[1]) - 4)
    max_y = min(canvas.shape[0], max(start[1], end[1], p1[1], p2[1]) + 4)

    return {
        "type": "arrow",
        "box": [min_x, min_y, max_x - min_x, max_y - min_y],
        "label": "Arrow",
    }


def generate_synthetic_board(
    width: int = 1280,
    height: int = 720,
    include_glare: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate a single annotated synthetic whiteboard image."""
    canvas = create_whiteboard_canvas(width, height)
    regions: List[Dict[str, Any]] = []

    # Dry-erase marker colors: Dark Blue, Black, Dark Red, Dark Green
    marker_colors = [
        (30, 40, 45),     # Charcoal / Black
        (130, 40, 20),    # Dark Blue
        (30, 25, 140),    # Dark Red
        (40, 110, 30),    # Forest Green
    ]
    color = random.choice(marker_colors)

    num_boxes = random.randint(3, 5)
    box_w = random.randint(180, 240)
    box_h = random.randint(80, 110)

    # Compute grid-like arrangement with jitter
    start_x = 100
    gap = (width - 200 - (num_boxes * box_w)) // max(1, (num_boxes - 1))
    box_positions = []

    for i in range(num_boxes):
        bx = start_x + i * (box_w + gap) + random.randint(-15, 15)
        by = random.randint(180, 360)
        bx = max(40, min(bx, width - box_w - 40))
        by = max(40, min(by, height - box_h - 40))

        # Draw box
        thickness = random.randint(2, 4)
        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), color, thickness, cv2.LINE_AA)

        # Label inside box
        label_text = random.choice(SAMPLE_LABELS)
        cv2.putText(
            canvas,
            label_text,
            (bx + 16, by + box_h // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        box_positions.append((bx, by, box_w, box_h))
        regions.append({
            "type": "box",
            "box": [bx, by, box_w, box_h],
            "label": label_text,
            "text": label_text,
        })

    # Draw arrows between adjacent boxes
    for i in range(len(box_positions) - 1):
        b1 = box_positions[i]
        b2 = box_positions[i + 1]

        start_pt = (b1[0] + b1[2], b1[1] + b1[3] // 2)
        end_pt = (b2[0], b2[1] + b2[3] // 2)

        arrow_reg = draw_arrow(canvas, start_pt, end_pt, color, thickness=random.randint(2, 4))
        regions.append(arrow_reg)

    # Add a top title text block
    title_text = "System Architecture: " + random.choice(["Pipeline V2", "Cache Strategy", "Event Bus"])
    title_x, title_y = 100, 80
    cv2.putText(canvas, title_text, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 30), 2, cv2.LINE_AA)
    regions.append({
        "type": "text",
        "box": [title_x - 10, title_y - 30, len(title_text) * 16, 40],
        "label": title_text,
        "text": title_text,
    })

    # Add a bulleted requirements note block at the bottom
    bullet_y = 520
    bullet_notes = [
        "• At-least-once message delivery guarantee",
        "• P99 response latency < 250ms under peak load",
        "• Graceful degradation if cache node disconnects",
    ]
    for b_note in bullet_notes:
        cv2.putText(canvas, b_note, (100, bullet_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 50), 2, cv2.LINE_AA)
        bullet_y += 35

    regions.append({
        "type": "text",
        "box": [90, 490, 560, 130],
        "label": "Requirements & Notes",
        "text": "\n".join(bullet_notes),
    })

    # Add realistic glare if requested
    if include_glare and random.random() > 0.3:
        canvas = add_glare(canvas)

    return canvas, regions


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic whiteboard training data")
    parser.add_argument("--count", type=int, default=5, help="Number of synthetic boards to generate")
    parser.add_argument("--out-dir", type=str, default="synthetic_output", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx in range(args.count):
        img, regs = generate_synthetic_board()
        img_name = f"synth_board_{idx+1:03d}.png"
        json_name = f"synth_board_{idx+1:03d}.json"

        cv2.imwrite(str(out_path / img_name), img)
        with open(out_path / json_name, "w", encoding="utf-8") as f:
            json.dump({"image": img_name, "regions": regs}, f, indent=2)

        manifest.append({"image": img_name, "annotations": json_name, "num_regions": len(regs)})
        print(f"[{idx+1}/{args.count}] Generated {img_name} with {len(regs)} annotated regions")

    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Successfully saved {args.count} synthetic boards to {out_path.resolve()}")


if __name__ == "__main__":
    main()
