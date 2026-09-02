"""Reusable helpers for Shelfmatic image inference and export."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "deployment.yaml"


def load_deployment_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and minimally validate the locked deployment configuration."""
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Deployment config must contain a mapping.")
    if "model" not in config or "inference" not in config:
        raise ValueError("Deployment config must define model and inference sections.")
    return config


def extract_detections(result: Any) -> list[dict[str, Any]]:
    """Convert one Ultralytics result into JSON/CSV-friendly records."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    image_height, image_width = result.orig_shape[:2]

    coordinates = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    names = result.names

    records = []
    for index, (xyxy, confidence, class_id) in enumerate(
        zip(coordinates, confidences, class_ids, strict=True),
        start=1,
    ):
        x1, y1, x2, y2 = (float(value) for value in xyxy)

        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        numeric_class_id = int(class_id)

        records.append(
            {
                "id": index,
                "class_id": numeric_class_id,
                "class_name": str(names[numeric_class_id]),
                "confidence": round(float(confidence), 6),

                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),

                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "width": round(width, 2),
                "height": round(height, 2),

                "x1_norm": round(x1 / image_width, 6),
                "y1_norm": round(y1 / image_height, 6),
                "x2_norm": round(x2 / image_width, 6),
                "y2_norm": round(y2 / image_height, 6),

                "center_x_norm": round(center_x / image_width, 6),
                "center_y_norm": round(center_y / image_height, 6),
                "width_norm": round(width / image_width, 6),
                "height_norm": round(height / image_height, 6),
            }
        )

    return records


def render_result(result: Any, show_details: bool = False) -> Image.Image:
    """Render boxes with clean defaults and return an RGB PIL image."""
    annotated_bgr = result.plot(
        labels=show_details,
        conf=show_details,
        line_width=2,
    )
    annotated_rgb = np.ascontiguousarray(annotated_bgr[..., ::-1])
    return Image.fromarray(annotated_rgb)


def image_to_jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    """Encode a PIL image as downloadable JPEG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def report_to_json_bytes(report: dict[str, Any]) -> bytes:
    """Encode an inference report as UTF-8 JSON."""
    return (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def detections_to_csv_bytes(detections: list[dict[str, Any]]) -> bytes:
    """Encode detection records as a UTF-8 CSV with a stable header."""
    fieldnames = [
    "id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "width",
    "height",
    "x1_norm",
    "y1_norm",
    "x2_norm",
    "y2_norm",
    "center_x_norm",
    "center_y_norm",
    "width_norm",
    "height_norm",
]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(detections)
    return buffer.getvalue().encode("utf-8-sig")
