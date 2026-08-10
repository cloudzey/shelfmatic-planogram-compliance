"""Run the selected Shelfmatic detector with its locked deployment settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "deployment.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect products with the selected YOLO11s Shelfmatic model."
    )
    parser.add_argument("source", type=Path, help="Image, video or directory to process.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Deployment config. Relative paths resolve from the repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "final_predictions",
        help="Directory for annotated media and prediction_summary.json.",
    )
    parser.add_argument(
        "--device",
        help="Optional Ultralytics device override such as cpu, 0 or mps.",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show the class name and confidence above each box.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding-box line width. Defaults to 2 pixels.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    """Prefer portable repository-relative paths in generated reports."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved_path)


def load_config(path: Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Deployment config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Deployment config must contain a mapping: {config_path}")
    return config


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    source = resolve_path(args.source)
    output_directory = resolve_path(args.output)
    model_path = resolve_path(Path(config["model"]["path"]))
    inference = config["inference"]

    if not source.exists():
        raise FileNotFoundError(f"Inference source not found: {source}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected model not found: {model_path}")
    if output_directory.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_directory}. "
            "Choose a new --output path to avoid mixing prediction files."
        )
    if args.line_width < 1:
        raise ValueError("Line width must be at least 1 pixel.")

    from ultralytics import YOLO

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path), task="detect")

    predict_args: dict[str, Any] = {
        "source": str(source),
        "imgsz": int(inference["imgsz"]),
        "conf": float(inference["confidence"]),
        "iou": float(inference["iou"]),
        "max_det": int(inference["max_det"]),
        "save": True,
        "save_txt": True,
        "save_conf": True,
        "show_labels": args.show_details,
        "show_conf": args.show_details,
        "line_width": args.line_width,
        "project": str(output_directory.parent),
        "name": output_directory.name,
        "exist_ok": True,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    results = model.predict(**predict_args)
    detections = []
    for result in results:
        confidences = (
            []
            if result.boxes is None
            else [round(float(value), 6) for value in result.boxes.conf.cpu().tolist()]
        )
        detections.append(
            {
                "source": display_path(Path(result.path)),
                "detections": len(confidences),
                "average_confidence": (
                    None
                    if not confidences
                    else round(sum(confidences) / len(confidences), 6)
                ),
                "minimum_confidence": None if not confidences else min(confidences),
                "maximum_confidence": None if not confidences else max(confidences),
            }
        )
    summary = {
        "model": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "model_family": config["model"]["family"],
        "settings": inference,
        "visualization": {
            "show_details": args.show_details,
            "line_width": args.line_width,
        },
        "processed_items": len(detections),
        "total_detections": sum(item["detections"] for item in detections),
        "items": detections,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "prediction_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Processed items: {summary['processed_items']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Outputs: {output_directory}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
