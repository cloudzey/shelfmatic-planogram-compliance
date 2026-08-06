"""Create a deterministic, single-class YOLO subset from an extracted SKU-110K dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
SELECTION_METHOD = "sha256_rank_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extracted SKU-110K CSV annotations and images are converted into a "
            "deterministic YOLO subset. No data is downloaded."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="SKU-110K root containing annotations/ and images/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_comparison.yaml"),
        help="Common comparison config, relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override dataset output path from the config.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Copy selected images or create hard links when source/output share a volume.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source files and deterministic selection without writing output.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_subset_config(config_path: Path) -> tuple[int, dict[str, int], Path]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    try:
        seed = int(config["experiment"]["seed"])
        subset = config["dataset"]["subset"]
        method = subset["selection_method"]
        counts = {split: int(subset[split]) for split in SPLITS}
        configured_output = Path(config["dataset"]["yaml"]).parent
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid subset settings in {config_path}: {error}") from error

    if method != SELECTION_METHOD:
        raise ValueError(
            f"Unsupported selection method {method!r}; expected {SELECTION_METHOD!r}."
        )
    if any(count <= 0 for count in counts.values()):
        raise ValueError("All subset sizes must be positive integers.")

    return seed, counts, configured_output


def annotation_path(source_root: Path, split: str) -> Path:
    return source_root / "annotations" / f"annotations_{split}.csv"


def scan_image_names(csv_path: Path) -> set[str]:
    image_names: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as annotation_file:
        for line_number, row in enumerate(csv.reader(annotation_file), start=1):
            if len(row) < 8:
                raise ValueError(
                    f"{csv_path}:{line_number} must contain at least 8 CSV columns."
                )
            image_name = row[0].strip()
            if not image_name or Path(image_name).name != image_name:
                raise ValueError(
                    f"{csv_path}:{line_number} has an invalid image name: {image_name!r}"
                )
            image_names.add(image_name)
    if not image_names:
        raise ValueError(f"No annotations found in {csv_path}.")
    return image_names


def deterministic_selection(
    image_names: set[str], split: str, seed: int, count: int
) -> list[str]:
    if count > len(image_names):
        raise ValueError(
            f"Requested {count} {split} images, but only {len(image_names)} are available."
        )

    def rank(image_name: str) -> tuple[bytes, str]:
        key = f"{SELECTION_METHOD}:{seed}:{split}:{image_name}".encode("utf-8")
        return hashlib.sha256(key).digest(), image_name

    return sorted(image_names, key=rank)[:count]


def load_selected_boxes(
    csv_path: Path, selected_names: set[str]
) -> dict[str, list[tuple[float, float, float, float, float, float]]]:
    boxes = {image_name: [] for image_name in selected_names}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as annotation_file:
        for line_number, row in enumerate(csv.reader(annotation_file), start=1):
            image_name = row[0].strip()
            if image_name not in boxes:
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in row[1:5])
                image_width = float(row[6])
                image_height = float(row[7])
            except ValueError as error:
                raise ValueError(
                    f"{csv_path}:{line_number} contains non-numeric box data."
                ) from error

            if image_width <= 0 or image_height <= 0:
                raise ValueError(f"{csv_path}:{line_number} has invalid image dimensions.")
            if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
                raise ValueError(f"{csv_path}:{line_number} has an out-of-range box.")

            boxes[image_name].append((x1, y1, x2, y2, image_width, image_height))

    missing = sorted(name for name, image_boxes in boxes.items() if not image_boxes)
    if missing:
        raise ValueError(f"Selected images without annotations: {missing[:5]}")
    return boxes


def yolo_label_content(
    image_boxes: list[tuple[float, float, float, float, float, float]],
) -> str:
    lines = []
    for x1, y1, x2, y2, image_width, image_height in image_boxes:
        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    return "\n".join(lines) + "\n"


def copy_image(source: Path, destination: Path, copy_mode: str) -> None:
    if copy_mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    source_root = resolve_repo_path(args.source)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"SKU-110K source root not found: {source_root}")

    seed, requested_counts, configured_output = load_subset_config(config_path)
    output_root = resolve_repo_path(args.output or configured_output)
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {output_root}"
        )

    selected: dict[str, list[str]] = {}
    selected_boxes: dict[
        str, dict[str, list[tuple[float, float, float, float, float, float]]]
    ] = {}
    available_counts: dict[str, int] = {}
    seen_across_splits: set[str] = set()

    for split in SPLITS:
        csv_path = annotation_path(source_root, split)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")
        image_names = scan_image_names(csv_path)
        overlap = seen_across_splits.intersection(image_names)
        if overlap:
            raise ValueError(f"Images occur in multiple source splits: {sorted(overlap)[:5]}")
        seen_across_splits.update(image_names)
        available_counts[split] = len(image_names)
        selected[split] = deterministic_selection(
            image_names, split, seed, requested_counts[split]
        )
        selected_boxes[split] = load_selected_boxes(csv_path, set(selected[split]))

        for image_name in selected[split]:
            source_image = source_root / "images" / image_name
            if not source_image.is_file():
                raise FileNotFoundError(f"Selected source image not found: {source_image}")

    print(
        f"Selection method={SELECTION_METHOD}, seed={seed}, output={output_root}"
    )
    for split in SPLITS:
        box_count = sum(len(boxes) for boxes in selected_boxes[split].values())
        print(
            f"{split}: selected={len(selected[split])}/"
            f"{available_counts[split]}, boxes={box_count}"
        )

    if args.dry_run:
        print("Dry-run successful; no files were written.")
        return 0

    records: dict[str, list[dict[str, object]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=False)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=False)

        for image_name in selected[split]:
            source_image = source_root / "images" / image_name
            destination_image = output_root / "images" / split / image_name
            destination_label = output_root / "labels" / split / f"{Path(image_name).stem}.txt"
            label_content = yolo_label_content(selected_boxes[split][image_name])

            copy_image(source_image, destination_image, args.copy_mode)
            destination_label.write_text(label_content, encoding="utf-8", newline="\n")

            records[split].append(
                {
                    "image": destination_image.relative_to(output_root).as_posix(),
                    "label": destination_label.relative_to(output_root).as_posix(),
                    "boxes": len(selected_boxes[split][image_name]),
                    "image_sha256": sha256_file(destination_image),
                    "label_sha256": hashlib.sha256(
                        label_content.encode("utf-8")
                    ).hexdigest(),
                }
            )

    data_yaml = (
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: object\n"
    )
    (output_root / "data.yaml").write_text(data_yaml, encoding="utf-8", newline="\n")

    selection_lines = [
        f"{split}/{record['image']}"
        for split in SPLITS
        for record in records[split]
    ]
    manifest = {
        "schema_version": 1,
        "dataset": "SKU-110K",
        "class_names": ["object"],
        "selection": {
            "method": SELECTION_METHOD,
            "seed": seed,
            "requested_counts": requested_counts,
            "selection_sha256": hashlib.sha256(
                "\n".join(selection_lines).encode("utf-8")
            ).hexdigest(),
        },
        "source_annotations": {
            split: {
                "file": f"annotations/annotations_{split}.csv",
                "sha256": sha256_file(annotation_path(source_root, split)),
                "available_images": available_counts[split],
            }
            for split in SPLITS
        },
        "splits": {
            split: {
                "image_count": len(records[split]),
                "box_count": sum(int(record["boxes"]) for record in records[split]),
                "records": records[split],
            }
            for split in SPLITS
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Dataset created: {output_root}")
    print(f"Manifest: {output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
