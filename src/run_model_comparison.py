"""Train and evaluate YOLO11s and YOLO26s under one reproducible protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_IDS = ("yolo11s", "yolo26s")
SPLITS = ("train", "val", "test")
SUMMARY_FIELDS = (
    "model_id",
    "initial_weights",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "parameters",
    "trainable_parameters",
    "model_size_bytes",
    "model_size_mib",
    "inference_ms_per_image",
    "train_seconds",
    "ultralytics_version",
    "seed",
    "imgsz",
    "train_batch",
    "eval_batch",
    "run_directory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a non-overwriting YOLO11s/YOLO26s comparison on one manifested "
            "SKU-110K subset."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_comparison.yaml"),
        help="Common experiment config; relative paths resolve from the repo root.",
    )
    parser.add_argument(
        "--model",
        choices=("all",) + MODEL_IDS,
        default="all",
        help="Run both models or only one model.",
    )
    parser.add_argument("--data", type=Path, help="Override dataset YAML path.")
    parser.add_argument(
        "--output-root", type=Path, help="Override non-overwriting run root."
    )
    parser.add_argument("--device", help="Override the shared device, e.g. 0 or cpu.")
    parser.add_argument("--yolo11-weights", help="Override YOLO11s weights name/path.")
    parser.add_argument("--yolo26-weights", help="Override YOLO26s weights name/path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, manifest, splits, model support and output paths only.",
    )
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Also recompute every selected image/label SHA-256 during dry-run.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    required_sections = (
        "experiment",
        "dataset",
        "models",
        "training",
        "augmentation",
        "evaluation",
    )
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Config sections missing: {', '.join(missing)}")
    return config


def dataset_root_from_yaml(data_yaml: Path, data_config: dict[str, Any]) -> Path:
    configured_root = data_config.get("path")
    if configured_root:
        configured_path = Path(configured_root)
        return (
            configured_path.resolve()
            if configured_path.is_absolute()
            else (data_yaml.parent / configured_path).resolve()
        )
    return data_yaml.parent.resolve()


def validate_dataset(
    data_yaml: Path,
    manifest_path: Path,
    expected_seed: int,
    expected_counts: dict[str, int],
    verify_hashes: bool,
) -> dict[str, Any]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    with data_yaml.open("r", encoding="utf-8") as data_file:
        data_config = yaml.safe_load(data_file)
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if manifest.get("dataset") != "SKU-110K":
        raise ValueError("Manifest dataset must be SKU-110K.")
    selection = manifest.get("selection", {})
    if selection.get("seed") != expected_seed:
        raise ValueError(
            f"Manifest seed {selection.get('seed')!r} does not match config seed "
            f"{expected_seed}."
        )
    if selection.get("method") != "sha256_rank_v1":
        raise ValueError("Manifest selection method must be sha256_rank_v1.")
    if manifest_path.parent.resolve() != data_yaml.parent.resolve():
        raise ValueError("Dataset YAML and manifest must be in the same dataset root.")
    if data_config.get("names") not in ({0: "object"}, ["object"]):
        raise ValueError("Dataset must contain the single class 'object'.")

    dataset_root = dataset_root_from_yaml(data_yaml, data_config)
    seen_image_names: set[str] = set()
    manifest_splits = manifest.get("splits", {})

    for split in SPLITS:
        split_data = manifest_splits.get(split, {})
        records = split_data.get("records", [])
        image_count = split_data.get("image_count")
        if image_count != expected_counts[split] or len(records) != expected_counts[split]:
            raise ValueError(
                f"{split} count mismatch: config={expected_counts[split]}, "
                f"manifest={image_count}, records={len(records)}."
            )

        split_path_value = data_config.get(split)
        if not isinstance(split_path_value, str):
            raise ValueError(f"Dataset YAML must define a string path for {split}.")
        split_image_dir = (dataset_root / split_path_value).resolve()
        if not split_image_dir.is_dir():
            raise FileNotFoundError(f"Dataset split directory not found: {split_image_dir}")

        manifest_images: set[str] = set()
        manifest_labels: set[str] = set()
        for record in records:
            image_relative = Path(record["image"])
            label_relative = Path(record["label"])
            image_path = (dataset_root / image_relative).resolve()
            label_path = (dataset_root / label_relative).resolve()
            if not image_path.is_relative_to(dataset_root):
                raise ValueError(f"Image escapes dataset root: {image_relative}")
            if not label_path.is_relative_to(dataset_root):
                raise ValueError(f"Label escapes dataset root: {label_relative}")
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Manifest file missing: {image_path} or {label_path}")
            if image_path.parent != split_image_dir:
                raise ValueError(f"Manifest image is outside the {split} image directory.")
            if verify_hashes and sha256_file(image_path) != record.get("image_sha256"):
                raise ValueError(f"Image hash differs from manifest: {image_path}")
            if verify_hashes and sha256_file(label_path) != record.get("label_sha256"):
                raise ValueError(f"Label hash differs from manifest: {label_path}")

            image_name = image_path.name
            if image_name in seen_image_names:
                raise ValueError(f"Image occurs in multiple splits: {image_name}")
            seen_image_names.add(image_name)
            manifest_images.add(image_name)
            manifest_labels.add(label_path.name)

        disk_images = {path.name for path in split_image_dir.iterdir() if path.is_file()}
        label_dir = dataset_root / "labels" / split
        disk_labels = {path.name for path in label_dir.iterdir() if path.is_file()}
        if disk_images != manifest_images or disk_labels != manifest_labels:
            raise ValueError(f"{split} files differ from the recorded manifest.")

    return manifest


def weight_value(configured: str, override: str | None) -> str:
    value = override or configured
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    if path.parent != Path("."):
        return str((PROJECT_ROOT / path).resolve())
    repo_candidate = PROJECT_ROOT / path
    return str(repo_candidate.resolve()) if repo_candidate.exists() else value


def validate_model_support(weights: dict[str, str]) -> tuple[str, dict[str, str]]:
    try:
        import ultralytics
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
    except ImportError as error:
        raise RuntimeError("Ultralytics is not installed in the active environment.") from error

    statuses: dict[str, str] = {}
    for model_id, value in weights.items():
        path = Path(value)
        asset_name = path.name
        if path.is_file():
            statuses[model_id] = f"local weights found: {relative_or_posix(path)}"
        elif asset_name in GITHUB_ASSETS_NAMES:
            statuses[model_id] = (
                f"supported Ultralytics asset: {asset_name} "
                "(download deferred until training)"
            )
        else:
            raise FileNotFoundError(
                f"Weights are neither a local file nor a supported asset: {value}. "
                f"Use --{model_id.replace('s', '')}-weights with a verified local file."
            )
    return ultralytics.__version__, statuses


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "model_id": result["model_id"],
        "initial_weights": result["initial_weights"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "parameters": result["parameters"],
        "trainable_parameters": result["trainable_parameters"],
        "model_size_bytes": result["model_size_bytes"],
        "model_size_mib": result["model_size_mib"],
        "inference_ms_per_image": result["inference_ms_per_image"],
        "train_seconds": result["train_seconds"],
        "ultralytics_version": result["environment"]["ultralytics"],
        "seed": result["common_settings"]["seed"],
        "imgsz": result["common_settings"]["imgsz"],
        "train_batch": result["common_settings"]["train_batch"],
        "eval_batch": result["common_settings"]["eval_batch"],
        "run_directory": result["run_directory"],
    }


def write_comparison_summary(output_root: Path) -> None:
    results = []
    for model_id in MODEL_IDS:
        result_path = output_root / model_id / "result.json"
        if result_path.is_file():
            with result_path.open("r", encoding="utf-8") as result_file:
                results.append(json.load(result_file))

    if not results:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (output_root / "comparison_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(result_row(result) for result in results)


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_config(config_path)

    seed = int(config["experiment"]["seed"])
    if seed != 42:
        raise ValueError("This comparison protocol requires seed=42.")
    subset_config = config["dataset"]["subset"]
    expected_counts = {split: int(subset_config[split]) for split in SPLITS}
    data_yaml = resolve_repo_path(args.data or Path(config["dataset"]["yaml"]))
    manifest_path = resolve_repo_path(Path(config["dataset"]["manifest"]))
    output_root = resolve_repo_path(
        args.output_root or Path(config["output_root"])
    )

    selected_models = list(MODEL_IDS) if args.model == "all" else [args.model]
    configured_models = config["models"]
    weights = {
        "yolo11s": weight_value(
            configured_models["yolo11s"]["weights"], args.yolo11_weights
        ),
        "yolo26s": weight_value(
            configured_models["yolo26s"]["weights"], args.yolo26_weights
        ),
    }
    selected_weights = {model_id: weights[model_id] for model_id in selected_models}

    manifest = validate_dataset(
        data_yaml=data_yaml,
        manifest_path=manifest_path,
        expected_seed=seed,
        expected_counts=expected_counts,
        verify_hashes=args.verify_hashes or not args.dry_run,
    )
    ultralytics_version, support_statuses = validate_model_support(selected_weights)

    training = dict(config["training"])
    augmentation = dict(config["augmentation"])
    evaluation = dict(config["evaluation"])
    if args.device is not None:
        training["device"] = args.device
    training["seed"] = seed
    training["deterministic"] = True

    for model_id in selected_models:
        run_directory = output_root / model_id
        if run_directory.exists():
            raise FileExistsError(
                f"Run directory already exists and will not be overwritten: {run_directory}"
            )

    print(f"Ultralytics: {ultralytics_version}")
    print(f"Dataset: {relative_or_posix(data_yaml)}")
    print(f"Manifest SHA256: {sha256_file(manifest_path)}")
    print(
        "Splits: "
        + ", ".join(f"{split}={expected_counts[split]}" for split in SPLITS)
    )
    print(
        f"Common settings: seed={seed}, imgsz={training['imgsz']}, "
        f"epochs={training['epochs']}, batch={training['batch']}, "
        f"device={training['device']}"
    )
    print(
        "Content hashes: "
        + ("verified" if args.verify_hashes or not args.dry_run else "skipped in light dry-run")
    )
    for model_id in selected_models:
        print(f"{model_id}: {support_statuses[model_id]}")
        print(f"  output: {relative_or_posix(output_root / model_id)}")

    if args.dry_run:
        print("Dry-run successful; no model was loaded, downloaded, trained or evaluated.")
        return 0

    import torch
    from ultralytics import YOLO

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256_file(manifest_path)
    for model_id in selected_models:
        print(f"\nStarting {model_id} training...")
        start_time = time.perf_counter()
        model = YOLO(selected_weights[model_id])
        model.train(
            data=str(data_yaml),
            project=str(output_root),
            name=model_id,
            exist_ok=False,
            **training,
            **augmentation,
        )
        train_seconds = time.perf_counter() - start_time
        run_directory = Path(model.trainer.save_dir).resolve()
        best_checkpoint = run_directory / "weights" / "best.pt"
        if not best_checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint was not produced: {best_checkpoint}")

        resolved_snapshot = {
            "config": relative_or_posix(config_path),
            "dataset_yaml": relative_or_posix(data_yaml),
            "dataset_manifest": relative_or_posix(manifest_path),
            "dataset_manifest_sha256": manifest_hash,
            "model_id": model_id,
            "initial_weights": (
                relative_or_posix(Path(selected_weights[model_id]))
                if Path(selected_weights[model_id]).is_absolute()
                else selected_weights[model_id]
            ),
            "training": training,
            "augmentation": augmentation,
            "evaluation": evaluation,
        }
        (run_directory / "resolved_experiment.json").write_text(
            json.dumps(resolved_snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        evaluated_model = YOLO(str(best_checkpoint))
        evaluation_device = training["device"]
        metrics = evaluated_model.val(
            data=str(data_yaml),
            imgsz=training["imgsz"],
            device=evaluation_device,
            project=str(run_directory),
            name="test_metrics",
            exist_ok=False,
            **evaluation,
        )

        parameters = sum(parameter.numel() for parameter in evaluated_model.model.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in evaluated_model.model.parameters()
            if parameter.requires_grad
        )
        model_size_bytes = best_checkpoint.stat().st_size
        speed = {key: float(value) for key, value in metrics.speed.items()}
        result = {
            "schema_version": 1,
            "model_id": model_id,
            "initial_weights": resolved_snapshot["initial_weights"],
            "best_checkpoint": relative_or_posix(best_checkpoint),
            "run_directory": relative_or_posix(run_directory),
            "metrics": {
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            },
            "parameters": int(parameters),
            "trainable_parameters": int(trainable_parameters),
            "model_size_bytes": model_size_bytes,
            "model_size_mib": round(model_size_bytes / (1024 * 1024), 4),
            "inference_ms_per_image": speed.get("inference"),
            "speed_ms_per_image": speed,
            "train_seconds": round(train_seconds, 3),
            "common_settings": {
                "seed": seed,
                "imgsz": training["imgsz"],
                "epochs": training["epochs"],
                "train_batch": training["batch"],
                "eval_batch": evaluation["batch"],
                "dataset_manifest_sha256": manifest_hash,
                "selection_sha256": manifest["selection"]["selection_sha256"],
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": str(torch.__version__),
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "ultralytics": ultralytics_version,
            },
        }
        (run_directory / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with (run_directory / "result.csv").open(
            "w", encoding="utf-8", newline=""
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow(result_row(result))

        write_comparison_summary(output_root)
        print(f"Completed {model_id}: {run_directory / 'result.json'}")

    print(f"Comparison summary: {output_root / 'comparison_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
