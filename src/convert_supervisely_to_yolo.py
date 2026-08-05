import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


RANDOM_SEED = 42

project_root = Path(__file__).resolve().parent.parent

source_root = (
    project_root
    / "data"
    / "raw"
    / "hitl_supermarket_shelves"
    / "Supermarket shelves"
    / "Supermarket shelves"
)

images_directory = source_root / "images"
annotations_directory = source_root / "annotations"

output_root = (
    project_root
    / "data"
    / "processed"
    / "hitl_product_yolo"
)

if output_root.exists():
    raise FileExistsError(
        f"Çıktı klasörü zaten var: {output_root}\n"
        "Dönüşümü yeniden yapmak istiyorsan önce klasörü kontrollü "
        "olarak kaldırmalısın."
    )

image_files = sorted(images_directory.glob("*.jpg"))

if not image_files:
    raise FileNotFoundError(
        f"Kaynak görüntü bulunamadı: {images_directory}"
    )

random_generator = random.Random(RANDOM_SEED)
random_generator.shuffle(image_files)

total_image_count = len(image_files)

validation_count = round(total_image_count * 0.15)
test_count = round(total_image_count * 0.15)
train_count = total_image_count - validation_count - test_count

splits = {
    "train": image_files[:train_count],
    "val": image_files[
        train_count:train_count + validation_count
    ],
    "test": image_files[
        train_count + validation_count:
    ],
}

statistics = {
    split_name: Counter()
    for split_name in splits
}

for split_name in splits:
    (output_root / "images" / split_name).mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_root / "labels" / split_name).mkdir(
        parents=True,
        exist_ok=True,
    )

for split_name, split_images in splits.items():
    for image_path in split_images:
        annotation_path = (
            annotations_directory
            / f"{image_path.name}.json"
        )

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Annotation bulunamadı: {annotation_path}"
            )

        with annotation_path.open(
            "r",
            encoding="utf-8",
        ) as annotation_file:
            annotation = json.load(annotation_file)

        annotation_width = int(annotation["size"]["width"])
        annotation_height = int(annotation["size"]["height"])

        with Image.open(image_path) as image:
            actual_width, actual_height = image.size

        if (
            annotation_width != actual_width
            or annotation_height != actual_height
        ):
            raise ValueError(
                f"Görüntü/annotation ölçüsü uyuşmuyor: "
                f"{image_path.name}"
            )

        yolo_lines = []

        for object_data in annotation["objects"]:
            class_name = object_data["classTitle"]

            if class_name != "Product":
                statistics[split_name][
                    f"ignored_{class_name}"
                ] += 1
                continue

            exterior = object_data["points"]["exterior"]

            if len(exterior) != 2:
                statistics[split_name]["invalid_boxes"] += 1
                continue

            first_x, first_y = exterior[0]
            second_x, second_y = exterior[1]

            x1 = min(float(first_x), float(second_x))
            y1 = min(float(first_y), float(second_y))
            x2 = max(float(first_x), float(second_x))
            y2 = max(float(first_y), float(second_y))

            x1 = max(0.0, min(x1, annotation_width))
            y1 = max(0.0, min(y1, annotation_height))
            x2 = max(0.0, min(x2, annotation_width))
            y2 = max(0.0, min(y2, annotation_height))

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                statistics[split_name]["invalid_boxes"] += 1
                continue

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            normalized_center_x = center_x / annotation_width
            normalized_center_y = center_y / annotation_height
            normalized_width = box_width / annotation_width
            normalized_height = box_height / annotation_height

            yolo_line = (
                f"0 "
                f"{normalized_center_x:.6f} "
                f"{normalized_center_y:.6f} "
                f"{normalized_width:.6f} "
                f"{normalized_height:.6f}"
            )

            yolo_lines.append(yolo_line)
            statistics[split_name]["product_boxes"] += 1

        destination_image = (
            output_root
            / "images"
            / split_name
            / image_path.name
        )

        destination_label = (
            output_root
            / "labels"
            / split_name
            / f"{image_path.stem}.txt"
        )

        shutil.copy2(image_path, destination_image)

        label_content = "\n".join(yolo_lines)

        if label_content:
            label_content += "\n"

        destination_label.write_text(
            label_content,
            encoding="utf-8",
        )

        statistics[split_name]["images"] += 1

yaml_content = f"""path: {output_root.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: product
"""

yaml_path = output_root / "data.yaml"
yaml_path.write_text(yaml_content, encoding="utf-8")

print("\nDönüşüm tamamlandı.")

for split_name in ("train", "val", "test"):
    split_statistics = statistics[split_name]

    print(f"\n{split_name.upper()}")
    print(f"- Görüntü: {split_statistics['images']}")
    print(
        f"- Product kutusu: "
        f"{split_statistics['product_boxes']}"
    )
    print(
        f"- Geçersiz kutu: "
        f"{split_statistics['invalid_boxes']}"
    )

    ignored_count = sum(
        count
        for key, count in split_statistics.items()
        if key.startswith("ignored_")
    )

    print(f"- Yok sayılan diğer sınıflar: {ignored_count}")

print(f"\nYOLO veri tanımı: {yaml_path}")