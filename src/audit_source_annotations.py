import csv
import json
from collections import Counter
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent

source_root = (
    project_root
    / "data"
    / "raw"
    / "hitl_supermarket_shelves"
    / "Supermarket shelves"
    / "Supermarket shelves"
)

annotations_directory = source_root / "annotations"

report_directory = (
    project_root
    / "outputs"
    / "data_quality"
)

report_directory.mkdir(parents=True, exist_ok=True)

report_path = report_directory / "invalid_product_boxes.csv"

class_counts = Counter()
invalid_counts_by_file = Counter()
issues = []

annotation_files = sorted(
    annotations_directory.glob("*.json")
)

for annotation_path in annotation_files:
    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        annotation = json.load(file)

    image_width = float(annotation["size"]["width"])
    image_height = float(annotation["size"]["height"])

    for object_data in annotation["objects"]:
        class_name = object_data["classTitle"]
        class_counts[class_name] += 1

        if class_name != "Product":
            continue

        exterior = object_data["points"]["exterior"]

        if len(exterior) != 2:
            issues.append(
                {
                    "annotation": annotation_path.name,
                    "object_id": object_data.get("id"),
                    "reason": "point_count_is_not_two",
                    "coordinates": repr(exterior),
                }
            )
            invalid_counts_by_file[annotation_path.name] += 1
            continue

        first_x, first_y = exterior[0]
        second_x, second_y = exterior[1]

        x1 = min(float(first_x), float(second_x))
        y1 = min(float(first_y), float(second_y))
        x2 = max(float(first_x), float(second_x))
        y2 = max(float(first_y), float(second_y))

        x1 = max(0.0, min(x1, image_width))
        y1 = max(0.0, min(y1, image_height))
        x2 = max(0.0, min(x2, image_width))
        y2 = max(0.0, min(y2, image_height))

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            issues.append(
                {
                    "annotation": annotation_path.name,
                    "object_id": object_data.get("id"),
                    "reason": "zero_area_after_clipping",
                    "coordinates": repr(exterior),
                }
            )
            invalid_counts_by_file[annotation_path.name] += 1

with report_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as report_file:
    writer = csv.DictWriter(
        report_file,
        fieldnames=[
            "annotation",
            "object_id",
            "reason",
            "coordinates",
        ],
    )

    writer.writeheader()
    writer.writerows(issues)

valid_product_count = class_counts["Product"] - len(issues)

print("KAYNAK VERİ KALİTE RAPORU")
print(f"- Annotation dosyası: {len(annotation_files)}")
print(f"- Toplam Product etiketi: {class_counts['Product']}")
print(f"- Geçerli Product etiketi: {valid_product_count}")
print(f"- Geçersiz Product etiketi: {len(issues)}")
print(f"- Price etiketi: {class_counts['Price']}")
print(
    f"- Geçersiz kutu bulunan görüntü: "
    f"{len(invalid_counts_by_file)}"
)

if invalid_counts_by_file:
    print("\nEn fazla geçersiz kutu bulunan dosyalar:")

    for file_name, count in invalid_counts_by_file.most_common(10):
        print(f"- {file_name}: {count}")

print(f"\nRapor kaydedildi: {report_path}")