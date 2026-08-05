import json
from collections import Counter
from pathlib import Path

import cv2

project_root = Path(__file__).resolve().parent.parent

dataset_root = (
    project_root
    / "data"
    / "raw"
    / "hitl_supermarket_shelves"
    / "Supermarket shelves"
    / "Supermarket shelves"
)

images_directory = dataset_root / "images"
annotations_directory = dataset_root / "annotations"

output_directory = project_root / "outputs" / "annotation_check"
output_directory.mkdir(parents=True, exist_ok=True)

annotation_files = sorted(annotations_directory.glob("*.json"))

if not annotation_files:
    raise FileNotFoundError(
        f"Annotation dosyası bulunamadı: {annotations_directory}"
    )

annotation_path = annotation_files[0]

image_name = annotation_path.stem
image_path = images_directory / image_name

if not image_path.exists():
    raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

with annotation_path.open("r", encoding="utf-8") as file:
    annotation = json.load(file)

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(f"Görüntü OpenCV ile açılamadı: {image_path}")

colors = {
    "Product": (0, 255, 0),   # Yeşil
    "Price": (0, 165, 255),   # Turuncu
}

class_counts = Counter()

for object_data in annotation["objects"]:
    class_name = object_data["classTitle"]
    exterior = object_data["points"]["exterior"]

    if len(exterior) != 2:
        continue

    x1, y1 = map(int, exterior[0])
    x2, y2 = map(int, exterior[1])

    color = colors.get(class_name, (255, 0, 255))

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness=2,
    )

    class_counts[class_name] += 1

legend_text = (
    f"Product: {class_counts['Product']} | "
    f"Price: {class_counts['Price']}"
)

cv2.rectangle(image, (10, 10), (520, 60), (0, 0, 0), thickness=-1)

cv2.putText(
    image,
    legend_text,
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    thickness=2,
)

output_path = output_directory / f"annotated_{image_name}"

success = cv2.imwrite(str(output_path), image)

if not success:
    raise RuntimeError(f"Sonuç görüntüsü kaydedilemedi: {output_path}")

print(f"Annotation dosyası: {annotation_path.name}")
print(f"Görüntü dosyası: {image_name}")
print(f"Görüntü ölçüsü: {annotation['size']}")
print(f"Toplam kutu: {len(annotation['objects'])}")

for class_name, count in class_counts.items():
    print(f"- {class_name}: {count}")

print(f"Sonuç kaydedildi: {output_path}")