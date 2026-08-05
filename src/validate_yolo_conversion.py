from pathlib import Path

import cv2


project_root = Path(__file__).resolve().parent.parent

dataset_root = (
    project_root
    / "data"
    / "processed"
    / "hitl_product_yolo"
)

output_directory = (
    project_root
    / "outputs"
    / "yolo_label_check"
)

output_directory.mkdir(parents=True, exist_ok=True)

total_errors = []
split_statistics = {}

for split_name in ("train", "val", "test"):
    images_directory = dataset_root / "images" / split_name
    labels_directory = dataset_root / "labels" / split_name

    image_paths = sorted(images_directory.glob("*.jpg"))
    label_paths = sorted(labels_directory.glob("*.txt"))

    image_names = {
        image_path.stem
        for image_path in image_paths
    }

    label_names = {
        label_path.stem
        for label_path in label_paths
    }

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    for name in sorted(missing_labels):
        total_errors.append(
            f"{split_name}: Etiketi olmayan görüntü: {name}"
        )

    for name in sorted(missing_images):
        total_errors.append(
            f"{split_name}: Görüntüsü olmayan etiket: {name}"
        )

    box_count = 0

    for label_path in label_paths:
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                total_errors.append(
                    f"{label_path}: {line_number}. satırda "
                    f"5 değer yok"
                )
                continue

            class_id = parts[0]

            try:
                center_x, center_y, width, height = map(
                    float,
                    parts[1:],
                )
            except ValueError:
                total_errors.append(
                    f"{label_path}: Sayıya çevrilemeyen değer"
                )
                continue

            if class_id != "0":
                total_errors.append(
                    f"{label_path}: Beklenmeyen sınıf: {class_id}"
                )

            if not (
                0 <= center_x <= 1
                and 0 <= center_y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                total_errors.append(
                    f"{label_path}: Aralık dışı değer: {line}"
                )

            box_count += 1

    split_statistics[split_name] = {
        "images": len(image_paths),
        "labels": len(label_paths),
        "boxes": box_count,
    }

print("YOLO VERİ SETİ KONTROLÜ")

for split_name, statistics in split_statistics.items():
    print(f"\n{split_name.upper()}")
    print(f"- Görüntü: {statistics['images']}")
    print(f"- Etiket dosyası: {statistics['labels']}")
    print(f"- Ürün kutusu: {statistics['boxes']}")

if total_errors:
    print(f"\nBulunan hata sayısı: {len(total_errors)}")

    for error in total_errors[:20]:
        print(f"- {error}")

    raise RuntimeError(
        "YOLO veri seti doğrulamasında hata bulundu."
    )

print("\nSayısal doğrulama başarılı: hata bulunamadı.")

# Test bölümündeki ilk görüntünün YOLO kutularını çiz
test_images_directory = dataset_root / "images" / "test"
test_labels_directory = dataset_root / "labels" / "test"

sample_image_path = sorted(
    test_images_directory.glob("*.jpg")
)[0]

sample_label_path = (
    test_labels_directory
    / f"{sample_image_path.stem}.txt"
)

image = cv2.imread(str(sample_image_path))

if image is None:
    raise RuntimeError(
        f"Örnek görüntü açılamadı: {sample_image_path}"
    )

image_height, image_width = image.shape[:2]

label_lines = sample_label_path.read_text(
    encoding="utf-8"
).splitlines()

for line in label_lines:
    _, center_x, center_y, width, height = line.split()

    center_x = float(center_x) * image_width
    center_y = float(center_y) * image_height
    width = float(width) * image_width
    height = float(height) * image_height

    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        thickness=2,
    )

text = f"YOLO product boxes: {len(label_lines)}"

cv2.rectangle(
    image,
    (10, 10),
    (550, 65),
    (0, 0, 0),
    thickness=-1,
)

cv2.putText(
    image,
    text,
    (20, 48),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    thickness=2,
)

output_path = (
    output_directory
    / f"yolo_{sample_image_path.name}"
)

cv2.imwrite(str(output_path), image)

print(f"Kontrol görseli: {output_path}")