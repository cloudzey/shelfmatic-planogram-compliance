import csv
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "product_detector_finetune_960"
    / "weights"
    / "best.pt"
)

TRAIN_IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hitl_product_yolo"
    / "images"
    / "train"
)

TRAIN_LABELS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hitl_product_yolo"
    / "labels"
    / "train"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "hard_negative_candidates_v2"
CANDIDATES_DIR = OUTPUT_ROOT / "candidates"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"

CONFIDENCE_THRESHOLD = 0.15
GROUND_TRUTH_IOU_LIMIT = 0.05
MAX_CANDIDATES = 50
MAX_CANDIDATES_PER_IMAGE = 3
PADDING_RATIO = 0.30
MINIMUM_CROP_WIDTH = 96
MINIMUM_CROP_HEIGHT = 80
MINIMUM_ASPECT_RATIO = 0.35
MAXIMUM_ASPECT_RATIO = 3.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def boxes_intersect(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_width = min(ax2, bx2) - max(ax1, bx1)
    intersection_height = min(ay2, by2) - max(ay1, by1)

    return intersection_width > 0 and intersection_height > 0


def read_yolo_boxes(label_path, image_width, image_height):
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        _, center_x, center_y, box_width, box_height = map(
            float, parts[:5]
        )

        center_x *= image_width
        center_y *= image_height
        box_width *= image_width
        box_height *= image_height

        x1 = center_x - box_width / 2
        y1 = center_y - box_height / 2
        x2 = center_x + box_width / 2
        y2 = center_y + box_height / 2

        boxes.append((x1, y1, x2, y2))

    return boxes


def expand_box(box, image_width, image_height):
    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    padding_x = box_width * PADDING_RATIO
    padding_y = box_height * PADDING_RATIO

    x1 = max(0, int(x1 - padding_x))
    y1 = max(0, int(y1 - padding_y))
    x2 = min(image_width, int(x2 + padding_x))
    y2 = min(image_height, int(y2 + padding_y))

    return x1, y1, x2, y2


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

if not TRAIN_IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Train görüntü klasörü bulunamadı: {TRAIN_IMAGES_DIR}"
    )

if not TRAIN_LABELS_DIR.exists():
    raise FileNotFoundError(
        f"Train etiket klasörü bulunamadı: {TRAIN_LABELS_DIR}"
    )

if CANDIDATES_DIR.exists() and any(CANDIDATES_DIR.iterdir()):
    raise RuntimeError(
        "Aday klasörü boş değil. Önce mevcut adayları kontrol et: "
        f"{CANDIDATES_DIR}"
    )

CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    path
    for path in TRAIN_IMAGES_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

model = YOLO(str(MODEL_PATH))

manifest_rows = []
total_candidates = 0

print(f"İncelenecek train görüntüsü: {len(image_paths)}")
print("Modelin yanlış pozitif adayları aranıyor...\n")

for image_path in image_paths:
    if total_candidates >= MAX_CANDIDATES:
        break

    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")

    image_width, image_height = image.size
    label_path = TRAIN_LABELS_DIR / f"{image_path.stem}.txt"

    ground_truth_boxes = read_yolo_boxes(
        label_path,
        image_width,
        image_height,
    )

    prediction = model.predict(
        source=str(image_path),
        imgsz=1280,
        conf=CONFIDENCE_THRESHOLD,
        iou=0.70,
        max_det=1000,
        device="cpu",
        verbose=False,
    )[0]

    if prediction.boxes is None:
        continue

    predicted_boxes = prediction.boxes.xyxy.cpu().tolist()
    confidence_scores = prediction.boxes.conf.cpu().tolist()

    predictions = sorted(
        zip(predicted_boxes, confidence_scores),
        key=lambda item: item[1],
        reverse=True,
    )

    saved_boxes = []
    candidates_from_this_image = 0

    for predicted_box, confidence in predictions:
        if total_candidates >= MAX_CANDIDATES:
            break

        if candidates_from_this_image >= MAX_CANDIDATES_PER_IMAGE:
            break

        maximum_ground_truth_iou = max(
            (
                calculate_iou(predicted_box, ground_truth_box)
                for ground_truth_box in ground_truth_boxes
            ),
            default=0.0,
        )

        if maximum_ground_truth_iou >= GROUND_TRUTH_IOU_LIMIT:
            continue

        crop_box = expand_box(
            predicted_box,
            image_width,
            image_height,
        )

        crop_width = crop_box[2] - crop_box[0]
        crop_height = crop_box[3] - crop_box[1]
        crop_aspect_ratio = crop_width / crop_height

        if (
            crop_width < MINIMUM_CROP_WIDTH
            or crop_height < MINIMUM_CROP_HEIGHT
            or crop_aspect_ratio < MINIMUM_ASPECT_RATIO
            or crop_aspect_ratio > MAXIMUM_ASPECT_RATIO
        ):
            continue

        if any(
            boxes_intersect(crop_box, ground_truth_box)
            for ground_truth_box in ground_truth_boxes
        ):
            continue

        if any(
            calculate_iou(crop_box, previous_box) > 0.30
            for previous_box in saved_boxes
        ):
            continue

        candidate = image.crop(crop_box)

        confidence_percent = int(round(confidence * 100))
        candidate_name = (
            f"{image_path.stem}_candidate_"
            f"{candidates_from_this_image + 1:02d}_"
            f"conf{confidence_percent:02d}.jpg"
        )

        candidate_path = CANDIDATES_DIR / candidate_name
        candidate.save(candidate_path, quality=92)

        manifest_rows.append(
            {
                "candidate": candidate_name,
                "source_image": image_path.name,
                "confidence": round(confidence, 4),
                "x1": crop_box[0],
                "y1": crop_box[1],
                "x2": crop_box[2],
                "y2": crop_box[3],
            }
        )

        saved_boxes.append(crop_box)
        candidates_from_this_image += 1
        total_candidates += 1

    print(
        f"{image_path.name}: "
        f"{candidates_from_this_image} aday kaydedildi"
    )

with MANIFEST_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as manifest_file:
    writer = csv.DictWriter(
        manifest_file,
        fieldnames=[
            "candidate",
            "source_image",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )

    writer.writeheader()
    writer.writerows(manifest_rows)

print("\nHARD-NEGATIVE ADAY TARAMASI TAMAMLANDI")
print(f"Toplam aday: {total_candidates}")
print(f"Aday klasörü: {CANDIDATES_DIR}")
print(f"Manifest: {MANIFEST_PATH}")
print(
    "\nUYARI: Bu adaylar henüz eğitim verisi değildir. "
    "Her görsel elle kontrol edilmelidir."
)