import shutil
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent

source_dataset = (
    project_root
    / "data"
    / "processed"
    / "hitl_product_yolo"
)

target_dataset = (
    project_root
    / "data"
    / "processed"
    / "hitl_product_yolo_hn_v1"
)

hard_negative_root = (
    project_root
    / "outputs"
    / "hard_negative_candidates_v2"
)

candidate_directory = hard_negative_root / "candidates"
accepted_file = hard_negative_root / "accepted.txt"


def count_images(directory):
    valid_extensions = {".jpg", ".jpeg", ".png"}

    return sum(
        1
        for file_path in directory.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in valid_extensions
    )


if not source_dataset.exists():
    raise FileNotFoundError(
        f"Ana YOLO veri seti bulunamadı: {source_dataset}"
    )

if not candidate_directory.exists():
    raise FileNotFoundError(
        f"Candidate klasörü bulunamadı: {candidate_directory}"
    )

if not accepted_file.exists():
    raise FileNotFoundError(
        f"accepted.txt bulunamadı: {accepted_file}"
    )

accepted_names = [
    line.strip()
    for line in accepted_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if not accepted_names:
    raise ValueError("accepted.txt boş.")

if len(accepted_names) != len(set(accepted_names)):
    raise ValueError("accepted.txt içinde tekrarlanan dosya adları var.")

missing_files = [
    name
    for name in accepted_names
    if not (candidate_directory / name).exists()
]

if missing_files:
    missing_text = "\n".join(missing_files)

    raise FileNotFoundError(
        f"Aşağıdaki candidate dosyaları bulunamadı:\n{missing_text}"
    )

if target_dataset.exists():
    raise FileExistsError(
        "Hedef veri seti zaten mevcut:\n"
        f"{target_dataset}\n\n"
        "Mevcut klasörü değiştirmemek için işlem durduruldu."
    )

print("Ana veri seti kopyalanıyor...")

shutil.copytree(source_dataset, target_dataset)

# Eski YOLO önbellekleri yeni veri setinde geçerli olmayacaktır.
for cache_file in target_dataset.rglob("*.cache"):
    cache_file.unlink()

train_images = target_dataset / "images" / "train"
train_labels = target_dataset / "labels" / "train"

added_images = []
added_labels = []

print("Hard-negative görüntüler ekleniyor...")

for index, candidate_name in enumerate(accepted_names, start=1):
    source_image = candidate_directory / candidate_name

    safe_original_stem = (
        source_image.stem
        .replace(",", "_")
        .replace(" ", "_")
    )

    destination_stem = (
        f"hn_{index:02d}_{safe_original_stem}"
    )

    destination_image = (
        train_images
        / f"{destination_stem}.jpg"
    )

    destination_label = (
        train_labels
        / f"{destination_stem}.txt"
    )

    shutil.copy2(source_image, destination_image)

    # Boş etiket dosyası:
    # Bu görüntüde hiçbir Product nesnesi bulunmadığını belirtir.
    destination_label.write_text("", encoding="utf-8")

    added_images.append(destination_image)
    added_labels.append(destination_label)

data_yaml = target_dataset / "data.yaml"

data_yaml.write_text(
    f"""path: "{target_dataset.as_posix()}"
train: images/train
val: images/val
test: images/test

names:
  0: product
""",
    encoding="utf-8",
)

train_image_count = count_images(train_images)
train_label_count = len(list(train_labels.glob("*.txt")))

val_image_count = count_images(
    target_dataset / "images" / "val"
)

test_image_count = count_images(
    target_dataset / "images" / "test"
)

empty_hard_negative_count = sum(
    1
    for label_path in added_labels
    if label_path.stat().st_size == 0
)

print("\nHARD-NEGATIVE VERİ SETİ HAZIRLANDI")
print(f"Hedef klasör: {target_dataset}")
print(f"Kabul edilen hard-negative: {len(accepted_names)}")
print(f"Train görüntüsü: {train_image_count}")
print(f"Train etiket dosyası: {train_label_count}")
print(
    "Boş hard-negative etiketi: "
    f"{empty_hard_negative_count}"
)
print(f"Val görüntüsü: {val_image_count}")
print(f"Test görüntüsü: {test_image_count}")
print(f"Yeni data.yaml: {data_yaml}")

if train_image_count != train_label_count:
    raise RuntimeError(
        "Train görüntü ve etiket sayıları eşit değil."
    )

if empty_hard_negative_count != len(accepted_names):
    raise RuntimeError(
        "Bazı hard-negative etiket dosyaları boş değil."
    )

print("\nSayısal kontroller başarılı.")
print("Ana veri seti değiştirilmedi.")