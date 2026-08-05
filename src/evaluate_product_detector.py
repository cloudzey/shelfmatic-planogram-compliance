from pathlib import Path

from ultralytics import YOLO


project_root = Path(__file__).resolve().parent.parent

model_path = (
    project_root
    / "outputs"
    / "training"
    / "product_detector_30epochs"
    / "weights"
    / "best.pt"
)

data_yaml = (
    project_root
    / "data"
    / "processed"
    / "hitl_product_yolo"
    / "data.yaml"
)

test_images = (
    project_root
    / "data"
    / "processed"
    / "hitl_product_yolo"
    / "images"
    / "test"
)

output_directory = project_root / "outputs" / "test_evaluation"


if not model_path.exists():
    raise FileNotFoundError(f"Model bulunamadı: {model_path}")

if not data_yaml.exists():
    raise FileNotFoundError(f"Veri tanımı bulunamadı: {data_yaml}")

if not test_images.exists():
    raise FileNotFoundError(f"Test görüntüleri bulunamadı: {test_images}")


print("Eğitilmiş model test veri setinde değerlendiriliyor...")

model = YOLO(str(model_path))

metrics = model.val(
    data=str(data_yaml),
    split="test",
    imgsz=640,
    batch=1,
    device="cpu",
    workers=0,
    max_det=1000,
    plots=True,
    project=str(output_directory),
    name="metrics",
    exist_ok=True,
)

print("\nTEST SONUÇLARI")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")

print("\nTest görüntüleri üzerinde tahmin yapılıyor...")

model.predict(
    source=str(test_images),
    imgsz=640,
    conf=0.25,
    iou=0.70,
    device="cpu",
    max_det=1000,
    save=True,
    save_txt=True,
    save_conf=True,
    show_labels=False,
    show_conf=False,
    line_width=1,
    project=str(output_directory),
    name="predictions_clean",
    exist_ok=True,
)

print(f"\nTahmin görüntüleri kaydedildi: {output_directory / 'predictions'}")