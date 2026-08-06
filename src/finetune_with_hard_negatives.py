import time
from pathlib import Path

from ultralytics import YOLO


project_root = Path(__file__).resolve().parent.parent

model_path = (
    project_root
    / "outputs"
    / "training"
    / "product_detector_finetune_960"
    / "weights"
    / "best.pt"
)

data_yaml = (
    project_root
    / "data"
    / "processed"
    / "hitl_product_yolo_hn_v1"
    / "data.yaml"
)

output_directory = project_root / "outputs" / "training"

if not model_path.exists():
    raise FileNotFoundError(f"Başlangıç modeli bulunamadı: {model_path}")

if not data_yaml.exists():
    raise FileNotFoundError(f"Hard-negative data.yaml bulunamadı: {data_yaml}")

print("Hard-negative düzeltme eğitimi başlatılıyor...")

start_time = time.perf_counter()

model = YOLO(str(model_path))

model.train(
    data=str(data_yaml),
    epochs=8,
    imgsz=960,
    batch=1,
    device="cpu",
    workers=0,
    max_det=1000,
    seed=42,
    deterministic=True,
    cache=False,
    plots=True,
    project=str(output_directory),
    name="product_detector_hn_finetune_960",
    patience=4,
    optimizer="AdamW",
    lr0=0.0002,
    lrf=0.01,
    close_mosaic=3,
    exist_ok=False,
)

elapsed_time = time.perf_counter() - start_time
save_directory = Path(model.trainer.save_dir)

minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("\nHARD-NEGATIVE DÜZELTME EĞİTİMİ TAMAMLANDI")
print(f"Süre: {minutes} dakika {seconds} saniye")
print(f"Sonuç klasörü: {save_directory}")
print(f"Best model: {save_directory / 'weights' / 'best.pt'}")
print(f"Last model: {save_directory / 'weights' / 'last.pt'}")