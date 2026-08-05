import time
from pathlib import Path

from ultralytics import YOLO


project_root = Path(__file__).resolve().parent.parent

model_path = project_root / "models" / "yolo11n.pt"

data_yaml = (
    project_root
    / "data"
    / "processed"
    / "hitl_product_yolo"
    / "data.yaml"
)

output_directory = project_root / "outputs" / "training"

if not model_path.exists():
    raise FileNotFoundError(f"Model bulunamadı: {model_path}")

if not data_yaml.exists():
    raise FileNotFoundError(f"data.yaml bulunamadı: {data_yaml}")

print("30 epoch tam model eğitimi başlatılıyor...")

start_time = time.perf_counter()

model = YOLO(str(model_path))

model.train(
    data=str(data_yaml),
    epochs=30,
    imgsz=640,
    batch=1,
    device="cpu",
    workers=0,
    max_det=1000,
    seed=42,
    deterministic=True,
    cache=False,
    plots=True,
    project=str(output_directory),
    name="product_detector_30epochs",
    patience=10,
    exist_ok=False,
)

elapsed_time = time.perf_counter() - start_time
save_directory = Path(model.trainer.save_dir)

minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("\nTAM MODEL EĞİTİMİ TAMAMLANDI")
print(f"Süre: {minutes} dakika {seconds} saniye")
print(f"Sonuç klasörü: {save_directory}")
print(f"Best model: {save_directory / 'weights' / 'best.pt'}")
print(f"Last model: {save_directory / 'weights' / 'last.pt'}")