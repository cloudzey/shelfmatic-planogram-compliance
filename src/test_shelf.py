from pathlib import Path

from ultralytics import YOLO


model_path = Path("models/yolo11n.pt")
image_path = Path("data/samples/legacy_supermarket_shelves.jpg")
output_directory = Path("outputs/shelf_baseline")
output_image = output_directory / "detected_shelf.jpg"

if not image_path.exists():
    raise FileNotFoundError(f"Raf görüntüsü bulunamadı: {image_path}")

output_directory.mkdir(parents=True, exist_ok=True)

print("YOLO modeli yükleniyor...")
model = YOLO(str(model_path))

print(f"Raf görüntüsü analiz ediliyor: {image_path}")

results = model.predict(
    source=str(image_path),
    conf=0.25,
    imgsz=960,
    verbose=True,
)

result = results[0]
boxes = result.boxes

result.save(filename=str(output_image))

print(f"\nTespit edilen toplam nesne: {len(boxes)}")

for index, class_id in enumerate(boxes.cls, start=1):
    class_name = result.names[int(class_id)]
    confidence = float(boxes.conf[index - 1])

    print(
        f"{index}. Nesne: {class_name}, "
        f"Güven: {confidence:.2f}"
    )

print(f"\nSonuç kaydedildi: {output_image}")
