from pathlib import Path

from ultralytics import YOLO

model_path = Path("models/yolo11n.pt")
output_path = Path("outputs")

image_url = "https://ultralytics.com/images/bus.jpg"

print("Model yükleniyor...")

model = YOLO(str(model_path))

print("Görüntü analiz ediliyor...")

results = model.predict(
    source=image_url,
    conf=0.25,
    save=True,
    project=str(output_path),
    name="first_test",
    exist_ok=True,
)

results[0].save(filename="outputs/first_test/detected_bus.jpg")

boxes = results[0].boxes
object_count = len(boxes)

print(f"Tespit edilen nesne sayısı: {object_count}")

for class_id in boxes.cls:
    class_name = results[0].names[int(class_id)]
    print(f"- {class_name}")

print("Sonuç: outputs/first_test/detected_bus.jpg")