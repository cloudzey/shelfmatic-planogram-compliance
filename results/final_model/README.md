# Final model artefaktları

Bu klasör, Kaggle'da tamamlanan final model seçiminin küçük ve denetlenebilir
metin kayıtlarını içerir. Ham `.pt` checkpointleri, eğitim grafikleri ve büyük
ZIP paketleri burada tekrarlanmaz.

- `experiment_config.json`: İki model için kullanılan ortak çalışma ayarları.
- `model_comparison.json`: YOLO11s ve YOLO26s test karşılaştırması.
- `confidence_sweep.csv`: Validation ve 20 hard-negative üzerindeki eşik taraması.
- `final_locked_test.json`: `0.33` eşiği kilitlendikten sonraki final test.
- `deployment_config.json`: Kaggle exportundaki kısa deployment özeti.
- `SHA256SUMS.txt`: Repoda sürümlenen seçilmiş `best.pt` checksumu.

Tam metodoloji ve yorumlar `docs/experiments.md` dosyasındadır.
