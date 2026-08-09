# Shelfmatic Planogram Compliance

Bu proje, raf görüntülerindeki ürünlerin görüntü işleme ve nesne tespiti yöntemleriyle otomatik olarak belirlenmesini amaçlayan bir PoC çalışmasıdır.

## Projenin amacı

İlk aşamada bir raf fotoğrafındaki ürünlerin:

- tespit edilmesi,
- kutularla işaretlenmesi,
- güven skorlarının hesaplanması,
- ürün sayısının belirlenmesi,
- sonuçların görsel ve JSON formatında kaydedilmesi

hedeflenmektedir.

İlerleyen aşamalarda tespit edilen raf düzeninin hedef planogramla karşılaştırılması planlanmaktadır.

## Kullanılacak teknolojiler

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- Streamlit

## Proje durumu

Proje ortamı hazırlanmış, YOLO kurulumu tamamlanmış ve genel amaçlı
YOLO11n modeliyle ilk raf baseline deneyi gerçekleştirilmiştir.

## Test görseli kaynağı

- “Supermarket shelves”, Frankie Fouganthin, Wikimedia Commons
- Lisans: CC BY-SA 4.0
- Kaynak: https://commons.wikimedia.org/wiki/File:Supermarket_shelves.jpg

## İlk baseline deneyi

İlk deneyde COCO veri setiyle önceden eğitilmiş YOLO11n modeli kullanılmıştır.

### Deney ayarları

- Görüntü boyutu: 1280 × 920
- Model giriş boyutu: 960
- Güven eşiği: 0.25
- Model: YOLO11n
- Test görüntüsü: Market rafı

### Sonuç

Model toplam 9 nesne tespit etmiş ve bütün nesneleri `bottle` olarak
sınıflandırmıştır. Güven skorları yaklaşık 0.26 ile 0.50 arasında
değişmektedir.

Görüntüde çok daha fazla ürün bulunmasına rağmen kutu, paket ve
kavanozların büyük bölümü tespit edilememiştir. Bazı kavanozlar ise
yanlış biçimde `bottle` olarak sınıflandırılmıştır.

### Değerlendirme

Genel amaçlı COCO modeli raf ürünlerinin tespiti için yeterli değildir.
Sonraki aşamada raf görüntüleriyle eğitilmiş, bütün ürün yüzlerini
`product` sınıfı altında tespit eden özel bir model hazırlanacaktır.

## Eğitim veri seti

İlk ürün tespit modeli için Humans in the Loop tarafından yayımlanan
Supermarket Shelves Dataset kullanılmıştır.

- Lisans: CC0 1.0
- Görüntü sayısı: 45
- Toplam annotation: 11.743
- Product annotation: 9.967
- Price annotation: 1.776
- Kaynak: https://humansintheloop.org/resources/datasets/supermarket-shelves-dataset/

İlk model yalnızca genel `product` sınıfını tespit edeceği için `Price`
etiketleri eğitim kapsamı dışında bırakılmıştır.

Kaynak veri kalite kontrolünde 57 adet geçersiz veya görüntü sınırları
dışında kalan Product etiketi belirlenmiş ve eğitime dahil edilmemiştir.
Toplam 9.910 geçerli ürün kutusu kullanılmıştır.

### Veri bölünmesi

Veriler sabit `42` random seed değeriyle aşağıdaki şekilde bölünmüştür:

- Train: 31 görüntü, 6.335 ürün kutusu
- Validation: 7 görüntü, 1.403 ürün kutusu
- Test: 7 görüntü, 2.172 ürün kutusu

Supervisely JSON etiketleri YOLO formatına dönüştürülmüş; görüntü-etiket
eşleşmeleri, normalize koordinatlar ve kutular görsel olarak
doğrulanmıştır.

## Eğitim sağlık testi

Veri ve eğitim hattını doğrulamak amacıyla YOLO11n modeliyle 1 epoch
eğitim gerçekleştirilmiştir.

### Eğitim ayarları

- Model: YOLO11n
- Epoch: 1
- Görüntü boyutu: 640
- Batch size: 1
- Cihaz: CPU
- İşlemci: Snapdragon X Elite
- Süre: 1 dakika 11 saniye

### İlk sonuçlar

- Precision: 0.103
- Recall: 0.418
- mAP50: 0.0622
- mAP50-95: 0.0258

Bu aşamanın amacı yüksek model performansı elde etmek değil; veri setinin
okunabildiğini, modelin eğitilebildiğini, validation işleminin
tamamlandığını ve ağırlık dosyalarının üretildiğini doğrulamaktır.

Eğitim sırasında dört adet yinelenen etiket Ultralytics tarafından
otomatik olarak kaldırılmıştır.

## Deney geçmişi

960 pikselde fine-tune edilmiş korunan YOLO11n baseline, reddedilen
hard-negative ablation, metrik farkları ve ret gerekçeleri
[`docs/experiments.md`](docs/experiments.md) dosyasında kayıtlıdır. Kök
dizindeki `best.pt` kabul edilen baseline checkpointidir. Hard-negative
checkpointi ve tanı kayıtları yalnızca tarihsel ablation kaydıdır; yeni
eğitimlerde başlangıç ağırlığı olarak kullanılmamalıdır.

## Tekrarlanabilir YOLO11s / YOLO26s karşılaştırması

Karşılaştırma altyapısı, yerel olarak çıkarılmış SKU-110K veri setinden aynı
train/validation/test alt kümesini deterministik biçimde üretir ve iki modeli
tek ortak protokolle çalıştırır. Scriptler veri setini indirmez. SKU-110K kök
dizini şu yapıda olmalıdır:

```text
data/raw/SKU-110K/
├── annotations/
│   ├── annotations_train.csv
│   ├── annotations_val.csv
│   └── annotations_test.csv
└── images/
```

Ortak seed, alt küme boyutları, görüntü boyutu, epoch, batch, optimizer ve
augmentation ayarları `configs/model_comparison.yaml` içindedir. Varsayılan
protokol seed 42 ile train/val/test için 1000/100/300 görüntü, 960 piksel, 50
epoch ve train batch 4 kullanır. Göreli yollar repo kökünden çözülür.

### 1. Alt kümeyi doğrula ve hazırla

Önce yalnızca kaynak düzenini ve seçimi kontrol et:

```bash
python src/prepare_sku110k_subset.py --source data/raw/SKU-110K --config configs/model_comparison.yaml --dry-run
```

Ardından aynı komutu `--dry-run` olmadan çalıştır:

```bash
python src/prepare_sku110k_subset.py --source data/raw/SKU-110K --config configs/model_comparison.yaml
```

Bu adım platformdan bağımsız `data.yaml` ile seçilen dosya, split, kutu ve
SHA-256 kayıtlarını içeren `manifest.json` üretir. Var olan çıktı klasörünün
üzerine yazılmaz. Aynı disk bölümünde veri kopyalamamak için isteğe bağlı
`--copy-mode hardlink` kullanılabilir.

### 2. Eğitim öncesi dry-run

```bash
python src/run_model_comparison.py --config configs/model_comparison.yaml --dry-run
```

Dry-run; configi, manifesti, split çakışmalarını, dosya sayılarını, kurulu
Ultralytics model desteğini ve çıktı klasörlerini doğrular. Model yüklemez,
ağırlık indirmez, eğitim veya değerlendirme başlatmaz. Görüntü ve etiket
hashlerini de yeniden hesaplamak için `--verify-hashes` eklenebilir; gerçek
eğitim öncesinde bu hash kontrolü otomatik olarak yapılır.

### 3. Modelleri aynı protokolle eğit ve değerlendir

Koşular ayrı ayrı başlatılabilir:

```bash
python src/run_model_comparison.py --config configs/model_comparison.yaml --model yolo26s
python src/run_model_comparison.py --config configs/model_comparison.yaml --model yolo11s
```

Ya da temiz bir çıktı kökünde ikisi sıralı çalıştırılabilir:

```bash
python src/run_model_comparison.py --config configs/model_comparison.yaml --model all
```

Varsayılan gerçek ağırlık adları `yolo11s.pt` ve `yolo26s.pt` olup dosyalar
ilk gerçek eğitimde `models/` altına alınır. Önceden indirilmiş doğrulanmış
ağırlıklar komut satırından verilebilir:

```bash
python src/run_model_comparison.py --config configs/model_comparison.yaml --model yolo26s --yolo26-weights /path/to/yolo26s.pt
```

Cihaz ayarı iki model için aynı tutulmalıdır; gerektiğinde her iki komuta da
örneğin `--device cpu` eklenebilir. Her modelin koşu klasörü mevcutsa script
üzerine yazmak yerine hata verir. Sonuçlar model klasörlerindeki `result.csv`
ve `result.json` yanında karşılaştırma kökünde `comparison_results.csv` ve
`comparison_results.json` olarak yan yana kaydedilir.

## Kaggle üzerinde çalıştırma

[`notebooks/kaggle_model_comparison.ipynb`](notebooks/kaggle_model_comparison.ipynb)
Kaggle GPU ortamı için sıralı ve güvenlik kilitli akışı içerir. Kaggle'da
Internet'i açın, Accelerator olarak NVIDIA GPU seçin ve gerekli
`annotations/*.csv` ile `images/` yapısını içeren SKU-110K veri setini input
olarak bağlayın. Hücreleri sırayla çalıştırın; notebook tek geçerli veri kökünü
bulur, splitleri doğrular, alt kümeyi `copy` modunda hazırlar ve hash doğrulamalı
dry-run yapar.

Eğitim varsayılan olarak `START_TRAINING = False` ile kapalıdır. İlk koşuda
`MODEL = "yolo26s"` bırakılmalı; kontroller incelendikten sonra eğitim bilinçli
olarak açılmalıdır. YOLO11s daha sonra aynı config ve ayrı çıktı klasörüyle
çalıştırılabilir. Configteki seed, split, görüntü boyutu, epoch, batch ve
augmentation değerleri iki model için değiştirilmemelidir.

SKU-110K'nin [orijinal proje sayfası](https://github.com/eg4000/SKU110K_CVPR19)
veri setinin yalnızca akademik ve ticari olmayan amaçlarla kullanılabileceğini
belirtir. Veri setini, hazırlanan alt kümeyi, model ağırlıklarını veya Kaggle'ın
ürettiği büyük ZIP/çıktı dosyalarını GitHub reposuna eklemeyin.
