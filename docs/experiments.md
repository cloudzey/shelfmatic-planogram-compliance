# Deney kayıtları

## Korunan YOLO11n baseline ve reddedilen hard-negative ablation

Bu tablo, aynı HITL Supermarket Shelves doğrulama bölümü üzerindeki en iyi
epoch metriklerini gösterir. Kök dizindeki `best.pt`, 960 pikselde fine-tune
edilmiş iyi YOLO11n baseline checkpointidir ve korunmalıdır. Hard-negative
checkpointi yalnızca reddedilmiş bir ablation kaydıdır; baseline veya yeni bir
eğitimin başlangıç ağırlığı değildir.

| Deney | Durum | En iyi epoch | Precision | Recall | mAP50 | mAP50–95 |
|---|---|---:|---:|---:|---:|---:|
| YOLO11n, 960 px fine-tune | Kabul edilen baseline | 18 | 0.6140 | 0.4989 | 0.5241 | 0.2930 |
| YOLO11n + hard-negative v1 | Reddedilen ablation | 3 | 0.5882 | 0.4001 | 0.4242 | 0.2112 |
| Fark (hard-negative − baseline) | Gerileme | — | -0.0258 | -0.0988 | -0.0999 | -0.0818 |

Kaynak kayıtlar:

- `diagnosis_bundle/good_args.yaml` ve `diagnosis_bundle/good_results.csv`
- `diagnosis_bundle/bad_args.yaml` ve `diagnosis_bundle/bad_results.csv`

`diagnosis_bundle/` değişmez tanı snapshotı olarak korunur. Bu nedenle paketteki
script kopyasında kalan tarihsel çıktı-mesajı yazım hatası değiştirilmemiş,
çalıştırılan aktif kopya `src/evaluate_hn_model_1280.py` içinde düzeltilmiştir.

Hard-negative koşusu en iyi değerini epoch 3'te aldı; `patience=4` ile epoch
7'de early stopping uygulandı. Deney aşağıdaki nedenlerle reddedildi:

- Normal train bölümündeki 31 görüntüye boş etiketli 20 küçük kırpım eklendi.
  Böylece 51 train örneğinin yaklaşık %39,2'si background-only oldu.
- Bazı kırpımlarda ürün parçaları bulunmasına rağmen etiketler boştu. Bu durum
  gerçek ürün piksellerini negatif örnek gibi göstererek false-negative label
  noise üretti.
- Küçük kırpımların 960 piksele büyütülmesi, normal raf görüntülerinden farklı
  ölçek ve doku dağılımı yarattı.
- Dört temel doğrulama metriğinin tamamı geriledi; özellikle Recall ve mAP50
  düşüşleri deneyin hedeflenen iyileştirmeyi sağlamadığını gösterdi.

Bu artefaktlar silinmemeli veya yeniden adlandırılmamalıdır. Gelecekteki bir
hard-negative denemesi yapılırsa ürün içeren kırpımlar eksiksiz etiketlenmeli,
negatif oranı kontrollü tutulmalı ve kırpım/ölçek dağılımı gerçek inference
girdilerine yaklaştırılmalıdır.

## YOLO11s ve YOLO26s karşılaştırma protokolü

Bu yeni karşılaştırma SKU-110K'nın deterministik bir alt kümesini kullanır.
HITL veri setindeki yukarıdaki YOLO11n metrikleri farklı bir veri setine ait
olduğu için yeni SKU-110K sonuçlarıyla doğrudan performans karşılaştırması
olarak sunulmamalıdır.

Ortak ayarlar `configs/model_comparison.yaml` dosyasındadır:

| Alan | Ortak değer |
|---|---:|
| Seed | 42 |
| Alt küme | train 1000 / val 100 / test 300 |
| Seçim | `sha256_rank_v1` |
| Görüntü boyutu | 960 |
| Epoch | 50 |
| Train batch | 4 |
| Test batch | 1 |
| Optimizer | AdamW |
| Workers | 0 |
| Çıktı politikası | Model başına ayrı klasör, mevcut klasörün üzerine yazma yok |

`src/prepare_sku110k_subset.py`, SKU-110K'nın resmî
`annotations_train.csv`, `annotations_val.csv` ve `annotations_test.csv`
bölümlerinden her split için seed ve dosya adına bağlı SHA-256 sıralamasıyla
örnek seçer. Üretilen `manifest.json`; seçilen dosyaları, splitleri, kutu
sayılarını, görüntü/etiket hashlerini ve kaynak annotation hashlerini kaydeder.
İki model de aynı `data.yaml` ve aynı manifest ile çalışır.

`src/run_model_comparison.py` şu değerleri model başına `result.json` ve
`result.csv` dosyalarına, yan yana görünümü de `comparison_results.json` ve
`comparison_results.csv` dosyalarına yazar:

- Precision, Recall, mAP50 ve mAP50–95
- fine-tune edilmiş en iyi `.pt` dosyasının bayt/MiB boyutu
- toplam ve trainable parametre sayısı
- Ultralytics test doğrulamasının görüntü başına inference süresi
- seed, görüntü boyutu, batch değerleri, ortam sürümleri ve manifest hashleri

Hafif dry-run dosya adlarını ve splitleri kontrol eder; isteğe bağlı
`--verify-hashes` bütün seçili içerik hashlerini de yeniden hesaplar. Gerçek
eğitim çalıştırmaları image/label hashlerini başlamadan önce zorunlu olarak
doğrular.

### YOLO26s desteği

Repo ortamında kurulu `ultralytics==8.4.115` paketinin asset listesi
`yolo11s.pt` ve `yolo26s.pt` adlarının ikisini de içerir. Paketle gelen YOLO26
mimari configi de `s` ölçeğini tanımlar. Ultralytics'in
[resmî YOLO26 model tablosu](https://docs.ultralytics.com/models/yolo26/)
`yolo26s.pt` dosyasını detection eğitimi ve doğrulaması desteklenen ağırlık
olarak listeler. Configte gerçek adlar kullanılmıştır; ağırlıkların indirilmesi
dry-run sırasında yapılmaz ve ancak gerçek eğitim başlatıldığında gerçekleşir.

Bu protokol kontrollü bir mimari karşılaştırmadır: augmentation ve optimizer
ayarları iki model için aynıdır. Model ailesine özel hiperparametre araması
yapılmadığı için sonuçlar her mimarinin ulaşabileceği mutlak en iyi skor olarak
yorumlanmamalıdır.
