# Deney kayıtları

## Tarihsel YOLO11n baseline ve reddedilen hard-negative ablation

Bu tablo, aynı HITL Supermarket Shelves doğrulama bölümü üzerindeki en iyi
epoch metriklerini gösterir. Bu YOLO11n baseline artık final model değildir;
metrikleri ve tanı artefaktları deney geçmişini korumak için saklanır.
Hard-negative checkpointi yalnızca reddedilmiş bir ablation kaydıdır ve yeni
bir eğitimin başlangıç ağırlığı değildir.

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
| Görüntü boyutu | 640 |
| Epoch | 30 |
| Train batch | 8 |
| Test batch | 8 |
| Optimizer | AdamW |
| Workers | 2 |
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

- Precision, Recall, mAP50, mAP75 ve mAP50–95
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

## Final model seçimi

İki model seed 42, aynı 1000/100/300 splitleri ve aynı eğitim ayarlarıyla 30
epoch çalıştırılmıştır. Test sonuçları:

| Model | Precision | Recall | mAP50 | mAP75 | mAP50–95 | Inference (ms/görsel) |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26s | 0.882640 | 0.830656 | 0.903605 | 0.591883 | 0.540087 | 7.546 |
| YOLO11s | **0.889042** | **0.842057** | **0.906829** | **0.602998** | **0.547441** | **7.153** |

YOLO11s; Precision, Recall, mAP50, mAP75 ve mAP50–95 değerlerinin tamamında
YOLO26s'ten daha yüksek sonuç verdiği için seçildi. Kök dizindeki `best.pt`
bu YOLO11s checkpointidir.

## Confidence seçimi ve kilitli test

Confidence eşiği yalnızca 100 görüntülük validation bölümü ve elle onaylanmış
20 hard-negative görüntü kullanılarak seçildi. Validation F1 eğrisinin ham
optimumu `0.33033` ve F1 değeri `0.84425` oldu. Raporlama/deployment değeri
`0.33` olarak kilitlendi:

| Confidence | Val Precision | Val Recall | Val F1 | Hard-negative FP görsel | FP kutu |
|---:|---:|---:|---:|---:|---:|
| 0.33 | 0.858835 | 0.830063 | 0.844204 | 0/20 | 0 |

Eşik seçildikten sonra 300 görüntülük test bölümü yalnızca bir kez final
değerlendirme için kullanıldı; test sonucuna bakılarak eşik değiştirilmedi.

| Metrik | Kilitli test sonucu |
|---|---:|
| Confidence | 0.33 |
| Precision | 0.867728 |
| Recall | 0.859087 |
| F1 | 0.863386 |
| mAP50 | 0.906829 |
| mAP75 | 0.602998 |
| mAP50–95 | 0.547441 |
| Inference | 7.456 ms/görüntü |

Kaynak metin artefaktları `results/final_model/` altındadır. Seçilen modelin
SHA-256 değeri:

```text
f0a0028b0f7e6ce4b597d9b422b4fa6003c5a08aec194942e2a6aedab08d8304  best.pt
```

## Deployment sanity kontrolleri

Kilitli deployment ayarlarıyla lisanslı market rafı örneğinde 149 ürün yüzü
tespit edildi. Ortalama confidence `0.672987`, minimum confidence `0.331646`
ve maksimum confidence `0.860646` oldu. Yazısız, yalnızca kutuları gösteren
çıktı `docs/demo/yolo11s/` altında saklanır.

Model ayrıca eğitim ve eşik seçiminde kullanılmayan, ürün içermeyen bir şehir
sahnesinde out-of-distribution negatif kontrol olarak çalıştırıldı. Model bu
görüntüde `0.458923` ve `0.333169` confidence değerleriyle iki yanlış pozitif
kutu üretti. Bu kontrol, eşik seçiminde kullanılan 20 hard-negative görüntüdeki
`0` yanlış pozitif sonucunu değiştirmez; farklı sahne dağılımlarında genelleme
sınırı bulunduğunu gösterir. Saha kullanımında raf-görüntüsü doğrulaması ve
daha çeşitli background-only örneklerle ek değerlendirme önerilir.
