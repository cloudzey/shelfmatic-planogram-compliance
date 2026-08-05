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