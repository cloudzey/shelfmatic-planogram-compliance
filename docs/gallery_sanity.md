# 20 görsellik demo galerisi sanity kontrolü

Streamlit galerisindeki 20 lisanslı Wikimedia Commons raf görseli, final
`best.pt` checkpointi ve kilitli deployment ayarlarıyla (`imgsz=640`,
`confidence=0.33`, `iou=0.70`, `max_det=1000`) tek tek çalıştırıldı.

Bu görüntülerin ground-truth kutuları yoktur. Aşağıdaki değerler bir doğruluk
metriği değil; galerideki bütün dosyaların açıldığını, modelin inference
üretebildiğini ve sonuçların nitel olarak incelendiğini gösteren deployment
sanity kaydıdır. Galeri eğitim verisinden bağımsızdır.

| # | Örnek | Tespit | Ortalama confidence | En yüksek confidence |
|---:|---|---:|---:|---:|
| 1 | İçecek rafı — önden görünüm | 49 | %79,0 | %87,1 |
| 2 | Sos ve konserve rafı | 260 | %63,5 | %84,0 |
| 3 | Enerji içeceği rafı | 202 | %64,4 | %81,4 |
| 4 | Şekerleme standı | 69 | %67,2 | %85,0 |
| 5 | Bakliyat ve makarna rafı | 22 | %57,4 | %85,3 |
| 6 | Paketli kek ve atıştırmalık rafı | 51 | %64,3 | %84,3 |
| 7 | Bebek bakım ürünleri rafı | 32 | %71,4 | %89,9 |
| 8 | Peynir ve şarküteri rafı | 66 | %61,1 | %80,1 |
| 9 | Gazlı içecek rafı — açılı görünüm | 61 | %63,6 | %82,7 |
| 10 | Kutu ve poşetli kiler ürünleri | 20 | %46,7 | %77,2 |
| 11 | Bal ve şurup rafı | 26 | %71,9 | %86,9 |
| 12 | Bebek mendili rafı | 24 | %70,7 | %83,1 |
| 13 | Kraker kutuları rafı | 44 | %66,5 | %83,4 |
| 14 | Su bidonları rafı | 21 | %80,0 | %86,0 |
| 15 | Şarap şişeleri rafı | 36 | %48,5 | %82,9 |
| 16 | Bira teşhir rafı | 49 | %70,2 | %88,6 |
| 17 | Yumurta rafı | 220 | %48,2 | %79,8 |
| 18 | Evcil hayvan ürünleri rafı | 139 | %58,2 | %82,9 |
| 19 | Mısır gevreği ve müsli rafı | 88 | %67,4 | %89,2 |
| 20 | Ketçap, hardal ve sos rafı | 162 | %63,4 | %84,8 |

Özet:

- 20/20 görselde deployment eşiğini geçen tespit üretildi.
- Toplam 1.641 ürün kutusu çizildi.
- Görsel başına tespit aralığı 20–260 oldu.
- Bütün çıktılar yazısız kutu görünümünde nitel olarak kontrol edildi.

Bu sonuçlar modelin farklı ürün kategorilerinde çalışabildiğini gösterir;
ancak gerçek saha başarısı için mağazadan çekilmiş, ground-truth etiketli yeni
bir değerlendirme seti hâlâ gereklidir.
