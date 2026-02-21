# 🤖 AI Futures Trading Bot - Kullanım Kılavuzu

## ⚠️ ÖNEMLİ UYARILAR

### YÜKSEK RİSK
- **Futures trading son derece risklidir**
- **Kaldıraç kullanımı riski katlar**
- **Tüm sermayenizi kaybedebilirsiniz**
- **Sadece kaybetmeyi göze alabileceğiniz para ile işlem yapın**

### SORUMLULUK REDDİ
Bu bot bir yazılım aracıdır. Kar garantisi vermez. Tüm kayıplardan kullanıcı sorumludur.

---

## 📋 İçindekiler

1. [Hızlı Başlangıç](#hızlı-başlangıç)
2. [Binance API Ayarları](#binance-api-ayarları)
3. [Bot Konfigürasyonu](#bot-konfigürasyonu)
4. [Risk Yönetimi](#risk-yönetimi)
5. [Bot Nasıl Çalışır](#bot-nasıl-çalışır)
6. [Sorun Giderme](#sorun-giderme)

---

## 🚀 Hızlı Başlangıç

### 1. Binance Hesabı Hazırlama

1. **Binance'de Futures hesabı açın**
   - Binance.com → Futures → USDⓈ-M Futures
   - KYC doğrulaması gereklidir

2. **API Key oluşturun**
   - Binance → Profil → API Management
   - "Create API" butonuna tıklayın
   - İki faktörlü doğrulama gereklidir

3. **API Yetkilerini Ayarlayın**
   - ✅ Enable Reading (Okuma yetkisi)
   - ✅ Enable Futures (Futures yetkisi)
   - ❌ Enable Withdrawals (VERMEY İN!)
   - ✅ IP Access Restriction (Mutlaka ekleyin)

### 2. Botu Başlatma

```bash
# 1. Sistemi başlat
cd ~/Desktop/borsa
npm run dev

# 2. Python servislerini başlat (ayrı terminallerde)
cd python-services/ai-models && source venv/bin/activate && python3 app.py
cd python-services/signal-generator && source venv/bin/activate && python3 app.py
cd python-services/talib-service && source venv/bin/activate && python3 app.py
```

### 3. Web Arayüzü

```
http://localhost:3000/futures-bot
```

---

## 🔐 Binance API Ayarları

### API Key ve Secret Girme

1. Futures Bot sayfasını açın
2. API Key ve API Secret'i girin
3. "Yapılandırmayı Tamamla" butonuna tıklayın
4. Güvenlik uyarısını okuyun ve onaylayın

### Güvenlik Kontrol Listesi

- [ ] Withdrawal yetkisi YOK
- [ ] IP kısıtlaması VAR
- [ ] API key yalnızca Futures yetkili
- [ ] API secret güvenli bir yerde saklanıyor
- [ ] İki faktörlü doğrulama aktif

---

## ⚙️ Bot Konfigürasyonu

### Temel Ayarlar

| Parametre | Açıklama | Önerilen | Minimum | Maksimum |
|-----------|----------|----------|---------|----------|
| **Symbol** | Trading çifti | BTCUSDT | - | - |
| **Leverage** | Kaldıraç oranı | 5x | 1x | 20x |
| **Max Position Size** | Pozisyon başına max USDT | 100 | 10 | 1000 |
| **Stop Loss %** | Zarar durdur oranı | 2% | 1% | 10% |
| **Take Profit %** | Kar al oranı | 5% | 1% | 20% |
| **Min Confidence** | Minimum AI güven eşiği | 70% | 60% | 100% |
| **Max Positions** | Aynı anda max açık pozisyon | 2 | 1 | 3 |

### Güvenli Başlangıç İçin Önerilen Ayarlar

```
Symbol: BTCUSDT
Leverage: 3x
Max Position Size: 50 USDT
Stop Loss: 2%
Take Profit: 4%
Min Confidence: 75%
Max Positions: 1
```

---

## 🛡️ Risk Yönetimi

### Otomatik Güvenlik Önlemleri

Bot aşağıdaki güvenlik kontrollerini otomatik yapar:

1. **Kaldıraç Limiti**: Max 20x (önerilen 5-10x)
2. **Pozisyon Limiti**: Max 1000 USDT
3. **Stop-Loss Zorunluluğu**: %1-%10 arası
4. **Take-Profit Zorunluluğu**: %1-%20 arası
5. **Güven Eşiği**: Minimum %60
6. **Max Pozisyon Sayısı**: Maximum 3

### Manuel Güvenlik Önlemleri

1. **Küçük Başlayın**
   - İlk 1 hafta max 50 USDT ile test edin
   - Stratejinin çalışmasını izleyin
   - Kademeli olarak artırın

2. **Bakiye Yönetimi**
   - Asla tüm bakiyenizi riske atmayın
   - Futures hesabınızda max %10'unu kullanın
   - Günlük zarar limitiniz olsun

3. **Sürekli İzleme**
   - Botu çalıştırırken bilgisayarınız açık olmalı
   - Pozisyonları düzenli kontrol edin
   - Ani piyasa hareketlerinde müdahale edin

### Acil Durum Prosedürü

```
1. ⏹️ "Botu Durdur" butonuna tıklayın
2. 🔴 Tüm açık pozisyonları manuel kapatın
3. ❌ API yetkilerini iptal edin (gerekirse)
4. 📧 Binance destek ile iletişime geçin
```

---

## 🧠 Bot Nasıl Çalışır

### İş Akışı

```
1. AI Tahmin
   ↓
2. Sinyal Üretimi (BUY/SELL/HOLD)
   ↓
3. Güven Eşiği Kontrolü (>70%)
   ↓
4. Risk Kontrolü (bakiye, pozisyon sayısı)
   ↓
5. Emir Gönderimi (Market Order)
   ↓
6. Stop-Loss & Take-Profit Ayarlama
   ↓
7. Pozisyon İzleme (10 saniyede bir)
```

### AI Karar Verme

Bot üç farklı kaynaktan sinyal alır:

1. **AI Models (Port 5003)**
   - 14 farklı ML modeli
   - LSTM, GRU, Transformer, Gradient Boosting
   - Fiyat tahmini yapar

2. **TA-Lib Indicators (Port 5005)**
   - 158 teknik indikatör
   - RSI, MACD, Bollinger Bands, EMA
   - Teknik analiz sinyalleri

3. **Signal Generator (Port 5004)**
   - AI + TA-Lib kombinasyonu
   - Ensemble sinyal üretimi

### Sinyal Örnekleri

**BUY Sinyali**
```
Symbol: BTCUSDT
Action: BUY (LONG)
Confidence: 78%
Reason: AI predicts +2.5% upward movement, RSI oversold
```

**SELL Sinyali**
```
Symbol: BTCUSDT
Action: SELL (SHORT)
Confidence: 82%
Reason: AI predicts -1.8% downward movement, RSI overbought
```

**HOLD Sinyali**
```
Symbol: BTCUSDT
Action: HOLD
Confidence: 55%
Reason: Low confidence, waiting for clearer signal
```

---

## 🎯 Örnek Kullanım Senaryoları

### Senaryo 1: Muhafazakar Trader

```
Hedef: Düşük riskle istikrarlı kazanç

Ayarlar:
- Leverage: 3x
- Position Size: 30 USDT
- Stop Loss: 1.5%
- Take Profit: 3%
- Confidence: 80%
- Max Positions: 1

Beklenen Sonuç:
- Günde 2-3 işlem
- %1-2 günlük getiri hedefi
- Düşük risk
```

### Senaryo 2: Agresif Trader

```
Hedef: Yüksek risk ile yüksek kazanç

Ayarlar:
- Leverage: 10x
- Position Size: 200 USDT
- Stop Loss: 3%
- Take Profit: 10%
- Confidence: 65%
- Max Positions: 3

Beklenen Sonuç:
- Günde 5-10 işlem
- %5-10 günlük getiri hedefi
- Yüksek risk (kayıp riski de yüksek)
```

### Senaryo 3: Test Modu

```
Hedef: Stratejiyi test etme

Ayarlar:
- Leverage: 2x
- Position Size: 10 USDT
- Stop Loss: 2%
- Take Profit: 4%
- Confidence: 75%
- Max Positions: 1

Beklenen Sonuç:
- Minimum risk
- Strateji validasyonu
- İstatistik toplama
```

---

## 📊 Performans Takibi

### Dashboard Metrikleri

1. **Bot Durumu**: Çalışıyor/Durduruldu
2. **Açık Pozisyonlar**: Aktif pozisyon sayısı
3. **Toplam P&L**: Kümülatif kar/zarar
4. **Win Rate**: Başarı oranı (%)
5. **Total Trades**: Toplam işlem sayısı

### Log Analizi

Bot her işlemi aşağıdaki formatta loglar:

```
[2025-10-02 14:30:15] 📡 Sinyal: BUY (Güven: 78.5%)
[2025-10-02 14:30:18] 🚀 YENİ POZİSYON AÇILIYOR
[2025-10-02 14:30:18] Yön: BUY LONG
[2025-10-02 14:30:18] Fiyat: 119250.00 USDT
[2025-10-02 14:30:18] Miktar: 0.004
[2025-10-02 14:30:20] ✅ Pozisyon açıldı - Order ID: 123456789
[2025-10-02 14:30:21] ✅ Stop-loss: 117307.50 USDT
[2025-10-02 14:30:22] ✅ Take-profit: 125212.50 USDT
```

---

## ⚠️ Sorun Giderme

### Bot Başlamıyor

**Problem**: "API bağlantısı başarısız" hatası

**Çözüm**:
1. API Key ve Secret'i kontrol edin
2. Binance API yetkilerini kontrol edin
3. IP kısıtlamasını kontrol edin
4. Binance'in bakımda olmadığını kontrol edin

---

### Yetersiz Bakiye

**Problem**: "Yetersiz bakiye" hatası

**Çözüm**:
1. Binance Futures hesabınıza USDT transfer edin
2. Max Position Size'ı düşürün
3. Kaldıracı azaltın

---

### Düşük Güven Sinyalleri

**Problem**: Bot hiç işlem yapmıyor

**Çözüm**:
1. Confidence Threshold'u %60'a düşürün
2. Farklı bir coin deneyin (ETH, BNB)
3. Piyasa volatilitesini kontrol edin

---

### API Rate Limit

**Problem**: "429 Too Many Requests" hatası

**Çözüm**:
1. Botu durdurup 1 dakika bekleyin
2. Bot döngü süresini 10 saniyeden 30 saniyeye çıkarın
3. Binance API limitlerini kontrol edin

---

## 🔧 Gelişmiş Konfigürasyon

### Trailing Stop Ekleme

```typescript
config.trailingStopPercent = 1.5; // %1.5 trailing stop
```

### Farklı Timeframe Kullanma

```typescript
// AI signal endpoint'inde
body: JSON.stringify({
  symbol,
  timeframe: '15m', // 1m, 5m, 15m, 1h, 4h, 1d
})
```

### Multiple Symbol Trading

```typescript
// Birden fazla bot instance oluşturun
const btcBot = new FuturesTradingBot(apiKey, apiSecret, btcConfig);
const ethBot = new FuturesTradingBot(apiKey, apiSecret, ethConfig);
```

---

## 📈 Başarı İçin İpuçları

### 1. Sabırlı Olun
- İlk hafta kar beklemeyin
- Stratejiyi test edin
- İstatistik toplayın

### 2. Overtrading Yapmayın
- Günde max 10-15 işlem
- Her sinyale girmeyin
- Yüksek güven eşiği kullanın

### 3. Piyasayı Anlayın
- Trend yönünde işlem yapın
- Önemli haberlere dikkat edin
- Volatiliteyi göz önünde bulundurun

### 4. Risk Yönetimi
- Asla %100 bakiye kullanmayın
- Günlük zarar limitiniz olsun
- Kazancın bir kısmını çekin

### 5. Sürekli İyileştirme
- İşlem loglarını analiz edin
- Başarılı stratejileri not edin
- Ayarları optimize edin

---

## 📞 Destek ve İletişim

### Binance Destek
- https://www.binance.com/en/support
- 7/24 canlı destek

### Bot Sorunları
- GitHub Issues açın
- Log dosyalarını paylaşın
- Konfigürasyonunuzu belirtin

---

## 📄 Lisans ve Yasal

Bu yazılım eğitim amaçlıdır. Kullanımdan doğacak her türlü zararda kullanıcı sorumludur.

**Yasal Uyarı**: Türkiye'de kripto para futures trading düzenlemeye tabidir. Yerel yasalarınızı kontrol edin.

---

## ✅ Başlamadan Önce Kontrol Listesi

- [ ] Binance hesabı KYC onaylı
- [ ] Futures hesabı açıldı
- [ ] API Key oluşturuldu ve yetkiler ayarlandı
- [ ] IP kısıtlaması eklendi
- [ ] Withdrawal yetkisi YOK
- [ ] Futures hesabında yeterli bakiye var
- [ ] Risk yönetimi kurallarını anladım
- [ ] Kaybetmeyi göze alabileceğim para ile başlıyorum
- [ ] Bot ayarlarını test modunda denedim
- [ ] Acil durum prosedürünü biliyorum

---

## 🚀 HADİ BAŞLAYALIM!

```bash
# 1. Sistemi başlat
npm run dev

# 2. Tarayıcıda aç
http://localhost:3000/futures-bot

# 3. API Key'leri gir

# 4. Ayarları yapılandır

# 5. BOTU BAŞLAT!
```

**Başarılar! 🎯**
