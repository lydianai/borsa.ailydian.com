# 🚀 LYDIAN TRADER - Hızlı Başlangıç Kılavuzu

## ⚡ 5 Dakikada Sistem Başlatma

### Ön Koşullar Kontrol Listesi

- [ ] Node.js 18+ kurulu mu? → `node --version`
- [ ] Python 3.10+ kurulu mu? → `python3 --version`
- [ ] npm kurulu mu? → `npm --version`
- [ ] Homebrew kurulu mu? (macOS) → `brew --version`

---

## 📦 Hızlı Kurulum (İlk Kez)

### 1. Projeyi Aç

```bash
cd ~/Desktop/borsa
```

### 2. Environment Dosyasını Kontrol Et

`.env` dosyası mevcut olmalı. Yoksa oluştur:

```bash
cat > .env << 'EOF'
NODE_ENV=development
NEXT_PUBLIC_APP_URL=http://localhost:3000
BINANCE_WS_URL=wss://stream.binance.com:9443/ws
BINANCE_API_URL=https://api.binance.com/api/v3
EOF
```

### 3. Frontend Dependencies Yükle

```bash
npm install
```

### 4. Python Virtual Environments Kontrol Et

```bash
# AI Models
ls python-services/ai-models/venv

# Signal Generator
ls python-services/signal-generator/venv

# TA-Lib
ls python-services/talib-service/venv
```

Eğer `venv` klasörleri yoksa, her biri için:

```bash
cd python-services/ai-models
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ../..
```

---

## 🎬 Sistemi Başlatma (Her Seferinde)

### Terminal 1: Frontend (Next.js)

```bash
cd ~/Desktop/borsa
npm run dev
```

✅ Başarı mesajı: `Ready started server on 0.0.0.0:3000`
🌐 URL: http://localhost:3000

### Terminal 2: AI Models Service

```bash
cd ~/Desktop/borsa/python-services/ai-models
source venv/bin/activate
python3 app.py
```

✅ Başarı mesajı: `Running on http://0.0.0.0:5003`
🤖 14 AI modeli yüklendi

### Terminal 3: Signal Generator Service

```bash
cd ~/Desktop/borsa/python-services/signal-generator
source venv/bin/activate
python3 app.py
```

✅ Başarı mesajı: `Running on http://0.0.0.0:5004`
📡 Sinyal motoru hazır

### Terminal 4: TA-Lib Service

```bash
cd ~/Desktop/borsa/python-services/talib-service
source venv/bin/activate
python3 app.py
```

✅ Başarı mesajı: `Running on http://0.0.0.0:5005`
📊 158 teknik indikatör yüklendi

---

## ✅ Sistem Kontrolü (Health Check)

### Otomatik Kontrol

Tarayıcıda aç:
```
http://localhost:3000/api/system/status
```

Beklenen sonuç:
```json
{
  "success": true,
  "system": {
    "status": "healthy",
    "healthy": 5,
    "total": 5
  }
}
```

### Manuel Servis Kontrolleri

```bash
# Frontend
curl http://localhost:3000

# AI Models
curl http://localhost:5003/health

# Signal Generator
curl http://localhost:5004/health

# TA-Lib
curl http://localhost:5005/health

# Binance API
curl "http://localhost:3000/api/binance/price?symbol=BTCUSDT"
```

Her biri `200 OK` dönmeli.

---

## 🎯 İlk Test: AI Analizi

### 1. Frontend'e Git

Tarayıcıda aç: http://localhost:3000

### 2. AI Testing Sayfasına Git

http://localhost:3000/ai-testing

### 3. Bitcoin Analizi Yap

- Coin listesinden **Bitcoin (BTC)** seç
- **"Analiz Et"** butonuna tıkla
- 14 model'den tahminler gelecek (5-10 saniye)

Beklenen çıktı:
```
Model: LSTM Basic → Tahmin: $120,500 (↗ Buy, %72 güven)
Model: GRU Deep → Tahmin: $119,800 (↗ Buy, %68 güven)
Model: Transformer → Tahmin: $121,200 (↗ Buy, %75 güven)
...
```

---

## 📈 Gerçek Zamanlı Fiyat Testi

### 1. Live Trading Sayfasına Git

http://localhost:3000/live-trading

### 2. BTC/USDT Seç

Varsayılan olarak seçili olmalı.

### 3. Fiyat Güncellemelerini İzle

Her 2 saniyede bir gerçek Binance fiyatı güncellenecek:

```
BTC/USDT
$119,076.46
+2.35%
```

Fiyatlar **gerçek zamanlı** Binance'ten gelir.

---

## 🤖 Trading Bot Testi

### 1. Bot Oluştur

```bash
curl -X POST http://localhost:3000/api/bot \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Bot",
    "symbol": "BTC/USDT",
    "strategy": "ai_consensus",
    "enabled": false,
    "riskManagement": {
      "maxPositionSize": 5,
      "stopLoss": 2,
      "takeProfit": 5,
      "maxDailyLoss": 10,
      "maxOpenPositions": 3
    },
    "aiModels": ["lstm_basic", "gru_deep"],
    "confidenceThreshold": 0.7
  }'
```

Beklenen sonuç:
```json
{
  "success": true,
  "bot": {
    "id": "bot_...",
    "name": "Test Bot",
    "paperTrading": true
  },
  "message": "Bot created successfully (PAPER TRADING MODE)"
}
```

### 2. Bot Listesini Kontrol Et

```bash
curl http://localhost:3000/api/bot
```

Oluşturduğun bot listede görünmeli.

### 3. Bot Engine Başlat

```bash
curl -X PUT http://localhost:3000/api/bot \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'
```

Bot motoru 60 saniyelik döngülerde çalışmaya başlar.

---

## 🛑 Sistemi Durdurma

Her terminal'de `Ctrl+C` ile servisleri durdur:

1. Terminal 1: Frontend durdur
2. Terminal 2: AI Models durdur
3. Terminal 3: Signal Generator durdur
4. Terminal 4: TA-Lib durdur

---

## 🔥 Hızlı Sorun Giderme

### Problem: Port zaten kullanımda

```bash
# Port 3000
lsof -ti:3000 | xargs kill -9

# Port 5003
lsof -ti:5003 | xargs kill -9

# Port 5004
lsof -ti:5004 | xargs kill -9

# Port 5005
lsof -ti:5005 | xargs kill -9
```

### Problem: Python modülü bulunamadı

```bash
cd python-services/[servis-adı]
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: TA-Lib yüklenemiyor

```bash
# macOS
brew install ta-lib

# Linux
sudo apt-get install ta-lib

# Sonra Python paketi
pip install TA-Lib
```

### Problem: Frontend build hatası

```bash
rm -rf .next node_modules
npm install
npm run dev
```

### Problem: Binance API timeout

- İnternet bağlantını kontrol et
- VPN kullanıyorsan kapat
- Binance API erişilebilir mi test et:
```bash
curl https://api.binance.com/api/v3/time
```

---

## 📊 Servis Port Referansı

| Servis | Port | URL |
|--------|------|-----|
| Frontend (Next.js) | 3000 | http://localhost:3000 |
| AI Models | 5003 | http://localhost:5003 |
| Signal Generator | 5004 | http://localhost:5004 |
| TA-Lib | 5005 | http://localhost:5005 |

---

## 🎓 Kullanıcı Arayüzü Turu

### Ana Sayfa (Dashboard)
- Market genel bakış
- Top 10 coinler
- AI sinyalleri özeti

### Live Trading
- Gerçek zamanlı fiyatlar (Binance)
- Order book (alış/satış emirleri)
- Trading panel (DEMO - gerçek işlem yapmaz)

### AI Testing
- 14 AI model'den tahmin
- Coin seçimi
- Analiz sonuçları ve grafikler

### Signals
- AI consensus sinyalleri
- Buy/Sell/Hold önerileri
- Güven skorları (%0-100)

---

## ⚠️ ÖNEMLİ HATIRLATMALAR

1. **Paper Trading Only**: Gerçek para ile işlem yapılmaz
2. **Educational Purpose**: Sadece eğitim amaçlıdır
3. **No Real API Keys**: Gerçek exchange API key'leri gerekmez
4. **White-Hat Compliant**: Tüm işlemler read-only ve güvenli
5. **Risk Disclaimer**: Finansal tavsiye değildir

---

## 📚 Daha Fazla Bilgi

Detaylı dokümantasyon için:
- `SYSTEM-ARCHITECTURE.md` - Tam sistem mimarisi
- `API-DOCUMENTATION.md` - API detayları (yakında)
- `DEVELOPMENT-GUIDE.md` - Geliştirici kılavuzu (yakında)

---

**✅ Hazırsın! Sistemin çalışır durumda. İyi analizler!** 🚀
