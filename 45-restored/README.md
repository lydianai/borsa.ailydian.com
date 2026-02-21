# 🚀 LYDIAN TRADER - AI-Powered Trading Analysis Platform

[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)]()
[![Version](https://img.shields.io/badge/Version-2.1.0-blue)]()
[![License](https://img.shields.io/badge/License-Educational-yellow)]()
[![White-Hat](https://img.shields.io/badge/Compliance-White--Hat-green)]()

**LYDIAN TRADER** - Yapay zeka destekli kripto para trading analiz platformu. 14 AI modeli, 158 teknik indikatör ve gerçek zamanlı veri akışı ile paper trading eğitim sistemi.

## ⚠️ ÖNEMLİ UYARI

**Bu sistem sadece eğitim amaçlıdır ve PAPER TRADING (simülasyon) modunda çalışır.**

- ❌ Gerçek para ile trading YAPMAZ
- ❌ Exchange hesaplarına erişim SAĞLAMAZ
- ❌ Finansal tavsiye DEĞİLDİR
- ✅ Sadece eğitim ve araştırma için
- ✅ White-Hat uyumlu (read-only, güvenli)

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Sistem Gereksinimleri](#-sistem-gereksinimleri)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Mimari](#-mimari)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Güvenlik](#-güvenlik)
- [Test](#-test)
- [Sorun Giderme](#-sorun-giderme)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## ✨ Özellikler

### 🤖 14 AI Modeli
- **3 LSTM** (Basic, Deep, Bidirectional)
- **5 GRU** (Basic, Deep, Bidirectional, Attention, Residual)
- **3 Transformer** (Basic, Multi-head, Deep)
- **3 Gradient Boosting** (XGBoost, LightGBM, CatBoost)

### 📊 158 Teknik İndikatör (TA-Lib)
- **Trend**: SMA, EMA, DEMA, TEMA, WMA, KAMA, MAMA, T3
- **Momentum**: RSI, STOCH, MACD, ADX, CCI, MFI, ROC
- **Volume**: OBV, AD, ADOSC
- **Volatility**: ATR, NATR, TRANGE, BBANDS
- **Pattern Recognition**: 50+ mum kalıpları

### ⚡ Gerçek Zamanlı Veri
- Binance API entegrasyonu
- WebSocket price streaming (read-only)
- 2 saniyede bir fiyat güncellemesi
- OHLCV candle data

### 🎯 AI Consensus Signals
- Multi-model voting algoritması
- Confidence scoring (%0-100)
- Buy/Sell/Hold önerileri
- Risk seviyesi değerlendirmesi

### 🔒 Paper Trading Bot Engine
- Otomatik trading bots (simülasyon)
- Risk yönetimi (stop-loss, take-profit)
- Position tracking
- Performance analytics

### 🛡️ White-Hat Security
- Gerçek trading engellendi (ENFORCED)
- Risk limitleri (max %10 position, %10 stop-loss)
- Read-only API access
- Güvenlik validasyonları

---

## 🚀 Hızlı Başlangıç

### 5 Dakikada Çalıştır

```bash
# 1. Projeyi klonla veya indir
cd ~/Desktop/borsa

# 2. Dependencies yükle
npm install

# 3. Environment dosyasını oluştur
cat > .env << 'EOF'
NODE_ENV=development
NEXT_PUBLIC_APP_URL=http://localhost:3000
BINANCE_WS_URL=wss://stream.binance.com:9443/ws
BINANCE_API_URL=https://api.binance.com/api/v3
EOF

# 4. Servisleri başlat (4 terminal gerekli)

# Terminal 1 - Frontend
npm run dev

# Terminal 2 - AI Models
cd python-services/ai-models
source venv/bin/activate
python3 app.py

# Terminal 3 - Signal Generator
cd python-services/signal-generator
source venv/bin/activate
python3 app.py

# Terminal 4 - TA-Lib
cd python-services/talib-service
source venv/bin/activate
python3 app.py

# 5. Tarayıcıda aç
open http://localhost:3000
```

### Sistem Sağlık Kontrolü

```bash
# Tüm servislerin durumunu kontrol et
curl http://localhost:3000/api/system/status
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

---

## 💻 Sistem Gereksinimleri

### Zorunlu
- **Node.js**: 18.x veya üzeri
- **npm**: 8.x veya üzeri
- **Python**: 3.10 veya üzeri
- **pip**: 21.x veya üzeri

### Önerilen
- **RAM**: 4GB+ (AI modelleri için)
- **Disk**: 2GB+ boş alan
- **İnternet**: Stabil bağlantı (Binance API için)

### macOS için Ek
```bash
# Homebrew ile TA-Lib kurulumu
brew install ta-lib
```

### Linux için Ek
```bash
# TA-Lib kurulumu
sudo apt-get install ta-lib
```

---

## 📦 Kurulum

### 1. Proje Yapısı

```
borsa/
├── src/                      # Next.js frontend
│   ├── app/                  # App router pages
│   ├── components/           # React components
│   ├── services/             # Business logic
│   └── lib/                  # Utilities
├── python-services/          # Python microservices
│   ├── ai-models/            # 14 AI models
│   ├── signal-generator/     # Signal service
│   └── talib-service/        # TA-Lib indicators
├── public/                   # Static assets
├── .env                      # Environment variables
├── package.json              # Node dependencies
└── README.md                 # Bu dosya
```

### 2. Frontend Kurulumu

```bash
# Dependencies yükle
npm install

# Development build
npm run dev

# Production build
npm run build
npm start
```

### 3. Python Servisleri Kurulumu

Her Python servisi için:

```bash
cd python-services/[servis-adı]

# Virtual environment oluştur
python3 -m venv venv

# Aktive et
source venv/bin/activate

# Dependencies yükle
pip install -r requirements.txt

# Servisi çalıştır
python3 app.py
```

---

## 🎯 Kullanım

### Ana Sayfa (Dashboard)
```
http://localhost:3000
```
- Market overview
- Top 10 cryptocurrencies
- AI signals summary
- Quick stats

### Live Trading
```
http://localhost:3000/live-trading
```
- Gerçek zamanlı BTC/USDT fiyatı
- Order book (bids/asks)
- Trading panel (DEMO - simülasyon)
- Portfolio overview

### AI Testing
```
http://localhost:3000/ai-testing
```
- Coin seçimi (BTC, ETH, BNB, vb.)
- 14 AI model'den tahmin
- Confidence scores
- Buy/Sell/Hold önerileri

### Signals Dashboard
```
http://localhost:3000/signals
```
- AI consensus signals
- Multi-model voting sonuçları
- Risk assessment
- Historical signals

---

## 🏗️ Mimari

```
┌─────────────────────────────────────────┐
│       FRONTEND (Next.js 15.1.6)         │
│         Port 3000                        │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           API GATEWAY LAYER             │
│  /api/ai/python | /api/binance | /api/bot
└─────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────┐    ┌──────────────────┐
│ AI Models    │    │  Signal Gen      │
│ Port 5003    │    │  Port 5004       │
│ 14 Models    │    │  Consensus Algo  │
└──────────────┘    └──────────────────┘
        ▼
┌──────────────┐    ┌──────────────────┐
│ TA-Lib      │    │  Binance API     │
│ Port 5005    │    │  (External)      │
│ 158 Indicators│   │  Market Data     │
└──────────────┘    └──────────────────┘
```

### Servis Portları

| Servis | Port | URL |
|--------|------|-----|
| Frontend | 3000 | http://localhost:3000 |
| AI Models | 5003 | http://localhost:5003 |
| Signal Generator | 5004 | http://localhost:5004 |
| TA-Lib | 5005 | http://localhost:5005 |

---

## 📡 API Dokümantasyonu

### System Status
```bash
GET /api/system/status
```

Response:
```json
{
  "success": true,
  "system": {
    "status": "healthy",
    "healthy": 5,
    "total": 5,
    "uptime": 3600
  },
  "services": [...]
}
```

### Binance Price
```bash
GET /api/binance/price?symbol=BTCUSDT
```

Response:
```json
{
  "success": true,
  "data": {
    "symbol": "BTCUSDT",
    "price": 119169,
    "change24h": 2.24,
    "volume": 18544.0935,
    "high24h": 119456.92,
    "low24h": 116399.5
  }
}
```

### AI Prediction
```bash
POST http://localhost:5003/predict
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "prices": [100, 101, 102, ...],
  "volumes": [1000, 1100, 1200, ...]
}
```

### Signal Generation
```bash
POST http://localhost:5004/signals/generate
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "timeframe": "1h"
}
```

Response:
```json
{
  "success": true,
  "signal": {
    "action": "buy",
    "confidence": 75,
    "models": ["lstm_basic", "gru_deep", ...],
    "current_price": 119169
  }
}
```

### Bot Management
```bash
# List bots
GET /api/bot

# Create bot
POST /api/bot
Content-Type: application/json

{
  "name": "BTC Scalper",
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
  "aiModels": ["lstm_basic"],
  "confidenceThreshold": 0.7
}

# Start/Stop bot engine
PUT /api/bot
Content-Type: application/json

{
  "action": "start"  // or "stop"
}
```

Detaylı API dokümantasyonu için: `SYSTEM-ARCHITECTURE.md`

---

## 🔒 Güvenlik

### White-Hat Uyumluluk

Bu sistem **tamamen white-hat uyumlu** ve eğitim amaçlıdır:

#### ✅ İzin Verilenler
- Public market data okuma (Binance API)
- Paper trading simülasyonu
- AI model eğitimi ve test
- Teknik analiz ve indikatörler
- Read-only WebSocket streams

#### ❌ Yasaklar
- Gerçek para ile trading (ENFORCED)
- Exchange hesap erişimi
- Write operations on exchanges
- API key requirement (public data only)
- Real money operations

### Güvenlik Özellikleri

1. **Paper Trading Enforcement**
```typescript
if (!config.paperTrading) {
  throw new Error('❌ SECURITY: Only paper trading is allowed');
}
```

2. **Risk Management Limits**
- Max position: %10
- Max stop-loss: %10
- Max open positions: 5
- Min confidence: %50

3. **Read-Only API**
- Binance: Public data only
- WebSocket: Price streams only
- No authentication required

4. **Input Validation**
- Bot configuration validation
- Risk parameter checking
- Symbol validation

---

## 🧪 Test

### Manuel Smoke Test

```bash
# 1. Port kontrolü
lsof -ti:3000  # Frontend
lsof -ti:5003  # AI Models
lsof -ti:5004  # Signal Generator
lsof -ti:5005  # TA-Lib

# 2. Health checks
curl http://localhost:5003/health
curl http://localhost:5004/health
curl http://localhost:5005/health

# 3. API tests
curl http://localhost:3000/api/system/status
curl "http://localhost:3000/api/binance/price?symbol=BTCUSDT"
curl http://localhost:3000/api/bot
```

### Otomatik Smoke Test

```bash
# Kapsamlı test script'i çalıştır
chmod +x COMPREHENSIVE-SMOKE-TEST.sh
./COMPREHENSIVE-SMOKE-TEST.sh
```

Beklenen çıktı:
```
🎉 TÜM TESTLER BAŞARILI! Sistem production'a hazır.
Başarı Oranı: 100%
```

### Frontend Tests

```bash
# Type check
npm run type-check

# Lint check
npm run lint

# Build test
npm run build
```

---

## 🔧 Sorun Giderme

### Problem: Port zaten kullanımda

```bash
# Portları temizle
lsof -ti:3000 | xargs kill -9
lsof -ti:5003 | xargs kill -9
lsof -ti:5004 | xargs kill -9
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
pip install TA-Lib

# Linux
sudo apt-get install ta-lib
pip install TA-Lib
```

### Problem: Binance API timeout

1. İnternet bağlantısını kontrol et
2. VPN kullanıyorsan kapat
3. Binance API erişilebilirliğini test et:
```bash
curl https://api.binance.com/api/v3/time
```

### Problem: Frontend build hatası

```bash
# Temizle ve yeniden yükle
rm -rf .next node_modules
npm install
npm run dev
```

### Problem: CoinGecko rate limit (429)

Bu normal bir durumdur (free tier limit). Binance API çalıştığı için kritik değil.

Çözüm (opsiyonel):
- Request caching ekle
- Alternative data source kullan
- CoinGecko Pro hesabı

---

## 📚 Ek Dokümantasyon

- **`SYSTEM-ARCHITECTURE.md`** - Detaylı sistem mimarisi ve API dokümantasyonu
- **`QUICK-START-GUIDE.md`** - 5 dakikada hızlı başlangıç kılavuzu
- **`FINAL-INTEGRATION-TEST-REPORT.md`** - Test raporu ve sonuçları
- **`COMPREHENSIVE-SMOKE-TEST.sh`** - Otomatik test script'i

---

## 🤝 Katkıda Bulunma

Bu proje eğitim amaçlıdır. Katkıda bulunmak isterseniz:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

### Geliştirme Kuralları

- ✅ White-hat uyumlu kalın
- ✅ Paper trading only
- ✅ No real money operations
- ✅ Security first
- ✅ Test coverage maintain edin

---

## 📄 Lisans

Bu proje **eğitim amaçlıdır** ve sadece öğrenme/araştırma için kullanılmalıdır.

**DİKKAT**:
- Finansal tavsiye değildir
- Gerçek para ile trading yapılmamalıdır
- Kullanımdan doğacak kayıplardan sorumluluk kabul edilmez
- White-hat etik kurallara uyulmalıdır

---

## 🙏 Teşekkürler

Bu proje aşağıdaki harika açık kaynak kütüphaneleri kullanmaktadır:

- [Next.js](https://nextjs.org/) - React framework
- [TensorFlow](https://www.tensorflow.org/) - Machine learning
- [TA-Lib](https://ta-lib.org/) - Technical analysis
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting
- [CatBoost](https://catboost.ai/) - Gradient boosting
- [Binance API](https://binance-docs.github.io/) - Market data

---

## 📞 İletişim ve Destek

**Proje**: LYDIAN TRADER (BORSA)
**Versiyon**: 2.1.0
**Durum**: Production Ready ✅
**Platform**: macOS, Linux (Windows WSL)

### Hızlı Linkler

- 🌐 **Frontend**: http://localhost:3000
- 🤖 **AI Models**: http://localhost:5003
- 📡 **Signal Generator**: http://localhost:5004
- 📊 **TA-Lib**: http://localhost:5005
- 📈 **System Status**: http://localhost:3000/api/system/status

---

**⭐ Eğer bu proje işinize yaradıysa, yıldız vermeyi unutmayın!**

**🚀 Happy Trading (Paper Only)!** 📊🤖

---

<div align="center">

Made with ❤️ for Education

**PAPER TRADING ONLY - NO REAL MONEY**

</div>
