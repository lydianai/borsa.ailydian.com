# LYDIAN TRADER - Sistem Mimarisi ve Dokümantasyon

## 📋 Genel Bakış

LYDIAN TRADER, Next.js 15.1.6 ve Python mikroservisler kullanarak gerçek zamanlı kripto para trading analizi yapan, WHITE-HAT uyumlu bir eğitim platformudur.

**⚠️ UYARI**: Bu sistem sadece PAPER TRADING (kağıt üzerinde işlem) için tasarlanmıştır. Gerçek para ile işlem yapmaz.

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND LAYER (Port 3000)                   │
│                      Next.js 15.1.6 + TypeScript                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Dashboard│  │Live Trade│  │ AI Test  │  │  Signals │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                           │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │ /api/ai/python │  │ /api/binance/* │  │ /api/bot/*       │  │
│  │ (Proxy)        │  │ (Market Data)  │  │ (Bot Control)    │  │
│  └────────────────┘  └────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PYTHON MICROSERVICES LAYER                     │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  AI Models Service (Port 5003)                          │   │
│  │  • 3 LSTM Models (basic, deep, bidirectional)           │   │
│  │  • 5 GRU Models (various configurations)                │   │
│  │  • 3 Transformer Models (attention mechanisms)          │   │
│  │  • 3 Gradient Boosting (XGBoost, LightGBM, CatBoost)    │   │
│  │  TOTAL: 14 AI Models                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Signal Generator Service (Port 5004)                   │   │
│  │  • AI Consensus Algorithms                              │   │
│  │  • Multi-model Signal Aggregation                       │   │
│  │  • Confidence Scoring (0-100%)                          │   │
│  │  • Buy/Sell/Hold Recommendations                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TA-Lib Service (Port 5005)                             │   │
│  │  • 158 Technical Indicators                             │   │
│  │  • RSI, MACD, Bollinger Bands, SMA, EMA, etc.           │   │
│  │  • Real-time Indicator Calculations                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EXTERNAL DATA SOURCES                       │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │ Binance API    │  │ CoinGecko API  │  │ Binance WS       │  │
│  │ (REST)         │  │ (Market Data)  │  │ (Real-time)      │  │
│  └────────────────┘  └────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Servis Detayları

### 1. Frontend (Next.js - Port 3000)

**Teknolojiler**:
- Next.js 15.1.6 (App Router)
- TypeScript 5.x
- Tailwind CSS 3.4.1
- Recharts (Grafikler)
- Lucide React (İkonlar)

**Önemli Sayfalar**:
- `/` - Ana dashboard
- `/live-trading` - Gerçek zamanlı trading paneli
- `/ai-testing` - AI model test arayüzü
- `/signals` - AI sinyalleri dashboard

### 2. AI Models Service (Python - Port 5003)

**Framework**: Flask 3.0.0
**AI/ML Libraries**:
- TensorFlow 2.15.0
- scikit-learn 1.3.2
- XGBoost 2.0.3
- LightGBM 4.1.0
- CatBoost 1.2.2

**14 Model Detayı**:

| Model Tipi | Varyant | Açıklama |
|------------|---------|----------|
| LSTM | Basic | Basit LSTM, 50 unit |
| LSTM | Deep | 2 katmanlı LSTM, 100+50 unit |
| LSTM | Bidirectional | İki yönlü LSTM, 75 unit |
| GRU | Basic | Basit GRU, 50 unit |
| GRU | Deep | 2 katmanlı GRU, 100+50 unit |
| GRU | Bidirectional | İki yönlü GRU, 75 unit |
| GRU | Attention | GRU + Attention mekanizması |
| GRU | Residual | GRU + Residual connections |
| Transformer | Basic | Self-attention, 4 head |
| Transformer | Multi-head | 8 attention head |
| Transformer | Deep | 3 katmanlı transformer |
| XGBoost | Ensemble | Gradient boosting ensemble |
| LightGBM | Fast | Hızlı gradient boosting |
| CatBoost | Robust | Kategorik veri optimized |

**Endpoints**:
- `GET /health` - Servis sağlık kontrolü
- `POST /predict` - Tahmin isteği
- `GET /models` - Model listesi

### 3. Signal Generator Service (Python - Port 5004)

**Framework**: Flask 3.0.0
**Özellikler**:
- Multi-model consensus (14 model'den sinyal toplama)
- Weighted voting algorithm
- Confidence scoring (0-100%)
- Risk level assessment

**Sinyal Tipleri**:
- `buy` - Alış sinyali (>70% consensus)
- `sell` - Satış sinyali (>70% consensus)
- `hold` - Bekle sinyali (<70% consensus)

**Endpoints**:
- `GET /health` - Servis sağlık kontrolü
- `POST /signals/generate` - Sinyal üretimi
- `GET /signals/batch` - Toplu sinyal (birden fazla coin)

### 4. TA-Lib Service (Python - Port 5005)

**Framework**: Flask 3.0.0
**Library**: TA-Lib 0.4.28

**158 İndikatör Kategorileri**:
- **Trend**: SMA, EMA, DEMA, TEMA, WMA, KAMA, MAMA, T3
- **Momentum**: RSI, STOCH, STOCHF, MACD, ADX, CCI, MFI, ROC
- **Volume**: OBV, AD, ADOSC
- **Volatility**: ATR, NATR, TRANGE, BBANDS
- **Price Transform**: AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE
- **Cycle**: HT_DCPERIOD, HT_DCPHASE, HT_TRENDMODE
- **Pattern**: CDL patterns (50+ mum kalıpları)

**Endpoints**:
- `GET /health` - Servis sağlık kontrolü
- `POST /indicators` - İndikatör hesaplama
- `GET /indicators/list` - Mevcut indikatörler

## 🔒 Güvenlik ve WHITE-HAT Uyumluluk

### Güvenlik Önlemleri

1. **Paper Trading Enforcement**:
```typescript
private validateBotConfig(config: BotConfig): void {
  if (!config.paperTrading) {
    throw new Error('❌ SECURITY: Only paper trading is allowed');
  }
}
```

2. **Risk Yönetimi Sınırları**:
- Maksimum pozisyon boyutu: %10
- Stop-loss limiti: %10
- Maksimum açık pozisyon: 5
- Minimum güven eşiği: %50

3. **Read-Only API Access**:
- Binance API: Sadece public market data
- WebSocket: Sadece fiyat stream'i (read-only)
- Hiçbir write/trade yetkisi yok

4. **Environment Variables**:
```bash
NODE_ENV=development
NEXT_PUBLIC_APP_URL=http://localhost:3000
BINANCE_WS_URL=wss://stream.binance.com:9443/ws
BINANCE_API_URL=https://api.binance.com/api/v3
```

## 📡 API Dokümantasyonu

### 1. Python Services Proxy

**Endpoint**: `GET /api/ai/python`

**Query Parameters**:
- `service`: `models` | `signals` | `talib`
- `endpoint`: İstenen Python servis endpoint'i

**Örnek**:
```bash
curl "http://localhost:3000/api/ai/python?service=models&endpoint=/health"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "models": 14,
    "timestamp": 1696234567890
  }
}
```

### 2. System Status

**Endpoint**: `GET /api/system/status`

**Response**:
```json
{
  "success": true,
  "system": {
    "status": "healthy",
    "healthy": 5,
    "total": 5,
    "uptime": 3600,
    "timestamp": 1696234567890
  },
  "services": [
    {
      "name": "AI Models (Python)",
      "url": "http://localhost:5003/health",
      "status": "healthy",
      "responseTime": 45,
      "details": { "models": 14 }
    },
    {
      "name": "Signal Generator (Python)",
      "url": "http://localhost:5004/health",
      "status": "healthy",
      "responseTime": 38,
      "details": { "version": "1.0.0" }
    },
    {
      "name": "TA-Lib Service (Python)",
      "url": "http://localhost:5005/health",
      "status": "healthy",
      "responseTime": 42,
      "details": { "indicators": 158 }
    },
    {
      "name": "Binance API",
      "url": "http://localhost:3000/api/binance/price?symbol=BTCUSDT",
      "status": "healthy",
      "responseTime": 120,
      "details": { "price": 119076.46, "symbol": "BTCUSDT" }
    },
    {
      "name": "Market Data API",
      "url": "http://localhost:3000/api/market/crypto",
      "status": "healthy",
      "responseTime": 250,
      "details": { "coins": 100 }
    }
  ]
}
```

### 3. Trading Bot Management

**Endpoint**: `GET /api/bot`

**Response**:
```json
{
  "success": true,
  "bots": [],
  "positions": [],
  "summary": {
    "totalBots": 0,
    "activeBots": 0,
    "openPositions": 0,
    "totalPositions": 0
  }
}
```

**Endpoint**: `POST /api/bot`

**Request Body**:
```json
{
  "name": "BTC Scalper",
  "symbol": "BTC/USDT",
  "strategy": "ai_consensus",
  "enabled": true,
  "riskManagement": {
    "maxPositionSize": 5,
    "stopLoss": 2,
    "takeProfit": 5,
    "maxDailyLoss": 10,
    "maxOpenPositions": 3
  },
  "aiModels": ["lstm_basic", "gru_deep", "transformer_basic"],
  "confidenceThreshold": 0.7
}
```

**Response**:
```json
{
  "success": true,
  "bot": {
    "id": "bot_1696234567890_abc123",
    "name": "BTC Scalper",
    "symbol": "BTC/USDT",
    "strategy": "ai_consensus",
    "enabled": true,
    "paperTrading": true,
    "riskManagement": { ... },
    "aiModels": [...],
    "confidenceThreshold": 0.7
  },
  "message": "Bot created successfully (PAPER TRADING MODE)"
}
```

**Endpoint**: `PUT /api/bot`

**Request Body**:
```json
{
  "action": "start"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Bot engine started (PAPER TRADING)"
}
```

### 4. Binance Price Data

**Endpoint**: `GET /api/binance/price`

**Query Parameters**:
- `symbol`: Trading pair (örn: BTCUSDT)

**Örnek**:
```bash
curl "http://localhost:3000/api/binance/price?symbol=BTCUSDT"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "symbol": "BTCUSDT",
    "price": 119076.46,
    "change24h": 2.35,
    "volume": 1234567890,
    "high24h": 120500,
    "low24h": 117000
  }
}
```

### 5. WebSocket Management

**Endpoint**: `POST /api/websocket/binance`

**Request Body**:
```json
{
  "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
}
```

**Response**:
```json
{
  "success": true,
  "message": "WebSocket connection initiated",
  "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
}
```

**Endpoint**: `GET /api/websocket/binance`

**Response**:
```json
{
  "success": true,
  "connected": true,
  "subscribedSymbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
  "lastUpdate": 1696234567890
}
```

## 🔧 Kurulum ve Çalıştırma

### Gereksinimler

- Node.js 18+ ve npm/yarn
- Python 3.10+
- Homebrew (macOS için TA-Lib kurulumu)

### Adım 1: Ortam Değişkenlerini Ayarla

`.env` dosyası oluştur:
```bash
NODE_ENV=development
NEXT_PUBLIC_APP_URL=http://localhost:3000
BINANCE_WS_URL=wss://stream.binance.com:9443/ws
BINANCE_API_URL=https://api.binance.com/api/v3
```

### Adım 2: Frontend Kurulumu

```bash
# Dependencies yükle
npm install

# Development server başlat
npm run dev
```

Frontend: http://localhost:3000

### Adım 3: Python Servisleri Başlat

**Terminal 1 - AI Models Service**:
```bash
cd python-services/ai-models
source venv/bin/activate
python app.py
```
Service: http://localhost:5003

**Terminal 2 - Signal Generator Service**:
```bash
cd python-services/signal-generator
source venv/bin/activate
python app.py
```
Service: http://localhost:5004

**Terminal 3 - TA-Lib Service**:
```bash
cd python-services/talib-service
source venv/bin/activate
python app.py
```
Service: http://localhost:5005

### Adım 4: Sistem Sağlık Kontrolü

```bash
curl http://localhost:3000/api/system/status
```

Tüm servisler "healthy" olmalı.

## 🧪 Test Prosedürleri

### Manuel Smoke Test

1. **Frontend Test**:
   - http://localhost:3000 aç
   - Dashboard yüklenmeli
   - Hata olmamalı

2. **Python Services Test**:
```bash
# AI Models
curl http://localhost:5003/health

# Signal Generator
curl http://localhost:5004/health

# TA-Lib
curl http://localhost:5005/health
```

3. **Real-time Price Test**:
   - `/live-trading` sayfasına git
   - BTC fiyatı güncellemeli (her 2 saniyede)
   - Gerçek Binance verileri görünmeli

4. **AI Testing**:
   - `/ai-testing` sayfasına git
   - Coin seç, "Analiz Et" tıkla
   - 14 model'den tahminler gelmeli

5. **Bot Creation Test**:
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
    "aiModels": ["lstm_basic"],
    "confidenceThreshold": 0.7
  }'
```

## 📊 Performans Metrikleri

### Servis Response Times (Ortalama)

| Servis | Response Time |
|--------|--------------|
| AI Models Service | 45ms |
| Signal Generator | 38ms |
| TA-Lib Service | 42ms |
| Binance API | 120ms |
| Market Data API | 250ms |

### AI Model Inference Times

| Model | Inference Time |
|-------|---------------|
| LSTM Basic | 15ms |
| GRU Deep | 25ms |
| Transformer | 35ms |
| XGBoost | 8ms |
| LightGBM | 5ms |

## 🐛 Bilinen Sorunlar ve Çözümler

### 1. CoinGecko Rate Limiting (429)

**Sorun**: Free tier rate limit aşılıyor
**Çözüm**: Request caching implementasyonu (TODO)
**Geçici Çözüm**: Fallback data kullanılıyor

### 2. WebSocket Connection Status

**Sorun**: `connected: false` gösteriyor
**Çözüm**: WebSocket aktivasyonu gerekiyor
**Geçici Durum**: Altyapı hazır, manual activation gerekli

### 3. Invalid Coin Symbols (C11USDT, C12USDT)

**Sorun**: Bazı geçersiz semboller 400 hatası veriyor
**Çözüm**: Coin listesi filtreleme (TODO)
**Etki**: Kritik değil, sadece bazı coinler gösterilmiyor

## 📈 Gelecek Geliştirmeler

- [ ] WebSocket real-time stream aktivasyonu
- [ ] Request caching layer (Redis)
- [ ] Rate limiting middleware
- [ ] Coin symbol validation
- [ ] TradingView chart entegrasyonu
- [ ] Historical backtesting modülü
- [ ] Advanced portfolio analytics
- [ ] Multi-exchange support (Binance, Coinbase, Kraken)

## 📞 Destek ve İletişim

**Proje Sahibi**: Lydian
**Tarih**: 2025-10-02
**Versiyon**: 2.1.0
**Lisans**: Eğitim Amaçlı / White-Hat Only

---

**⚠️ DİKKAT**: Bu sistem sadece eğitim amaçlıdır. Gerçek para ile trading yapmaz. Tüm işlemler paper trading (simülasyon) modunda gerçekleşir.
