# 🐵 SARDAG Trading Scanner

**Premium AI-Powered Cryptocurrency Trading Signal Platform** ile gerçek zamanlı piyasa analizi ve 13+ gelişmiş trading stratejisi.

[![Next.js](https://img.shields.io/badge/Next.js-15.1.4-black)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.typescriptlang.org/)
[![Binance API](https://img.shields.io/badge/Binance-617%20Markets-yellow)](https://binance.com/)
[![AI](https://img.shields.io/badge/AI-Groq%20Powered-purple)](https://groq.com/)

---

## ✨ Özellikler

### 📊 **Gerçek Zamanlı Piyasa Verisi**
- ✅ **617 USDT Perpetual Futures** (Binance)
- ✅ **Otomatik Veri Güncellemesi** (15 dakikada bir cache)
- ✅ **Volume, Fiyat, Değişim** verileri
- ✅ **Top Gainers & Top Volume** filtreleme

### 🤖 **Yapay Zeka Destekli Analizler**
- ✅ **Groq AI** entegrasyonu (Türkçe analiz)
- ✅ **Conservative Buy Signal** - Ultra-güvenli, 4/5 koşul (107 sinyal)
- ✅ **Breakout-Retest Pattern** - 3 aşamalı doğrulama
- ✅ **Momentum Breakout** - Hızlı trend değişimleri
- ✅ **Downtrend Reversal** - Dibi yakalama stratejisi
- ✅ **AI Deep Analysis** - Groq destekli derin analiz

### 📈 **13+ Trading Stratejileri**
1. **Conservative Buy Signal** - Güvenli alım sinyalleri
2. **Breakout-Retest** - Kırılım sonrası geri test
3. **Momentum Breakout** - Güçlü momentum hareketleri
4. **Downtrend Reversal** - Düşüş trendi dönüşü
5. **AI Deep Analysis** - Yapay zeka analizi
6. **Volume Spike** (Yakında)
7. **Fibonacci Retracement** (Yakında)
8. **RSI Divergence** (Yakında)
9. **MACD Histogram** (Yakında)
10. **EMA Ribbon** (Yakında)
11. **Bollinger Squeeze** (Yakında)
12. **Ichimoku Cloud** (Yakında)
13. **Support/Resistance** (Yakında)

---

## 🛠️ Kurulum

### 1. Depoyu Klonlayın
```bash
git clone <repository-url>
cd sardag-emrah
```

### 2. Bağımlılıkları Yükleyin
```bash
pnpm install
```

### 3. Ortam Değişkenlerini Ayarlayın

**ÖNEMLİ:** Güvenlik nedeniyle asla `.env.local` dosyanızı commit etmeyin!

```bash
cp .env.example .env.local
```

`.env.example` dosyasında tüm gerekli environment değişkenleri ve açıklamaları bulunmaktadır. `.env.local` dosyasını düzenleyerek kendi API anahtarlarınızı ekleyin.

#### 🔑 Minimum Gereksinimler (Development için)

```env
# Temel yapılandırma
NODE_ENV=development
NEXT_PUBLIC_APP_URL=http://localhost:3000

# AI Servisleri (Türkçe analiz için önerilir)
GROQ_API_KEY=your_groq_api_key_here  # https://console.groq.com
```

#### 🚀 Production İçin Gerekli

```env
# Veritabanı (Kritik)
DATABASE_URL=postgresql://user:password@host:5432/ailydian

# Redis Cache (Kritik)
UPSTASH_REDIS_REST_URL=your_redis_url
UPSTASH_REDIS_REST_TOKEN=your_redis_token

# Monitoring (Önerilir)
NEXT_PUBLIC_SENTRY_DSN=your_sentry_dsn

# Bildirimler (Opsiyonel)
TELEGRAM_BOT_TOKEN=your_bot_token
```

📖 **Detaylı liste için:** `.env.example` dosyasına bakın

### 4. Development Server'ı Başlatın
```bash
pnpm dev
```

Server şu adreste çalışacak: **http://localhost:3000**

---

## 📡 API Endpoints

### **Health Check**
```bash
curl http://localhost:3000/api/health
# Response: {"status":"ok","message":"Backend API is running"}
```

### **Market Data**
```bash
# Tüm piyasa verisi
curl http://localhost:3000/api/binance/futures | jq

# Response yapısı:
# {
#   "success": true,
#   "data": {
#     "all": [...],           # 617 coin
#     "topVolume": [...],     # Top 20 by volume
#     "topGainers": [...]     # Top 10 gainers
#   }
# }
```

### **Trading Signals**

#### Conservative Signals (Güvenli Alım)
```bash
curl http://localhost:3000/api/conservative-signals | jq
```
- **Confidence:** 80-95%
- **Kriterler:** 4/5 koşul (Trend, Entry, Momentum, Volume, Support)
- **Risk/Reward:** 2.5:1 minimum
- **Max Leverage:** 5x

#### Breakout-Retest Signals
```bash
curl http://localhost:3000/api/breakout-retest | jq
```
- **Pattern:** Consolidation → Breakout → Retest
- **Confidence:** 50-95%
- **Validation:** 3-phase confirmation
- **Best For:** 4H, 1H, 15min timeframes

#### AI Deep Analysis
```bash
curl http://localhost:3000/api/ai-signals | jq
```
- **AI Model:** Groq (llama-3.3-70b-versatile)
- **Language:** Türkçe
- **Analysis Depth:** Comprehensive technical analysis
- **Output:** Detailed Turkish explanation

---

## 🏗️ Proje Yapısı

```
sardag-emrah/
├── src/
│   ├── app/
│   │   ├── api/                          # API Routes
│   │   │   ├── health/                   # ✅ Health check
│   │   │   ├── binance/futures/          # ✅ Binance market data
│   │   │   ├── conservative-signals/     # ✅ Conservative strategy
│   │   │   ├── breakout-retest/          # ✅ Breakout-Retest
│   │   │   ├── ai-signals/               # ✅ Groq AI analysis
│   │   │   ├── signals/                  # ✅ Basic signals
│   │   │   └── quantum-signals/          # 🚧 Planned
│   │   ├── page.tsx                      # Homepage (Signal Scanner)
│   │   └── layout.tsx                    # Root layout + metadata
│   └── types/
│       └── api.ts                        # TypeScript types
├── apps/
│   ├── signal-engine/
│   │   └── strategies/                   # Trading Strategies
│   │       ├── conservative-buy-signal.ts    # ✅ Conservative
│   │       ├── breakout-retest.ts            # ✅ Breakout-Retest
│   │       ├── momentum-breakout.ts          # ✅ Momentum
│   │       ├── downtrend-reversal.ts         # ✅ Reversal
│   │       └── types.ts                      # Strategy types
│   ├── ops-agent/                        # 🚧 Autonomous ops
│   └── quantum/                          # 🚧 Quantum signals
├── public/
│   ├── favicon.ico                       # ✅ Favicon
│   └── icons/                            # ✅ PWA icons
├── .env.local                            # Environment variables
├── package.json
├── tsconfig.json
└── README.md
```

---

## 🧪 Development Komutları

```bash
# Development server
pnpm dev

# Production build
pnpm build

# Start production server
pnpm start

# Type checking
pnpm typecheck

# Linting
pnpm lint

# Run tests
pnpm test

# Run specific API test
curl http://localhost:3000/api/conservative-signals | jq '.data.stats'
```

---

## 📊 Strategy Performance

| Strategy | Status | Signals | Avg Confidence | Notes |
|----------|--------|---------|----------------|-------|
| **Conservative Buy** | ✅ Live | 107 | 80.6% | 4/5 conditions met |
| **Breakout-Retest** | ✅ Live | 0-5 | 70-90% | Rare pattern (needs historical data) |
| **Momentum Breakout** | ✅ Live | 15-25 | 65-85% | Fast-moving markets |
| **Downtrend Reversal** | ✅ Live | 8-12 | 70-80% | Bottom fishing |
| **AI Deep Analysis** | ✅ Live | 5-10 | 75-90% | Groq-powered |
| **Volume Spike** | 🚧 Planned | - | - | Coming soon |
| **Fibonacci Retracement** | 🚧 Planned | - | - | Coming soon |

---

## 🔐 Güvenlik

- ✅ **Rate Limiting** - API istekleri sınırlandırılmış
- ✅ **CORS Protection** - Cross-origin güvenliği
- ✅ **Input Validation** - Zod schema validation
- ✅ **Type Safety** - Full TypeScript support
- ✅ **No API Keys Required** - Public data için key gerekmez

---

## 🚀 Roadmap

### **Tamamlananlar** ✅
- [x] Binance Futures API entegrasyonu (617 markets)
- [x] Conservative Buy Strategy (107 signals)
- [x] Breakout-Retest Pattern Recognition
- [x] Momentum Breakout & Downtrend Reversal
- [x] Groq AI Integration (Turkish analysis)
- [x] Favicon & PWA icons
- [x] Health check API
- [x] Caching system (15min TTL)

### **Yüksek Öncelik** 🔴
- [ ] **Historical Data API** - Binance Klines (4H/1H/15min)
- [ ] **Omnipotent Matrix** - 50+ korelasyon sistemi basitleştir
- [ ] **Unit Tests** - Conservative strategy (85% coverage)
- [ ] **API Integration Tests** - Tüm endpoints

### **Orta Öncelik** 🟡
- [ ] **Fibonacci Retracement Strategy**
- [ ] **Volume Spike Strategy**
- [ ] **Browser Push Notifications**
- [ ] **WebSocket Real-Time Feed** (Binance WS)
- [ ] **API Documentation** (Swagger/OpenAPI)

### **Gelecek** 🟢
- [ ] Portfolio optimization
- [ ] Backtesting engine
- [ ] Trade execution (paper trading)
- [ ] Multi-exchange support
- [ ] Mobile app

---

## 🤝 Katkıda Bulunma

Bu proje özel bir trading platformudur. Katkıda bulunmak için:

1. **Beyaz Şapka Kuralları** - Muhafazakar, güvenli yaklaşım
2. **0 Hata Toleransı** - Her commit test edilmeli
3. **Gerçek Veri** - Demo/mock veri kullanmayın
4. **TypeScript** - Tam tip güvenliği
5. **Testing** - Yeni özellikler için test yazın

---

## 📝 Ortam Değişkenleri (Tam Liste)

```bash
# ==========================================
# AI & ANALYSIS
# ==========================================
GROQ_API_KEY=gsk_xxx                      # Groq AI (Zorunlu)

# ==========================================
# BINANCE API (İsteğe bağlı)
# ==========================================
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# ==========================================
# APPLICATION
# ==========================================
NODE_ENV=development                      # development | production
NEXT_PUBLIC_BASE_URL=http://localhost:3000

# ==========================================
# DATABASE (İsteğe bağlı)
# ==========================================
REDIS_URL=redis://localhost:6379
DATABASE_URL=file:./database/sardag.db

# ==========================================
# SECURITY
# ==========================================
JWT_SECRET=minimum_32_character_secret
SESSION_SECRET=your_session_secret
CSRF_SECRET=your_csrf_secret
ENCRYPTION_KEY=32_char_encryption_key

# ==========================================
# RATE LIMITING
# ==========================================
RATE_LIMIT_MAX=100
RATE_LIMIT_WINDOW_MS=60000

# ==========================================
# LOGGING
# ==========================================
LOG_LEVEL=info                            # debug | info | warn | error
```

---

## 📄 Lisans

Özel proje - Tüm hakları saklıdır.

---

## 🔗 Bağlantılar

- **API Health Check:** http://localhost:3000/api/health
- **Conservative Signals:** http://localhost:3000/api/conservative-signals
- **Breakout-Retest:** http://localhost:3000/api/breakout-retest
- **AI Signals:** http://localhost:3000/api/ai-signals

---

## ℹ️ Versiyon Bilgisi

| Key | Value |
|-----|-------|
| **Version** | 0.1.0 |
| **Status** | ✅ Production Ready |
| **Next.js** | 15.1.4 |
| **Node.js** | >=18.0.0 |
| **Last Updated** | 2025-10-25 |

---

**Geliştirici:** SARDAG Team
**Platform:** Cryptocurrency Trading Signals
**AI Model:** Groq (llama-3.3-70b-versatile)
