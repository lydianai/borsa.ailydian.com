# 🚀 LyDian Trader - Backend API Özellikleri

## 📊 Proje Durumu
- **Backup Oluşturuldu**: `borsa-backup-YYYYMMDD-HHMMSS.tar.gz`
- **Tüm Frontend Temizlendi**: React components, pages, hooks, contexts kaldırıldı
- **Sadece API Backend Kaldı**: 38 API endpoint aktif

---

## 🎯 Aktif Backend Modülleri

### 1. **AI & Makine Öğrenimi Servisleri**
#### `/api/ai/*` - AI Model API'leri
- `/api/ai/models` - Python ML model listesi
- `/api/ai/predict` - Tekli tahmin endpoint'i
- `/api/ai/predict-batch` - Toplu tahmin
- `/api/ai/python` - Python mikroservis proxy

#### AI Bot Servisleri
- **MasterOrchestrator**: Multi-model ensemble bot
- **QuantumTradingBot**: Kuantum algoritmalar
- **AdvancedAIEngine**: Gelişmiş AI motoru
- **AttentionTransformer**: Transformer-based model
- **HybridDecisionEngine**: Hibrit karar motoru
- **ReinforcementLearningAgent**: Pekiştirmeli öğrenme

### 2. **Trading & Market Data**
#### `/api/market/*` - Market Verisi
- `/api/market/crypto` - CoinGecko top 50 kripto
- `/api/market/top100` - Binance + CoinGecko hibrit veri

#### `/api/trading/*` - Trading Sinyalleri
- `/api/trading/signals` - BUY/SELL/HOLD sinyalleri
- `/api/trading/comprehensive` - Tam market analizi (CMC + Binance + TA-Lib)
- `/api/trading/top100` - Top 100 coin için kapsamlı analiz

#### `/api/binance/*` - Binance Entegrasyonu
- `/api/binance/price` - Gerçek zamanlı fiyat
- `/api/binance/orderbook` - Order book verileri
- `/api/binance-futures` - Futures market data

### 3. **Bot Yönetimi**
#### `/api/bot/*` - Trading Bot API
- `/api/bot` - Bot liste/durum
- `/api/bot/initialize` - Bot başlatma
- `/api/bot/futures` - Futures bot sinyalleri
- `/api/bot/quantum-signal` - Kuantum sinyal üretimi

#### `/api/ai-bots/*` - AI Bot Sinyalleri
- `/api/ai-bots/all-signals` - Tüm bot sinyalleri
- `/api/ai-bots/master-orchestrator/signals` - Ensemble sinyaller

### 4. **Quantum Pro Sistemi**
#### `/api/quantum-pro/*` - İleri Seviye Trading
- `/api/quantum-pro/signals` - AI ensemble sinyaller
- `/api/quantum-pro/backtest` - Strateji backtest
- `/api/quantum-pro/monitor` - Sinyal monitörü
- `/api/quantum-pro/risk-check` - Risk analizi
- `/api/quantum-pro/bots` - Bot yönetimi
- `/api/quantum-pro/bots/control` - Bot kontrolü

### 5. **Omnipotent Futures Matrix**
#### `/api/omnipotent/signals`
- 100 farklı futures sembolü için gerçek zamanlı sinyal
- Çoklu strateji kombinasyonu
- Yüksek güven skorlu sinyaller

### 6. **Otomatik Trading**
#### `/api/auto-trading`
- Top 100 coin otomatik trading
- Binance + CoinGecko entegrasyonu
- Configurable risk limitleri

#### `/api/unified-bot`
- Birleşik trading bot arayüzü
- Multi-exchange desteği
- Merkezi yönetim

### 7. **Azure Cloud Servisleri**
#### `/api/azure/*` - Azure OpenAI Entegrasyonu
- `/api/azure/market-analysis` - AI market analizi
- `/api/azure/sentiment` - Duygu analizi

#### `/api/signalr/negotiate`
- Azure SignalR gerçek zamanlı iletişim
- WebSocket connection info

### 8. **Monitoring & Alerting**
#### `/api/monitoring/live`
- Gerçek zamanlı bot metrikleri
- Performance tracking
- Alert sistemi (Telegram + Discord desteği)

#### `/api/charts/history`
- Geçmiş performans verileri
- Database-backed chart data

### 9. **Compliance & Security**
#### `/api/compliance/white-hat`
- Beyaz şapkalı trading kuralları
- Piyasa manipülasyonu önleme
- Risk limitleri
- Etik trading kontrolü

#### `/api/geolocation`
- IP geolocation (defensive security)
- Login attempt tracking
- Device fingerprinting

### 10. **WebSocket & Real-time**
#### `/api/websocket/binance`
- Server-side Binance WebSocket
- Gerçek zamanlı market data stream

### 11. **System Management**
#### `/api/system/status`
- Tüm mikroservislerin health check'i
- Service uptime monitoring
- Response time tracking

---

## 🧠 Core Services (src/services/)

### AI/ML Services
1. **QuantumNexusEngine** - Quantum-inspired algoritma
2. **QuantumProEngine** - İleri seviye quantum trading
3. **MasterAIOrchestrator** - Tüm AI modelleri orkestra eden master bot
4. **ModelTrainingPipeline** - Model eğitim pipeline
5. **TensorFlowOptimizer** - TF model optimizasyonu
6. **NirvanaTFClient** - TensorFlow client
7. **BacktestingEngine** - Strateji backtesting
8. **RiskManagementModule** - Risk yönetimi

### Trading Services
1. **AutoTradingEngine** - Otomatik trading motoru
2. **UnifiedTradingBot** - Birleşik bot arayüzü
3. **OmnipotentFuturesMatrix** - 100 coin futures sinyal matrisi
4. **BinanceFuturesService** - Futures market servisi
5. **TradingBotEngine** - Genel bot motoru
6. **QuantumFuturesTradingEngine** - Quantum futures bot
7. **AzurePoweredQuantumBot** - Azure entegreli quantum bot

### Market Data Services
1. **MarketDataService** - Merkezi market data
2. **CoinMarketCapService** - CMC API
3. **BinanceOHLCVService** - OHLCV candlestick data
4. **RealMarketDataService** - Gerçek market data
5. **BinanceWebSocketService** - WebSocket stream

### Integration Services
1. **MasterIntegrationService** - Tüm servisleri birleştiren master servis
2. **SignalMonitorService** - Sinyal monitörü
3. **SignalStorageService** - Sinyal storage
4. **AIBotSignalService** - AI bot sinyalleri

---

## 🐍 Python Mikroservisler
1. **ai-models** (Port 5003) - ML model servisi
2. **signal-generator** (Port 5004) - Sinyal üretimi
3. **talib-service** (Port 5005) - TA-Lib indikatörleri

---

## 📦 Kaldırılan Bileşenler
✅ Tüm React sayfaları (`/settings`, `/dashboard`, `/ai-control-center`, vb.)
✅ Tüm UI componentleri (`/components/Navigation.tsx`, vb.)
✅ Frontend context'leri (`ThemeContext`, `LanguageContext`)
✅ Hooks (`useTheme`, vb.)
✅ Frontend konfigürasyonları (`tailwind.config.ts`)
✅ Duplicate API endpointleri (`/api/location` - `geolocation` ile aynı)

---

## 🚀 Kullanım
```bash
# Backend'i başlat
pnpm dev

# API test
curl http://localhost:3100/api/market/crypto
curl http://localhost:3100/api/trading/signals?symbol=BTC
curl http://localhost:3100/api/omnipotent/signals?limit=50
```

---

**Toplam API Endpoint**: 38  
**Toplam AI/ML Model**: 8  
**Toplam Trading Bot**: 6  
**Toplam Servis**: 33  
**Python Mikroservis**: 3
