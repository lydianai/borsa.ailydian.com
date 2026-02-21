# 🧠 AI/ML SÜREKLI ÖĞRENME MİMARİSİ

**Version:** 1.0
**Date:** 2025-11-19
**Status:** Production Ready Architecture

---

## 📋 İÇİNDEKİLER

1. [Mimari Özet](#mimari-özet)
2. [Sürekli Çalışma Mantığı](#sürekli-çalışma-mantığı)
3. [Python Servisleri Entegrasyonu](#python-servisleri-entegrasyonu)
4. [Model Persistence & Learning](#model-persistence--learning)
5. [PM2 Background Jobs](#pm2-background-jobs)
6. [Data Flow & Pipeline](#data-flow--pipeline)
7. [Implementation Plan](#implementation-plan)

---

## 🏗️ MİMARİ ÖZET

### Sistem Bileşenleri

```
┌─────────────────────────────────────────────────────────────────┐
│                     AILYDIAN SIGNAL PLATFORM                     │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
         ┌──────────▼───┐  ┌────▼─────┐  ┌──▼───────────┐
         │   Next.js    │  │   APIs   │  │  WebSocket   │
         │   Frontend   │  │  Routes  │  │   Server     │
         └──────────────┘  └──────────┘  └──────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │                         │
         ┌──────────▼──────────┐   ┌─────────▼──────────┐
         │  AI/ML Learning Hub │   │  Existing Services │
         │    (Port 5020)      │   │  (Various Ports)   │
         └─────────────────────┘   └────────────────────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   CONTINUOUS LEARNING   │
                    │     ORCHESTRATOR        │
                    │     (PM2 Managed)       │
                    └─────────────────────────┘
                                 │
        ┌────────────┬───────────┼───────────┬────────────┐
        │            │           │           │            │
   ┌────▼───┐  ┌────▼───┐  ┌────▼───┐  ┌────▼───┐  ┌────▼───┐
   │   RL   │  │ Online │  │ Multi  │  │ AutoML │  │  Meta  │
   │ Agent  │  │Learning│  │ Agent  │  │Optimize│  │Learning│
   │ Worker │  │ Worker │  │ Worker │  │ Worker │  │ Worker │
   └────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘
        │           │           │           │           │
        └───────────┴───────────┴───────────┴───────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   SHARED DATA LAYER     │
                    │  - Redis Cache          │
                    │  - PostgreSQL Models    │
                    │  - File-based Checkpts  │
                    └─────────────────────────┘
```

---

## ⚙️ SÜREKLI ÇALIŞMA MANTIĞI

### 1. **7/24 Background Learning**

AI/ML Hub servisleri **kesintisiz çalışarak**:

#### A. **Passive Learning (Gözlem Modu)**
```python
# Her 5 dakikada bir
- Binance'den yeni fiyat verisi çek
- TA-Lib servisinden indikatör hesapla
- Signal generator'dan sinyal al
- Feature engineering servisinden özellik çıkar
- Bu verileri model'e besle (incremental learning)
```

**Örnek Flow:**
```
05:00 → Yeni BTC fiyatı geldi ($43,250)
05:01 → TA-Lib: RSI=65, MACD=+120
05:02 → Signal: BUY sinyali (confidence: 78%)
05:03 → RL Agent: Bu durumda BUY yaptım → Reward: +$150
05:04 → RL Agent Q-Table güncellendi (öğrendi!)
05:05 → Model checkpoint kaydedildi
```

#### B. **Active Learning (Doğrulama Modu)**
```python
# Her saat başı
- Son 1 saatteki tahminlerimi kontrol et
- Gerçek sonuçlarla karşılaştır (doğru mu yanlış mı?)
- Hata varsa model parametrelerini ayarla
- Accuracy ve loss metriklerini güncelle
```

**Örnek Flow:**
```
06:00 → 05:00'daki BUY tahminim doğruydu (fiyat $43,250 → $43,800)
06:01 → Reward: +1 (doğru tahmin bonusu)
06:02 → Q-Table: Bu state-action pair'inin değeri arttı
06:03 → Win Rate: 73.2% → 73.5% (gelişti!)
```

---

### 2. **Mevcut Servislerle Entegrasyon**

AI/ML Hub, **tüm mevcut Python servislerini kullanarak** öğrenir:

#### Entegre Edilecek Servisler:

| Servis | Port | AI/ML Kullanımı |
|--------|------|-----------------|
| **TA-Lib Service** | 5001 | Teknik indikatörler (RSI, MACD, Bollinger) → Feature input |
| **Signal Generator** | 5002 | Oluşturulan sinyaller → Label/Target olarak |
| **Risk Management** | 5003 | Risk skorları → Reward shaping için |
| **Feature Engineering** | 5004 | Hazırlanmış özellikler → Model input |
| **SMC Strategy** | 5005 | Order block verileri → Context bilgisi |
| **Transformer AI** | 5006 | Attention weights → Meta-learning için |
| **Online Learning** | 5007 | Drift detection → Model güncelleme tetikleyici |
| **Multi-Timeframe** | 5008 | Farklı zaman dilimi verileri → Ensemble için |
| **Order Flow** | 5009 | Volume profil → Market regime detection |
| **Continuous Monitor** | 5010 | Real-time alerts → Active learning trigger |
| **MFI Monitor** | 5011 | Money Flow Index → Liquidity features |

#### Entegrasyon Kodu Örneği:
```python
# AI Learning Hub içinde
async def collect_training_data(symbol: str):
    # 1. TA-Lib'den indikatörler al
    indicators = await fetch('http://localhost:5001/indicators', {
        'symbol': symbol,
        'timeframe': '1h'
    })

    # 2. Feature Engineering'den özellikler al
    features = await fetch('http://localhost:5004/features', {
        'symbol': symbol
    })

    # 3. Signal Generator'dan label al
    signal = await fetch('http://localhost:5002/signals', {
        'symbol': symbol
    })

    # 4. Hepsini birleştir
    training_sample = {
        'features': {**indicators, **features},
        'label': signal['type'],  # BUY, SELL, HOLD
        'confidence': signal['confidence']
    }

    # 5. Model'e besle
    model.partial_fit([training_sample['features']],
                      [training_sample['label']])

    return training_sample
```

---

## 💾 MODEL PERSISTENCE & LEARNING

### 1. **Checkpoint Sistemi**

Her AI modeli düzenli olarak kaydedilir:

```
/45-backend/python-services/ai-learning-hub/
├── models/
│   ├── rl_agent/
│   │   ├── q_table_20251119_0600.pkl        # Sabah 06:00 checkpoint
│   │   ├── q_table_20251119_1200.pkl        # Öğlen 12:00 checkpoint
│   │   ├── q_table_20251119_1800.pkl        # Akşam 18:00 checkpoint
│   │   └── q_table_latest.pkl               # En son checkpoint
│   ├── online_learning/
│   │   ├── model_v247.pkl
│   │   ├── model_v248.pkl
│   │   └── model_latest.pkl
│   ├── multi_agent/
│   │   ├── agent_momentum_v15.pkl
│   │   ├── agent_rsi_v12.pkl
│   │   └── ensemble_weights.pkl
│   └── ...
├── logs/
│   ├── training_history.json               # Tüm eğitim kayıtları
│   ├── performance_metrics.json            # Accuracy, loss, etc.
│   └── predictions_log.json                # Tahmin geçmişi
```

### 2. **Incremental Learning Pipeline**

```python
class ContinuousLearner:
    def __init__(self):
        self.models = self.load_latest_checkpoints()
        self.training_buffer = []
        self.checkpoint_interval = 3600  # 1 saat

    async def run_forever(self):
        while True:
            # 1. Yeni veri topla
            new_data = await self.collect_data()
            self.training_buffer.append(new_data)

            # 2. Buffer dolduğunda eğit
            if len(self.training_buffer) >= 100:
                await self.train_batch()
                self.training_buffer = []

            # 3. Düzenli checkpoint
            if self.should_checkpoint():
                await self.save_checkpoints()

            # 4. Performance izle
            await self.log_metrics()

            await asyncio.sleep(300)  # 5 dakika bekle
```

---

## 🔄 PM2 BACKGROUND JOBS

### PM2 Configuration

Her AI sistemi için ayrı bir PM2 job:

```json
// ecosystem.config.js
{
  "apps": [
    {
      "name": "ai-learning-orchestrator",
      "script": "orchestrator.py",
      "cwd": "/Users/sardag/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub",
      "interpreter": "./venv/bin/python3",
      "instances": 1,
      "exec_mode": "fork",
      "autorestart": true,
      "watch": false,
      "max_memory_restart": "1G",
      "env": {
        "NODE_ENV": "production",
        "AI_LEARNING_MODE": "continuous"
      },
      "cron_restart": "0 4 * * *"  // Her gün 04:00'de restart
    },
    {
      "name": "rl-agent-worker",
      "script": "workers/rl_agent_worker.py",
      "cwd": "/Users/sardag/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub",
      "interpreter": "./venv/bin/python3",
      "instances": 1,
      "autorestart": true
    },
    {
      "name": "online-learning-worker",
      "script": "workers/online_learning_worker.py",
      "cwd": "/Users/sardag/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub",
      "interpreter": "./venv/bin/python3",
      "instances": 1,
      "autorestart": true
    },
    {
      "name": "multi-agent-worker",
      "script": "workers/multi_agent_worker.py",
      "cwd": "/Users/sardag/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub",
      "interpreter": "./venv/bin/python3",
      "instances": 1,
      "autorestart": true
    },
    {
      "name": "automl-optimizer-worker",
      "script": "workers/automl_worker.py",
      "cwd": "/Users/sardag/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub",
      "interpreter": "./venv/bin/python3",
      "instances": 1,
      "autorestart": true,
      "cron_restart": "0 */6 * * *"  // Her 6 saatte bir restart
    }
  ]
}
```

### PM2 Başlatma:
```bash
cd /Users/sardag/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub
pm2 start ecosystem.config.js
pm2 save
pm2 startup  # Boot'ta otomatik başlat
```

---

## 📊 DATA FLOW & PIPELINE

### Real-time Learning Flow:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA COLLECTION (Her 5 dakika)                          │
├─────────────────────────────────────────────────────────────┤
│  Binance API → Yeni fiyat verileri                         │
│  TA-Lib Service (5001) → Teknik indikatörler               │
│  Feature Engineering (5004) → İşlenmiş özellikler          │
│  Signal Generator (5002) → Sinyal tahminleri               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING & FEATURE ENGINEERING                     │
├─────────────────────────────────────────────────────────────┤
│  • Normalize features (0-1 scaling)                        │
│  • Handle missing values                                   │
│  • Create rolling windows (5min, 15min, 1h, 4h)          │
│  • Calculate momentum, volatility metrics                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. PARALLEL AI TRAINING (10 AI sistemleri eş zamanlı)     │
├─────────────────────────────────────────────────────────────┤
│  ⚡ RL Agent        → Q-Learning update                     │
│  🔄 Online Learning → Incremental fit                      │
│  👥 Multi-Agent     → Ensemble voting                      │
│  ⚙️ AutoML          → Hyperparameter tuning (her 6 saat)   │
│  🏗️ NAS             → Architecture evolution (günlük)      │
│  ✨ Meta-Learning   → Few-shot adaptation                  │
│  🛡️ Federated       → Privacy-preserving update           │
│  🔀 Causal AI       → Causal graph update                 │
│  📈 Regime Detect   → State transition update              │
│  🔍 Explainable AI  → SHAP value calculation              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. MODEL EVALUATION & VALIDATION (Her saat)                │
├─────────────────────────────────────────────────────────────┤
│  • Compare predictions vs actual outcomes                  │
│  • Calculate accuracy, precision, recall                   │
│  • Update win rate, Sharpe ratio                          │
│  • Detect concept drift                                    │
│  • Trigger re-training if needed                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. MODEL PERSISTENCE (Her 6 saat)                         │
├─────────────────────────────────────────────────────────────┤
│  • Save model checkpoints                                  │
│  • Log training history                                    │
│  • Update performance metrics                              │
│  • Backup to cloud storage (optional)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. FEEDBACK LOOP (Continuous)                              │
├─────────────────────────────────────────────────────────────┤
│  • Good prediction → Positive reward → Strengthen policy   │
│  • Bad prediction → Negative reward → Adjust parameters    │
│  • Concept drift detected → Re-initialize model            │
│  • New market regime → Adapt strategy                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 IMPLEMENTATION PLAN

### Phase 1: Infrastructure Setup (2 gün)

**Görevler:**
1. ✅ PM2 ecosystem.config.js oluştur
2. ✅ Model persistence klasör yapısı kur
3. ✅ Logging infrastructure kur
4. ✅ Redis cache entegrasyonu (opsiyonel)
5. ✅ PostgreSQL model metadata tablosu

**Dosyalar:**
- `/45-backend/python-services/ai-learning-hub/ecosystem.config.js`
- `/45-backend/python-services/ai-learning-hub/models/` (klasörler)
- `/45-backend/python-services/ai-learning-hub/logs/` (klasörler)
- `/45-backend/python-services/ai-learning-hub/utils/persistence.py`
- `/45-backend/python-services/ai-learning-hub/utils/logging.py`

---

### Phase 2: Orchestrator Development (3 gün)

**Görevler:**
1. ✅ Ana orchestrator.py yazılımı
2. ✅ Data collection pipeline
3. ✅ Tüm Python servisleri ile entegrasyon
4. ✅ Batch processing logic
5. ✅ Error handling & retry mechanism

**Dosyalar:**
- `/45-backend/python-services/ai-learning-hub/orchestrator.py`
- `/45-backend/python-services/ai-learning-hub/services/service_integrator.py`
- `/45-backend/python-services/ai-learning-hub/utils/data_collector.py`

---

### Phase 3: Worker Development (5 gün)

Her AI sistemi için ayrı worker:

**Görevler:**
1. ✅ `workers/rl_agent_worker.py` - Reinforcement Learning worker
2. ✅ `workers/online_learning_worker.py` - Online Learning worker
3. ✅ `workers/multi_agent_worker.py` - Multi-Agent worker
4. ✅ `workers/automl_worker.py` - AutoML optimizer worker
5. ✅ Diğer 6 AI worker

**Her Worker'ın Yapması Gerekenler:**
```python
# Worker template
class AIWorker:
    def __init__(self):
        self.model = self.load_or_create_model()

    async def run_forever(self):
        while True:
            try:
                # 1. Veri topla
                data = await self.collect_data()

                # 2. Model eğit
                result = await self.train(data)

                # 3. Sonuçları logla
                await self.log_result(result)

                # 4. Checkpoint kaydet
                if self.should_checkpoint():
                    await self.save_checkpoint()

                await asyncio.sleep(self.interval)
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(60)  # 1 dakika bekle
```

---

### Phase 4: Monitoring Dashboard (2 gün)

**Görevler:**
1. ✅ Frontend monitoring sayfası: `/ai-learning-hub/monitoring`
2. ✅ Real-time metrics API
3. ✅ Training progress charts
4. ✅ Model comparison dashboard
5. ✅ Alert system (Slack/Telegram)

**UI Elements:**
- 📊 Live training metrics (accuracy, loss, reward)
- 📈 Historical performance charts
- 🤖 Worker status (online/offline/error)
- 💾 Model checkpoint timeline
- ⚠️ Alert notifications
- 🔄 Manual trigger buttons (force re-train, checkpoint, etc.)

---

### Phase 5: Testing & Optimization (3 gün)

**Görevler:**
1. ✅ End-to-end pipeline test
2. ✅ Load testing (1000 req/sec)
3. ✅ Memory leak detection
4. ✅ Performance optimization
5. ✅ Production deployment

---

## 📈 EXPECTED BENEFITS

### 1. **Sürekli İyileşme**
- Modeller her gün daha iyi tahmin yapar
- Piyasa değişikliklerine otomatik adapte olur
- Manual intervention gerektirmez

### 2. **Sistem Genelinde Entegrasyon**
- Tüm Python servisleri birbirine bağlı çalışır
- Veri tekrarı olmaz (central data lake)
- Resource kullanımı optimize edilir

### 3. **Sağlam Altyapı**
- PM2 ile otomatik restart
- Model persistence ile veri kaybı olmaz
- Monitoring ile sorunlar hızla tespit edilir

### 4. **Scalability**
- Yeni AI sistemi eklemek kolay
- Worker sayısı ihtiyaca göre artırılabilir
- Distributed training mümkün

---

## 🚀 DEPLOYMENT CHECKLIST

### Development Environment:
- [ ] PM2 kurulumu
- [ ] Tüm Python servisleri health check
- [ ] Model persistence klasörleri oluşturuldu
- [ ] Logging infrastructure hazır
- [ ] Orchestrator test edildi

### Staging Environment:
- [ ] PM2 jobs başlatıldı
- [ ] Workers çalışıyor
- [ ] Data collection aktif
- [ ] Model training başladı
- [ ] Monitoring dashboard açıldı

### Production Environment:
- [ ] PM2 startup configured
- [ ] Auto-restart policies set
- [ ] Monitoring alerts configured
- [ ] Backup strategy implemented
- [ ] Performance benchmarks met

---

## 📞 SUPPORT

**Documentation:** `/AI_ML_CONTINUOUS_LEARNING_ARCHITECTURE.md`
**Code Location:** `/45-backend/python-services/ai-learning-hub/`
**Monitoring:** `http://localhost:3000/ai-learning-hub/monitoring`

---

**Created by:** Claude Code
**Last Updated:** 2025-11-19
**Version:** 1.0
