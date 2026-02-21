# 🚀 AI/ML SÜREKLI ÖĞRENME SİSTEMİ - BAŞLATMA REHBERİ

**Tüm Binance Futures USDT-M coinleri (538 adet) için 7/24 sürekli öğrenme**

---

## 📋 HIZLI BAŞLANGIÇ

### 1. Gereksinimler

```bash
# PM2 yüklü mü kontrol et
pm2 --version

# Yoksa yükle
npm install -g pm2

# Python bağımlılıkları yüklü mü?
cd /Users/sardag/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub
./venv/bin/pip list
```

### 2. Tek Komut ile Tüm Sistemi Başlat

```bash
cd /Users/sardag/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub

# PM2 ile tüm servisleri başlat
pm2 start ecosystem.config.js

# Durumu kontrol et
pm2 status

# Logları izle
pm2 logs
```

### 3. Sistem Durumunu İzle

```bash
# Tüm worker'ların durumu
pm2 list

# Belirli bir worker'ın logları
pm2 logs ai-learning-orchestrator
pm2 logs rl-agent-worker
pm2 logs data-collector

# Real-time monitoring
pm2 monit
```

---

## 🔧 MANUEL BAŞLATMA (Test İçin)

### Adım 1: Data Collector'ı Başlat

```bash
cd /Users/sardag/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub

# Arka planda çalıştır
./venv/bin/python3 services/data_collector.py > logs/data-collector-manual.log 2>&1 &

# PID'yi kaydet
echo $! > data_collector.pid

# Log izle
tail -f logs/data-collector-manual.log
```

**Beklenen çıktı:**
```
📡 Data Collector initialized
🚀 Data Collector started
⏱️ Collection interval: 60s
📦 Batch size: 50 symbols
📊 Loaded 538 USDT-M perpetual symbols
🔄 Iteration #1 - Collecting 538 symbols...
  ✓ Batch 1: 50/50 collected
  ✓ Batch 2: 50/50 collected
  ...
✅ Iteration #1 completed in 12.3s | Collected: 538/538
```

### Adım 2: Orchestrator'ı Başlat

```bash
# Orchestrator başlat
./venv/bin/python3 orchestrator.py > logs/orchestrator-manual.log 2>&1 &

# PID kaydet
echo $! > orchestrator.pid

# Log izle
tail -f logs/orchestrator-manual.log
```

**Beklenen çıktı:**
```
======================================================================
🤖 AI/ML LEARNING HUB - ORCHESTRATOR
======================================================================
📍 Mode: Continuous Learning
🌐 Market: Binance Futures USDT-M
⚡ Workers: 10 AI/ML systems
======================================================================
🚀 Starting main orchestration loop...
📊 Total Binance Futures USDT-M symbols: 538
✅ Loaded 538 symbols
🔄 Iteration #1 - Processing 538 symbols...
```

### Adım 3: RL Agent Worker'ı Başlat

```bash
# RL Agent worker başlat
./venv/bin/python3 workers/rl_agent_worker.py > logs/rl-agent-manual.log 2>&1 &

# PID kaydet
echo $! > rl_worker.pid

# Log izle
tail -f logs/rl-agent-manual.log
```

**Beklenen çıktı:**
```
⚡ RL Agent Worker initialized
🚀 RL Agent Worker started
⚙️ Training interval: 300s
💾 Checkpoint interval: 3600s
📊 Processing 50 data points...
✅ Processed 50 | Episodes: 50 | Win Rate: 65.2% | Symbols: 50
```

---

## 📊 SİSTEM MİMARİSİ

### Aktif Servisler (PM2)

| Servis | Açıklama | Port | Interval |
|--------|----------|------|----------|
| **ai-learning-orchestrator** | Ana koordinatör | - | 5 dakika |
| **data-collector** | Binance veri toplayıcı | - | 1 dakika |
| **ai-learning-api** | Flask API server | 5020 | - |
| **rl-agent-worker** | RL öğrenme worker | - | 5 dakika |
| **online-learning-worker** | Online learning | - | 10 dakika |
| **multi-agent-worker** | Multi-agent system | - | 5 dakika |
| **automl-optimizer-worker** | AutoML optimizer | - | 6 saat |
| **nas-worker** | Architecture search | - | 24 saat |
| **meta-learning-worker** | Meta-learning | - | 1 saat |
| **federated-learning-worker** | Federated learning | - | 2 saat |
| **causal-ai-worker** | Causal inference | - | 1 saat |
| **regime-detection-worker** | Regime detection | - | 5 dakika |
| **explainable-ai-worker** | Explainability | - | 10 dakika |
| **service-integrator** | Servis entegratörü | - | 5 dakika |

**Toplam: 13 servis**

### Data Flow

```
1️⃣ DATA COLLECTION (Her 1 dakika)
   ↓
   Data Collector → 538 coin verisini toplar
   ↓
   queue/ klasörüne JSON dosyaları yazar

2️⃣ ORCHESTRATION (Her 5 dakika)
   ↓
   Orchestrator → queue'dan okur
   ↓
   Mevcut Python servislerinden ek veri toplar (TA-Lib, signals, etc.)
   ↓
   Zenginleştirilmiş veriyi tekrar queue'ya yazar

3️⃣ AI/ML TRAINING (Her worker kendi intervalinde)
   ↓
   Worker'lar → queue'dan okur
   ↓
   Model'leri güncellerler (incremental learning)
   ↓
   Checkpoint'leri kaydederler (models/ klasörüne)

4️⃣ PERSISTENCE (Her 1 saat)
   ↓
   Her worker kendi model checkpointlarını kaydeder
   ↓
   models/<worker_name>/ klasörüne .pkl dosyaları
```

---

## 💾 CHECKPOINT & MODEL STORAGE

### Klasör Yapısı

```
ai-learning-hub/
├── models/                      # Model checkpointları
│   ├── rl_agent/
│   │   ├── BTCUSDT_q_table.pkl
│   │   ├── ETHUSDT_q_table.pkl
│   │   └── stats.json
│   ├── online_learning/
│   │   ├── model_v1.pkl
│   │   └── drift_stats.json
│   ├── multi_agent/
│   │   ├── agent_momentum_v1.pkl
│   │   ├── agent_rsi_v1.pkl
│   │   └── ensemble_weights.pkl
│   └── ...
│
├── queue/                       # Data queue (JSON files)
│   ├── BTCUSDT_20251119_120000.json
│   ├── ETHUSDT_20251119_120001.json
│   └── ...
│
├── logs/                        # Log dosyaları
│   ├── orchestrator.log
│   ├── data-collector.log
│   ├── rl-agent.log
│   └── ...
```

### Model Yükleme

Her worker başlatıldığında:
1. `models/<worker_name>/` klasörüne bakar
2. Eğer checkpoint varsa yükler (öğrenmeye kaldığı yerden devam)
3. Yoksa yeni model oluşturur

**Örnek:**
```python
# RL Agent başlatıldığında
if os.exists("models/rl_agent/BTCUSDT_q_table.pkl"):
    # Önceki eğitimden devam et
    q_table = pickle.load(open("models/rl_agent/BTCUSDT_q_table.pkl", "rb"))
    logger.info("✅ Loaded existing Q-table for BTCUSDT")
else:
    # Yeni başla
    q_table = {}
    logger.info("🆕 Created new Q-table for BTCUSDT")
```

---

## 🔄 PM2 YÖNETİMİ

### Temel Komutlar

```bash
# Tüm servisleri başlat
pm2 start ecosystem.config.js

# Tüm servisleri durdur
pm2 stop all

# Tüm servisleri yeniden başlat
pm2 restart all

# Belirli bir servisi yeniden başlat
pm2 restart ai-learning-orchestrator

# Servisleri sil
pm2 delete all

# Durumu kaydet (reboot sonrası otomatik başlat)
pm2 save

# Boot'ta otomatik başlat
pm2 startup

# Monitoring
pm2 monit

# Logları temizle
pm2 flush
```

### Belirli Worker'ları Başlat

```bash
# Sadece data collector
pm2 start ecosystem.config.js --only data-collector

# Sadece RL agent
pm2 start ecosystem.config.js --only rl-agent-worker

# Sadece API server
pm2 start ecosystem.config.js --only ai-learning-api
```

---

## 📈 PERFORMANS İZLEME

### 1. PM2 Dashboard

```bash
pm2 monit
```

Gösterir:
- CPU kullanımı
- Memory kullanımı
- Restart sayısı
- Uptime
- Real-time loglar

### 2. Log Analizi

```bash
# Tüm logları göster
pm2 logs

# Son 100 satır
pm2 logs --lines 100

# Sadece error'lar
pm2 logs --err

# Belirli bir worker
pm2 logs rl-agent-worker --lines 50
```

### 3. Sistem Metrikleri

```bash
# Worker istatistikleri
cat models/rl_agent/stats.json

# Data collector istatistikleri
grep "✅ Iteration" logs/data-collector.log | tail -10

# Toplanan veri sayısı
ls -1 queue/ | wc -l
```

---

## 🛠️ TROUBLESHOOTING

### Problem: Worker çalışmıyor

```bash
# 1. PM2 durumunu kontrol et
pm2 status

# 2. Worker loglarına bak
pm2 logs <worker-name> --err

# 3. Manuel başlat ve hata mesajını gör
cd /Users/sardag/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub
./venv/bin/python3 workers/rl_agent_worker.py

# 4. Worker'ı restart et
pm2 restart <worker-name>
```

### Problem: Queue dolmuyor

```bash
# 1. Data collector çalışıyor mu?
pm2 status data-collector

# 2. Data collector logları
pm2 logs data-collector

# 3. Manuel test
./venv/bin/python3 services/data_collector.py
```

### Problem: Binance API rate limit

```bash
# ecosystem.config.js içinde COLLECTION_INTERVAL'i artır
COLLECTION_INTERVAL: '120'  # 2 dakika (1 dakika yerine)
SYMBOLS_PER_BATCH: '25'     # 25 coin (50 yerine)

# Sonra restart
pm2 restart data-collector
```

### Problem: Memory kullanımı yüksek

```bash
# Max memory limit ayarla (ecosystem.config.js)
max_memory_restart: '1G'  # 1GB'den fazla olursa restart

# Veya manuel restart
pm2 restart all
```

---

## ✅ TEST & DOĞRULAMA

### 1. Data Collection Test

```bash
# 1 dakika bekle ve queue'yu kontrol et
sleep 60
ls -lh queue/ | head -20

# En az 50 dosya olmalı (50 coin/batch)
```

### 2. RL Agent Training Test

```bash
# 5 dakika bekle (training interval)
sleep 300

# Model checkpoint oluştu mu?
ls -lh models/rl_agent/

# Stats dosyası kontrol
cat models/rl_agent/stats.json
```

### 3. API Health Check

```bash
# Flask API çalışıyor mu?
curl http://localhost:5020/health

# System stats
curl http://localhost:5020/system/stats
```

### 4. End-to-End Test

```bash
# Tüm pipeline'ı test et
echo "🧪 Testing full pipeline..."

# 1. Data collector çalışıyor mu?
pm2 describe data-collector | grep "status"

# 2. Queue'da veri var mı?
queue_count=$(ls -1 queue/ 2>/dev/null | wc -l)
echo "📊 Queue size: $queue_count files"

# 3. Worker'lar işliyor mu?
pm2 logs rl-agent-worker --lines 5 --nostream

# 4. Model checkpoints var mı?
ls -R models/ | grep ".pkl" | wc -l
```

---

## 🎯 PRODUCTION DEPLOYMENT

### 1. System Startup'a Ekle

```bash
# PM2 startup script oluştur
pm2 startup

# Gösterilen komutu çalıştır (sudo ile)

# Mevcut servisleri kaydet
pm2 save
```

### 2. Auto-restart Politikası

Zaten ecosystem.config.js içinde ayarlı:
- `autorestart: true` - Crash olursa otomatik restart
- `max_memory_restart: '1G'` - 1GB'den fazla memory kullanırsa restart
- `cron_restart` - Bazı worker'lar düzenli restart (bakım için)

### 3. Log Rotation

```bash
# PM2 log rotation modülü yükle
pm2 install pm2-logrotate

# Ayarlar
pm2 set pm2-logrotate:max_size 100M        # Max 100MB per log
pm2 set pm2-logrotate:retain 10            # Son 10 log dosyasını tut
pm2 set pm2-logrotate:compress true        # Compress old logs
```

---

## 📞 DESTEK

**Dokümantasyon:**
- Mimari: `AI_ML_CONTINUOUS_LEARNING_ARCHITECTURE.md`
- UI Rehberi: `AI_LEARNING_HUB_UI_GUIDE.md`
- Implementation: `AI_LEARNING_HUB_COMPLETE.md`

**Kod Konumu:**
- Backend: `/45-backend/python-services/ai-learning-hub/`
- Frontend: `/src/app/ai-learning-hub/`
- API Routes: `/src/app/api/ai-learning/`

**Monitoring:**
- PM2 Dashboard: `pm2 monit`
- Web UI: `http://localhost:3000/ai-learning-hub`
- API Health: `http://localhost:5020/health`

---

**Created:** 2025-11-19
**Version:** 1.0
**Status:** Production Ready 🚀
