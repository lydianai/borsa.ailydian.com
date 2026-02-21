# 🤖 TELEGRAM ENTİRE KURULUM REHBERİ

## ✅ TAMAMLANAN İŞLER

### 1. **API Entegrasyonları** ✅
- ✅ Nirvana Dashboard API test edildi - 5 strategies, 50 signals
- ✅ Omnipotent Futures (Wyckoff) API test edildi
- ✅ BTC-ETH Analysis API test edildi - 0.958 correlation
- ✅ Market Correlation API test edildi - 10 correlations
- ✅ Crypto News API düzeltildi - Groq fallback eklendi

### 2. **Telegram Bridge Fonksiyonları** ✅
**Dosya**: `/Telegram/telegram 2/unified-notification-bridge.ts`

5 yeni Türkçe bridge fonksiyonu eklendi:

1. **notifyNirvanaOverview()** - Nirvana dashboard günlük özet
2. **notifyOmnipotentFuturesSignal()** - Wyckoff phase signals
3. **notifyBTCETHAnalysis()** - BTC-ETH korelasyon analizi
4. **notifyMarketCorrelationDetail()** - Detaylı market correlation
5. **notifyCryptoNews()** - Kripto haberleri (Türkçe)

### 3. **Scheduler Sistemi** ✅
**Dosyalar**:
- `/Telegram/schedulers/telegram-signal-scheduler.ts`
- `/Telegram/schedulers/cron-service.ts`

**Zaman Dilimleri**:
- 🕐 **Saatlik** (Her saat başı): Market Correlation yüksek confidence sinyalleri
- 🕓 **4 Saatlik** (00:00, 04:00, 08:00, 12:00, 16:00, 20:00): Omnipotent Futures + Crypto News
- 📅 **Günlük** (UTC 00:00 / TR 03:00): Nirvana Dashboard + BTC-ETH Analysis + News Özeti
- 📆 **Haftalık** (Pazartesi UTC 00:00): Nirvana haftalık özet

### 4. **PM2 Configuration** ✅
**Dosya**: `/Telegram/schedulers/ecosystem.config.js`

- ✅ 7/24 otomatik çalışma
- ✅ Bilgisayar restart'ta otomatik başlatma
- ✅ Hata yönetimi ve auto-restart
- ✅ Log dosyaları
- ✅ Memory limit (500MB)

---

## 📦 KURULUM ADIMLARI

### Adım 1: Dependencies Yükleme

```bash
cd /Users/sardag/Documents/sardag-emrah-final.bak-20251030-170900

# Node-cron ve dependencies
pnpm add node-cron @types/node-cron ts-node -D

# PM2 global install (eğer yoksa)
npm install -g pm2
```

### Adım 2: Environment Variables

`.env.local` dosyasında şunlar olmalı:

```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ALLOWED_CHAT_IDS=your_chat_id_here

# Groq AI (optional - fallback var)
GROQ_API_KEY=your_groq_key_here

# CryptoPanic (optional - mock data var)
CRYPTOPANIC_API_KEY=your_cryptopanic_key_here

# App URL
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

**Telegram Bot Token Alma**:
1. Telegram'da @BotFather'ı aç
2. `/newbot` komutunu gönder
3. Bot adını ve username'ini belirle
4. Alınan token'ı `TELEGRAM_BOT_TOKEN` olarak kaydet

**Chat ID Alma**:
1. Bot'u Telegram'da başlat
2. @userinfobot'u aç
3. Chat ID'ni kopyala
4. `TELEGRAM_ALLOWED_CHAT_IDS` olarak kaydet

### Adım 3: PM2 Servislerini Başlatma

```bash
cd /Users/sardag/Documents/sardag-emrah-final.bak-20251030-170900/Telegram/schedulers

# Servisleri başlat
pm2 start ecosystem.config.js

# Otomatik başlatma için (bilgisayar restart'ta)
pm2 startup
pm2 save

# Durumu kontrol
pm2 list
pm2 monit
```

### Adım 4: Log'ları İzleme

```bash
# Tüm log'lar
pm2 logs

# Sadece scheduler
pm2 logs telegram-scheduler

# Sadece bot
pm2 logs telegram-bot

# Error log'ları
pm2 logs telegram-scheduler --err
```

---

## 🧪 TEST

### Manuel Test (Scheduler'sız)

```bash
cd /Users/sardag/Documents/sardag-emrah-final.bak-20251030-170900/Telegram/schedulers

# Test scheduler fonksiyonunu çalıştır
ts-node -e "import('./telegram-signal-scheduler').then(m => m.testAllSchedulers())"
```

### Tek Tek Test

```bash
# Nirvana günlük özet
ts-node -e "import('./telegram-signal-scheduler').then(m => m.sendNirvanaDaily())"

# Omnipotent Futures
ts-node -e "import('./telegram-signal-scheduler').then(m => m.sendOmnipotentFuturesSignals())"

# BTC-ETH Analysis
ts-node -e "import('./telegram-signal-scheduler').then(m => m.sendBTCETHDaily())"

# Market Correlation
ts-node -e "import('./telegram-signal-scheduler').then(m => m.sendMarketCorrelationSignals())"

# Crypto News
ts-node -e "import('./telegram-signal-scheduler').then(m => m.sendCryptoNews())"
```

---

## 📊 TELEGRAM MESAJ ÖRNEKLERİ

### Nirvana Dashboard
```
╭━━━━━━━━━━━━━━━━━━━━╮
┃ 🌟 NİRVANA ÖZET 🌟
├━━━━━━━━━━━━━━━━━━━━┤
┃ 📊 Aktif Strateji: 5/8
┃ 🎯 Toplam Sinyal: 50
┃ 🟢 YÜKSELİŞ (25)
├━━━━━━━━━━━━━━━━━━━━┤
┃ 🔝 EN İYİ FIRSATLAR:
┃ 🟢 BTCUSDT
┃   Unified (5 strategies)
┃   Güven: %85
├━━━━━━━━━━━━━━━━━━━━┤
┃ ⌚ 11:30
╰━━━━━━━━━━━━━━━━━━━━╯
```

### Wyckoff Analizi
```
🟢 YENİ BUY SİNYALİ ↗

📊 Sembol: BTCUSDT
💰 Fiyat: $109835.90
🎯 Güven: %85 ⭐⭐⭐⭐
⚙️ 🎯 Wyckoff Analizi
⏰ 31.10.2025 11:30

📝 En Güçlü Stratejiler:
🟢 TOPLAMA
Omnipotent Skor: 85/100
Hacim: HIGH

⚠️ Eğitim amaçlıdır, finansal tavsiye değildir.
```

### BTC-ETH Korelasyon
```
╭━━━━━━━━━━━━━━━━━━━━╮
┃ 🔗 BTC-ETH ANALİZ 🔗
├━━━━━━━━━━━━━━━━━━━━┤
┃ 📊 30 Günlük Korelasyon: %95.8
┃ 📊 STABIL
├━━━━━━━━━━━━━━━━━━━━┤
┃ ⌚ 11:30
╰━━━━━━━━━━━━━━━━━━━━╯
```

---

## 🔧 YÖNETİM KOMUTLARI

### PM2 Komutları

```bash
# Tüm servisleri restart
pm2 restart all

# Sadece scheduler'ı restart
pm2 restart telegram-scheduler

# Servisleri durdur
pm2 stop telegram-scheduler
pm2 stop telegram-bot

# Servisleri sil
pm2 delete telegram-scheduler
pm2 delete telegram-bot

# Log'ları temizle
pm2 flush

# Monitoring
pm2 monit
```

### Scheduler Manuel Çalıştırma

Eğer PM2 kullanmak istemezseniz:

```bash
cd /Users/sardag/Documents/sardag-emrah-final.bak-20251030-170900/Telegram/schedulers

# Manuel başlat
ts-node cron-service.ts

# Background'da başlat
ts-node cron-service.ts > logs/cron.log 2>&1 &
```

---

## ⚠️ ÖNEMLİ NOTLAR

1. **Internet Bağlantısı**: Scheduler'ın çalışması için sürekli internet gerekli
2. **Next.js Dev Server**: API endpoint'leri için `pnpm dev` çalışıyor olmalı
3. **Ta-Lib Service**: Python Flask servisi (port 5005) çalışıyor olmalı
4. **Environment Variables**: `.env.local` dosyası her zaman güncel olmalı
5. **Spam Önleme**: Her mesaj arası 500ms-1s bekleme var
6. **Rate Limiting**: Telegram API limitleri: 30 mesaj/saniye, 20 mesaj/dakika grup başına

---

## 🎯 SINYAL FİLTRELEME

### Omnipotent Futures
- Minimum Confidence: %75
- Signal Type: BUY veya SELL (WAIT hariç)
- Maksimum 5 sinyal / 4 saatte

### Market Correlation
- Minimum Confidence: %80
- Minimum Omnipotent Score: 85/100
- Maksimum 3 sinyal / saatte

### Crypto News
- Minimum Impact Score: 8/10
- Otomatik Türkçe çeviri (Groq fallback ile)

---

## 📁 DOSYA YAPISI

```
/Telegram
├── telegram 2/
│   ├── bot.ts                           # Ana Telegram bot
│   ├── config.ts                        # Bildirim konfigürasyonu
│   ├── notifications.ts                 # Bildirim servisi
│   ├── unified-notification-bridge.ts   # ✨ 5 YENİ FONKSİYON
│   ├── premium-formatter.ts             # Mesaj formatları
│   └── ...
├── schedulers/                          # ✨ YENİ
│   ├── telegram-signal-scheduler.ts     # Scheduler fonksiyonları
│   ├── cron-service.ts                  # Cron servisi
│   ├── ecosystem.config.js              # PM2 config
│   └── logs/                            # Log dosyaları
└── TELEGRAM-SETUP-GUIDE-TR.md           # Bu dosya
```

---

## 🆘 SORUN GİDERME

### Problem: Telegram mesaj gitmiyor
**Çözüm**:
1. `TELEGRAM_BOT_TOKEN` ve `TELEGRAM_ALLOWED_CHAT_IDS` kontrol et
2. Bot'u Telegram'da başlattığından emin ol (/start)
3. Log'ları kontrol et: `pm2 logs telegram-scheduler --err`

### Problem: API hatası alıyorum
**Çözüm**:
1. Next.js dev server çalışıyor mu? `pnpm dev`
2. Ta-Lib service çalışıyor mu? `curl http://localhost:5005/health`
3. API endpoint'leri test et: `curl http://localhost:3000/api/nirvana`

### Problem: Scheduler çalışmıyor
**Çözüm**:
1. PM2 durumu: `pm2 list`
2. Cron syntax kontrol: `pm2 logs telegram-scheduler`
3. Manuel test: `ts-node -e "import('./telegram-signal-scheduler').then(m => m.testAllSchedulers())"`

### Problem: node-cron install hatası
**Çözüm**:
1. İnternet bağlantısını kontrol et
2. NPM registry'yi kontrol et: `npm config get registry`
3. Alternatif: `npm install node-cron @types/node-cron ts-node --save-dev`

---

## ✅ KONTROL LİSTESİ

Kurulumdan önce:
- [ ] Telegram Bot oluşturuldu (BotFather)
- [ ] Chat ID alındı (userinfobot)
- [ ] .env.local dosyası güncellendi
- [ ] Next.js dev server çalışıyor (`pnpm dev`)
- [ ] Ta-Lib service çalışıyor
- [ ] node-cron ve dependencies yüklendi
- [ ] PM2 global olarak yüklendi

Kurulumdan sonra:
- [ ] PM2 servisleri başlatıldı
- [ ] `pm2 list` komutu çalışıyor
- [ ] `pm2 startup` ve `pm2 save` yapıldı
- [ ] Log'lar kontrol edildi (`pm2 logs`)
- [ ] Manuel test yapıldı
- [ ] İlk Telegram mesajı alındı

---

## 🚀 HIZLI BAŞLANGIÇ (Özet)

```bash
# 1. Dependencies
pnpm add node-cron @types/node-cron ts-node -D
npm install -g pm2

# 2. Environment (.env.local)
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_ALLOWED_CHAT_IDS=123456789

# 3. PM2 Başlat
cd /Users/sardag/Documents/sardag-emrah-final.bak-20251030-170900/Telegram/schedulers
pm2 start ecosystem.config.js
pm2 startup
pm2 save

# 4. Kontrol
pm2 list
pm2 logs telegram-scheduler
```

---

**🎉 KURULUM TAMAMLANDI!**

Artık Telegram botunuz 7/24 otomatik olarak sinyal bildirimleri gönderecek! 🤖📊✨
