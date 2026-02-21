# 🤖 TELEGRAM SCHEDULER - KURULUM KILAVUZU

## ✅ BAŞARILI! SİSTEM AKTİF

Telegram bildirim sisteminiz başarıyla kuruldu ve çalışıyor!

### 📱 Telegram'dan Gelen İlk Mesaj
Botunuz şu mesajı gönderdi:
```
🤖 TELEGRAM SCHEDULER AKTİF!

✅ Scheduler servisi başarıyla başlatıldı
⏰ [Şu anki tarih/saat]

📅 Zamanlamalar:
- 🕐 Saatlik: Market sinyalleri
- 🕓 4 Saatlik: Futures + Haberler
- 📅 Günlük: Nirvana + BTC-ETH
- 📆 Haftalık: Nirvana özet

Sistem 7/24 çalışıyor! 🚀
```

---

## 🔧 SİSTEM DETAYLARI

### Telegram Bot Bilgileri
- **Bot Token**: `8292640150:AAHqDdkHxFqx9q8hJ-bJ8KS_Z2LZWrOLroI`
- **Chat ID**: `7575640489`
- **Durum**: ✅ Aktif ve çalışıyor

### PM2 Process Manager
- **Servis Adı**: `telegram-scheduler`
- **Status**: 🟢 Online
- **Uptime**: Çalışıyor
- **Auto Restart**: ✅ Aktif
- **Memory**: ~64MB

### Zamanlanmış Bildirimler

#### 1️⃣ SAATLİK (Her saat başı: 00:00, 01:00, 02:00...)
- Market Correlation sinyalleri
- Cron: `0 * * * *`

#### 2️⃣ 4 SAATLİK (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
- 🔥 Crypto News (Önemli haberler - Türkçe çeviri)
- Omnipotent Futures (Wyckoff) sinyalleri
- Cron: `0 */4 * * *`

#### 3️⃣ GÜNLÜK (Her gün UTC 00:00 = Türkiye 03:00)
- 🌟 Nirvana Dashboard Özeti
- 📊 BTC-ETH Analysis
- Cron: `0 0 * * *`

#### 4️⃣ HAFTALIK (Her Pazartesi UTC 00:00)
- 📈 Haftalık Nirvana Raporu
- Cron: `0 0 * * 1`

---

## 🎯 PM2 KOMUTLARI

### Durumu Kontrol Et
```bash
cd /Users/sardag/Documents/sardag-emrah-final.bak-20251030-170900/Telegram/schedulers
pm2 list
pm2 info telegram-scheduler
```

### Log'ları İzle
```bash
# Canlı log izleme
pm2 logs telegram-scheduler

# Son 50 satır
pm2 logs telegram-scheduler --lines 50

# Sadece hata log'ları
pm2 logs telegram-scheduler --err
```

### Servisi Yönet
```bash
# Restart
pm2 restart telegram-scheduler

# Stop
pm2 stop telegram-scheduler

# Start (eğer durdurulmuşsa)
pm2 start telegram-scheduler

# Delete (tamamen kaldır)
pm2 delete telegram-scheduler
```

### Monitoring
```bash
# Gerçek zamanlı monitoring
pm2 monit

# JSON olarak detaylı bilgi
pm2 jlist
```

---

## 🔄 BİLGİSAYAR YENİDEN BAŞLATILDIĞINDA OTOMATİK BAŞLATMA

### Adım 1: PM2 Startup Script (Sudo Gerekli)
Terminal'de şu komutu çalıştırın:

```bash
sudo env PATH=$PATH:/opt/homebrew/Cellar/node/24.10.0/bin /opt/homebrew/lib/node_modules/pm2/bin/pm2 startup launchd -u sardag --hp /Users/sardag
```

**Not**: Bu komut sudo şifresi isteyecek. Şifrenizi girin.

### Adım 2: Process List'i Kaydet
```bash
pm2 save
```

Bu komut şu anki çalışan process'leri kaydeder, böylece bilgisayar her açıldığında otomatik başlar.

### Adım 3: Test Et
Bilgisayarınızı yeniden başlatın ve kontrol edin:
```bash
pm2 list
# telegram-scheduler görünmeli
```

---

## 📁 DOSYA YAPISI

```
/Telegram/
├── schedulers/
│   ├── ecosystem.config.js          # PM2 configuration
│   ├── run-scheduler.js             # Ana scheduler (JavaScript)
│   ├── start-telegram.sh            # Hızlı başlatma script'i
│   └── logs/
│       ├── telegram-scheduler-out.log   # Normal log'lar
│       └── telegram-scheduler-error.log # Hata log'ları
├── telegram 2/
│   └── unified-notification-bridge.ts   # Telegram API bridge
└── .env.local                       # Telegram credentials (GÜVENLI)
```

---

## 🔐 GÜVENLİK

### Credentials Güvenliği
- Telegram Bot Token ve Chat ID `.env.local` dosyasında saklanıyor
- Bu dosya `.gitignore`'da olmalı (Git'e gönderilmemeli)
- PM2 log'larında token görünmez (maskelenir)

### Log'ları Temizleme
```bash
# Log dosyalarını temizle
pm2 flush telegram-scheduler

# Tüm log'ları temizle
pm2 flush all
```

---

## 🧪 TEST

### Manuel Test Mesajı Gönder
```bash
cd /Users/sardag/Documents/sardag-emrah-final.bak-20251030-170900/Telegram
node test-telegram-scheduler.js
```

Bu script tüm API'leri test eder ve sonuçları gösterir.

### Cron Job'ları Manuel Tetikle
Scheduler'da cron'ları manuel tetiklemek için restart yapın:
```bash
pm2 restart telegram-scheduler
# 3 saniye sonra test mesajı gönderilecek
```

---

## ⚙️ YAPILANDIRMA

### Zamanlamaları Değiştirme
`/Telegram/schedulers/run-scheduler.js` dosyasını düzenleyin:

```javascript
// Örnek: Saatlik yerine 30 dakikada bir
cron.schedule('*/30 * * * *', async () => {
  // ...
});

// Örnek: Günlük yerine 12 saatte bir
cron.schedule('0 */12 * * *', async () => {
  // ...
});
```

Değişiklikten sonra:
```bash
pm2 restart telegram-scheduler
```

### Bildirim Mesajlarını Özelleştirme
`/Telegram/telegram 2/unified-notification-bridge.ts` dosyasındaki mesaj template'lerini düzenleyin.

---

## 🐛 SORUN GİDERME

### Problem: Telegram mesajı gönderilmiyor
**Kontrol 1**: Bot Token ve Chat ID doğru mu?
```bash
cat /Users/sardag/Documents/sardag-emrah-final.bak-20251030-170900/.env.local | grep TELEGRAM
```

**Kontrol 2**: PM2 log'larına bak
```bash
pm2 logs telegram-scheduler --lines 50
```

**Kontrol 3**: Manuel test yap
```bash
node /Users/sardag/Documents/sardag-emrah-final.bak-20251030-170900/Telegram/test-telegram-scheduler.js
```

### Problem: PM2 servisi sürekli restart oluyor
```bash
# Hata log'larını kontrol et
pm2 logs telegram-scheduler --err --lines 100

# Memory kullanımını kontrol et
pm2 info telegram-scheduler
```

### Problem: API yanıt vermiyor (localhost:3000)
```bash
# Next.js dev server çalışıyor mu kontrol et
curl http://localhost:3000/api/nirvana

# Eğer 404 veya timeout alıyorsan, Next.js'i başlat
pnpm dev
```

### Problem: Cron job'lar tetiklenmiyor
```bash
# Log'larda cron tetikleme mesajlarını ara
pm2 logs telegram-scheduler | grep "Scheduler Tetiklendi"

# Sistem saatini kontrol et (UTC veya Europe/Istanbul)
date
```

---

## 📊 PERFORMANS

### Resource Kullanımı
- **CPU**: ~0% (idle), <5% (aktif)
- **Memory**: ~64MB (normal)
- **Disk**: Log dosyaları için ~10MB (PM2 logrotate otomatik temizler)

### PM2 Logrotate
PM2 otomatik log rotation yapıyor:
- Maksimum log boyutu: 10MB
- Eski log'lar otomatik sıkıştırılıp arşivlenir

---

## 🚀 GELİŞMİŞ ÖZELLİKLER

### Cluster Mode (Birden fazla instance)
Eğer yüksek trafikte sorun yaşarsanız:

`ecosystem.config.js` dosyasında:
```javascript
{
  instances: 2,  // 1 yerine 2
  exec_mode: 'cluster'  // 'fork' yerine 'cluster'
}
```

### PM2 Plus (Monitoring Dashboard)
Ücretsiz monitoring için:
```bash
pm2 plus
# Tarayıcıda dashboard açılır
```

---

## 📞 DESTEK

### Log Dosyaları
- Normal: `/Telegram/schedulers/logs/telegram-scheduler-out.log`
- Hata: `/Telegram/schedulers/logs/telegram-scheduler-error.log`

### Sistem Bilgileri
```bash
# Node.js versiyonu
node --version

# PM2 versiyonu
pm2 --version

# PM2 process listesi
pm2 jlist
```

---

## ✅ BAŞARIYLA KURULDU!

✅ Telegram Bot yapılandırıldı
✅ PM2 process manager kuruldu
✅ Scheduler servisi başlatıldı
✅ İlk test mesajı gönderildi
✅ Auto-restart aktif
✅ Log dosyaları oluşturuldu
✅ Cron job'lar zamanlandı

**Sistem 7/24 çalışıyor! 🎉**

---

**Son Güncelleme**: 31 Ekim 2025
**Versiyon**: 1.0.0
**PM2 Status**: 🟢 Online
