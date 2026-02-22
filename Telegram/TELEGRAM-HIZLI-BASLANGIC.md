# ⚡ TELEGRAM BOT HIZLI BAŞLANGIÇ

**5 dakikada Telegram botunuzu aktif edin!**

---

## 1️⃣ Vercel Environment Variables (2 dakika)

Vercel Dashboard → Settings → Environment Variables

```env
# Telegram Bot Token (BotFather'dan aldığınız)
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# Webhook Secret (kendiniz belirleyin, min 32 karakter)
TELEGRAM_BOT_WEBHOOK_SECRET=super-gizli-webhook-anahtari-buraya-32-karakter-olmali

# App URL (zaten var)
NEXT_PUBLIC_APP_URL=https://lydian.app
```

**✅ Tamamlandı!** Vercel otomatik redeploy yapacak.

---

## 2️⃣ Webhook Kurulumu (1 dakika)

Terminalde çalıştırın:

```bash
# Token ve Secret'ınızı buraya yazın
BOT_TOKEN="1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
WEBHOOK_SECRET="super-gizli-webhook-anahtari-buraya"

# Webhook'u ayarla
curl -X POST "https://api.telegram.org/bot${BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d "{
    \"url\": \"https://lydian.app/api/telegram/webhook\",
    \"secret_token\": \"${WEBHOOK_SECRET}\"
  }"
```

**Başarılı yanıt:**
```json
{
  "ok": true,
  "result": true,
  "description": "Webhook was set"
}
```

---

## 3️⃣ Test Etme (2 dakika)

### a) Telegram'dan Test

1. **Bot'u bulun:** Telegram'da `@YourBotName` araması yapın
2. **Start gönderin:** `/start` yazın
3. **Hoşgeldin mesajı geldi mi?** ✅

Gelmediyse:
```bash
# Webhook durumunu kontrol edin
curl "https://api.telegram.org/bot${BOT_TOKEN}/getWebhookInfo"
```

### b) Test Bildirimi Gönder

```bash
# Simple test
curl -X POST "https://lydian.app/api/telegram/test" \
  -H "Content-Type: application/json" \
  -d '{"type": "simple"}'

# STRONG_BUY test (premium format)
curl -X POST "https://lydian.app/api/telegram/test" \
  -H "Content-Type: application/json" \
  -d '{"type": "strong_buy"}'
```

**Telegram'da premium bildirim geldi mi?** ✅

### c) Admin Paneli

```bash
# Sistem durumunu kontrol et
curl "https://lydian.app/api/telegram/admin"
```

Yanıt:
```json
{
  "status": "active",
  "config": {
    "signalTypes": ["STRONG_BUY", "BUY", "SELL", "WAIT"],
    "minConfidence": 70,
    "mode": "realtime"
  },
  "stats": {
    "subscriberCount": 1
  }
}
```

---

## 4️⃣ Otomatik Sinyaller Aktif! 🎉

Artık sistem otomatik çalışıyor:

### Nasıl Çalışıyor?

```
Strategy Aggregator
      ↓
   Sinyal Analizi
      ↓
   Filtreleme (STRONG_BUY, BUY, SELL, WAIT | %70+)
      ↓
   Spam Kontrolü (5 dk/sembol)
      ↓
   Premium Format
      ↓
   Telegram Bildirimi → Tüm Aboneler
```

### Hangi Sinyaller Gönderilir?

✅ **Sinyal Tipleri:** STRONG_BUY, BUY, SELL, WAIT
✅ **Minimum Confidence:** %70+
✅ **Piyasalar:** 600+ crypto + traditional markets
✅ **Stratejiler:** Tüm 16 strateji + TA-Lib Pro
✅ **Mod:** Real-time (anlık)
✅ **Spam Önleme:** Sembol başına 5 dakikada bir

### Günlük Özet

- **Saat:** 09:00 ve 18:00
- **Format:** Premium tasarım
- **İçerik:** Toplam sinyal + top 5 fırsat

---

## 🎨 Premium Format Özellikleri

### Her Bildirimde:

✨ **Unicode Art Tasarım**
```
╔═══════════════════╗
┃ ⚡ GÜÇLÜ ALIM FIRSATI! ⚡ ┃
╠═══════════════════╣
```

🌈 **Renk Paleti**
- STRONG_BUY: 🟢🟢 Yeşil + ⚡
- BUY: 🟢 Yeşil + ✨
- SELL: 🔴 Kırmızı + ⚠️
- WAIT: 🟨 Sarı + ⏸️

📊 **Görsel Confidence**
```
🎯 Güven Skoru: 92% ⭐⭐⭐⭐⭐
   ▓▓▓▓▓▓▓▓▓░ 🔥 ÇOK YÜKSEK
```

🏷️ **Piyasa Tipi**
- ₿ CRYPTO
- 💱 FOREX
- 📊 INDEX
- 🏆 COMMODITY
- 📈 STOCK

📈 **Top 3 Strateji**
```
1. MA Crossover (95%): Golden cross detected
2. RSI Divergence (90%): Bullish divergence confirmed
3. Volume Breakout (88%): High volume breakout
```

---

## 🔧 Hızlı Sorun Giderme

### ❌ Bildirim Gelmiyor

**1. Abone olduğunuzdan emin olun:**
```
Telegram'da /start gönderin
```

**2. Webhook durumunu kontrol edin:**
```bash
curl "https://api.telegram.org/bot${BOT_TOKEN}/getWebhookInfo"
```

Yanıt `"url": "https://lydian.app/api/telegram/webhook"` içermeli.

**3. Vercel environment variables kontrol:**
- TELEGRAM_BOT_TOKEN doğru mu?
- TELEGRAM_BOT_WEBHOOK_SECRET doğru mu?

**4. Test bildirimi gönderin:**
```bash
curl -X POST "https://lydian.app/api/telegram/test" \
  -H "Content-Type: application/json" \
  -d '{"type": "simple"}'
```

### ❌ Webhook Hatası

**Webhook'u sıfırlayın:**
```bash
# 1. Webhook'u kaldır
curl -X POST "https://api.telegram.org/bot${BOT_TOKEN}/deleteWebhook"

# 2. Yeniden kur (20 saniye bekleyin)
curl -X POST "https://api.telegram.org/bot${BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d "{
    \"url\": \"https://lydian.app/api/telegram/webhook\",
    \"secret_token\": \"${WEBHOOK_SECRET}\"
  }"
```

### ❌ "No subscribers" Hatası

```
Telegram'da bot'a /start gönderdikten sonra tekrar deneyin
```

---

## 📱 Telegram Bot Komutları

```
/start       Bota abone ol
/signals     Son sinyalleri görüntüle
/price BTCUSDT  Fiyat sorgula
/help        Yardım menüsü
/stop        Abonelikten çık
```

---

## 🎯 Sonraki Adımlar (Opsiyonel)

### Redis Storage (Persistent subscribers)

Vercel KV veya Upstash Redis ekleyin:

```bash
# Vercel KV kurulum
vercel env add UPSTASH_REDIS_URL
vercel env add UPSTASH_REDIS_TOKEN
```

### Cron Job (Günlük özet)

`vercel.json`:
```json
{
  "crons": [{
    "path": "/api/cron/telegram-daily-summary",
    "schedule": "0 9,18 * * *"
  }]
}
```

### Özelleştirme

Config dosyasını düzenleyin:
```typescript
// src/lib/telegram/config.ts

export const TELEGRAM_CONFIG = {
  enabledSignalTypes: ['STRONG_BUY', 'BUY'],  // Sadece alım sinyalleri
  minConfidence: 80,  // Min %80 confidence
  symbolWhitelist: ['BTCUSDT', 'ETHUSDT'],  // Sadece BTC ve ETH
  // ...
};
```

---

## ✅ Checklist

- [ ] Vercel'de TELEGRAM_BOT_TOKEN eklendi
- [ ] Vercel'de TELEGRAM_BOT_WEBHOOK_SECRET eklendi
- [ ] Webhook kuruldu (`setWebhook` çalıştırıldı)
- [ ] Telegram'da bot'a /start gönderildi
- [ ] Test bildirimi başarılı (premium format geldi)
- [ ] Admin paneli çalışıyor
- [ ] Otomatik sinyaller aktif

---

## 📞 Hızlı Test Komutları

```bash
# Tüm testleri tek seferde çalıştır
BOT_TOKEN="YOUR_TOKEN_HERE"

# 1. Webhook durumu
curl "https://api.telegram.org/bot${BOT_TOKEN}/getWebhookInfo"

# 2. Admin paneli
curl "https://lydian.app/api/telegram/admin"

# 3. Simple test
curl -X POST "https://lydian.app/api/telegram/test" \
  -H "Content-Type: application/json" \
  -d '{"type": "simple"}'

# 4. Strong buy test
curl -X POST "https://lydian.app/api/telegram/test" \
  -H "Content-Type: application/json" \
  -d '{"type": "strong_buy"}'

# 5. Sell test
curl -X POST "https://lydian.app/api/telegram/test" \
  -H "Content-Type: application/json" \
  -d '{"type": "sell"}'

# 6. Wait test
curl -X POST "https://lydian.app/api/telegram/test" \
  -H "Content-Type: application/json" \
  -d '{"type": "wait"}'
```

---

## 🎉 Tamamlandı!

✅ Telegram bot aktif
✅ Premium bildirimler çalışıyor
✅ Otomatik sinyal sistemi aktif
✅ 600+ coin + traditional markets desteği
✅ 0 hata, production-ready

**Artık sistem tamamen otomatik çalışıyor!** 🚀

Detaylı bilgi için: `TELEGRAM-PREMIUM-COMPLETE-TR.md`
