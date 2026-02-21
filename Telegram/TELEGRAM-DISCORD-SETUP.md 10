# 📱 TELEGRAM & DISCORD SETUP GUIDE

## 🤖 Telegram Bot Setup

### 1. BotFather ile Bot Oluştur

1. Telegram'da **@BotFather**'ı aç
2. `/newbot` komutunu gönder
3. Bot için bir isim seç (örn: "Borsa Trading Alert Bot")
4. Bot için benzersiz bir username seç (örn: "borsa_trading_alerts_bot")
5. BotFather sana bir **token** verecek, kaydet!

Örnek token: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

### 2. Chat ID Bul

#### Yöntem 1: @userinfobot kullan
1. Telegram'da **@userinfobot**'u aç
2. Bot'a `/start` gönder
3. Bot sana **Chat ID**'ni verecek

#### Yöntem 2: Manuel olarak bul
1. Telegram Web'de botunu aç: `https://web.telegram.org`
2. Bot'una bir mesaj gönder
3. Bu URL'yi ziyaret et:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
4. JSON'da `"chat":{"id":123456789}` kısmını bul

### 3. .env'e Ekle

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```

### 4. Test Et

```bash
curl -X POST "https://api.telegram.org/bot<YOUR_TOKEN>/sendMessage" \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "<YOUR_CHAT_ID>", "text": "Test mesajı!"}'
```

---

## 💬 Discord Webhook Setup

### 1. Webhook Oluştur

1. Discord sunucunda **Server Settings** > **Integrations** > **Webhooks**'a git
2. **New Webhook** tıkla
3. Webhook için bir isim ver (örn: "Trading Alerts")
4. Hangi kanala mesaj göndereceğini seç
5. **Copy Webhook URL** tıkla

Örnek URL: `https://discord.com/api/webhooks/123456789/ABCdefGHIjklMNOpqrsTUVwxyz`

### 2. .env'e Ekle

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/123456789/ABCdefGHIjklMNOpqrsTUVwxyz
```

### 3. Test Et

```bash
curl -X POST "YOUR_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test mesajı!"}'
```

---

## ✅ Alert System Test

API üzerinden alert test et:

```bash
# Emergency stop alert gönder (CRITICAL)
curl -X POST http://localhost:3000/api/monitoring/live \
  -H 'Content-Type: application/json' \
  -d '{"action":"emergency_stop"}'
```

Bu komut şu kanallara alert gönderecek:
- ✅ Email (hazırsa)
- ✅ SMS (hazırsa)
- ✅ Telegram
- ✅ Discord (hazırsa)
- ✅ Azure Event Hub

---

## 📊 Alert Severity Levels

| Severity | Telegram | Discord | Email | SMS | Azure |
|----------|----------|---------|-------|-----|-------|
| CRITICAL | ✅       | ✅      | ✅    | ✅  | ✅    |
| HIGH     | ✅       | ✅      | ✅    | ❌  | ✅    |
| MEDIUM   | ✅       | ✅      | ❌    | ❌  | ✅    |
| LOW      | ❌       | ❌      | ❌    | ❌  | ✅    |

---

## 🎯 Örnek Alert Mesajları

### Telegram
```
🚨 Emergency Stop Activated

Emergency stop has been triggered. All positions closed.

2025-10-03 15:30:45
```

### Discord
```json
{
  "embeds": [{
    "title": "⚠️ Maximum Drawdown Warning",
    "description": "Current drawdown: 18% (Max: 20%)",
    "color": 16753920,
    "timestamp": "2025-10-03T15:30:45Z",
    "footer": {
      "text": "Severity: HIGH"
    }
  }]
}
```

---

## 🔧 Troubleshooting

### Telegram Bot yanıt vermiyor
- Bot'un `/start` ile başlatıldığından emin ol
- Chat ID'nin doğru olduğunu kontrol et
- Token'ın güncel olduğunu doğrula

### Discord webhook çalışmıyor
- Webhook URL'sinin doğru kopyalandığını kontrol et
- Discord sunucusunda webhooks izninin olduğunu doğrula
- Webhook'un silinmediğinden emin ol

### Alerts gelmiyor
- `.env` dosyasının yüklendiğini kontrol et
- Server'ı restart et
- Console log'larını kontrol et: `console.log('[TELEGRAM] Alert:', ...)`

---

**🎉 Setup tamamlandıktan sonra:**
- Dev server'ı restart et
- Live monitor'dan test alertleri gönder
- Telegram ve Discord'da mesajları kontrol et

**Next:** WebSocket Client + Historical Charts ekleyelim!
