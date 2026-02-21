# 🤖 TELEGRAM BOT SETUP - ADIM ADIM REHBERİ

## 📋 Gerekli Bilgiler
- Telegram Bot Token (BotFather'dan alınacak)
- Telegram Chat ID (kendi kullanıcı ID'niz)

---

## 1️⃣ TELEGRAM BOT OLUŞTURMA

### Adım 1: BotFather'ı Aç
1. Telegram uygulamasını aç
2. Arama çubuğuna **@BotFather** yaz
3. Resmi BotFather botunu aç (mavi tik işaretli)

### Adım 2: Yeni Bot Oluştur
1. `/newbot` komutunu gönder
2. Bot için bir **isim** belirle (örnek: "Borsa Trading Alert Bot")
3. Bot için benzersiz bir **username** belirle (örnek: "borsa_trading_alerts_bot")
   - Username **mutlaka** "bot" ile bitmeli
   - Sadece harf, rakam ve alt çizgi içermeli

### Adım 3: Token'ı Kaydet
BotFather şuna benzer bir mesaj gönderecek:
```
Done! Congratulations on your new bot.
You will find it at t.me/borsa_trading_alerts_bot

Use this token to access the HTTP API:
7891234567:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw

For a description of the Bot API, see this page: https://core.telegram.org/bots/api
```

**Bu token'ı kopyala ve kaydet!** (örnek: `7891234567:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw`)

---

## 2️⃣ CHAT ID BULMA

### Yöntem 1: @userinfobot (EN KOLAY)
1. Telegram'da **@userinfobot** ara
2. Botu aç ve `/start` gönder
3. Bot sana **Chat ID**'ni verecek (örnek: `123456789`)

### Yöntem 2: getUpdates API (Manuel)
1. Önce kendi botuna bir mesaj gönder (örnek: "test")
2. Tarayıcıda bu URL'yi aç (TOKEN yerine kendi token'ını yaz):
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
3. JSON response'da `"chat":{"id":123456789}` kısmını bul
4. Bu sayı senin Chat ID'n

---

## 3️⃣ .ENV DOSYASINA EKLEME

1. Projenin root klasöründeki `.env` dosyasını aç
2. Şu satırları bul:
   ```bash
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   TELEGRAM_CHAT_ID=your_telegram_chat_id_here
   ```

3. Gerçek değerlerle değiştir:
   ```bash
   TELEGRAM_BOT_TOKEN=7891234567:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw
   TELEGRAM_CHAT_ID=123456789
   ```

4. Dosyayı kaydet

---

## 4️⃣ TEST ETME

### Terminal Test (Quick)
```bash
curl -X POST "https://api.telegram.org/bot<TOKEN>/sendMessage" \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "<CHAT_ID>", "text": "🚀 Test mesajı! Bot çalışıyor!"}'
```

**TOKEN** ve **CHAT_ID** yerine kendi değerlerini yaz.

### Başarılı Response:
```json
{
  "ok": true,
  "result": {
    "message_id": 123,
    "from": {...},
    "chat": {...},
    "text": "🚀 Test mesajı! Bot çalışıyor!"
  }
}
```

Telegram'da mesajı göreceksin! ✅

---

## 5️⃣ SİSTEM ENTEGRASYONU TEST

Dev server'ı restart et:
```bash
cd ~/Desktop/borsa
npm run dev
```

Emergency stop alert test et (CRITICAL alert tetikler):
```bash
curl -X POST http://localhost:3000/api/monitoring/live \
  -H 'Content-Type: application/json' \
  -d '{"action":"emergency_stop"}'
```

**Beklenen Sonuç:**
- Telegram'da alert mesajı alacaksın
- Format: `🚨 Emergency Stop Activated\n\nEmergency stop has been triggered...`

---

## 📊 ALERT SEVİYELERİ VE TELEGRAM

| Severity | Telegram Gönderilir mi? | Format |
|----------|------------------------|--------|
| CRITICAL | ✅ Evet | 🚨 + Kalın metin |
| HIGH | ✅ Evet | ⚠️ + Normal metin |
| MEDIUM | ✅ Evet | 📊 + Normal metin |
| LOW | ❌ Hayır | - |

---

## 🔧 SORUN GİDERME

### "Unauthorized" Hatası
- Token'ın doğru kopyalandığını kontrol et
- Token'da boşluk veya ekstra karakter olmamalı

### "Chat not found" Hatası
- Chat ID'nin doğru olduğunu kontrol et
- Önce bota `/start` mesajı gönder

### Alert Gelmiyor
- `.env` dosyasını kaydettin mi?
- Dev server'ı restart ettin mi?
- Console log'larında `[TELEGRAM] Alert:` var mı?

---

## ✅ SON KONTROL LİSTESİ

- [ ] BotFather'dan bot oluşturdun
- [ ] Token'ı aldın
- [ ] Chat ID'ni buldun
- [ ] .env dosyasına ekledin
- [ ] curl ile test ettin
- [ ] Dev server'ı restart ettin
- [ ] Emergency stop alert test ettin
- [ ] Telegram'da mesaj aldın

---

**🎉 Tamamlandı!** Artık trading bot'undan Telegram'a real-time alert alacaksın!

**Sonraki Adım:** Discord Webhook Setup
