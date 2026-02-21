# 💬 DISCORD WEBHOOK SETUP - ADIM ADIM REHBERİ

## 📋 Gerekli Bilgiler
- Discord Webhook URL (Discord Server ayarlarından alınacak)

---

## 1️⃣ DISCORD WEBHOOK OLUŞTURMA

### Adım 1: Discord Server'ı Aç
1. Discord uygulamasını aç
2. Kendi server'ına git (yoksa yeni bir server oluştur)
3. Alert mesajlarının gitmesini istediğin kanalı seç (örnek: #trading-alerts)

### Adım 2: Webhook Oluştur
1. Kanal adına sağ tıkla > **Edit Channel**
2. Sol menüden **Integrations** sekmesine git
3. **Webhooks** bölümünü bul
4. **New Webhook** (veya **Create Webhook**) butonuna tıkla

### Adım 3: Webhook Ayarları
1. Webhook için bir **isim** ver (örnek: "Borsa Trading Alerts")
2. Webhook için bir **avatar** (profil fotoğrafı) seç (opsiyonel)
3. Hangi **kanala** mesaj göndereceğini seç (#trading-alerts)
4. **Copy Webhook URL** butonuna tıkla

Webhook URL şuna benzer:
```
https://discord.com/api/webhooks/1234567890123456789/ABCdefGHIjklMNOpqrsTUVwxyz-123456789
```

### Adım 4: Test Et (Tarayıcıdan)
Terminal veya PowerShell'de test et:
```bash
curl -X POST "YOUR_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"content": "🚀 Test mesajı! Webhook çalışıyor!"}'
```

Discord kanalında mesajı göreceksin! ✅

---

## 2️⃣ .ENV DOSYASINA EKLEME

1. Projenin root klasöründeki `.env` dosyasını aç
2. Şu satırı bul:
   ```bash
   DISCORD_WEBHOOK_URL=your_discord_webhook_url_here
   ```

3. Gerçek URL ile değiştir:
   ```bash
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1234567890123456789/ABCdefGHIjklMNOpqrsTUVwxyz-123456789
   ```

4. Dosyayı kaydet

---

## 3️⃣ SİSTEM ENTEGRASYONU TEST

### Terminal Test (Advanced)
```bash
curl -X POST "YOUR_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "embeds": [{
      "title": "🚨 Emergency Stop Activated",
      "description": "Bot stopped due to emergency condition",
      "color": 16711680,
      "timestamp": "2025-10-03T10:00:00Z",
      "footer": {
        "text": "Severity: CRITICAL"
      }
    }]
  }'
```

### Dev Server Test
1. Dev server'ı restart et:
   ```bash
   cd ~/Desktop/borsa
   npm run dev
   ```

2. Emergency stop alert test et:
   ```bash
   curl -X POST http://localhost:3000/api/monitoring/live \
     -H 'Content-Type: application/json' \
     -d '{"action":"emergency_stop"}'
   ```

**Beklenen Sonuç:**
- Discord kanalında **embed** (güzel formatlı) mesaj alacaksın
- Format: Başlık, açıklama, renk, zaman damgası

---

## 📊 DISCORD EMBED RENK KODLARI

| Severity | Renk | Hex | Decimal |
|----------|------|-----|---------|
| CRITICAL | 🔴 Kırmızı | #FF0000 | 16711680 |
| HIGH | 🟠 Turuncu | #FF9900 | 16750848 |
| MEDIUM | 🟡 Sarı | #FFFF00 | 16776960 |
| LOW | 🟢 Yeşil | #00FF00 | 65280 |
| INFO | 🔵 Mavi | #0099FF | 39423 |

---

## 📱 ÖRNEK DISCORD MESAJLARI

### Basit Mesaj
```json
{
  "content": "🚀 Test mesajı!"
}
```

### Embed Mesaj (Güzel Format)
```json
{
  "embeds": [{
    "title": "⚠️ Maximum Drawdown Warning",
    "description": "Current drawdown: 18% (Max: 20%)",
    "color": 16750848,
    "fields": [
      { "name": "Bot", "value": "BTCUSDT Quantum Bot", "inline": true },
      { "name": "P&L", "value": "-180 USDT", "inline": true }
    ],
    "timestamp": "2025-10-03T10:00:00Z",
    "footer": {
      "text": "Severity: HIGH"
    }
  }]
}
```

### Multiple Embeds
```json
{
  "embeds": [
    {
      "title": "📊 Position Opened",
      "description": "LONG BTCUSDT",
      "color": 39423
    },
    {
      "title": "📈 Win Rate Update",
      "description": "Current: 68.5%",
      "color": 65280
    }
  ]
}
```

---

## 🔧 SORUN GİDERME

### "Invalid Webhook Token" Hatası
- Webhook URL'sinin tamamen kopyalandığını kontrol et
- URL'de boşluk veya ekstra karakter olmamalı
- Webhook silinmemiş olmalı

### "Unknown Webhook" Hatası
- Webhook'un hala aktif olduğunu kontrol et (Discord'da Integrations > Webhooks)
- Yeni bir webhook oluştur ve URL'yi yenile

### Mesaj Gelmiyor
- `.env` dosyasını kaydettin mi?
- Dev server'ı restart ettin mi?
- Console log'larında `[DISCORD] Alert:` var mı?
- Webhook URL doğru kopyalandı mı?

### Embed Görünmüyor
- JSON formatının doğru olduğunu kontrol et
- `embeds` array içinde olmalı: `{"embeds": [...]}`
- Color değeri decimal olmalı (hex değil!)

---

## 🎨 ALERT SEVİYELERİ VE DISCORD

| Severity | Discord Gönderilir mi? | Format | Renk |
|----------|----------------------|--------|------|
| CRITICAL | ✅ Evet | Embed | 🔴 Kırmızı |
| HIGH | ✅ Evet | Embed | 🟠 Turuncu |
| MEDIUM | ✅ Evet | Embed | 🟡 Sarı |
| LOW | ❌ Hayır | - | - |

---

## 🎯 TEST SCRİPTİ

Discord webhook test script'i oluştur:

```bash
node test-discord-webhook.js
```

Script otomatik olarak:
- ✅ Env variable kontrolü yapar
- ✅ 3 farklı alert türü gönderir
- ✅ Embed formatında mesaj gönderir
- ✅ Renk kodlarını test eder

---

## ✅ SON KONTROL LİSTESİ

- [ ] Discord server'ında kanal oluşturdun (#trading-alerts)
- [ ] Webhook oluşturdun (Edit Channel > Integrations)
- [ ] Webhook URL'sini kopyaladın
- [ ] .env dosyasına ekledin
- [ ] curl ile test ettin
- [ ] Dev server'ı restart ettin
- [ ] Emergency stop alert test ettin
- [ ] Discord kanalında mesaj aldın

---

**🎉 Tamamlandı!** Artık trading bot'undan Discord'a real-time alert alacaksın!

**Sonraki Adım:** Bot Initialization & WebSocket → Azure SignalR Migration

---

## 📚 KAYNAKLAR

- Discord Webhook API Docs: https://discord.com/developers/docs/resources/webhook
- Embed Visualizer: https://discohook.org/
- Color Picker (Decimal): https://www.spycolor.com/
