# 🔒 TELEGRAM GIZLI MOD KULLANIM KILAVUZU

**Telegram botunu sadece sen kullan, kimse giremesin!**

---

## ❓ İki Soru - İki Cevap

### 1️⃣ Localhost'ta Telegram Webhook Çalışır mı?

**HAYIR! ❌** Telegram webhook'ları sadece public HTTPS URL'leri kabul eder.

`http://localhost:3000` webhook için çalışmaz çünkü:
- Telegram sunucuları localhost'una erişemez
- Webhook sadece internet üzerinden erişilebilir URL'ler gerektirir

### ✅ Çözüm Seçenekleri:

#### **Seçenek A: Ngrok ile Localhost'u Public Yap (Test için)**

```bash
# 1. Ngrok'u indir: https://ngrok.com/download
# 2. Ngrok'u başlat
ngrok http 3000

# 3. Ngrok'un verdiği URL'i not al (örnek: https://abc123.ngrok.io)

# 4. Webhook'u ngrok URL'ine ayarla
curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://abc123.ngrok.io/api/telegram/webhook",
    "secret_token": "your_webhook_secret"
  }'

# 5. Localhost'u başlat
pnpm dev
```

**⚠️ Dikkat:** Ngrok her yeniden başlatıldığında URL değişir, webhook'u tekrar ayarlamanız gerekir.

---

#### **Seçenek B: Vercel'e Deploy Et (Önerilen)** ⭐

En kolay ve güvenli yöntem:

```bash
# 1. Vercel'e deploy et
vercel --prod

# 2. Webhook'u production URL'ine ayarla
curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://lydian.app/api/telegram/webhook",
    "secret_token": "your_webhook_secret"
  }'

# 3. Artık Telegram bot production'da çalışıyor!
```

---

### 2️⃣ Telegram Bot'u Sadece Ben Kullanmak İstiyorum (Gizli Mod)

**✅ Gizli mod artık aktif!** Sadece senin chat ID'ne bildirim gönderir.

---

## 🔐 Gizli Mod Nasıl Çalışır?

Bot şu an **iki modda** çalışabiliyor:

### **A) Herkese Açık Mod** (Varsayılan)
- `TELEGRAM_ALLOWED_CHAT_IDS` boş bırakılırsa
- Herkes `/start` göndererek abone olabilir
- Tüm abonelere bildirim gider

### **B) Gizli Mod** 🔒 (Önerilen - Sadece Sen)
- `TELEGRAM_ALLOWED_CHAT_IDS` doldurulursa
- Sadece listedeki chat ID'ler bot kullanabilir
- Başkaları `/start` gönderirse "Bu bot gizli moddadır" mesajı alır
- Sadece senin chat ID'ne bildirim gider

---

## 🚀 Gizli Mod Kurulum (3 Adım)

### **Adım 1: Chat ID'ni Öğren**

#### Yöntem A: @userinfobot Kullan (En Kolay)

1. Telegram'da **@userinfobot** ara
2. `/start` gönder
3. Chat ID'ni göreceksin (örnek: `123456789`)

#### Yöntem B: Kendi Bot'undan Öğren

1. Bot'unu geçici olarak herkese açık modda deploy et (TELEGRAM_ALLOWED_CHAT_IDS boş)
2. Bot'a `/start` gönder
3. Admin API'sini çağır:
   ```bash
   curl https://lydian.app/api/telegram/admin
   ```
4. `subscribers` listesinde chat ID'ni göreceksin

---

### **Adım 2: Chat ID'ni Environment Variable'a Ekle**

#### Local Development (.env.local):

```env
# .env.local dosyasını oluştur/düzenle
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_BOT_WEBHOOK_SECRET=your_webhook_secret_here

# 🔒 GIZLI MOD: Senin Chat ID'ni buraya yaz
TELEGRAM_ALLOWED_CHAT_IDS=123456789
```

Birden fazla kişi eklemek istersen:
```env
# Virgülle ayırarak ekle
TELEGRAM_ALLOWED_CHAT_IDS=123456789,987654321,555444333
```

#### Vercel Production:

1. **Vercel Dashboard** → Proje seç → **Settings** → **Environment Variables**
2. Yeni variable ekle:
   - **Name:** `TELEGRAM_ALLOWED_CHAT_IDS`
   - **Value:** `123456789` (senin chat ID'n)
   - **Environment:** Production
3. **Save**
4. Vercel otomatik redeploy yapacak

---

### **Adım 3: Test Et**

```bash
# 1. Vercel'e deploy et (otomatik redeploy olacak)
vercel --prod

# 2. Telegram'da bot'a /start gönder
# 3. Şu mesajı göreceksin:

# ✅ İzin verilen chat ID isen:
# "👋 Hoş geldin!
#  🔒 (Gizli Mod Aktif)
#  ✅ Bildirimler aktif edildi!"

# ❌ İzin verilmeyen chat ID isen:
# "🔒 Bu bot gizli moddadır
#  Bu bot sadece yetkili kullanıcılar tarafından kullanılabilir.
#  Chat ID: 123456789"
```

---

## 🧪 Test Senaryoları

### **Test 1: Gizli Mod Aktif - Sen Bot'a Giriyorsun**

```bash
# Chat ID'n: 123456789
# .env: TELEGRAM_ALLOWED_CHAT_IDS=123456789

# Telegram'da /start gönder
# Sonuç: ✅ Başarılı - Bildirimler aktif
```

### **Test 2: Gizli Mod Aktif - Başkası Bot'a Girmeye Çalışıyor**

```bash
# Başkasının Chat ID'si: 999888777
# .env: TELEGRAM_ALLOWED_CHAT_IDS=123456789

# Başkası /start gönderir
# Sonuç: ❌ "Bu bot gizli moddadır" mesajı alır
```

### **Test 3: Test Bildirimi Gönder**

```bash
# Sadece senin chat ID'ne gider
curl -X POST "https://lydian.app/api/telegram/test" \
  -H "Content-Type: application/json" \
  -d '{"type":"strong_buy"}'

# Sonuç: ✅ Premium bildirim Telegram'da gelir
```

---

## 📊 Gizli Mod Özeti

| Özellik | Herkese Açık Mod | Gizli Mod 🔒 |
|---------|-----------------|-------------|
| TELEGRAM_ALLOWED_CHAT_IDS | Boş | Dolu (örn: 123456789) |
| Kim `/start` gönderebilir? | Herkes | Sadece listedekiler |
| Bildirimi kim alır? | Tüm aboneler | Sadece listedekiler |
| Başkaları girebilir mi? | ✅ Evet | ❌ Hayır |
| Güvenlik | Düşük | Yüksek |

---

## 🔧 Sorun Giderme

### ❌ "Bu bot gizli moddadır" mesajı alıyorum ama ben sahibiyim!

**Çözüm:**
1. Admin API'sini çağır:
   ```bash
   curl https://lydian.app/api/telegram/admin
   ```
2. `config.allowedChatIds` listesini kontrol et
3. Chat ID'nin doğru olduğundan emin ol
4. Vercel'de environment variable'ı kontrol et

---

### ❌ Gizli mod aktif değil, herkes girebiliyor!

**Çözüm:**
1. `.env.local` veya Vercel'de `TELEGRAM_ALLOWED_CHAT_IDS` değişkeninin dolu olduğundan emin ol
2. Chat ID'nin doğru formatda olduğundan emin ol (sadece rakam, boşluk yok)
3. Vercel'de değişkeni kaydettikten sonra redeploy yap:
   ```bash
   vercel --prod
   ```

---

### ❌ Localhost'ta test edemiyorum!

**Çözüm:**
Localhost'ta test etmek için **Ngrok** kullanmalısın:

```bash
# 1. Ngrok'u başlat
ngrok http 3000

# 2. Ngrok URL'ini not al (örn: https://abc123.ngrok.io)

# 3. Webhook'u ngrok URL'ine ayarla
curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
  -d "url=https://abc123.ngrok.io/api/telegram/webhook" \
  -d "secret_token=<SECRET>"

# 4. Localhost'u başlat
pnpm dev

# 5. Telegram'da test et
```

---

## 📱 Komutlar

### Admin API (Durumu Kontrol Et)

```bash
# Sistem durumunu ve config'i görüntüle
curl https://lydian.app/api/telegram/admin

# Örnek Yanıt:
{
  "status": "active",
  "config": {
    "signalTypes": ["STRONG_BUY", "BUY", "SELL", "WAIT"],
    "minConfidence": 70,
    "mode": "realtime",
    "allowedChatIds": [123456789]  // 🔒 Gizli Mod aktif
  },
  "stats": {
    "subscriberCount": 1,
    "subscribers": [123456789]
  }
}
```

### Test Notification

```bash
# Simple test
curl -X POST "https://lydian.app/api/telegram/test" \
  -H "Content-Type: application/json" \
  -d '{"type":"simple"}'

# Strong buy test
curl -X POST "https://lydian.app/api/telegram/test" \
  -H "Content-Type: application/json" \
  -d '{"type":"strong_buy"}'
```

### Webhook Durumu

```bash
# Webhook'un kurulu olup olmadığını kontrol et
curl "https://api.telegram.org/bot<TOKEN>/getWebhookInfo"
```

---

## ✅ Checklist

Gizli mod için tamamlanması gereken adımlar:

- [ ] Chat ID'mi öğrendim (@userinfobot veya admin API)
- [ ] `.env.local` veya Vercel'de `TELEGRAM_ALLOWED_CHAT_IDS` ekledim
- [ ] Vercel'e deploy ettim (`vercel --prod`)
- [ ] Webhook'u kurdum (`setWebhook`)
- [ ] Telegram'da bot'a `/start` gönderdim
- [ ] "Gizli Mod Aktif" mesajını gördüm
- [ ] Test bildirimi gönderip aldım
- [ ] Başka bir hesaptan denedim ve "gizli moddadır" mesajını gördüm

---

## 🎉 Sonuç

✅ **Gizli mod aktif!**
✅ **Sadece sen kullanabilirsin**
✅ **Başkaları giremez**
✅ **Premium bildirimler sadece sana gelir**

### Otomatik Sinyal Akışı:

```
Strategy Aggregator
      ↓
Sinyal Analizi (600+ coin)
      ↓
Filtreleme (STRONG_BUY, BUY, SELL, WAIT | %70+)
      ↓
Spam Kontrolü (5 dk/sembol)
      ↓
🔒 Gizli Mod Kontrolü (Sadece sen mi?)
      ↓
Premium Format (Unicode Art)
      ↓
Telegram Bildirimi → Sadece Senin Chat ID'ne ✅
```

---

## 📚 Ek Kaynaklar

- **Detaylı Dokümantasyon:** `TELEGRAM-PREMIUM-COMPLETE-TR.md`
- **Hızlı Başlangıç:** `TELEGRAM-HIZLI-BASLANGIC.md`
- **Teknik Mimari:** `TELEGRAM-BOT-INTEGRATION-BRIEF-TR.md`

---

**Artık Telegram botun tamamen gizli ve sadece senin! 🚀**
