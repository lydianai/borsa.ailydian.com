# 🚀 BORSA.AILYDIAN.COM DEPLOYMENT TALİMATLARI

## ✅ TAMAMLANAN ADIMLAR

1. **Git Commit**: Tüm değişiklikler commit edildi
   - Quantum ladder console hataları düzeltildi
   - Telegram NaN hataları giderildi
   - DecisionPanel AbortError düzeltildi
   - React hydration hataları çözüldü

2. **Vercel CLI**: Kurulu ve giriş yapılmış
   - Version: 48.0.0
   - Kullanıcı: lydianlydian-9142

3. **Deployment Script**: Hazır
   - Dosya: `borsa-ailydian-deploy.sh`

## 📋 DEPLOYMENT ADIMLARI

### Seçenek 1: Otomatik Deployment (Önerilen)

```bash
./borsa-ailydian-deploy.sh
```

### Seçenek 2: Manuel Deployment

```bash
# 1. Production build test et
pnpm build

# 2. Vercel'e deploy et
vercel --prod

# 3. Sorulan sorulara cevaplar:
# - Set up and deploy? Yes
# - Which scope? lydianlydian-9142
# - Link to existing project? No (yeni proje)
# - Project name? lytrade-borsa
# - Directory? ./ 
# - Override settings? No
```

## 🌐 CUSTOM DOMAIN YAPILANDIRMASI

### borsa.ailydian.com Domain Ekleme

1. **Vercel Dashboard'a git**:
   ```
   https://vercel.com/lydianlydian-9142
   ```

2. **Projenizi seçin** (lytrade-borsa)

3. **Settings > Domains**'e gidin

4. **Add Domain**'e tıklayın
   - Domain: `borsa.ailydian.com`
   - Add butonuna tıklayın

5. **DNS Ayarları**:
   Vercel size şu DNS kayıtlarını eklemenizi isteyecek:
   
   ```
   Type: A
   Name: borsa (veya @)
   Value: 76.76.21.21
   
   Type: CNAME  
   Name: borsa
   Value: cname.vercel-dns.com
   ```

6. **DNS Propagation** bekleyin (5-10 dakika)

## 🔐 ENVIRONMENT VARIABLES

Vercel Dashboard'da Settings > Environment Variables'a gidin ve şu değişkenleri ekleyin:

```env
# Public Variables
NEXT_PUBLIC_PERSONAL_AUTH_ENABLED=0
NEXT_PUBLIC_FREEZE_TIME_TO=2025-10-27T10:00:00+03:00
NEXT_PUBLIC_APP_URL=https://borsa.ailydian.com

# Server Variables (Sensitive - Mark as "Secret")
GROQ_API_KEY=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
TELEGRAM_BOT_TOKEN=8292640150:AAHqDdkHxFqx9q8hJ-bJ8KS_Z2LZWrOLroI
TELEGRAM_ALLOWED_CHAT_IDS=7575640489

# System Variables
NODE_ENV=production
FETCH_INTERVAL_MS=60000
```

**ÖNEMLİ**: 
- Server variables'ı mutlaka "Secret" olarak işaretleyin
- Her değişkeni ekledikten sonra environment'ı seçin (Production, Preview, Development)

## 🔄 SON DEPLOYMENT

Environment variables ekledikten sonra tekrar deploy edin:

```bash
vercel --prod
```

## ✅ DOĞRULAMA

Deployment tamamlandıktan sonra test edin:

```bash
# 1. Health check
curl https://borsa.ailydian.com/api/nirvana | jq '.success'

# 2. Signals check
curl https://borsa.ailydian.com/api/signals?limit=5 | jq '.data.signals | length'

# 3. Browser'da aç
open https://borsa.ailydian.com
```

## 🐛 SORUN GİDERME

### Build Hataları

```bash
# Local build test
pnpm build

# Hata loglarını kontrol et
vercel logs
```

### Domain SSL Hataları

- DNS propagation bekleyin (24 saate kadar)
- Vercel Dashboard > Domains'de SSL durumunu kontrol edin
- "Renew Certificate" butonunu deneyin

### Environment Variable Hataları

- Vercel Dashboard'da değişkenlerin doğru environment'lara eklendiğini kontrol edin
- Tekrar deploy edin: `vercel --prod`

## 📞 DESTEK

- Vercel Docs: https://vercel.com/docs
- Vercel Support: support@vercel.com

---

**Not**: Tüm değişiklikler git'e commit edildi. Deployment script hazır. Sadece yukarıdaki adımları takip edin.

✅ **TARİH**: 2025-11-04 13:57 (Türkiye Saati)
✅ **SON ÇALIŞAN HAL**: localhost:3000 - Kusursuz Çalışıyor
