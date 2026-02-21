# 📰 CryptoPanic API Kurulumu

**ÜCRETSİZ** - Kredi kartı gerektirmez

## Neden CryptoPanic?

- ✅ **Tamamen ücretsiz** (kredi kartı yok)
- ✅ **1000+ kaynak** (CoinDesk, Cointelegraph, Bloomberg, vb.)
- ✅ **Gerçek zamanlı** kripto haberleri
- ✅ **Kategori filtreleme** (Bitcoin, Ethereum, DeFi, vb.)
- ✅ **Güvenilir** ve stabil API

## API Key Alma (2 dakika)

### 1. Kayıt Ol
https://cryptopanic.com/developers/api/

### 2. Ücretsiz Hesap Oluştur
- Email ile kayıt ol
- Email'ini doğrula
- **Kredi kartı gerekmiyor!**

### 3. API Token Al
- Dashboard'a git
- "API Tokens" bölümünde token'ını görürsün
- Token'ı kopyala

### 4. Projeye Ekle
`.env.local` dosyasına ekle:

```bash
CRYPTOPANIC_API_KEY=your_actual_token_here
```

### 5. Server'ı Yeniden Başlat
```bash
# Dev server'ı durdur (Ctrl+C) ve tekrar başlat
pnpm dev
```

## Kullanım Limitleri (FREE Tier)

| Özellik | Limit |
|---------|-------|
| **Requests/Ay** | 10,000 |
| **Requests/Dakika** | ~7 |
| **Geçmiş Data** | 7 gün |
| **Haberler** | Sınırsız |

Bizim kullanımımız:
- 10 dakikada 1 istek
- Ayda ~4,320 istek
- ✅ **Limitlerin içinde!**

## Test Et

API key'i ekledikten sonra:

```bash
# API'yi test et
curl "http://localhost:3001/api/crypto-news"

# Veya browser'da aç
http://localhost:3001/haberler
```

## Sorun Giderme

### "No news available" hatası
- `.env.local` dosyasında `CRYPTOPANIC_API_KEY` var mı kontrol et
- Server'ı yeniden başlat: `pnpm dev`
- Token'ın doğru olduğundan emin ol

### Mock data görüyorum
- Normal! API key yoksa otomatik mock data gösterir
- API key ekle ve server'ı yeniden başlat

## API Özellikleri

CryptoPanic API şunları sağlar:

- 📊 **Trending haberler**
- 📈 **Market sentiment**
- 🏷️ **Coin bazlı filtreleme**
- ⏰ **Real-time güncellemeler**
- 🔖 **Kategorize edilmiş haberler**

## Daha Fazla Bilgi

- [CryptoPanic API Docs](https://cryptopanic.com/developers/api/)
- [Örnek Kullanım](https://cryptopanic.com/developers/api/examples/)
