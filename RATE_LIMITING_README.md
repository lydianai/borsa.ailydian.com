# 🛡️ Groq API Rate Limiting Sistemi

## Genel Bakış

Ailydian Signal platformu için kapsamlı rate limiting sistemi. Groq API'nin free tier limitlerini (30 RPM, 14,400 RPD, 7,000 TPM) korumak ve aşırı kullanımı önlemek için tasarlanmıştır.

## 🎯 Özellikler

### 1. **Sliding Window Algorithm**
- ✅ Redis tabanlı distributed rate limiting
- ✅ Memory fallback (development için)
- ✅ Token bazlı rate limiting (TPM - Tokens Per Minute)
- ✅ Dakikalık ve günlük limit kontrolü
- ✅ IP bazlı tracking

### 2. **Groq API Limitleri**
```typescript
Free Tier Limits:
- 30 requests/minute (RPM)
- 14,400 requests/day (RPD)
- 7,000 tokens/minute (TPM)
```

### 3. **Endpoint Bazlı Rate Limits**
| Endpoint | Limit | Açıklama |
|----------|-------|----------|
| `/api/ai-assistant` | 30/min | Groq AI assistant |
| `/api/ai-signals` | 30/min | AI trading signals |
| `/api/traditional-markets-analysis/*` | 30/min | Market analysis |
| `/api/auth/*` | 20/min | Authentication |
| `/api/binance/*` | 200/min | Market data |
| `/api/bot-analysis/*` | 50/min | Bot analysis |
| Global API | 100/min | Tüm diğer endpoints |

## 📦 Kurulum

### 1. Redis Kurulumu (Opsiyonel)

```bash
# Upstash Redis kullanıyorsanız
# .env dosyasına ekleyin:
UPSTASH_REDIS_REST_URL=your_redis_url
UPSTASH_REDIS_REST_TOKEN=your_redis_token
```

Redis yoksa sistem otomatik olarak memory-based fallback kullanır.

### 2. Ngrok Kurulumu

```bash
# Ngrok indir ve yükle
https://ngrok.com/download

# Ngrok auth token ekle
ngrok authtoken YOUR_AUTH_TOKEN
```

## 🚀 Kullanım

### Development Modunda

**Terminal 1** - App'i çalıştır:
```bash
pnpm dev
```

**Terminal 2** - Ngrok başlat:
```bash
./start-ngrok.sh
```

Veya custom domain ile:
```bash
export NGROK_DOMAIN=your-endpoint.ngrok.app
./start-ngrok.sh
```

### Production Modunda

Vercel'de rate limiting otomatik çalışır. Ngrok sadece development/testing için gereklidir.

## 🔧 Yapılandırma

### rate-limit.yml

Ngrok traffic policy dosyası. Her endpoint için farklı limitler tanımlayabilirsiniz:

```yaml
on_http_request:
  - expressions:
      - req.url.contains('/api/ai-assistant')
    actions:
      - type: rate_limit
        config:
          name: groq_api_rate_limit
          algorithm: sliding_window
          capacity: 30
          rate: 60s
          bucket_key:
            - conn.client_ip
```

### src/lib/groq-rate-limiter.ts

Programatik rate limiter. Redis veya memory kullanarak rate limiting yapar.

```typescript
import { withGroqRateLimit } from '@/lib/groq-rate-limiter'

// API route'da kullanım
const { allowed, headers } = await withGroqRateLimit(clientIp, tokens)

if (!allowed) {
  return NextResponse.json(
    { error: 'Rate limit exceeded' },
    { status: 429, headers }
  )
}
```

## 📊 Rate Limit Headers

Her response'da aşağıdaki headerlar döner:

```
X-RateLimit-Limit: 30           # Limit (requests/minute)
X-RateLimit-Remaining: 25       # Kalan istek sayısı
X-RateLimit-Reset: 1234567890   # Reset zamanı (Unix timestamp)
Retry-After: 45                 # Kaç saniye sonra tekrar denenebilir
```

## 🧪 Test Etme

### 1. Rate Limit Test Script

```bash
# 35 request gönder (limit 30)
for i in {1..35}; do
  curl -s http://localhost:3000/api/ai-assistant \
    -H "Content-Type: application/json" \
    -d '{"message":"test"}' | jq .
  echo "Request $i"
done
```

### 2. İstatistikleri Görüntüleme

```typescript
import { groqRateLimiter } from '@/lib/groq-rate-limiter'

const stats = await groqRateLimiter.getUsageStats(clientIp)
console.log(stats)
// {
//   minuteUsage: 25,
//   dayUsage: 1234,
//   tokenUsage: 4567
// }
```

### 3. Limit Sıfırlama (Admin)

```typescript
await groqRateLimiter.resetLimit(clientIp)
```

## 🔍 Monitoring

### Ngrok Dashboard

```
http://127.0.0.1:4040
```

- Real-time requests
- Rate limit hits
- Traffic analytics

### Application Logs

```bash
# Rate limit logs
[GroqRateLimiter] Redis check passed: 25/30 remaining
[GroqRateLimiter] Rate limit exceeded for IP: 192.168.1.1
```

## ⚠️ Best Practices

### 1. **IP Extraction**
```typescript
const clientIp = request.headers.get('x-forwarded-for')?.split(',')[0] ||
                 request.headers.get('x-real-ip') ||
                 'unknown'
```

### 2. **Cache Stratejisi**
Cache kullanarak API çağrılarını azaltın:
```typescript
const AI_SIGNALS_CACHE_TTL = 5 * 60 * 1000 // 5 dakika
```

### 3. **Graceful Degradation**
Redis hatası durumunda memory fallback kullanın:
```typescript
try {
  return await checkWindowRedis(...)
} catch (error) {
  return checkWindowMemory(...)
}
```

### 4. **Token Tracking**
Groq API token kullanımını takip edin:
```typescript
const estimatedTokens = message.length / 4 // Rough estimate
await withGroqRateLimit(clientIp, estimatedTokens)
```

## 🚨 Troubleshooting

### Problem: "Rate limit exceeded" çok sık
**Çözüm**:
- Redis kullanıyorsanız connection'ı kontrol edin
- Limit değerlerini `rate-limit.yml`'de artırın
- Cache TTL'i uzatın

### Problem: Ngrok bağlanamıyor
**Çözüm**:
- Port 3000'in açık olduğundan emin olun
- `rate-limit.yml` dosyasının doğru olduğunu kontrol edin
- Ngrok auth token'ı doğrulayın

### Problem: Redis connection error
**Çözüm**:
- Upstash Redis URL ve token'ı kontrol edin
- Memory fallback kullanılacak (development için yeterli)

## 📚 Kaynaklar

- [Groq API Documentation](https://console.groq.com/docs/rate-limits)
- [Ngrok Traffic Policy](https://ngrok.com/docs/http/traffic-policy/)
- [Sliding Window Algorithm](https://en.wikipedia.org/wiki/Sliding_window_protocol)
- [Upstash Redis](https://upstash.com/)

## 🎓 Örnek Kullanım Senaryoları

### Senaryo 1: High Traffic Period
```typescript
// Peak saatlerde cache'i artır
const PEAK_CACHE_TTL = 10 * 60 * 1000 // 10 dakika

if (isPeakHour()) {
  cacheTimestamp = PEAK_CACHE_TTL
}
```

### Senaryo 2: Premium Users
```typescript
// Premium kullanıcılar için daha yüksek limit
const limit = user.isPremium ? 60 : 30
```

### Senaryo 3: Batch Processing
```typescript
// Batch işlemler için özel handling
if (isBatchRequest) {
  const tokensPerRequest = 100
  const { allowed } = await withGroqRateLimit(
    'batch-processor',
    tokensPerRequest
  )
}
```

## 📝 Lisans

MIT License - Ailydian Signal Platform

---

**Son Güncelleme**: 2025-12-16
**Versiyon**: 1.0.0
**Durum**: ✅ Production Ready
