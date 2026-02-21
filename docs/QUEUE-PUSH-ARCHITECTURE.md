# Queue & Push Bildirim Mimari Taslağı

## Amaç
- Binance ve geleneksel piyasa taramalarını sürekli ve ölçeklenebilir şekilde çalıştırmak.
- Yüksek öncelikli sinyalleri gerçek zamanlı push sağlayıcıları (FCM/APNs) üzerinden ulaştırmak.
- Beyaz şapka prensipleri: rate limit koruması, kişisel verinin minimal tutulması, güvenli kimlik doğrulama.

## 1. Queue Katmanı (BullMQ + Redis)
- **Queue’lar**
  - `signals:scan`: Top N futures/spot sembolleri tarar; payload `{ symbol, timeframe }`.
  - `signals:aggregate`: Strateji sonuçlarını tekilleştirip `SignalMatrix` deposuna yazar.
  - `notifications:dispatch`: Öncelik filtresinden geçen sinyalleri push servisine yollar.
- **Workerlar**
  - `scan-worker`: `strategy-aggregator.analyzeSymbol` kullanır. Redis rate limiter + resilientFetch katmanı.
  - `aggregate-worker`: Çoklu strateji sonuçlarını birleştirir, cache’i (marketDataCache) günceller.
  - `dispatch-worker`: Kullanıcı tercihlerini (notif preferences) kontrol eder, push kuyruğuna gönderir.
- **Redis**
  - Ayrı DB index: `0` queue, `1` rate limiter sayaçları.
  - `BullMQ` queue başına concurrency ayarı (örn. scan: 5, aggregate: 2, dispatch: 10).

## 2. Kalıcı Depo
- **Seçenek**: PostgreSQL (`signals`, `jobs`, `device_tokens` tabloları).
- `signals` şeması: `id`, `symbol`, `timeframe`, `confidence`, `strategies`, `status`, `created_at`.
- `device_tokens`: Kullanıcı bazlı FCM/APNs token, platform, son kullanım, izin flag’leri.

## 3. Push Servisi
- **FCM (Firebase Cloud Messaging)**
  - Server key `.env` (`FCM_SERVER_KEY`), `firebase-admin` SDK.
  - Mesaj payload: `{ title, body, data: { symbol, confidence, strategies } }`.
- **APNs (Apple Push Notification service)**
  - Token-based auth (`APNS_KEY_ID`, `APNS_TEAM_ID`, `APNS_AUTH_KEY_PATH`).
  - `node-apn` ya da `apn` paketi.
- **Queue Akışı**
  1. `dispatch-worker` -> `pushService.send`.
  2. Sinyal tipine göre kanal (`signals`, `price-alerts`).
  3. Başarısız teslimatlar için retry queue (`notifications:retry`) ve exponential backoff.

## 4. Güvenlik & Uyumluluk
- Device token’lar AES-256 ile şifrelenmiş halde depolanacak.
- Her push isteği rate limit (örn. kullanıcı başına dakikada 5).
- Audit log: Hangi sinyal hangi cihazlara gönderildi (PII içermeden).
- Kullanıcı tercihi zorunlu: `opt-in`/`opt-out`, quiet hours, kanal bazlı izinler.

## 5. Entegrasyon Adımları
1. **Altyapı**: Redis + PostgreSQL docker config, BullMQ setup (`src/lib/queues`).
2. **Worker Kodları**: `scanWorker.ts`, `aggregateWorker.ts`, `dispatchWorker.ts` (TS).
3. **API**: `/api/notifications/register` (token kaydı), `/api/notifications/preferences`.
4. **Push Service**: `src/lib/notifications/push-provider.ts` -> FCM/APNs adapter katmanı.
5. **Monitoring**: Bull Board veya custom dashboard, queue health endpoint (`/api/health/queue`).
6. **Deployment**: Redis + PostgreSQL için managed servis veya Docker Compose; Vercel edge function’ları queue tetikleyicisi olarak kullanılmayacak (worker separate runtime).

## 6. Riskler & Mitigasyon
- **Rate Limit**: Binance rate limiter + queue concurrency ayarı.
- **Push Delivery Failure**: Retry queue + dead-letter queue.
- **Veri Gizliliği**: Token ve sinyal log’larında kişisel veri tutulmamalı.
- **Monitoring**: BullMQ metrics + Prometheus exporter planlanmalı.
