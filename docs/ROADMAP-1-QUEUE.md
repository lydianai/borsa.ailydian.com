# Roadmap-1: Queue Bazlı Market Tarayıcı Altyapısı

**Amaç:** 7/24 çalışan tarama altyapısının kuyruğa taşınması, rate-limit ve güvenlik kontrolleriyle beyaz şapkalı uyumun sağlanması.

## Mimari Bileşenler

1. **ScanQueue Hizmeti**
   - `src/lib/queue/scan-queue.ts` dosyasında başlangıç iskeleti oluşturuldu.
   - BullMQ sürücüsü opsiyonel import ile izole edilerek build kırılmaları önlendi.
   - Bellek tabanlı fallback ile yerel geliştirmede çalışma garantisi sağlandı.
2. **Güvenlik Politikası**
   - Payload doğrulama, yetki kontrolü ve maskeleme kuralları `QueueSecurityPolicy` altında toplandı.
   - Her iş için `requestId`, `requestedBy`, `scopes` zorunlu tutuldu; loglarda anonimleştirme yapılıyor.
3. **Gözlemlenebilirlik**
   - Kuyruk metriklerini dönen `getMetrics` metodu eklendi.
   - İleride Prometheus exporter ile entegre edilecek.

## Yapılacaklar

- [x] BullMQ bağımlılığını `pnpm` ile ekle ve bağlantı yapılandırmasını `.env` üzerinde dokümante et.
- [x] `.env.example` dosyasına `QUEUE_DRIVER`, `QUEUE_REDIS_*` alanlarını ekle.
- [x] `ScanQueue` için unit test ve memory sürücüsü için fixture yaz.
- [x] Autonomous sistem içerisindeki `strategy-analysis` görevini kuyruk üzerinden tetikle.
- [x] BullMQ worker (`strategy-worker`) ile kuyruk işlerini Groq/Nirvana aggregator çıktılarıyla işle.
- [x] Queue metriklerinin çekilmesi için `GET /api/queue/metrics` endpoint’ini `INTERNAL_SERVICE_TOKEN` ile koru.
- [x] API katmanında güvenli enqueue endpoint’i (`POST /api/queue/enqueue`) için HMAC imza doğrulaması, rate-limit ve audit loglarını ekle.
- [ ] Executor plan kuyruğu için BullMQ consumer’ı (testnet doğrulamalı) devreye al.

## Güvenlik Notları

- Redis bağlantısı TLS ve kullanıcı adı/parola ile çalışacak; varsayılan açık değer bırakılmayacak.
- `requestedBy` değeri loglarda maskeleme ile tutulacak.
- Kuyruğa sadece `scan:enqueue` scope’una sahip servis hesapları erişebilecek.
