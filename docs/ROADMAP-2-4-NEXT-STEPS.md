# Roadmap Maddeleri 2-4: Uyum ve Doğrulama Çerçevesi

## Roadmap-2: Push Sağlayıcı Entegrasyonu

**Hedef:** FCM/APNs ile güvenli push bildirim dağıtımı, AI özetlerinin maskeleme kurallarıyla gönderilmesi.

- [ ] FCM & APNs kimlik bilgileri için `config/push/README.md` ve `.env.example` güncelle.
- [ ] `NotificationManager` üzerine imzalı token desteği ve cihaz bazlı yetki kontrolü ekle.
- [ ] Payload içerisinde gömülü AI özetlerini kullanıcı gizlilik seviyesine göre (Free/Premium) filtrele.
- [ ] `tests/ai/groq-enhancer.test.ts` içerisinde payload maskeleme regresyon testleri ekle.
- [x] Groq plan kuyruğu için imza ve rate-limit dokümantasyonunu hazırla (`docs/GROQ-PLAN-ENQUEUE.md`).

## Roadmap-3: Gözlemlenebilirlik ve Alarm

**Hedef:** Prometheus/Grafana metrikleri, alarm eşikleri ve erişim kontrollü gösterge tabloları.

- [ ] `scripts/` altında `export-metrics.ts` oluşturup `ScanQueue` ve real-time motor metriklerini expose et.
- [ ] Prometheus endpoint’ini `middleware` katmanında throttle + IP allowlist ile koru.
- [ ] Grafana dashboard tanımlarını `docs/observability/` dizininde JSON olarak sakla, erişim seviyelerini tanımla.
- [ ] Alarm şablonları için incident runbook hazırlayıp `docs/security/incident-runbook.md` ekle.

## Roadmap-4: Strateji Doğrulama Suite’i

**Hedef:** Fixture, snapshot ve pozitif senaryolarla Trend Reversal, Bollinger, ATR stratejilerinin üst düzey validasyonu.

- [ ] `tests/signals/fixtures/` dizinini oluşturup yüksek volatilite verilerini kaydet.
- [ ] Vitest snapshot testi ile `strategy-aggregator.analyzeSymbol` sonuçlarını 50 sembol için karşılaştır.
- [ ] CI aşamasında fixture güncellemelerini zorunlu inceleme sürecine bağlamak için `scripts/verify-fixtures.sh` yaz.
- [ ] Doğrulama raporlarını `test-results/roadmap-4/` altında zaman damgasıyla sakla.
