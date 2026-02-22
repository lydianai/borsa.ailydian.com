# Groq + Binance Futures Entegrasyon Yol Haritası

## Amaç
- Groq tabanlı planlayıcı (PLAN-only) ile deterministik Binance USDT-M yürütücüsünü (EXECUTOR) LyTrade mimarisine entegre etmek.
- Nirvana aggregator’ı (analyzeFromCandles) koruyup Walk-Forward backtest kapılarıyla uyumlu hale getirmek.
- Tüm bileşenleri white-hat güvenlik prensipleri, log maskeleme ve testnet doğrulama adımlarıyla güvence altına almak.

## Sprint Görevleri
1. **Monorepo Hazırlığı**
   - `groq-binance-futures-stack/` pnpm monorepo iskeletini oluştur.
   - `apps/orchestrator`, `apps/executor`, `packages/shared`, `ops` dizinlerini LyTrade repo içinde `integrations/` altında template olarak tut.
   - Typescript, eslint, vitest, ts-node bağımlılıklarını ve scriptlerini (build/dev/typecheck/test/smoke/backtest) ekle.
2. **Shared Paket**
   - `packages/shared` içinde fiyat/lot filtreleri, HMAC/Hash yardımcıları, `Plan` ve `RiskCaps` şemalarını tanımla.
   - uuidv7 tabanlı clientOrderId üretimini ekle.
3. **Executor**
   - `undici` + `ws` ile Binance testnet REST/WS adaptörünü yaz.
   - Listen key keepalive, position mode ayarı, normalize edilmiş order gönderimi ve risk `approvePlan` fonksiyonunu uygula.
   - Pino loglama, zod doğrulama, stop-loss/take-profit ve reduceOnly kontrollerini ekle.
4. **Orchestrator**
   - Groq (OpenAI uyumlu) client’ı ile sadece JSON PLAN üretimini sağla.
   - Zod şema validasyonu, not/risk alanları ve signature alanı ekle.
   - PLAN → `POST /api/queue/enqueue` endpointine HMAC imza ile gönder; kuyruğa düşen kayıtları audit log’la.
5. **Aggregator + Backtest**
   - analyzeFromCandles fonksiyonunu WFA senaryolarıyla uyumlu halde tut ve AI guardrail’i koru.
   - `scripts/backtest-wfa.ts` ile Sharpe/Sortino/MaxDD/PF eşiklerini enforce et; CI modunda kapıları ihlal ederse çıkış kodu > 0 olsun.
6. **Monitoring & Ops**
   - `ops/.env`, `docker-compose.dev.yml`, `Makefile` komutlarını Groq + Executor için uyumlu tut.
   - Testnet kanıtı olmadan prod ortamına geçişi engelle (flag/tarihçeleme).
   - Pino loglarını maskele, queue ve redis bağlantılarını TLS + yetki kontrolü ile yapılandır; `GET /api/queue/metrics` ve `POST /api/queue/enqueue` çağrılarını rate-limit et.

## Güvenlik Notları
- Testnet ortamı haricinde gerçek API anahtarları kullanılmayacak; prod açılmadan önce CI → sim → testnet → prod zincirine ait kayıtlar saklanacak.
- Risk guardrail’leri (max leverage, per-trade cap, daily loss cap) 0 toleranslı çalışmalı.
- PLAN çıktıları yatırım tavsiyesi değildir; kullanıcı tarafında mevzuat/vergiler dokümante edilmelidir.
