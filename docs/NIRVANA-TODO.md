# Nirvana Trading Stack Roadmap

Organizes the steps needed to stabilize and scale the trading platform to "always-on" quality with white-hat compliance.

## 1. Market Data Reliability
- [ ] Stand up dedicated `data-service` (Node) that wraps Binance Futures/Spot, traditional markets (Yahoo, CME, etc.) with retry/backoff and circuit breaker.
- [ ] Add websocket listeners for real-time price streams; persist latest tick per symbol (Redis).
- [ ] Implement REST fallbacks and health endpoints (`/health`, `/latency`, `/ratelimit`).
- [ ] Create integration tests that replay recorded candles to validate responses (Jest + Supertest).

## 2. Strategy Verification Suite
- [ ] Build fixture-driven unit tests for each indicator (MA pullback, RSI divergence, Bollinger squeeze, EMA ribbon, Volume profile, Fibonacci, Ichimoku, ATR, Trend reversal).
- [ ] Add integration tests for `strategy-aggregator.analyzeSymbol` covering: cache hit/miss, Groq AI available/unavailable, mixed signal scenarios.
- [ ] Snapshot expected outputs for top 50 coins (precision thresholds) to detect regressions.
- [ ] Create regression harness that replays historical data and compares strategy scoring vs. ground truth.

## 3. Continuous Scanning Infrastructure
- [ ] Introduce job queue (BullMQ + Redis) to schedule scans for 522+ crypto instruments, BTC/ETH spotlight, and traditional tickers.
- [ ] Implement batched worker that aggregates signals into unified `SignalMatrix` and writes to persistent store (PostgreSQL/Redis).
- [ ] Track job latency, failure rate, retries; expose metrics to monitoring dashboard.
- [ ] Support dynamic scheduling (configurable intervals per asset class).

## 4. Notification & Mobile Delivery
- [ ] Replace browser-only notifications with push provider (FCM/APNs) and store device tokens securely.
- [ ] Build notification service that publishes alerts after deduplication + throttle + user preference checks.
- [ ] Integrate LyTrade Groq AI summary into notification payload when confidence ≥ threshold.
- [ ] Provide fallback channels (email/SMS) for high-priority signals.

## 5. Monitoring, Security, Compliance
- [ ] Instrument services with Prometheus/Grafana (CPU, memory, queue depth, API latency, error budgets).
- [ ] Add alerting rules (PagerDuty/Slack) for data drift, scan failures, Groq downtime.
- [ ] Enforce rate limiting, input sanitization, CSP/helmet headers, and dependency audits.
- [ ] Document white-hat checklist (logging policies, key rotation, incident response).

## 6. QA, CI/CD, Release Process
- [ ] Establish CI pipeline running lint, unit, integration, e2e (Playwright/Cypress), performance (k6) suites.
- [ ] Add canary/feature-flag workflow to roll out new strategies to a subset of users.
- [ ] Create automated data validation before deployment (compare live vs. cached vs. third-party reference).
- [ ] Maintain release notes and rollback scripts.

## 7. User Experience Enhancements
- [ ] Build real-time status page (uptime, scan cadence, last update per instrument).
- [ ] Surface AI commentary, confidence scores, MA7 pullback states in dashboards and mobile.
- [ ] Add in-app preference center (notification frequency, strategy filters, market segments).
- [ ] Ensure accessibility, localization, and responsive design for mobile-first users.

---

## Current Priority TODOs
- [ ] **Roadmap-1:** Queue tabanlı market tarayıcı altyapısını BullMQ + Redis ile kur; health endpoint ve rate-limit ölçümleri ekle (white-hat gereksinimleri: kimlik doğrulaması, yetkilendirme, log maskeleme).
- [ ] **Roadmap-2:** Push sağlayıcı (FCM/APNs) entegrasyonunu yap, bildirim payload’larında AI özetlerini maskele ve kullanıcı izin yönetimini belgeye bağla.
- [ ] **Roadmap-3:** Prometheus/Grafana tabanlı gözlemleme + alarm kurallarını ekle, veri gizliliği ve erişim kontrollü gösterge tabloları sağla.
- [ ] **Roadmap-4:** Strateji doğrulama suite’ini fixture/snapshot destekli hale getir, Trend Reversal/Bollinger/ATR için pozitif senaryoları üret ve sonuçları raporla.
- [ ] **Groq-Orchestrator:** LyTrade AI Groq planlayıcısını VS Code + executor mimarisiyle entegre et; PLAN/EXECUTION ayrımını koru, signature ve pino loglama gerekliliklerini uygula.
- [ ] **Groq-Executor:** Deterministik Binance USDT-M futures yürütücüsünü (undici/ws/zod) testnet ortamında devreye al; risk guardrail’leri, HMAC imzalama ve listen key keepalive’ı doğrula.
- [ ] **Aggregator-WFA:** analyzeFromCandles tabanlı aggregator’ı Walk-Forward backtest metrik kapıları (Sharpe ≥ 1.5, Sortino ≥ 2.0, PF ≥ 1.4, MaxDD ≤ 0.15) ile CI’ye bağla; white-hat raporlama ve dökümantasyon hazırla.
- [ ] **Real-Time Dashboard:** Groq orchestrator + executor + Nirvana aggregator çıktılarının tek panelde izlenmesi için gerçek zamanlı (localhost) durum izleyici oluştur; risk ve uyarıları maskele.
- [x] ✅ Veriye dayanıklı servis mimarisini kur: dış kaynaklardan veri toplamak için retry/backoff destekli `data-service` (Section 1-1) + mevcut strateji testlerini genişlet (Section 2). _(completed 2025-10-21)_
- [x] 🔁 Tüm veri adaptörlerini (Binance klines, Yahoo Finance, MetalpriceAPI, Turkish Gold, Commodities) resilient fetch katmanına taşı. _(completed 2025-10-21)_
- [ ] 🔄 Queue tabanlı sürekli tarama altyapısını (BullMQ/Redis) devreye al, sinyal özetlerini kalıcı depoya yaz (Section 3).
- [ ] 📣 Push sağlayıcı (FCM/APNs) ile bildirim katmanını modernize et ve AI özetini entegre et (Section 4).

**Next Steps Recommendation**
1. Begin with Section 1 + 2 to secure data and strategy accuracy (`data-service` + test suites).
2. Parallelize queue infrastructure (Section 3) once data-source is dependable.
3. Move to notifications (Section 4) and monitoring (Section 5) after scan pipeline stabilizes.
4. Layer QA/CI and UX enhancements progressively.
- [x] 🧪 Groq Vitest senaryolarını çevresel değişken ve fetch mock’ları ile stabilize et. _(completed 2025-10-21)_
- [x] 🧪 Sinyal strateji testlerini (EMA/MACD/RSI/Volume) temel kontrol senaryoları ile kapsa; ileri seviye pozitif fixture’lar backlog’a taşındı.
- [ ] 🧪 Gelişmiş sinyal doğrulaması için pozitif fixture üretimi (Trend Reversal, Bollinger, ATR) ve güç koşullarını simüle et.
