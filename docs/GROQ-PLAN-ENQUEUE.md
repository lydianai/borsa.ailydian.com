# Groq Plan Enqueue Akışı

## 1. İmza Oluşturma
```ts
import { signPayload } from "@/lib/crypto/signature";

const plan = {
  symbol: "BTCUSDT",
  intent: "OPEN",
  side: "LONG",
  qtyMode: "CONTRACT",
  qty: 0.01,
  orderType: "MARKET",
  risk: { maxLeverage: 10 }
};

const signature = signPayload({
  secret: process.env.QUEUE_ENQUEUE_SECRET!,
  payload: plan,
});
```

## 2. API Çağrısı
```bash
curl -X POST http://localhost:3002/api/queue/enqueue \
  -H "Content-Type: application/json" \
  -d '{
    "plan": { ... },
    "signature": { ... },
    "meta": {
      "source": "groq-planner",
      "orchestrator": "sardag-groq",
      "priority": "high",
      "timeframe": "4h"
    }
  }'
```

## 3. Testnet Workflow
1. Groq orkestratörü PLAN üretir ve imzalar (HMAC).
2. `POST /api/queue/enqueue` çağrısı ile PLAN testnet kuyruğuna düşer (`PLAN_QUEUE_NAME`).
3. Executor worker (`pnpm plan-worker`) testnette planı loglar; gerçek emir entegrasyonu hazır olduğunda aynı worker üzerinden uygulanacak.
4. Testnet kanıtı sağlanmadan `QUEUE_DRIVER=bullmq` + prod anahtarları açılmaz.

### Worker Başlatma
```bash
# TESTNET=1 ve gerekli Binance testnet key'leri ile
pnpm plan-worker
```
