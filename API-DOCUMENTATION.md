# 📚 SarDag Trading Signals API Documentation

Production-ready RESTful API for cryptocurrency trading signals powered by 18 advanced strategies, AI analysis, and real-time market data.

## 🚀 Quick Start

### Base URLs
- **Development**: `http://localhost:3001`
- **Production**: `https://your-domain.vercel.app`

### Example Request
```bash
# Get unified signal for BTC
curl -X POST http://localhost:3001/api/unified-signals \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT"}'
```

### Example Response
```json
{
  "symbol": "BTCUSDT",
  "finalDecision": "BUY",
  "overallConfidence": 82,
  "buyPercentage": 75,
  "waitPercentage": 25,
  "strategyBreakdown": [
    {
      "name": "Conservative Buy Signal",
      "signal": "BUY",
      "confidence": 92,
      "weight": 1.5
    },
    {
      "name": "Volume Spike",
      "signal": "BUY",
      "confidence": 85,
      "weight": 1.2
    }
  ],
  "topRecommendations": [
    "Conservative Buy Signal (92%)",
    "Volume Spike (85%)",
    "Breakout-Retest (78%)"
  ],
  "riskAssessment": {
    "level": "LOW",
    "factors": ["Strong consensus across strategies"]
  },
  "aggregatedTargets": [50500, 52000, 54500],
  "aggregatedStopLoss": 48750
}
```

## 📖 OpenAPI Specification

Full OpenAPI 3.0.3 specification: [`openapi.yaml`](./openapi.yaml)

**View Interactive Documentation:**
- **Swagger UI**: Import `openapi.yaml` to [Swagger Editor](https://editor.swagger.io/)
- **Postman**: Import `openapi.yaml` to [Postman](https://www.postman.com/)
- **Redoc**: Use [Redoc](https://redocly.github.io/redoc/) for beautiful API docs

## 🎯 Core Endpoints

### 1. Unified Signals (Recommended)
**Endpoint**: `POST /api/unified-signals`

Analyzes a symbol using **all 18 strategies** and returns a unified BUY/WAIT decision with percentage-based confidence.

**Strategies Included:**
1. Conservative Buy Signal ✅
2. Breakout-Retest ✅
3. Volume Spike ✅
4. MA Crossover Pullback
5. RSI Divergence
6. Volume Breakout
7. Bollinger Squeeze
8. EMA Ribbon
9. Fibonacci Retracement
10. Ichimoku Cloud
11. ATR Volatility
12. Trend Reversal
13. MACD Histogram
14. Support/Resistance
15. Red Wick Green Closure
16. MA7 Pullback
17. BTC-ETH Correlation
18. Omnipotent Futures Matrix

**Request:**
```json
{
  "symbol": "ETHUSDT"
}
```

**Response Fields:**
- `finalDecision`: BUY | WAIT
- `overallConfidence`: 0-100%
- `buyPercentage`: % of strategies voting BUY
- `waitPercentage`: % of strategies voting WAIT
- `strategyBreakdown`: Individual strategy results
- `topRecommendations`: Top 3 best-performing strategies
- `riskAssessment`: Risk level (LOW/MEDIUM/HIGH) + factors
- `aggregatedTargets`: Average take-profit targets
- `aggregatedStopLoss`: Conservative stop-loss price

---

### 2. AI Signals (3-Layer Analysis)
**Endpoint**: `POST /api/ai-signals`

Groq AI-powered signal generation with 3-layer system:
- **Layer 1**: Conservative Buy Signal (9-condition system)
- **Layer 2**: Breakout-Retest Pattern (historical data)
- **Layer 3**: Unified 18-strategy aggregation

**Request:**
```json
{
  "symbol": "SOLUSDT"
}
```

**Response:**
```json
{
  "symbol": "SOLUSDT",
  "analysis": "AI-generated market analysis...",
  "layers": [
    { "name": "Conservative Buy Signal", "signal": "BUY", "confidence": 88 },
    { "name": "Breakout-Retest", "signal": "BUY", "confidence": 75 },
    { "name": "Unified Aggregator", "signal": "BUY", "confidence": 82 }
  ],
  "finalRecommendation": "BUY",
  "confidence": 85,
  "timestamp": "2025-10-25T12:00:00Z"
}
```

---

### 3. Conservative Signals
**Endpoint**: `POST /api/conservative-signals`

Ultra-safe signals with **9-condition confirmation**. Only returns BUY signals with 80%+ confidence.

**9 Conditions:**
1. Price within 40% of 24h low (support bounce)
2. Price below 20% of 24h range (oversold)
3. Positive 24h change (momentum confirmation)
4. Volume > $100M (sufficient liquidity)
5. Price change 0.5% - 5% (not overheated)
6. Volume > 2x average (volume spike)
7. Not near resistance (< 70% of range)
8. Moderate volatility (< 8%)
9. Strong support confirmation

**Request:**
```json
{
  "symbol": "BNBUSDT"
}
```

**Response:**
```json
{
  "name": "Conservative Buy Signal",
  "signal": "BUY",
  "confidence": 92,
  "reason": "9/9 conditions met: Support bounce at $580, Volume spike 2.8x, Strong momentum +3.2%",
  "targets": [600, 610, 625],
  "stopLoss": 575,
  "timeframe": "4H",
  "indicators": {
    "conditionsMet": 9,
    "supportDistance": 38,
    "volumeRatio": 2.8,
    "momentum": 3.2,
    "riskScore": 95
  }
}
```

---

### 4. Breakout-Retest Signals
**Endpoint**: `POST /api/breakout-retest`

Detects breakout-retest patterns using **real historical candlestick data** from Binance.

**How It Works:**
1. Fetches last 100 candles (4H, 1H, 15min)
2. Identifies recent breakouts (price > resistance)
3. Detects pullbacks to broken resistance (now support)
4. Confirms volume surge during breakout
5. Validates support holding on retest

**Request:**
```json
{
  "symbol": "ADAUSDT"
}
```

**Response:**
```json
{
  "name": "Breakout-Retest",
  "signal": "BUY",
  "confidence": 78,
  "reason": "Breakout at $0.52, clean retest at $0.50, volume confirmed",
  "targets": [0.54, 0.56, 0.58],
  "stopLoss": 0.49,
  "timeframe": "4H",
  "indicators": {
    "breakoutLevel": 0.52,
    "retestLevel": 0.50,
    "volumeConfirmation": true,
    "distanceFromBreakout": 3.8
  }
}
```

---

### 5. All Signals
**Endpoint**: `GET /api/signals?limit=20`

Returns signals from all strategies for top trading pairs. Useful for scanning multiple symbols.

**Query Parameters:**
- `limit`: Number of symbols (1-100, default: 20)

**Response:**
```json
{
  "signals": [
    {
      "symbol": "BTCUSDT",
      "signal": "BUY",
      "confidence": 85,
      "strategies": ["Conservative", "Volume Spike"]
    },
    {
      "symbol": "ETHUSDT",
      "signal": "WAIT",
      "confidence": 62,
      "strategies": []
    }
  ],
  "timestamp": "2025-10-25T12:00:00Z",
  "count": 20
}
```

---

## 🤖 AI & Analysis Endpoints

### AI Assistant
**Endpoint**: `POST /api/ai-assistant`

Interactive AI trading assistant for market insights.

**Request:**
```json
{
  "message": "Should I buy ETH now?",
  "context": {
    "symbol": "ETHUSDT",
    "userPortfolio": {}
  }
}
```

### Market Correlation
**Endpoint**: `GET /api/market-correlation`

BTC-ETH correlation and market phase detection.

---

## 🌍 Traditional Markets

### Get Traditional Market Data
**Endpoint**: `GET /api/traditional-markets`

Returns stock indices, gold, oil, and forex data.

### Analyze Symbol
**Endpoint**: `GET /api/traditional-markets-analysis/AAPL`

Detailed analysis for stocks, commodities, or forex pairs.

**Supported Symbols:**
- **Stocks**: AAPL, GOOGL, MSFT, TSLA, AMZN
- **Commodities**: GC (Gold), CL (Oil), SI (Silver)
- **Forex**: EUR/USD, GBP/USD, USD/JPY

---

## 🔔 Push Notifications

### Register Device
**Endpoint**: `POST /api/push/register`

Register FCM token for browser push notifications.

**Request:**
```json
{
  "token": "FCM_DEVICE_TOKEN",
  "userId": "optional_user_id"
}
```

### Send Notification
**Endpoint**: `POST /api/push/send`

Send notification to registered devices.

### Get Stats
**Endpoint**: `GET /api/push/stats`

Returns delivery stats and registered device count.

---

## 🏥 Health & Monitoring

### Service Health
**Endpoint**: `GET /api/health`

Overall system health status.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-10-25T12:00:00Z",
  "uptime": 3600000,
  "version": "1.0.0"
}
```

### Data Service Health
**Endpoint**: `GET /api/data-service/health`

Binance data service and queue health.

### Queue Metrics
**Endpoint**: `GET /api/queue/metrics`

Message queue performance metrics.

### Security Audit
**Endpoint**: `GET /api/security/audit`

Recent security events and audit trail.

---

## ⚙️ Settings

### Get Settings
**Endpoint**: `GET /api/settings`

Retrieve user preferences.

### Update Settings
**Endpoint**: `POST /api/settings`

Save user preferences.

**Request:**
```json
{
  "notifications": {
    "strongBuy": true,
    "sell": true,
    "aiUpdates": false
  },
  "theme": "dark",
  "language": "tr"
}
```

---

## 📊 Response Schemas

### Signal Schema
```typescript
interface Signal {
  name: string;              // Strategy name
  signal: 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';
  confidence: number;        // 0-100%
  reason: string;            // Human-readable explanation
  targets?: number[];        // Take-profit targets
  stopLoss?: number;         // Stop-loss price
  timeframe?: string;        // Recommended timeframe
  indicators?: Record<string, any>; // Technical indicators
}
```

### UnifiedSignal Schema
```typescript
interface UnifiedSignal {
  symbol: string;
  finalDecision: 'BUY' | 'WAIT';
  overallConfidence: number;
  buyPercentage: number;
  waitPercentage: number;
  strategyBreakdown: {
    name: string;
    signal: string;
    confidence: number;
    weight: number;
  }[];
  topRecommendations: string[];
  riskAssessment: {
    level: 'LOW' | 'MEDIUM' | 'HIGH';
    factors: string[];
  };
  aggregatedTargets?: number[];
  aggregatedStopLoss?: number;
}
```

---

## 🔐 Authentication

Currently, all endpoints are open. For production:

1. **API Key Authentication** (Optional)
   ```bash
   curl -H "X-API-Key: YOUR_API_KEY" http://localhost:3001/api/signals
   ```

2. **Rate Limiting**
   - Standard: 100 requests/minute
   - Authenticated: 500 requests/minute

---

## 🛠️ Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Invalid symbol",
  "code": "INVALID_SYMBOL",
  "timestamp": "2025-10-25T12:00:00Z"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error

---

## 💡 Best Practices

1. **Always specify symbol in uppercase**: `BTCUSDT`, not `btcusdt`
2. **Use unified signals for comprehensive analysis**: Includes all 18 strategies
3. **Check confidence levels**: Only act on 70%+ confidence signals
4. **Monitor risk assessment**: LOW risk = safer entries
5. **Set stop-losses**: Use `aggregatedStopLoss` for risk management
6. **Use multiple timeframes**: Cross-reference 4H, 1H, 15min signals

---

## 📝 Example Use Cases

### 1. Get Quick BUY/WAIT Decision
```bash
curl -X POST http://localhost:3001/api/unified-signals \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT"}'
```

### 2. Ultra-Safe Conservative Entry
```bash
curl -X POST http://localhost:3001/api/conservative-signals \
  -H "Content-Type: application/json" \
  -d '{"symbol": "ETHUSDT"}'
```

### 3. AI-Powered Multi-Layer Analysis
```bash
curl -X POST http://localhost:3001/api/ai-signals \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SOLUSDT"}'
```

### 4. Scan Top 50 Symbols
```bash
curl "http://localhost:3001/api/signals?limit=50"
```

---

## 🚀 Integration Examples

### JavaScript/TypeScript
```typescript
async function getUnifiedSignal(symbol: string) {
  const response = await fetch('http://localhost:3001/api/unified-signals', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol })
  });

  const data = await response.json();

  if (data.finalDecision === 'BUY' && data.overallConfidence >= 75) {
    console.log(`🎯 Strong BUY signal for ${symbol}`);
    console.log(`Confidence: ${data.overallConfidence}%`);
    console.log(`Targets: ${data.aggregatedTargets}`);
    console.log(`Stop Loss: ${data.aggregatedStopLoss}`);
  }

  return data;
}

// Usage
await getUnifiedSignal('BTCUSDT');
```

### Python
```python
import requests

def get_unified_signal(symbol):
    url = 'http://localhost:3001/api/unified-signals'
    response = requests.post(url, json={'symbol': symbol})
    data = response.json()

    if data['finalDecision'] == 'BUY' and data['overallConfidence'] >= 75:
        print(f"🎯 Strong BUY signal for {symbol}")
        print(f"Confidence: {data['overallConfidence']}%")
        print(f"Targets: {data['aggregatedTargets']}")
        print(f"Stop Loss: {data['aggregatedStopLoss']}")

    return data

# Usage
get_unified_signal('BTCUSDT')
```

---

## 📞 Support

- **Documentation**: [openapi.yaml](./openapi.yaml)
- **Issues**: Open an issue on GitHub
- **Email**: support@example.com

---

## 📜 License

MIT License - see [LICENSE](./LICENSE) for details

---

**Last Updated**: October 25, 2025
**API Version**: 1.0.0
**Status**: Production Ready ✅
