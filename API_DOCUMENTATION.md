# 📘 API Documentation

## Ailydian Signal - REST API Reference

**Version:** 1.0.0
**Base URL:** `https://your-domain.com/api`
**Authentication:** Not required for public endpoints

---

## 📋 Table of Contents

1. [Market Data](#market-data)
2. [Trading Signals](#trading-signals)
3. [AI Signals](#ai-signals)
4. [Decision Engine](#decision-engine)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)

---

## 🔐 Authentication

Currently, the API does not require authentication for read-only endpoints. Future versions will implement API key authentication.

---

## 📊 Standard Response Format

### Success Response
```json
{
  "success": true,
  "data": { /* response data */ },
  "isMockData": false,
  "timestamp": "2025-01-19T14:30:00Z",
  "cached": false
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "message": "Error description",
    "code": "ERROR_CODE",
    "statusCode": 400,
    "timestamp": "2025-01-19T14:30:00Z"
  }
}
```

---

## 🌍 Market Data

### GET `/api/binance/futures`

Get real-time market data from Binance Futures.

**Query Parameters:**
- None

**Response:**
```json
{
  "success": true,
  "data": {
    "all": [
      {
        "symbol": "BTCUSDT",
        "price": 42500.50,
        "changePercent24h": 2.45,
        "volume24h": 1234567890,
        "high24h": 43000,
        "low24h": 41500,
        "lastUpdate": "2025-01-19T14:30:00Z"
      }
    ],
    "gainers": [ /* top gaining coins */ ],
    "losers": [ /* top losing coins */ ],
    "highVolume": [ /* high volume coins */ ],
    "totalMarkets": 150,
    "lastUpdate": "2025-01-19T14:30:00Z"
  }
}
```

**Cache:** 60 seconds

**Example:**
```bash
curl https://your-domain.com/api/binance/futures
```

---

## 📈 Trading Signals

### GET `/api/signals`

Get trading signals from all strategies.

**Query Parameters:**
- `limit` (optional): Number of signals to return (1-200, default: 50)
- `minConfidence` (optional): Minimum confidence level (0-100, default: 70)
- `timeframe` (optional): Timeframe filter (1m, 5m, 15m, 1h, 4h, 1d, 1w)
- `symbol` (optional): Filter by symbol (e.g., BTCUSDT)

**Response:**
```json
{
  "success": true,
  "data": {
    "signals": [
      {
        "id": "sig_123",
        "symbol": "BTCUSDT",
        "type": "BUY",
        "strategy": "SMC Strategy",
        "confidence": 85,
        "price": 42500.50,
        "targets": [43000, 43500, 44000],
        "stopLoss": 42000,
        "timestamp": "2025-01-19T14:30:00Z",
        "reason": "Strong bullish order block detected"
      }
    ],
    "stats": {
      "totalCoinsScanned": 150,
      "buySignalsFound": 12,
      "avgConfidence": "82.5",
      "lastUpdate": "2025-01-19T14:30:00Z"
    }
  }
}
```

**Cache:** 300 seconds (5 minutes)

**Example:**
```bash
curl "https://your-domain.com/api/signals?limit=10&minConfidence=80"
```

---

### GET `/api/conservative-signals`

Ultra-conservative buy signals with strict criteria.

**Query Parameters:**
- `minConfidence` (optional): Min confidence (default: 80)
- `limit` (optional): Max signals (default: 50)
- `minRiskReward` (optional): Min risk/reward ratio (default: 2.5)
- `maxLeverage` (optional): Max recommended leverage (default: 5)

**Response:**
```json
{
  "success": true,
  "data": {
    "signals": [
      {
        "symbol": "ETHUSDT",
        "type": "BUY",
        "confidence": 87,
        "price": 2250.50,
        "highlightYellow": true,
        "priority": 950,
        "targets": [2280, 2320, 2350],
        "stopLoss": 2210,
        "indicators": {
          "rsi": 58.5,
          "volumeRatio": 12.5,
          "stopLossPercent": 2.0,
          "riskRewardRatio": 3.2,
          "leverageMax": 5
        },
        "reason": "🎯 ULTRA-CONSERVATIVE BUY SIGNAL..."
      }
    ],
    "stats": {
      "totalCoinsScanned": 150,
      "buySignalsFound": 8,
      "strictCriteria": [
        "Volume > $10M",
        "Change: 3-15%",
        "Confidence > 85%",
        "Risk < 2%"
      ]
    }
  }
}
```

**Cache:** 900 seconds (15 minutes)

---

## 🤖 AI Signals

### GET `/api/ai-signals`

AI-generated trading signals using machine learning models.

**Query Parameters:**
- `limit` (optional): Number of signals (1-200, default: 50)
- `minConfidence` (optional): Min confidence (0-100, default: 70)

**Response:**
```json
{
  "success": true,
  "data": {
    "signals": [
      {
        "symbol": "BTCUSDT",
        "type": "BUY",
        "confidence": 88,
        "aiModel": "Transformer + LSTM Ensemble",
        "price": 42500,
        "predictedMove": 2.5,
        "timeHorizon": "4h",
        "reasoning": "Strong bullish pattern detected by transformer model"
      }
    ],
    "modelInfo": {
      "version": "v2.1.0",
      "lastTrained": "2025-01-15",
      "accuracy": "78.5%"
    }
  }
}
```

**Cache:** 900 seconds (15 minutes)

---

### GET `/api/quantum-signals`

Quantum-inspired signals using advanced algorithms.

**Response:**
```json
{
  "success": true,
  "data": {
    "signals": [
      {
        "symbol": "BTCUSDT",
        "type": "BUY",
        "confidence": 92,
        "quantumScore": 0.87,
        "entanglementStrength": 0.92,
        "coherenceLevel": 0.85
      }
    ]
  }
}
```

---

## 🧠 Decision Engine

### GET `/api/decision-engine`

Get trading decision for a specific symbol based on all strategies.

**Query Parameters:**
- `symbol` (required): Trading symbol (e.g., BTCUSDT)

**Response:**
```json
{
  "success": true,
  "data": {
    "decision": "BUY",
    "confidence": 84,
    "symbol": "BTCUSDT",
    "price": 42500,
    "strategies": [
      {
        "name": "SMC Strategy",
        "signal": "BUY",
        "confidence": 85,
        "weight": 0.20
      },
      {
        "name": "AI Transformer",
        "signal": "BUY",
        "confidence": 88,
        "weight": 0.25
      }
    ],
    "reasoning": "Consensus of 8/10 strategies recommends BUY with high confidence"
  }
}
```

**Cache:** 120 seconds (2 minutes)

**Example:**
```bash
curl "https://your-domain.com/api/decision-engine?symbol=BTCUSDT"
```

---

## ⚠️ Error Handling

### Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `AUTH_ERROR` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Access denied |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT` | 429 | Rate limit exceeded |
| `EXTERNAL_API_ERROR` | 502 | External service error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### Error Response Example

```json
{
  "success": false,
  "error": {
    "message": "Invalid symbol format. Must be uppercase and end with USDT",
    "code": "VALIDATION_ERROR",
    "statusCode": 400,
    "details": [
      {
        "path": ["symbol"],
        "message": "Invalid symbol format"
      }
    ],
    "timestamp": "2025-01-19T14:30:00Z"
  }
}
```

---

## 🚦 Rate Limiting

### Limits

- **Anonymous:** 100 requests per minute
- **Authenticated:** 500 requests per minute

### Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642608000
```

### Rate Limit Exceeded

```json
{
  "success": false,
  "error": {
    "message": "Rate limit exceeded",
    "code": "RATE_LIMIT",
    "statusCode": 429,
    "details": {
      "resetTime": 1642608000
    }
  }
}
```

---

## 📦 Response Metadata

### Mock Data Warning

When using demo/mock data, responses include:

```json
{
  "success": true,
  "isMockData": true,
  "mockDataWarning": "⚠️ DEMO DATA: This is simulated data for development/testing purposes.",
  "data": { /* ... */ }
}
```

### Cache Information

Cached responses include:

```json
{
  "success": true,
  "cached": true,
  "nextUpdate": 240,  // seconds until next refresh
  "data": { /* ... */ }
}
```

---

## 🔧 SDK Examples

### JavaScript/TypeScript

```typescript
// Fetch signals
const response = await fetch('https://your-domain.com/api/signals?limit=10');
const data = await response.json();

if (data.success) {
  console.log('Signals:', data.data.signals);
} else {
  console.error('Error:', data.error.message);
}
```

### Python

```python
import requests

# Get market data
response = requests.get('https://your-domain.com/api/binance/futures')
data = response.json()

if data['success']:
    print(f"Total markets: {data['data']['totalMarkets']}")
else:
    print(f"Error: {data['error']['message']}")
```

### cURL

```bash
# Get AI signals with high confidence
curl -X GET "https://your-domain.com/api/ai-signals?minConfidence=85&limit=5" \
  -H "Accept: application/json"
```

---

## 📞 Support

For API issues or questions:
- **Documentation**: This file
- **GitHub Issues**: Report bugs
- **Email**: support@your-domain.com

---

**Last Updated:** 2025-01-19
