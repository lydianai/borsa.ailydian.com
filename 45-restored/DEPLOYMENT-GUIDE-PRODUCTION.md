# 🚀 PRODUCTION DEPLOYMENT GUIDE
## Lydian Trader - Quantum AI Trading Platform

**Last Updated:** 2025-10-02

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Deploy Python AI Services to Railway](#step-1-deploy-python-ai-services-to-railway)
4. [Step 2: Deploy Next.js Frontend to Vercel](#step-2-deploy-nextjs-frontend-to-vercel)
5. [Step 3: Configure Environment Variables](#step-3-configure-environment-variables)
6. [Step 4: Enable Azure OpenAI (Optional)](#step-4-enable-azure-openai-optional)
7. [Step 5: Verification & Testing](#step-5-verification--testing)
8. [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│         VERCEL (Frontend)                           │
│   - Next.js 15.1.6                                  │
│   - React 19                                        │
│   - TradingView Charts                              │
│   - Real-time Binance API                           │
└─────────────────────────────────────────────────────┘
                    ↓ ↑
┌─────────────────────────────────────────────────────┐
│       RAILWAY (Python AI Services)                  │
│                                                     │
│  Service 1: AI Models (Port 5003)                  │
│   - 3 LSTM Models                                   │
│   - 5 GRU Models                                    │
│   - 3 Transformer Models                            │
│   - 3 Gradient Boosting Models                      │
│                                                     │
│  Service 2: TA-Lib Service (Port 5005)             │
│   - 158 Technical Indicators                        │
│   - Real-time calculations                          │
└─────────────────────────────────────────────────────┘
                    ↓ ↑
┌─────────────────────────────────────────────────────┐
│      AZURE OPENAI (Optional)                        │
│   - GPT-4 Market Analysis                           │
│   - Sentiment Analysis                              │
│   - Advanced Insights                               │
└─────────────────────────────────────────────────────┘
```

---

## ✅ Prerequisites

### Required Accounts:
- ✅ [Railway Account](https://railway.app) (Free tier available)
- ✅ [Vercel Account](https://vercel.com) (Free tier available)
- ✅ [GitHub Account](https://github.com) (for deployment)

### Optional:
- 🔵 [Azure Account](https://azure.microsoft.com) (for Azure OpenAI)
- 🟡 [Binance API Keys](https://binance.com) (for live trading)

### Local Requirements:
- ✅ Node.js 18+ installed
- ✅ Git installed
- ✅ Railway CLI installed: `npm install -g @railway/cli`
- ✅ Vercel CLI installed: `npm install -g vercel`

---

## 🛤️ Step 1: Deploy Python AI Services to Railway

### 1.1 Deploy AI Models Service

```bash
# Navigate to AI models directory
cd python-services/ai-models

# Login to Railway
railway login

# Initialize new Railway project
railway init

# Deploy
railway up

# Note the deployment URL (e.g., https://your-ai-models.up.railway.app)
```

**Environment Variables to Set in Railway Dashboard:**
- `PORT=5003`
- `FLASK_ENV=production`

### 1.2 Deploy TA-Lib Service

```bash
# Navigate to TA-Lib directory
cd ../talib-service

# Initialize new Railway project
railway init

# Deploy
railway up

# Note the deployment URL (e.g., https://your-talib-service.up.railway.app)
```

**Environment Variables to Set in Railway Dashboard:**
- `PORT=5005`
- `FLASK_ENV=production`

### 1.3 Verify Railway Deployments

```bash
# Test AI Models Service
curl https://your-ai-models.up.railway.app/health

# Test TA-Lib Service
curl https://your-talib-service.up.railway.app/health
```

Expected response: `{"status": "healthy"}`

---

## 🌐 Step 2: Deploy Next.js Frontend to Vercel

### 2.1 Push to GitHub

```bash
# Go to project root
cd /Users/sardag/Desktop/borsa

# Initialize git (if not already)
git init
git add .
git commit -m "Production deployment ready"

# Create GitHub repository and push
git remote add origin https://github.com/YOUR_USERNAME/lydian-trader.git
git push -u origin main
```

### 2.2 Deploy to Vercel

**Option A: Via Vercel Dashboard**
1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your GitHub repository
3. Configure project:
   - Framework Preset: **Next.js**
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Install Command: `npm install`

**Option B: Via Vercel CLI**
```bash
# Login to Vercel
vercel login

# Deploy
vercel --prod
```

---

## 🔐 Step 3: Configure Environment Variables

### 3.1 Vercel Environment Variables

Go to **Vercel Dashboard → Your Project → Settings → Environment Variables**

Add the following:

```bash
# Python Services (Railway URLs from Step 1)
NEXT_PUBLIC_AI_MODELS_URL=https://your-ai-models.up.railway.app
NEXT_PUBLIC_TALIB_SERVICE_URL=https://your-talib-service.up.railway.app

# External APIs
NEXT_PUBLIC_BINANCE_API_URL=https://api.binance.com
NEXT_PUBLIC_COINGECKO_API_URL=https://api.coingecko.com/api/v3

# Application
NODE_ENV=production
NEXT_TELEMETRY_DISABLED=1
```

### 3.2 Redeploy After Adding Variables

```bash
vercel --prod
```

---

## 🔵 Step 4: Enable Azure OpenAI (Optional)

### 4.1 Create Azure OpenAI Resource

1. Go to [Azure Portal](https://portal.azure.com)
2. Create a new **Azure OpenAI Service**
3. Deploy a model (e.g., **gpt-4** or **gpt-35-turbo**)
4. Get your API credentials:
   - Endpoint URL
   - API Key
   - Deployment Name

### 4.2 Add Azure Variables to Vercel

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

### 4.3 Test Azure Integration

```bash
curl -X POST https://your-vercel-app.vercel.app/api/azure/market-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "marketData": {
      "price": 42000,
      "volume": 1234567890,
      "change24h": 2.5
    }
  }'
```

---

## ✅ Step 5: Verification & Testing

### 5.1 Automated Smoke Test

```bash
# Test all production endpoints
./dev-smoke-test.sh
```

### 5.2 Manual Tests

**Test Frontend:**
```bash
# Open in browser
open https://your-vercel-app.vercel.app

# Test live trading page
open https://your-vercel-app.vercel.app/live-trading

# Test quantum pro page
open https://your-vercel-app.vercel.app/quantum-pro
```

**Test AI Models API:**
```bash
curl -X POST https://your-ai-models.up.railway.app/predict/single \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "timeframe": "1h"}'
```

**Test TA-Lib API:**
```bash
curl -X POST https://your-talib-service.up.railway.app/indicators/batch \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "indicators": ["rsi", "macd", "bbands"]
  }'
```

**Test Quantum Signal API:**
```bash
curl -X POST https://your-vercel-app.vercel.app/api/bot/quantum-signal \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "config": {
      "multiTimeframe": true,
      "adaptivePositionSizing": true,
      "aiModelWeights": {
        "lstm": 0.25,
        "gru": 0.25,
        "transformer": 0.25,
        "gradientBoosting": 0.25
      }
    },
    "apiKey": "test",
    "apiSecret": "test"
  }'
```

---

## 🐛 Troubleshooting

### Issue 1: Railway Service Not Starting

**Solution:**
```bash
# Check Railway logs
railway logs

# Common issues:
# - Missing dependencies in requirements.txt
# - Port not exposed correctly
# - Memory limits exceeded
```

### Issue 2: Vercel Build Failing

**Solution:**
```bash
# Check build logs in Vercel dashboard
# Common issues:
# - TypeScript errors
# - Missing environment variables
# - Out of memory (increase Node memory)
```

**Fix TypeScript errors:**
```bash
# Locally
npm run type-check
```

**Increase memory in `package.json`:**
```json
{
  "scripts": {
    "build": "NODE_OPTIONS='--max-old-space-size=4096' next build"
  }
}
```

### Issue 3: CORS Errors

**Solution:**
Already configured in `vercel.json`:
```json
{
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        {
          "key": "Access-Control-Allow-Origin",
          "value": "*"
        }
      ]
    }
  ]
}
```

### Issue 4: API Timeouts

**Railway:**
- Increase memory/CPU in Railway dashboard
- Check service health: `railway logs`

**Vercel:**
- Serverless functions timeout after 10s (Hobby) or 60s (Pro)
- Consider upgrading to Pro plan for longer timeouts

---

## 📊 Production Checklist

- [ ] Railway AI Models Service deployed and healthy
- [ ] Railway TA-Lib Service deployed and healthy
- [ ] Vercel Next.js app deployed
- [ ] All environment variables configured
- [ ] All pages load without errors (18/18)
- [ ] Real-time data working (Order Book, Prices, Charts)
- [ ] AI predictions working
- [ ] TA-Lib indicators working
- [ ] Quantum Signal API working
- [ ] Azure OpenAI integrated (optional)
- [ ] Mobile responsive
- [ ] Security headers configured
- [ ] HTTPS enabled (automatic on Vercel/Railway)

---

## 🎉 Success Metrics

After successful deployment:

✅ **Frontend:** https://your-app.vercel.app
✅ **AI Models:** https://your-ai-models.up.railway.app
✅ **TA-Lib:** https://your-talib-service.up.railway.app
✅ **Azure OpenAI:** Integrated (if enabled)

**Performance Targets:**
- Page load time: < 2s
- API response time: < 500ms
- Chart rendering: < 1s
- Real-time updates: Every 1-2s

---

## 🔒 Security Best Practices

1. ✅ **Environment Variables:** Never commit API keys to git
2. ✅ **HTTPS:** Enforced on all services
3. ✅ **CORS:** Configured properly in vercel.json
4. ✅ **Rate Limiting:** Implement in production
5. ✅ **API Key Rotation:** Regular key rotation for Binance/Azure
6. ✅ **Secret Management:** Use Railway/Vercel secret stores

---

## 📞 Support

**Issues?** Create an issue on GitHub
**Updates?** Check deployment logs
**Monitoring?** Use Vercel Analytics + Railway Metrics

---

**Deployment Completed:** 🎉🚀
**Platform:** Vercel + Railway + Azure OpenAI
**Status:** Production Ready
**Zero Downtime:** ✅
**Real Data:** ✅
**White Hat Security:** ✅
