# 🚀 PRODUCTION DEPLOYMENT GUIDE

**Vercel & Railway - Zero Error Deployment**

---

## 📋 PRE-DEPLOYMENT CHECKLIST

### ✅ 1. Environment Variables Hazır
- [ ] `.env.production` dosyası oluşturuldu
- [ ] API keys hazır (Anthropic, Groq)
- [ ] Database URL hazır (PostgreSQL)
- [ ] Alert channels configure (Telegram, Discord)

### ✅ 2. Production Build Test
```bash
npm run build
```
**Beklenen:** 0 error, 0 warning

### ✅ 3. Database Migration
```bash
# PostgreSQL için
npx prisma migrate deploy
```

### ✅ 4. Git Commit
```bash
git add .
git commit -m "Production ready deployment"
git push origin main
```

---

## 🔵 VERCEL DEPLOYMENT

### Step 1: Vercel CLI Install
```bash
npm i -g vercel
```

### Step 2: Login
```bash
vercel login
```

### Step 3: Environment Variables Set
```bash
# Tek tek ekle
vercel env add ANTHROPIC_API_KEY production
vercel env add GROQ_API_KEY production
vercel env add DATABASE_URL production
vercel env add NEXT_PUBLIC_APP_URL production

# Veya .env.production'dan import et
vercel env pull .env.vercel.production
```

### Step 4: Deploy
```bash
# Production deploy
vercel --prod

# Preview deploy (test için)
vercel
```

### Step 5: Environment Variables (Vercel Dashboard)
1. https://vercel.com/dashboard
2. Project seç → Settings → Environment Variables
3. Ekle:
   - `ANTHROPIC_API_KEY` = your_key
   - `GROQ_API_KEY` = your_key
   - `DATABASE_URL` = postgresql://...
   - `TELEGRAM_BOT_TOKEN` = your_token
   - `DISCORD_WEBHOOK_URL` = your_webhook
   - `NEXT_PUBLIC_APP_URL` = https://your-app.vercel.app

### Step 6: Redeploy
```bash
vercel --prod
```

---

## 🟣 RAILWAY DEPLOYMENT

### Step 1: Railway CLI Install
```bash
npm i -g @railway/cli
```

### Step 2: Login
```bash
railway login
```

### Step 3: Initialize Project
```bash
# Yeni proje
railway init

# Veya existing proje link
railway link
```

### Step 4: Add PostgreSQL Database
```bash
railway add --database postgres
```

### Step 5: Environment Variables Set
```bash
# Railway dashboard'dan ekle
railway open

# Veya CLI ile
railway variables set ANTHROPIC_API_KEY=your_key
railway variables set GROQ_API_KEY=your_key
railway variables set TELEGRAM_BOT_TOKEN=your_token
railway variables set DISCORD_WEBHOOK_URL=your_webhook
```

### Step 6: Deploy
```bash
railway up
```

### Step 7: Get URLs
```bash
# Production URL
railway open

# Database URL
railway variables
```

---

## 🗄️ DATABASE SETUP

### PostgreSQL (Vercel)
1. Vercel Storage → Postgres → Create
2. Copy connection string
3. Set as `DATABASE_URL` env variable
4. Run migration:
```bash
DATABASE_URL="postgresql://..." npx prisma migrate deploy
```

### PostgreSQL (Railway)
1. Railway otomatik PostgreSQL oluşturur
2. DATABASE_URL otomatik set edilir
3. Run migration:
```bash
railway run npx prisma migrate deploy
```

---

## 🔐 SECURITY CHECKLIST

### ✅ Environment Variables
- [ ] No hardcoded secrets in code
- [ ] All sensitive data in env vars
- [ ] `.env` in `.gitignore`
- [ ] Production secrets different from dev

### ✅ API Security
- [ ] CORS configured
- [ ] Rate limiting enabled
- [ ] HTTPS only
- [ ] Security headers set (vercel.json)

### ✅ Database Security
- [ ] Connection string encrypted
- [ ] SSL enabled
- [ ] Firewall rules set
- [ ] Backups enabled

---

## 🧪 POST-DEPLOYMENT TESTING

### 1. Health Check
```bash
curl https://your-app.vercel.app/api/health
```

### 2. Live Monitor Test
```
https://your-app.vercel.app/live-monitor
```
**Beklenen:**
- Charts render
- No console errors
- Metrics updating

### 3. Bot Integration Test
```bash
curl -X POST https://your-app.vercel.app/api/bot/initialize \
  -H 'Content-Type: application/json' \
  -d '{
    "apiKey": "test",
    "apiSecret": "test",
    "config": {
      "symbol": "BTCUSDT",
      "leverage": 10,
      "maxPositionSize": 100,
      "stopLossPercent": 2,
      "takeProfitPercent": 3,
      "maxDailyLoss": 50,
      "riskPerTrade": 1
    },
    "testnet": true
  }'
```

### 4. Alert Test
```bash
# Emergency stop alert (CRITICAL)
curl -X POST https://your-app.vercel.app/api/monitoring/live \
  -H 'Content-Type: application/json' \
  -d '{"action":"emergency_stop"}'
```
**Beklenen:** Telegram/Discord'da alert gelir

---

## 📊 MONITORING

### Vercel Analytics
```bash
# Enable analytics
vercel --prod --with-analytics
```

### Railway Logs
```bash
# View logs
railway logs

# Follow logs
railway logs --follow
```

### Custom Monitoring
- Vercel: https://vercel.com/dashboard → Project → Analytics
- Railway: https://railway.app → Project → Observability

---

## 🔄 CI/CD SETUP

### GitHub Actions (Vercel)
Create `.github/workflows/vercel.yml`:
```yaml
name: Vercel Production Deployment
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          vercel-args: '--prod'
```

### GitHub Actions (Railway)
Create `.github/workflows/railway.yml`:
```yaml
name: Railway Deployment
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: bervProject/railway-deploy@main
        with:
          railway_token: ${{ secrets.RAILWAY_TOKEN }}
          service: "production"
```

---

## 🚨 TROUBLESHOOTING

### Build Fails
```bash
# Local build test
npm run build

# Check logs
vercel logs
railway logs
```

### Database Connection Issues
```bash
# Test connection
npx prisma db push

# Check URL format
echo $DATABASE_URL
```

### Environment Variables Not Loading
```bash
# Verify vars
vercel env ls
railway variables

# Re-deploy
vercel --prod --force
railway up --force
```

### 502/504 Errors
- Check function timeout (Vercel: 30s max)
- Check memory limits
- Check database connection pool

---

## 📈 PERFORMANCE OPTIMIZATION

### 1. Image Optimization
```javascript
// next.config.ts
module.exports = {
  images: {
    domains: ['your-cdn.com'],
    formats: ['image/avif', 'image/webp'],
  },
}
```

### 2. Caching
```javascript
// API routes
export const config = {
  runtime: 'edge',
}

// Static pages
export const revalidate = 3600 // 1 hour
```

### 3. Bundle Size
```bash
# Analyze bundle
npm run build -- --analyze
```

---

## ✅ DEPLOYMENT COMPLETE CHECKLIST

- [ ] Vercel deployment successful
- [ ] Railway deployment successful
- [ ] Database migrations applied
- [ ] Environment variables set
- [ ] Health check passing
- [ ] Live monitor working
- [ ] Bot integration working
- [ ] Alerts working (Telegram/Discord)
- [ ] Charts rendering
- [ ] Mobile responsive
- [ ] No console errors
- [ ] SSL certificate active
- [ ] Custom domain configured (optional)
- [ ] Monitoring enabled
- [ ] Backups configured

---

## 🎉 PRODUCTION URLs

### Vercel
```
https://your-app.vercel.app
```

### Railway
```
https://your-app.railway.app
```

### Custom Domain (Optional)
```
https://borsa.yourdomain.com
```

---

## 📞 SUPPORT

### Vercel Support
- Docs: https://vercel.com/docs
- Discord: https://vercel.com/discord

### Railway Support
- Docs: https://docs.railway.app
- Discord: https://discord.gg/railway

---

**🚀 DEPLOYMENT HAZIR!**

**Next Steps:**
1. Deploy to Vercel → Test
2. Deploy to Railway → Test
3. Configure custom domain
4. Enable monitoring
5. Setup alerts
6. Production launch! 🎊
