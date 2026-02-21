# 🚀 BORSA.AILYDIAN.COM - PRODUCTION DEPLOYMENT GUIDE

**%100 Seviye Deployment - Zero-Error Protocol**

## 📋 İçindekiler

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [8-Step Validation Protocol](#8-step-validation-protocol)
3. [Mobile Optimization Checklist](#mobile-optimization-checklist)
4. [Performance Testing](#performance-testing)
5. [Security Verification](#security-verification)
6. [Vercel Deployment](#vercel-deployment)
7. [Post-Deployment Monitoring](#post-deployment-monitoring)
8. [Troubleshooting](#troubleshooting)

---

## ✅ Pre-Deployment Checklist

### 1. Code Quality

- [ ] **TypeScript Strict Mode:** tsconfig.json'da `"strict": true`
- [ ] **No TypeScript Errors:** `pnpm typecheck` 0 error
- [ ] **ESLint Clean:** `pnpm lint` 0 warnings
- [ ] **Build Success:** `pnpm build` başarılı

### 2. Environment Variables

**Required (.env.production):**

```env
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...

# Authentication
NEXTAUTH_SECRET=min_32_characters_here
NEXTAUTH_URL=https://borsa.ailydian.com

# AI Services
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...

# Payment
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Monitoring
NEXT_PUBLIC_SENTRY_DSN=https://...
```

### 3. Security

- [ ] **No Hardcoded Secrets:** Tüm API keys .env'de
- [ ] **HTTPS Only:** Production URL https://
- [ ] **CORS Configured:** Sadece allowed origins
- [ ] **Rate Limiting:** API endpoints korumalı
- [ ] **Security Headers:** CSP, HSTS, X-Frame-Options aktif

### 4. Performance

- [ ] **Bundle Size:** < 5MB (check: `du -sh .next/static`)
- [ ] **Images Optimized:** next/image kullanılıyor
- [ ] **Code Splitting:** Dynamic imports var
- [ ] **Lazy Loading:** Below-fold content lazy
- [ ] **Caching:** Redis + ISR configured

---

## 🎯 8-STEP VALIDATION PROTOCOL

**ZERO-ERROR PROTOCOL: Tüm adımlar başarılı olmalı!**

### Step 1: Syntax Check

```bash
npx tsc --noEmit
```

**Expected:** 0 errors
**If Fails:** Fix TypeScript errors

### Step 2: Dev Server Start

```bash
pnpm dev
```

**Expected:** Server starts on http://localhost:3000
**If Fails:** Check port 3000 availability, check dependencies

### Step 3: Browser Test

```bash
open http://localhost:3000
```

**Expected:**
- ✅ Page loads successfully
- ✅ No JavaScript errors in console (F12)
- ✅ Main content visible
- ✅ Navigation works

**If Fails:** Check browser console for errors

### Step 4: Console Check

**Open DevTools (F12) → Console**

**Expected:**
- ✅ 0 errors (red messages)
- ✅ 0 warnings (yellow messages)
- ⚠️ Info messages OK (blue)

**Common Issues:**
- Hydration errors → Check SSR/Client mismatch
- 404 errors → Check API routes
- CORS errors → Check next.config.js headers

### Step 5: Interaction Test

**Manual Testing:**

- [ ] Click navigation links → Pages load
- [ ] Submit forms → Data saves
- [ ] Toggle filters → UI updates
- [ ] Search functionality → Results show
- [ ] User actions → No errors

**If Fails:** Debug specific component

### Step 6: Network Test (API)

**Open DevTools → Network Tab**

```bash
# Test API endpoints
curl http://localhost:3000/api/health
curl http://localhost:3000/api/binance/futures | jq
```

**Expected:**
- ✅ All APIs return 200 OK
- ✅ Response time < 500ms
- ✅ Valid JSON responses
- ✅ No CORS errors

**If Fails:** Check API route implementations

### Step 7: Production Build

```bash
NODE_ENV=production pnpm build
```

**Expected:**
- ✅ Build completes without errors
- ✅ .next/ directory created
- ✅ Bundle size < 5MB

**If Fails:** Check build logs for errors

### Step 8: Production Test

```bash
pnpm start
# In another terminal:
open http://localhost:3000
```

**Expected:**
- ✅ Production server starts
- ✅ Page loads < 2 seconds
- ✅ All features work
- ✅ No console errors

**If Fails:** Check production build configuration

---

## 📱 MOBILE OPTIMIZATION CHECKLIST

### Responsive Design

- [ ] **Breakpoints Tested:**
  - 375px (iPhone SE)
  - 390px (iPhone 12/13/14)
  - 414px (iPhone Plus)
  - 768px (iPad)
  - 1024px (Desktop)
  - 1920px (Large Desktop)

- [ ] **Touch Targets:** Min 44x44px
- [ ] **Font Sizes:** Readable on mobile (min 14px)
- [ ] **Scrolling:** Smooth, no janky animations
- [ ] **Hamburger Menu:** Works properly

### Mobile Testing Commands

```bash
# Simulate mobile in Chrome DevTools
# 1. Open DevTools (F12)
# 2. Toggle Device Toolbar (Ctrl+Shift+M)
# 3. Test different devices

# Test on real device:
# Find your local IP:
ip addr show | grep "inet " | grep -v 127.0.0.1

# Access from mobile:
# http://YOUR_LOCAL_IP:3000
```

### PWA Features

- [ ] **Manifest:** public/manifest.json exists
- [ ] **Icons:** 192x192 and 512x512 PNG
- [ ] **Service Worker:** Offline support (optional)
- [ ] **Install Prompt:** "Add to Home Screen" works
- [ ] **Splash Screen:** Shows on app launch

### Performance on Mobile

**Target Metrics:**
- FCP (First Contentful Paint): < 1.8s
- LCP (Largest Contentful Paint): < 2.5s
- TTI (Time to Interactive): < 3.8s
- CLS (Cumulative Layout Shift): < 0.1
- FID (First Input Delay): < 100ms

**Test with:**
```bash
# Lighthouse Mobile
npx lighthouse http://localhost:3000 --preset=mobile --view
```

---

## ⚡ PERFORMANCE TESTING

### Lighthouse Audit

```bash
# Desktop
npx lighthouse http://localhost:3000 --preset=desktop --view

# Mobile
npx lighthouse http://localhost:3000 --preset=mobile --view

# Target Scores:
# Performance: > 90
# Accessibility: > 95
# Best Practices: > 95
# SEO: > 90
```

### Bundle Analysis

```bash
# Analyze bundle size
ANALYZE=true pnpm build

# View report
open .next/analyze.html

# Recommendations:
# - Keep vendor bundle < 300KB
# - Keep page bundles < 150KB each
# - Use dynamic imports for heavy libraries
```

### Core Web Vitals

**Tools:**
- Chrome DevTools → Lighthouse
- web.dev/measure
- Vercel Analytics (production)

**Target Metrics:**
- **LCP:** < 2.5s (Good)
- **FID:** < 100ms (Good)
- **CLS:** < 0.1 (Good)

### Load Testing

```bash
# Install autocannon
npm install -g autocannon

# Test API endpoint
autocannon -c 100 -d 30 http://localhost:3000/api/health

# Expected:
# - Latency p99 < 200ms
# - No errors
# - Throughput > 1000 req/sec
```

---

## 🔒 SECURITY VERIFICATION

### Security Headers Check

```bash
# Check security headers
curl -I https://borsa.ailydian.com

# Must include:
# Strict-Transport-Security: max-age=63072000
# X-Frame-Options: SAMEORIGIN
# X-Content-Type-Options: nosniff
# Content-Security-Policy: ...
# Referrer-Policy: strict-origin-when-cross-origin
```

### Dependency Audit

```bash
# Check for vulnerabilities
pnpm audit

# Fix vulnerabilities
pnpm audit --fix

# Expected: 0 high/critical vulnerabilities
```

### SQL Injection Test

```bash
# Test with malicious input (safely)
curl -X POST http://localhost:3000/api/test \
  -H "Content-Type: application/json" \
  -d '{"input": "'; DROP TABLE users; --"}'

# Expected: 400 Bad Request (input validation works)
```

### XSS Test

```bash
# Test script injection
curl "http://localhost:3000/search?q=<script>alert('XSS')</script>"

# Expected: Script tag sanitized/escaped
```

### Rate Limiting Test

```bash
# Rapid requests
for i in {1..150}; do
  curl http://localhost:3000/api/health &
done
wait

# Expected: Some requests return 429 Too Many Requests
```

---

## 🚀 VERCEL DEPLOYMENT

### Prerequisites

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login
```

### Deploy Steps

#### 1. Link Project

```bash
cd ~/Desktop/PROJELER/borsa.ailydian.com
vercel link
```

#### 2. Set Environment Variables

```bash
# Add production env vars
vercel env add DATABASE_URL production
vercel env add NEXTAUTH_SECRET production
vercel env add GROQ_API_KEY production
# ... (all required vars)

# Or import from .env.production
vercel env pull .env.production
```

#### 3. Deploy to Production

```bash
# Deploy
vercel --prod

# Expected output:
# ✅ Deployment Complete
# 🔗 https://borsa-ailydian.vercel.app
```

#### 4. Custom Domain

```bash
# Add custom domain
vercel domains add borsa.ailydian.com

# Configure DNS:
# Type: CNAME
# Name: borsa
# Value: cname.vercel-dns.com
```

### Vercel Dashboard Configuration

**Settings → General:**
- Framework Preset: Next.js
- Node.js Version: 20.x
- Build Command: `pnpm build`
- Output Directory: `.next`
- Install Command: `pnpm install`

**Settings → Environment Variables:**
- Add all production variables
- Scope: Production

**Settings → Deployment Protection:**
- Enable Vercel Authentication (optional)
- Configure allowed emails/domains

---

## 📊 POST-DEPLOYMENT MONITORING

### Health Check

```bash
# Check deployment
curl https://borsa.ailydian.com/api/health

# Expected:
# {"status":"ok","timestamp":"2025-01-09T..."}
```

### Monitoring Setup

#### 1. Vercel Analytics

```typescript
// Already configured in src/app/layout.tsx
import { Analytics } from '@vercel/analytics/react';

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
```

#### 2. Sentry Error Tracking

```typescript
// sentry.client.config.ts
import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  tracesSampleRate: 1.0,
});
```

#### 3. Uptime Monitoring

**Options:**
- UptimeRobot (free)
- Pingdom
- Better Uptime
- Vercel Monitoring (paid)

**Setup:**
- Monitor: https://borsa.ailydian.com/api/health
- Interval: 5 minutes
- Alert: Email/SMS on failure

#### 4. Performance Monitoring

**Tools:**
- Vercel Speed Insights
- Google Analytics 4
- Lighthouse CI
- WebPageTest

### Daily Checks

```bash
# Daily health check script
#!/bin/bash

URL="https://borsa.ailydian.com"

# 1. Health endpoint
curl -s "$URL/api/health" | jq

# 2. Response time
curl -o /dev/null -s -w "Response Time: %{time_total}s\n" "$URL"

# 3. Check status code
STATUS=$(curl -o /dev/null -s -w "%{http_code}" "$URL")
if [ "$STATUS" == "200" ]; then
  echo "✅ Site is up!"
else
  echo "❌ Site is down! Status: $STATUS"
fi
```

---

## 🐛 TROUBLESHOOTING

### Common Issues

#### 1. Build Fails

**Error:** `Type error: ...`

**Solution:**
```bash
# Check TypeScript errors
pnpm typecheck

# Fix errors in reported files
# Then rebuild
pnpm build
```

#### 2. Page Not Found (404)

**Error:** 404 on production but works locally

**Solution:**
- Check `next.config.js` redirects
- Verify file structure in `src/app/`
- Check dynamic routes `[slug]` configuration

#### 3. API Returns 500

**Error:** Internal Server Error

**Solution:**
```bash
# Check Vercel logs
vercel logs --follow

# Check environment variables
vercel env ls

# Test API locally
pnpm dev
curl http://localhost:3000/api/problematic-endpoint
```

#### 4. Slow Performance

**Symptoms:** LCP > 4s, slow loading

**Solution:**
1. Run Lighthouse audit
2. Check bundle size: `du -sh .next/static`
3. Enable caching in Redis
4. Optimize images with next/image
5. Use dynamic imports

#### 5. Mobile Layout Broken

**Symptoms:** Horizontal scroll, overlapping elements

**Solution:**
- Check Tailwind responsive classes (sm:, md:, lg:)
- Verify `max-w-full` on containers
- Test with DevTools mobile emulation
- Check `overflow-x: hidden` on body

#### 6. Database Connection Timeout

**Error:** `Prisma Client initialization timeout`

**Solution:**
```bash
# Verify DATABASE_URL in production
vercel env ls

# Check Prisma client generation
pnpm prisma generate

# Redeploy
vercel --prod
```

### Debug Mode

```bash
# Enable debug logging
DEBUG=* pnpm dev

# Check Next.js build output
pnpm build --debug

# Analyze bundle
ANALYZE=true pnpm build
```

---

## 📈 SUCCESS METRICS

### Performance Targets

- ✅ Lighthouse Score: > 90
- ✅ FCP: < 1.5s
- ✅ LCP: < 2.5s
- ✅ TTI: < 3.5s
- ✅ CLS: < 0.1
- ✅ FID: < 100ms

### Reliability Targets

- ✅ Uptime: > 99.9%
- ✅ Error Rate: < 0.1%
- ✅ API Response Time: < 200ms (p99)
- ✅ Build Success Rate: 100%

### User Experience Targets

- ✅ Mobile Score: > 85
- ✅ Accessibility Score: > 95
- ✅ SEO Score: > 90
- ✅ PWA Installable: ✅

---

## 🎉 DEPLOYMENT COMPLETE!

Your production deployment is now %100 optimized and ready!

**Next Steps:**
1. Monitor Vercel Analytics
2. Set up alerting (Sentry, PagerDuty)
3. Regular security audits
4. Performance monitoring
5. User feedback collection

**Support:**
- Vercel Docs: https://vercel.com/docs
- Next.js Docs: https://nextjs.org/docs
- Project Issues: Check logs and troubleshooting section

---

**Last Updated:** 2025-01-09
**Version:** 1.0.0
**ZERO-ERROR PROTOCOL:** Enforced ✅
