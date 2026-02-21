# 🚀 Production Deployment Guide

## Ailydian Signal - Production Deployment Checklist

**Last Updated:** 2025-01-19
**Version:** 1.0.0

---

## 📋 Pre-Deployment Checklist

### 1. Environment Variables

Ensure all required environment variables are configured in production:

#### **Critical (Required)**
```bash
# Application
NODE_ENV=production
NEXT_PUBLIC_APP_URL=https://your-domain.com

# Database (PostgreSQL)
DATABASE_URL=postgresql://user:password@host:5432/database

# AI Services (at least one required)
GROQ_API_KEY=your_groq_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

#### **Recommended**
```bash
# Caching (Redis/Upstash)
UPSTASH_REDIS_REST_URL=your_redis_url
UPSTASH_REDIS_REST_TOKEN=your_redis_token

# Monitoring (Sentry)
SENTRY_DSN=your_sentry_dsn
NEXT_PUBLIC_SENTRY_DSN=your_public_sentry_dsn
SENTRY_AUTH_TOKEN=your_auth_token

# Security
CSRF_SECRET=your_random_32_char_secret
```

#### **Optional**
```bash
# External APIs
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET=your_binance_secret
TELEGRAM_BOT_TOKEN=your_telegram_token
```

### 2. Security Checklist

- [ ] ✅ All API keys stored in environment variables (NOT in code)
- [ ] ✅ CSRF protection enabled
- [ ] ✅ Security headers configured (CSP, HSTS, X-Frame-Options)
- [ ] ✅ Input validation active (Zod schemas)
- [ ] ✅ Rate limiting configured
- [ ] ✅ Error messages sanitized (no stack traces in production)
- [ ] ✅ Sentry configured for error monitoring

### 3. Database Setup

```bash
# Run Prisma migrations
pnpm prisma migrate deploy

# Generate Prisma client
pnpm prisma generate

# Verify connection
pnpm prisma db pull
```

### 4. Build and Test

```bash
# Type check
pnpm typecheck

# Run tests
pnpm test

# Build for production
pnpm build

# Test production build locally
pnpm start
```

---

## 🌐 Deployment Platforms

### Option 1: Vercel (Recommended)

#### **Advantages**
- Zero-config deployment
- Automatic SSL
- Edge network CDN
- Preview deployments
- Serverless functions

#### **Setup**

1. **Install Vercel CLI**
```bash
npm i -g vercel
```

2. **Login**
```bash
vercel login
```

3. **Configure Project**
```bash
# Link project
vercel link

# Set environment variables
vercel env add DATABASE_URL
vercel env add GROQ_API_KEY
# ... add all required env vars
```

4. **Deploy**
```bash
# Production deployment
vercel --prod

# Preview deployment
vercel
```

#### **Vercel Configuration** (`vercel.json`)
```json
{
  "buildCommand": "pnpm build",
  "outputDirectory": ".next",
  "framework": "nextjs",
  "regions": ["iad1"],
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=60, stale-while-revalidate=30"
        }
      ]
    }
  ]
}
```

---

### Option 2: Docker + VPS

#### **Dockerfile**
```dockerfile
FROM node:20-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Copy package files
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# Build stage
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build application
RUN pnpm build

# Production stage
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000

CMD ["node", "server.js"]
```

#### **docker-compose.yml**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - GROQ_API_KEY=${GROQ_API_KEY}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=ailydian
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### **Deploy to VPS**
```bash
# Copy files to VPS
scp -r . user@your-server:/app

# SSH to server
ssh user@your-server

# Navigate to app directory
cd /app

# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f
```

---

## 🔧 Post-Deployment

### 1. Verify Deployment

```bash
# Health check
curl https://your-domain.com/api/health

# Test API endpoints
curl https://your-domain.com/api/signals?limit=5

# Check error reporting (Sentry dashboard)
# Visit: https://sentry.io/organizations/your-org/issues/
```

### 2. Performance Monitoring

- **Vercel Analytics**: Auto-enabled on Vercel
- **Sentry Performance**: Check transaction traces
- **Custom Monitoring**: Set up alerts for high error rates

### 3. Database Backups

#### **Automated Backups (Recommended)**
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump $DATABASE_URL > backups/ailydian_$DATE.sql
```

#### **Manual Backup**
```bash
# Export database
pnpm prisma db push --force-reset
```

---

## 📊 Monitoring & Alerts

### Key Metrics to Monitor

1. **API Response Time**
   - Target: p95 < 500ms
   - Alert: p95 > 1000ms

2. **Error Rate**
   - Target: < 1%
   - Alert: > 5%

3. **Database Connections**
   - Target: < 80% of max
   - Alert: > 90%

4. **Cache Hit Rate**
   - Target: > 70%
   - Alert: < 50%

### Sentry Alerts

Configure alerts for:
- New error types
- Error rate spike (>10 errors/min)
- Performance regression (>500ms p95)

---

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: pnpm/action-setup@v2
        with:
          version: 8
      - uses: actions/setup-node@v3
        with:
          node-version: 20
          cache: 'pnpm'

      - run: pnpm install
      - run: pnpm typecheck
      - run: pnpm test
      - run: pnpm build

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          vercel-args: '--prod'
```

---

## 🛡️ Security Best Practices

### Production Hardening

1. **Environment Secrets**
   - Never commit `.env.local`
   - Use platform secret managers
   - Rotate secrets regularly

2. **Database Security**
   - Enable SSL connections
   - Use connection pooling
   - Restrict IP access

3. **API Security**
   - Enable rate limiting
   - Validate all inputs
   - Use HTTPS only

4. **Monitoring**
   - Enable Sentry in production
   - Set up uptime monitoring
   - Configure log aggregation

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Build Fails
```bash
# Clear cache and rebuild
rm -rf .next
rm -rf node_modules
pnpm install
pnpm build
```

#### 2. Database Connection Issues
```bash
# Verify connection string
pnpm prisma db pull

# Check Prisma client generation
pnpm prisma generate
```

#### 3. High Memory Usage
- Enable Redis caching
- Optimize bundle size with ANALYZE=true
- Review Sentry for memory leaks

#### 4. Slow API Responses
- Check cache hit rate
- Enable CDN caching
- Optimize database queries

---

## 📞 Support

For issues or questions:
- **Documentation**: Check README.md
- **Logs**: Review Sentry dashboard
- **Monitoring**: Check Vercel analytics

---

## ✅ Final Checklist

Before going live:

- [ ] ✅ All environment variables configured
- [ ] ✅ Database migrations applied
- [ ] ✅ SSL certificate active
- [ ] ✅ Sentry configured and tested
- [ ] ✅ Cache strategy validated
- [ ] ✅ Load testing completed
- [ ] ✅ Backup strategy in place
- [ ] ✅ Monitoring alerts configured
- [ ] ✅ Error handling tested
- [ ] ✅ API documentation complete

---

**🎉 Ready for Production!**
