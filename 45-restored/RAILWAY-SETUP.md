# Railway Environment Variable Setup

## 🔐 CRITICAL: Railway AI API Secret Configuration

**Action Required**: Add the following environment variable to Railway Dashboard

### Step 1: Copy the Secret Key

```bash
RAILWAY_AI_API_SECRET=16b18cd1da71fd50a3b86f0f2bef7433a2b49ac2b98200d19bc0fbb1fe264d53
```

### Step 2: Add to Railway Dashboard

1. Go to Railway Dashboard: https://railway.app
2. Select project: **borsa-production**
3. Click on **Variables** tab
4. Add new variable:
   - **Name**: `RAILWAY_AI_API_SECRET`
   - **Value**: `16b18cd1da71fd50a3b86f0f2bef7433a2b49ac2b98200d19bc0fbb1fe264d53`
5. Click **Add** and **Deploy**

### Step 3: Verify Configuration

After deploying, verify the environment variable is set:

```bash
curl https://borsa-production.up.railway.app/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "ailydian-ai-core",
  "version": "1.0.0",
  "security": {
    "hmac": true,
    "rateLimit": true,
    "inputValidation": true
  }
}
```

---

## 🔒 Security Notes

⚠️ **NEVER commit this secret to Git!**

- Already added to `.gitignore`: `.env*.local`
- Local development: `.env.local` (already configured)
- Railway production: Environment Variables (manual setup required)
- Vercel production: Will be added later in deployment

---

## ✅ Configuration Checklist

- [x] Secret key generated (64-character hex string)
- [x] Added to Vercel `.env.local` (local development)
- [ ] **TODO**: Add to Railway Dashboard (production)
- [ ] **TODO**: Add to Vercel Environment Variables (production deployment)

---

## 📝 Next Steps

Once Railway environment variable is set:

1. Railway will automatically redeploy with the new secret
2. Test HMAC authentication from Vercel
3. Verify end-to-end integration

**Current Status**: Waiting for Railway environment variable configuration
