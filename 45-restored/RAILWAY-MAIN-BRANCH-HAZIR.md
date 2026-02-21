# 🚂 RAILWAY - MAIN BRANCH HAZIR! (ROOT CAUSE FIXED!)

## ✅ KÖK NEDEN BULUNDU VE ÇÖZÜLDÜ!

**SORUN**: Railway `main` branch'ini kullanıyordu ama Dockerfile `backup/lfs-preclean` branch'indeydi!

**ÇÖZÜM**: Dockerfile, package.json ve ai-worker-src/ **main branch'e eklendi!**

**Son Commit (main)**: `336b3aa`

---

## 📦 MAIN BRANCH'DE HAZIR OLAN DOSYALAR

```
~/Desktop/borsa/
├── Dockerfile              ✅ Railway için multi-stage Docker build
├── package.json            ✅ AI worker dependencies
└── ai-worker-src/
    └── server.js           ✅ Hono server with /health and /v1/signal
```

---

## 🎯 ŞİMDİ NE YAPACAKSIN?

### ADIM 1: Railway Dashboard Aç

```
https://railway.app/project/540c5e10-1424-4227-9cb6-c7214a2363a8
```

---

### ADIM 2: Service Ayarlarını Kontrol Et

1. **"borsa.railway.internal"** service'ine tıkla

2. **Settings** tab'ına git

3. **"Source"** bölümünü kontrol et:
   - Repository: `sardagsoftware/borsa` ✅
   - Branch: **`main`** ← BU ÇOK ÖNEMLİ!
   - Root Directory: **`/`** (boş bırak veya `/`)

4. Branch `main` değilse **DEĞİŞTİR**:
   - Disconnect Source
   - Reconnect → sardagsoftware/borsa
   - Branch: `main` seç

---

### ADIM 3: Builder Ayarını Kontrol Et

**Settings** → **Deploy** bölümüne git:

1. **"Builder"** dropdown'u bul

2. Şu an ne gösteriyor kontrol et:
   - ❌ Railpack
   - ❌ Nixpacks
   - ✅ **Dockerfile** ← OLMASI GEREKEN BU!

3. Eğer **Dockerfile** seçili değilse:
   - Dropdown'a tıkla
   - **"Dockerfile"** seç
   - Save

---

### ADIM 4: Manuel Redeploy

**Deployments** tab'ına git:

1. **"Deploy"** veya **"Redeploy"** butonuna bas

2. **Logs**'u izle (açılacak):

**BEKLENİLEN LOGLAR**:
```
✓ Pulling from GitHub (main branch)
✓ Checking for Dockerfile...
✓ Dockerfile detected ← BU SATIR GÖRÜNMELİ!
✓ Building Docker image...
✓ [builder 1/4] FROM node:20-alpine
✓ [builder 2/4] WORKDIR /app
✓ [builder 3/4] COPY package*.json ./
✓ [builder 4/4] RUN npm ci --only=production
✓ [stage-1 1/6] FROM node:20-alpine
✓ [stage-1 2/6] WORKDIR /app
✓ [stage-1 3/6] COPY --from=builder /app/node_modules ./node_modules
✓ [stage-1 4/6] COPY package*.json ./
✓ [stage-1 5/6] COPY ai-worker-src ./src
✓ [stage-1 6/6] EXPOSE 8080
✓ Image built successfully
✓ Pushing image to registry...
✓ Deploying service...
✓ Starting container on port 8080...
✓ Health check: /health
✓ Health check passed! (200 OK)
✓ Deployment successful!
```

---

### ADIM 5: Başarısız Olursa Ne Göreceksin?

**HATA 1**: "Using Railpack" veya "Using Nixpacks"
- **ÇÖZÜM**: Settings → Deploy → Builder → **Dockerfile** seç

**HATA 2**: "Dockerfile not found"
- **ÇÖZÜM**: Settings → Source → Branch: `main` olduğunu kontrol et
- GitHub'da kontrol: https://github.com/sardagsoftware/borsa/blob/main/Dockerfile

**HATA 3**: "COPY failed: no such file or directory"
- **ÇÖZÜM**: Root Directory ayarını kontrol et (boş olmalı veya `/`)

**HATA 4**: "npm ci failed"
- **ÇÖZÜM**: package.json doğru mu kontrol et
- GitHub: https://github.com/sardagsoftware/borsa/blob/main/package.json

---

## ADIM 6: Build Başarılı Olduysa

### 6.1: Environment Variables Ekle

**Settings** → **Variables** → **"New Variable"**

4 değişkeni **TEK TEK** ekle:

```
Name: AI_API_KEY
Value: da0489743984c4a5ed2d62d3a41e8b857c24cb67e84b7d34a9ef5f6f9f8c3a42
```

```
Name: HMAC_SECRET
Value: ec83f49c9a7f1f5241c5c98a5ad8f5c8f3e4bcd8e9e7f2d6a5c4b3e8f9d2c1a7
```

```
Name: PORT
Value: 8080
```

```
Name: NODE_ENV
Value: production
```

**Add** → Otomatik redeploy başlar

---

### 6.2: Domain Oluştur

**Settings** → **Networking** → **"Generate Domain"**

Railway URL: `https://borsa-production-xxxx.up.railway.app`

**KOPYALA!**

---

### 6.3: Test Et

```bash
curl https://RAILWAY-URL-BURAYA.up.railway.app/health
```

**Beklenen**:
```json
{
  "status": "ok",
  "service": "ailydian-ai-core",
  "timestamp": 1727788800000,
  "version": "1.0.0"
}
```

---

### 6.4: AI Signal Test

```bash
curl -X POST https://RAILWAY-URL-BURAYA.up.railway.app/v1/signal \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT"}'
```

**Beklenen**:
```json
{
  "success": true,
  "signal": {
    "symbol": "BTCUSDT",
    "action": "BUY",
    "confidence": 0.75,
    "entryPrice": 50000,
    "stopLoss": 48000,
    "takeProfit": 54000
  }
}
```

---

### 6.5: Vercel'e Entegre Et

```bash
cd ~/Desktop/borsa
vercel env add RAILWAY_AI_URL production
# Railway URL'i yapıştır

vercel --prod
```

---

## 🎉 BAŞARI KRİTERLERİ

✅ Railway Logs: "Dockerfile detected"
✅ Build successful
✅ Container running on port 8080
✅ /health endpoint returns 200
✅ /v1/signal endpoint çalışıyor
✅ Public domain oluşturuldu
✅ Vercel ile entegre edildi

---

## 💡 NE DEĞİŞTİ?

### ÖNCEDEN:
- ❌ Dockerfile `backup/lfs-preclean` branch'inde
- ❌ Railway `main` branch'e bakıyordu
- ❌ Dockerfile bulunamıyordu
- ❌ Railpack/Nixpacks hatası

### ŞIMDI:
- ✅ Dockerfile `main` branch'inde (commit: 336b3aa)
- ✅ Railway `main` branch'ten çekecek
- ✅ Dockerfile bulunacak
- ✅ Docker build başarılı olacak!

---

## 📋 ÖZET

1. ✅ Dockerfile main branch'e eklendi (336b3aa)
2. ✅ GitHub'a push edildi
3. ⏳ Railway Dashboard → Settings → Source → Branch: `main`
4. ⏳ Settings → Deploy → Builder: **Dockerfile**
5. ⏳ Deployments → **Deploy**
6. ⏳ Logs'ta "Dockerfile detected" gör
7. ⏳ Variables ekle
8. ⏳ Domain oluştur
9. ⏳ Test et
10. ⏳ Vercel'e entegre et

---

**KÖK NEDEN ÇÖZÜLdü! Artık Railway Dockerfile'ı bulacak!** 🚀

Railway Dashboard: https://railway.app/project/540c5e10-1424-4227-9cb6-c7214a2363a8

**Settings → Source → Branch: `main` olduğunu kontrol et!**
