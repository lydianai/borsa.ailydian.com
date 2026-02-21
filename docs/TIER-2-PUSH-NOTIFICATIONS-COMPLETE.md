# 📨 TIER 2: PUSH NOTIFICATIONS (FCM/APNs) - COMPLETE

**Tarih:** 24 Ekim 2025
**Durum:** ✅ %100 Tamamlandı
**Güvenlik:** White-hat uyumlu (Token-based auth, audit logging, invalid token cleanup)

---

## 📦 İçerik

### 1. **Firebase Admin Initialization**
Firebase Admin SDK entegrasyonu (FCM + APNs desteği).

**Dosya:** `src/lib/push/firebase-admin.ts` (~100 satır)

**Özellikler:**
- ✅ Singleton pattern (tek instance)
- ✅ Lazy initialization
- ✅ Service account authentication
- ✅ Environment-based configuration
- ✅ Graceful shutdown support
- ✅ White-hat logging

**API:**
```typescript
import { getFirebaseAdmin, getMessaging, isFirebaseAvailable } from '@/lib/push/firebase-admin';

// Get Firebase Admin instance
const app = getFirebaseAdmin();

// Get Messaging service
const messaging = getMessaging();

// Check availability
if (isFirebaseAvailable()) {
  // Firebase ready
}

// Graceful shutdown
await shutdownFirebase();
```

---

### 2. **Device Token Manager**
FCM device token yönetim sistemi (in-memory storage).

**Dosya:** `src/lib/push/device-token-manager.ts` (~250 satır)

**Özellikler:**
- ✅ In-memory token storage (production'da database'e upgrade edilebilir)
- ✅ User-device mapping
- ✅ Platform tracking (iOS/Android/Web)
- ✅ Token expiration handling (90 gün)
- ✅ Invalid token cleanup
- ✅ Metadata support (device model, OS version)
- ✅ Statistics tracking

**API:**
```typescript
import deviceTokenManager from '@/lib/push/device-token-manager';

// Register token
deviceTokenManager.registerToken(
  'fcm-token-here',
  'user123',
  'ios',
  {
    deviceModel: 'iPhone 15 Pro',
    osVersion: 'iOS 17.0',
    appVersion: '1.0.0'
  }
);

// Get user's tokens
const tokens = deviceTokenManager.getUserTokens('user123');
// ['token1', 'token2', ...]

// Get all tokens (for broadcast)
const allTokens = deviceTokenManager.getAllTokens();

// Cleanup expired tokens
const removed = deviceTokenManager.cleanupExpiredTokens(90); // 90 days

// Statistics
const stats = deviceTokenManager.getStats();
// {
//   totalTokens: 150,
//   totalUsers: 50,
//   platformBreakdown: { ios: 75, android: 60, web: 15 }
// }
```

---

### 3. **Push Notification Service**
FCM push notification gönderim servisi.

**Dosya:** `src/lib/push/push-notification-service.ts` (~300 satır)

**Özellikler:**
- ✅ Send to single device
- ✅ Send to multiple devices (batch)
- ✅ Send to user (all devices)
- ✅ Broadcast to all users
- ✅ Trading signal notifications (özel template)
- ✅ Test notifications
- ✅ Invalid token handling
- ✅ Platform-specific config (iOS/Android/Web)
- ✅ Delivery tracking

**Signal Notification Template:**
```
🟢 BUY Signal - BTCUSDT
🔥 Confidence: 92% | Price: $67,234 | MA Crossover Pullback
```

**API:**
```typescript
import pushNotificationService from '@/lib/push/push-notification-service';

// Send trading signal
await pushNotificationService.sendSignalNotification({
  symbol: 'BTCUSDT',
  signal: 'BUY',
  confidence: 92,
  price: 67234,
  strategy: 'MA Crossover Pullback',
  reason: 'Strong uptrend with pullback to 7-day MA'
});

// Send to specific user
await pushNotificationService.sendToUser('user123', {
  title: 'Test Notification',
  body: 'Hello from Sardag AI!',
  data: { type: 'custom' }
});

// Broadcast to all users
await pushNotificationService.broadcast({
  title: '🚨 Market Alert',
  body: 'BTC just crossed $70,000!',
  data: { type: 'market-alert', symbol: 'BTCUSDT' }
});

// Test notification
await pushNotificationService.sendTestNotification('fcm-token');
```

---

### 4. **API Endpoints**

#### **POST /api/push/register**
Device token kaydı.

**Request:**
```json
{
  "token": "fcm-token-here",
  "userId": "user123",
  "platform": "ios",
  "metadata": {
    "deviceModel": "iPhone 15 Pro",
    "osVersion": "iOS 17.0",
    "appVersion": "1.0.0"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Device token registered successfully",
  "userId": "user123",
  "platform": "ios"
}
```

#### **DELETE /api/push/register**
Device token silme.

**Request:**
```json
{
  "token": "fcm-token-here"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Device token unregistered successfully"
}
```

#### **POST /api/push/send**
Push notification gönder (requires `INTERNAL_SERVICE_TOKEN`).

**Authentication:** `x-service-token` header required.

**Request (Signal):**
```json
{
  "type": "signal",
  "signal": {
    "symbol": "BTCUSDT",
    "signal": "BUY",
    "confidence": 92,
    "price": 67234,
    "strategy": "MA Crossover Pullback"
  },
  "userIds": ["user1", "user2"]  // Optional, omit for broadcast
}
```

**Request (Test):**
```json
{
  "type": "test",
  "token": "fcm-token-here"
}
```

**Request (Custom):**
```json
{
  "type": "custom",
  "payload": {
    "title": "Custom Title",
    "body": "Custom Body",
    "data": { "key": "value" }
  },
  "userIds": ["user1", "user2"]  // Optional
}
```

**Response:**
```json
{
  "success": true,
  "messageId": "projects/your-project/messages/1234567890",
  "invalidTokens": ["token1", "token2"]  // If any
}
```

#### **GET /api/push/stats**
Push notification istatistikleri.

**Response:**
```json
{
  "timestamp": "2025-10-24T12:00:00.000Z",
  "firebase": {
    "available": true,
    "status": "connected"
  },
  "devices": {
    "totalTokens": 150,
    "totalUsers": 50,
    "platformBreakdown": {
      "ios": 75,
      "android": 60,
      "web": 15
    }
  }
}
```

---

## 🔧 Configuration (.env)

### Firebase Service Account Setup

1. **Firebase Console'a Git:**
   - https://console.firebase.google.com/
   - Projenizi seçin

2. **Service Account Key Oluştur:**
   - Project Settings → Service Accounts
   - "Generate New Private Key" butonuna tıkla
   - JSON dosyasını indir

3. **JSON'u Minify Et:**
   ```bash
   # Minify JSON (tek satır yap)
   cat service-account.json | jq -c > service-account-minified.json
   ```

4. **.env Dosyasına Ekle:**
   ```bash
   FIREBASE_SERVICE_ACCOUNT='{"type":"service_account","project_id":"your-project-id",...}'
   ```

**⚠️ Güvenlik Uyarısı:**
- Service account JSON'u asla Git'e commit etmeyin!
- `.env` dosyasını `.gitignore`'a ekleyin
- Production'da environment variable olarak ayarlayın

---

## 📊 Platform-Specific Features

### iOS (APNs via FCM)

```typescript
apns: {
  payload: {
    aps: {
      sound: 'default',
      badge: 1,
      'content-available': 1,  // Background notification
      category: 'TRADING_SIGNAL'
    }
  }
}
```

### Android (FCM)

```typescript
android: {
  priority: 'high',
  notification: {
    sound: 'default',
    channelId: 'trading-signals',
    color: '#00FF00',
    icon: 'ic_notification'
  },
  data: {
    click_action: 'FLUTTER_NOTIFICATION_CLICK'
  }
}
```

### Web Push

```typescript
webpush: {
  notification: {
    icon: '/icon-192x192.png',
    badge: '/icon-96x96.png',
    vibrate: [200, 100, 200],
    requireInteraction: true
  },
  fcm_options: {
    link: '/signals'  // Click destination
  }
}
```

---

## 🧪 Testing

### 1. Register Device Token

```bash
curl -X POST http://localhost:3000/api/push/register \
  -H "Content-Type: application/json" \
  -d '{
    "token": "your-fcm-token-here",
    "userId": "test-user",
    "platform": "web",
    "metadata": {
      "deviceModel": "Chrome",
      "osVersion": "macOS 14.0"
    }
  }' | jq
```

### 2. Send Test Notification

```bash
curl -X POST http://localhost:3000/api/push/send \
  -H "Content-Type: application/json" \
  -H "x-service-token: your_token_here" \
  -d '{
    "type": "test",
    "token": "your-fcm-token-here"
  }' | jq
```

### 3. Send Trading Signal

```bash
curl -X POST http://localhost:3000/api/push/send \
  -H "Content-Type: application/json" \
  -H "x-service-token: your_token_here" \
  -d '{
    "type": "signal",
    "signal": {
      "symbol": "BTCUSDT",
      "signal": "BUY",
      "confidence": 92,
      "price": 67234,
      "strategy": "MA Crossover Pullback"
    }
  }' | jq
```

### 4. Check Stats

```bash
curl http://localhost:3000/api/push/stats | jq
```

---

## 🚀 Production Usage

### Client-Side Integration (Web)

```typescript
// public/firebase-messaging-sw.js (Service Worker)
importScripts('https://www.gstatic.com/firebasejs/10.7.0/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.7.0/firebase-messaging-compat.js');

firebase.initializeApp({
  apiKey: "your-api-key",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abcdef"
});

const messaging = firebase.messaging();

messaging.onBackgroundMessage((payload) => {
  console.log('Background message received:', payload);

  const { title, body } = payload.notification;
  self.registration.showNotification(title, {
    body,
    icon: '/icon-192x192.png',
    badge: '/icon-96x96.png'
  });
});
```

```typescript
// app.tsx (Client)
import { getMessaging, getToken, onMessage } from 'firebase/messaging';

async function requestNotificationPermission() {
  const permission = await Notification.requestPermission();

  if (permission === 'granted') {
    const messaging = getMessaging();
    const token = await getToken(messaging, {
      vapidKey: 'your-vapid-key'
    });

    // Register token
    await fetch('/api/push/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        token,
        userId: currentUser.id,
        platform: 'web'
      })
    });

    // Listen for foreground messages
    onMessage(messaging, (payload) => {
      console.log('Foreground message:', payload);
      showToast(payload.notification.title, payload.notification.body);
    });
  }
}
```

---

## 📂 Dosya Yapısı

```
src/
├── lib/
│   └── push/
│       ├── firebase-admin.ts                 # Firebase Admin SDK (~100 satır)
│       ├── device-token-manager.ts           # Token Manager (~250 satır)
│       └── push-notification-service.ts      # Push Service (~300 satır)
├── app/
│   └── api/
│       └── push/
│           ├── register/
│           │   └── route.ts                  # POST/DELETE /api/push/register (~110 satır)
│           ├── send/
│           │   └── route.ts                  # POST /api/push/send (~140 satır)
│           └── stats/
│               └── route.ts                  # GET /api/push/stats (~40 satır)
```

**Toplam:** ~950 satır kod

---

## 📊 Metrics Summary

| Metric | Value |
|--------|-------|
| **Lines of Code** | 950 (services: 650, API: 300) |
| **Files Created** | 7 |
| **API Endpoints** | 4 (register, send, delete, stats) |
| **Supported Platforms** | 3 (iOS, Android, Web) |
| **Notification Types** | 3 (signal, test, custom) |
| **Token Storage** | In-memory (upgradeable to DB) |
| **Max Token Age** | 90 days |
| **Batch Send Support** | ✅ Yes |
| **Invalid Token Cleanup** | ✅ Automatic |

---

## 🎉 Conclusion

**TIER 2: Push Notifications %100 tamamlandı!**

- ✅ Firebase Admin SDK entegrasyonu
- ✅ Device token yönetimi (in-memory)
- ✅ Push notification servisi (FCM + APNs)
- ✅ Trading signal templates
- ✅ API endpoints (register, send, stats)
- ✅ Platform-specific configuration
- ✅ Invalid token cleanup
- ✅ White-hat compliance (all operations logged)

**Dependency:**
- `firebase-admin@^12.0.0` ✅ Installed

**Next Integration:**
Scanner → Signal Detection → Push Notification

---

## 🔗 Scanner Integration Example

```typescript
// src/lib/queue/strategy-worker.ts (enhancement)
import pushNotificationService from '@/lib/push/push-notification-service';

async function processJob(job: Job<ScanJobData>): Promise<JobResult> {
  // ... existing strategy analysis code ...

  for (const result of results) {
    const { symbol, analysis } = result;

    // Check if any strategy gave a strong BUY/SELL signal
    const strongSignals = analysis.strategies.filter(
      (s) => (s.signal === 'BUY' || s.signal === 'SELL') && s.confidence >= 85
    );

    if (strongSignals.length > 0) {
      const topSignal = strongSignals[0];

      // Send push notification
      await pushNotificationService.sendSignalNotification({
        symbol,
        signal: topSignal.signal as 'BUY' | 'SELL',
        confidence: topSignal.confidence,
        price: priceData.price,
        strategy: topSignal.name,
        reason: topSignal.reason
      });

      console.log(`[Worker] 📨 Push notification sent for ${symbol} ${topSignal.signal} signal`);
    }
  }

  return { jobId, requestId, results, duration };
}
```

---

**Status:** Ready for production deployment after Firebase setup.
