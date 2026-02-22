/**
 * LyTrade Trading Scanner - Service Worker
 * Offline-First PWA with Smart Caching
 */

const CACHE_VERSION = 'lydian-v1.0.0';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;
const API_CACHE = `${CACHE_VERSION}-api`;

// Static assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/market-scanner',
  '/trading-signals',
  '/ai-signals',
  '/quantum-signals',
  '/conservative-signals',
  '/settings',
  '/manifest.json',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png',
];

// Cache duration for different resource types
const CACHE_DURATION = {
  static: 7 * 24 * 60 * 60 * 1000,  // 7 days
  dynamic: 24 * 60 * 60 * 1000,     // 24 hours
  api: 60 * 1000,                    // 1 minute
};

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[SW] Installing Service Worker...', CACHE_VERSION);

  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating Service Worker...', CACHE_VERSION);

  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => name.startsWith('lydian-') && name !== CACHE_VERSION)
            .map((name) => {
              console.log('[SW] Deleting old cache:', name);
              return caches.delete(name);
            })
        );
      })
      .then(() => self.clients.claim())
  );
});

// Fetch event - network first for API, cache first for static
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') return;

  // Skip chrome extension requests
  if (url.protocol === 'chrome-extension:') return;

  // API requests - Network First with cache fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirstStrategy(request, API_CACHE));
    return;
  }

  // Static assets - Cache First with network fallback
  if (
    url.pathname.match(/\.(js|css|png|jpg|jpeg|svg|woff2|woff|ttf)$/) ||
    url.pathname.startsWith('/icons/') ||
    url.pathname === '/manifest.json'
  ) {
    event.respondWith(cacheFirstStrategy(request, STATIC_CACHE));
    return;
  }

  // HTML pages - Network First with cache fallback
  event.respondWith(networkFirstStrategy(request, DYNAMIC_CACHE));
});

// Network First Strategy (for fresh data)
async function networkFirstStrategy(request, cacheName) {
  try {
    const networkResponse = await fetch(request);

    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(cacheName);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.log('[SW] Network failed, trying cache:', request.url);
    const cachedResponse = await caches.match(request);

    if (cachedResponse) {
      return cachedResponse;
    }

    // Return offline page or error
    return new Response(
      JSON.stringify({
        error: 'Offline',
        message: 'İnternet bağlantısı yok. Lütfen bağlantınızı kontrol edin.'
      }),
      {
        status: 503,
        statusText: 'Service Unavailable',
        headers: new Headers({ 'Content-Type': 'application/json' }),
      }
    );
  }
}

// Cache First Strategy (for static assets)
async function cacheFirstStrategy(request, cacheName) {
  const cachedResponse = await caches.match(request);

  if (cachedResponse) {
    // Check if cache is stale
    const cacheDate = new Date(cachedResponse.headers.get('date'));
    const now = new Date();
    const cacheAge = now - cacheDate;

    if (cacheAge < CACHE_DURATION.static) {
      return cachedResponse;
    }
  }

  try {
    const networkResponse = await fetch(request);

    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(cacheName);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.log('[SW] Network failed, returning cache:', request.url);
    return cachedResponse || new Response('Offline');
  }
}

// Background Sync - for failed requests
self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync:', event.tag);

  if (event.tag === 'sync-signals') {
    event.waitUntil(syncSignals());
  }
});

async function syncSignals() {
  try {
    const cache = await caches.open(API_CACHE);
    const requests = await cache.keys();

    for (const request of requests) {
      try {
        const response = await fetch(request);
        if (response && response.status === 200) {
          await cache.put(request, response);
        }
      } catch (error) {
        console.log('[SW] Sync failed for:', request.url);
      }
    }
  } catch (error) {
    console.error('[SW] Background sync error:', error);
  }
}

// Push Notifications - ENHANCED with vibration and actions
self.addEventListener('push', (event) => {
  console.log('[SW] 🔔 Push notification received');

  let data = {
    title: 'LyTrade Trading Scanner',
    body: 'Yeni sinyal tespit edildi!',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/icon-96x96.png',
    tag: 'signal-notification',
    requireInteraction: true,
    priority: 'high',
  };

  if (event.data) {
    try {
      data = { ...data, ...event.data.json() };
    } catch (e) {
      data.body = event.data.text();
    }
  }

  const notificationOptions = {
    body: data.body,
    icon: data.icon,
    badge: data.badge,
    tag: data.tag,
    requireInteraction: data.requireInteraction,
    vibrate: [150, 80, 250, 80, 150], // Enhanced vibration pattern (Short-Long-Short)
    silent: false, // Ensure notification has sound
    renotify: true, // Re-alert for same tag
    data: data,
    actions: [
      {
        action: 'view',
        title: '👁️ Görüntüle',
        icon: '/icons/action-view.png'
      },
      {
        action: 'dismiss',
        title: '❌ Kapat',
        icon: '/icons/action-close.png'
      }
    ],
    // Visual enhancements
    image: data.image || null, // Optional large image
    timestamp: Date.now(),
  };

  event.waitUntil(
    self.registration.showNotification(data.title, notificationOptions)
  );
});

// Notification Click - ENHANCED with action handling
self.addEventListener('notificationclick', (event) => {
  console.log('[SW] 🔔 Notification clicked:', event.notification.tag, 'Action:', event.action);

  event.notification.close();

  // Handle different actions
  if (event.action === 'dismiss') {
    // Just close the notification
    return;
  }

  // Default action or 'view' action - open the app
  const urlToOpen = event.notification.data?.url || '/';

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then((clientList) => {
        // Check if there's already a window open
        for (const client of clientList) {
          if (client.url.includes(self.registration.scope) && 'focus' in client) {
            return client.focus().then(() => {
              // Navigate to the URL if different
              if (client.navigate && client.url !== urlToOpen) {
                return client.navigate(urlToOpen);
              }
              return client;
            });
          }
        }
        // Otherwise open new window
        if (clients.openWindow) {
          return clients.openWindow(urlToOpen);
        }
      })
  );
});

console.log('[SW] Service Worker loaded:', CACHE_VERSION);
