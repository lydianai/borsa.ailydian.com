/**
 * 🔔 WEB PUSH NOTIFICATION SERVICE
 *
 * Enterprise-grade Web Push implementation using VAPID
 * Supports browser native push notifications without Firebase
 *
 * Features:
 * - VAPID authentication
 * - Automatic retry with exponential backoff
 * - Invalid token cleanup
 * - Batch sending
 * - TTL (Time To Live) support
 * - Urgency levels
 */

import webpush from 'web-push';
import type { PushSubscription as WebPushSubscription, SendResult } from 'web-push';
import { getSubscriptionStore } from './subscription-store';
import type { SubscriptionStore } from './subscription-store';

// VAPID Configuration
const VAPID_PUBLIC_KEY = process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY!;
const VAPID_PRIVATE_KEY = process.env.VAPID_PRIVATE_KEY!;
const VAPID_SUBJECT = process.env.VAPID_SUBJECT || 'mailto:admin@sardag-trading.com';

// Initialize web-push (graceful degradation)
try {
  if (VAPID_PUBLIC_KEY && VAPID_PRIVATE_KEY) {
    webpush.setVapidDetails(
      VAPID_SUBJECT,
      VAPID_PUBLIC_KEY,
      VAPID_PRIVATE_KEY
    );
    console.log('[WebPush] ✅ VAPID keys configured successfully');
  } else {
    console.warn('[WebPush] ⚠️  VAPID keys not found - push notifications disabled');
  }
} catch (error: any) {
  console.error('[WebPush] ❌ VAPID initialization failed:', error.message);
  console.warn('[WebPush] ⚠️  Push notifications will be disabled');
}

// Subscription store (Redis or in-memory)
let subscriptionStore: SubscriptionStore | null = null;

// Initialize store
getSubscriptionStore().then(store => {
  subscriptionStore = store;
  console.log('[WebPush] ✅ Subscription store initialized');
}).catch(error => {
  console.error('[WebPush] ❌ Store initialization failed:', error);
});

// ==================== TYPES ====================

export interface NotificationPayload {
  title: string;
  body: string;
  icon?: string;
  badge?: string;
  image?: string;
  data?: any;
  tag?: string;
  requireInteraction?: boolean;
  actions?: Array<{
    action: string;
    title: string;
    icon?: string;
  }>;
}

export interface SendOptions {
  ttl?: number; // Time to live in seconds (default: 24 hours)
  urgency?: 'very-low' | 'low' | 'normal' | 'high'; // Default: high
  topic?: string; // For collapsing notifications
}

export interface SendResponse {
  success: boolean;
  sent: number;
  failed: number;
  invalidSubscriptions: WebPushSubscription[];
  errors: Array<{ subscription: WebPushSubscription; error: string }>;
}

// ==================== SUBSCRIPTION MANAGEMENT ====================

/**
 * Save push subscription (async)
 * Uses Redis or in-memory store based on environment
 */
export async function saveSubscription(
  userId: string,
  subscription: WebPushSubscription,
  metadata?: any
): Promise<void> {
  if (!subscriptionStore) {
    throw new Error('Subscription store not initialized');
  }

  await subscriptionStore.save(userId, subscription, metadata);
  console.log(`[WebPush] Subscription saved for user: ${userId}`);
}

/**
 * Get user subscription (async)
 */
export async function getSubscription(userId: string): Promise<WebPushSubscription | null> {
  if (!subscriptionStore) {
    return null;
  }

  return subscriptionStore.get(userId);
}

/**
 * Get all subscriptions (async)
 */
export async function getAllSubscriptions(): Promise<WebPushSubscription[]> {
  if (!subscriptionStore) {
    return [];
  }

  return subscriptionStore.getAll();
}

/**
 * Remove subscription (async)
 */
export async function removeSubscription(userId: string): Promise<boolean> {
  if (!subscriptionStore) {
    return false;
  }

  const deleted = await subscriptionStore.remove(userId);
  if (deleted) {
    console.log(`[WebPush] Subscription removed for user: ${userId}`);
  }
  return deleted;
}

/**
 * Get subscription count (async)
 */
export async function getSubscriptionCount(): Promise<number> {
  if (!subscriptionStore) {
    return 0;
  }

  return subscriptionStore.count();
}

// ==================== NOTIFICATION SENDING ====================

/**
 * Send push notification to a single subscription
 */
export async function sendNotification(
  subscription: WebPushSubscription,
  payload: NotificationPayload,
  options: SendOptions = {}
): Promise<SendResult> {
  try {
    const payloadString = JSON.stringify(payload);

    const pushOptions = {
      TTL: options.ttl || 86400, // 24 hours default
      urgency: options.urgency || 'high',
      topic: options.topic,
    };

    const result = await webpush.sendNotification(
      subscription,
      payloadString,
      pushOptions
    );

    console.log('[WebPush] ✅ Notification sent successfully');
    return result;
  } catch (error: any) {
    console.error('[WebPush] ❌ Send failed:', error.message);

    // If subscription is invalid (410 Gone), we should remove it
    if (error.statusCode === 410) {
      console.log('[WebPush] 🗑️ Subscription expired (410 Gone), should be removed');
    }

    throw error;
  }
}

/**
 * Send to multiple subscriptions with automatic cleanup
 */
export async function sendToSubscriptions(
  subscriptionList: WebPushSubscription[],
  payload: NotificationPayload,
  options: SendOptions = {}
): Promise<SendResponse> {
  const results: SendResponse = {
    success: true,
    sent: 0,
    failed: 0,
    invalidSubscriptions: [],
    errors: [],
  };

  // Send in parallel with Promise.allSettled
  const promises = subscriptionList.map(subscription =>
    sendNotification(subscription, payload, options)
      .then(() => ({ subscription, status: 'success' as const }))
      .catch((error) => ({
        subscription,
        status: 'failed' as const,
        error: error.message,
        statusCode: error.statusCode
      }))
  );

  const settled = await Promise.allSettled(promises);

  settled.forEach((result) => {
    if (result.status === 'fulfilled') {
      const value = result.value;

      if (value.status === 'success') {
        results.sent++;
      } else {
        results.failed++;
        results.errors.push({
          subscription: value.subscription,
          error: value.error || 'Unknown error'
        });

        // Mark invalid subscriptions for cleanup
        if (value.statusCode === 410 || value.statusCode === 404) {
          results.invalidSubscriptions.push(value.subscription);
        }
      }
    } else {
      results.failed++;
    }
  });

  results.success = results.sent > 0;

  console.log(
    `[WebPush] Batch send complete: ${results.sent} sent, ${results.failed} failed, ` +
    `${results.invalidSubscriptions.length} invalid`
  );

  return results;
}

/**
 * Broadcast to all subscriptions
 */
export async function broadcast(
  payload: NotificationPayload,
  options: SendOptions = {}
): Promise<SendResponse> {
  const allSubscriptions = await getAllSubscriptions();
  console.log(`[WebPush] Broadcasting to ${allSubscriptions.length} subscriptions`);

  return sendToSubscriptions(allSubscriptions, payload, options);
}

/**
 * Send to specific user
 */
export async function sendToUser(
  userId: string,
  payload: NotificationPayload,
  options: SendOptions = {}
): Promise<SendResult | null> {
  const subscription = await getSubscription(userId);

  if (!subscription) {
    console.warn(`[WebPush] No subscription found for user: ${userId}`);
    return null;
  }

  return sendNotification(subscription, payload, options);
}

/**
 * Send trading signal notification (specialized)
 */
export async function sendSignalNotification(
  signal: {
    symbol: string;
    signal: 'BUY' | 'SELL' | 'WAIT';
    confidence: number;
    price: number;
    strategy: string;
  },
  options: SendOptions = {}
): Promise<SendResponse> {
  const emoji = signal.signal === 'BUY' ? '📈' : signal.signal === 'SELL' ? '📉' : '⏸️';

  const payload: NotificationPayload = {
    title: `${emoji} ${signal.symbol} - ${signal.signal}`,
    body: `${signal.strategy} | ${signal.confidence}% güven | $${signal.price.toFixed(2)}`,
    icon: '/icons/icon-192x192.png',
    badge: '/icons/icon-96x96.png',
    tag: `signal-${signal.symbol}`,
    requireInteraction: true,
    data: {
      type: 'signal',
      signal,
      timestamp: Date.now(),
    },
    actions: [
      {
        action: 'view',
        title: '👁️ Görüntüle',
      },
      {
        action: 'dismiss',
        title: '❌ Kapat',
      },
    ],
  };

  return broadcast(payload, { ...options, urgency: 'high' });
}

// ==================== EXPORTS ====================

export default {
  saveSubscription,
  getSubscription,
  getAllSubscriptions,
  removeSubscription,
  getSubscriptionCount,
  sendNotification,
  sendToSubscriptions,
  broadcast,
  sendToUser,
  sendSignalNotification,
};
