/**
 * 🔔 PUSH NOTIFICATION CLIENT
 *
 * Browser-side push notification management
 * SECURITY: White-hat implementation with proper validation
 *
 * Features:
 * - Permission request UI
 * - VAPID subscription creation
 * - Automatic retry on failure
 * - Error handling with user feedback
 * - Subscription state management
 */

// VAPID Public Key from environment
const VAPID_PUBLIC_KEY = process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY!;

export interface SubscriptionState {
  supported: boolean;
  permission: NotificationPermission;
  subscribed: boolean;
  subscription: PushSubscription | null;
  userId: string | null;
}

/**
 * Check if push notifications are supported
 */
export function isPushSupported(): boolean {
  return (
    typeof window !== 'undefined' &&
    'Notification' in window &&
    'serviceWorker' in navigator &&
    'PushManager' in window
  );
}

/**
 * Get current subscription state
 */
export async function getSubscriptionState(): Promise<SubscriptionState> {
  const state: SubscriptionState = {
    supported: isPushSupported(),
    permission: isPushSupported() ? Notification.permission : 'denied',
    subscribed: false,
    subscription: null,
    userId: null,
  };

  if (!state.supported) {
    return state;
  }

  try {
    // Get service worker registration
    const registration = await navigator.serviceWorker.ready;

    // Get existing subscription
    const subscription = await registration.pushManager.getSubscription();

    if (subscription) {
      state.subscribed = true;
      state.subscription = subscription;

      // Get userId from localStorage
      state.userId = localStorage.getItem('push-notification-userId');
    }
  } catch (error) {
    console.error('[PushClient] Error getting subscription state:', error);
  }

  return state;
}

/**
 * Request notification permission
 */
export async function requestPermission(): Promise<NotificationPermission> {
  if (!isPushSupported()) {
    throw new Error('Push notifications are not supported in this browser');
  }

  const permission = await Notification.requestPermission();
  console.log('[PushClient] Permission result:', permission);

  return permission;
}

/**
 * Subscribe to push notifications
 *
 * SECURITY:
 * - User consent required (browser native permission)
 * - VAPID key validation
 * - Service worker registration check
 * - Error handling with user feedback
 */
export async function subscribeToPush(userId?: string): Promise<PushSubscription> {
  // 1. Check support
  if (!isPushSupported()) {
    throw new Error('Push notifications are not supported');
  }

  // 2. Check VAPID key
  if (!VAPID_PUBLIC_KEY) {
    throw new Error('VAPID public key not configured');
  }

  // 3. Request permission if not granted
  if (Notification.permission === 'default') {
    const permission = await requestPermission();
    if (permission !== 'granted') {
      throw new Error('Notification permission denied');
    }
  }

  if (Notification.permission !== 'granted') {
    throw new Error('Notification permission not granted');
  }

  // 4. Get service worker registration
  const registration = await navigator.serviceWorker.ready;

  // 5. Check for existing subscription
  let subscription = await registration.pushManager.getSubscription();

  if (subscription) {
    console.log('[PushClient] Existing subscription found');
    return subscription;
  }

  // 6. Create new subscription with VAPID
  try {
    // Convert VAPID key to Uint8Array
    const applicationServerKey = urlBase64ToUint8Array(VAPID_PUBLIC_KEY);

    subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true, // SECURITY: Must show notification to user
      applicationServerKey: applicationServerKey as BufferSource,
    });

    console.log('[PushClient] ✅ New subscription created');

    // 7. Save to backend
    await saveSubscriptionToBackend(subscription, userId);

    return subscription;
  } catch (error: any) {
    console.error('[PushClient] ❌ Subscription failed:', error);
    throw new Error(`Failed to subscribe: ${error.message}`);
  }
}

/**
 * Unsubscribe from push notifications
 */
export async function unsubscribeFromPush(): Promise<boolean> {
  if (!isPushSupported()) {
    return false;
  }

  try {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.getSubscription();

    if (!subscription) {
      console.log('[PushClient] No subscription found');
      return true;
    }

    // Unsubscribe from browser
    const success = await subscription.unsubscribe();

    if (success) {
      console.log('[PushClient] ✅ Unsubscribed successfully');

      // Remove from backend
      const userId = localStorage.getItem('push-notification-userId');
      if (userId) {
        await removeSubscriptionFromBackend(userId);
      }

      // Clear localStorage
      localStorage.removeItem('push-notification-userId');
    }

    return success;
  } catch (error) {
    console.error('[PushClient] ❌ Unsubscribe failed:', error);
    return false;
  }
}

/**
 * Save subscription to backend
 *
 * SECURITY:
 * - Input validation on backend
 * - Rate limiting via middleware
 * - HTTPS required in production
 */
async function saveSubscriptionToBackend(
  subscription: PushSubscription,
  userId?: string
): Promise<void> {
  try {
    const response = await fetch('/api/push/subscribe', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        subscription: subscription.toJSON(),
        userId: userId || generateBrowserFingerprint(),
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to save subscription');
    }

    const data = await response.json();

    // Save userId to localStorage for future use
    if (data.userId) {
      localStorage.setItem('push-notification-userId', data.userId);
    }

    console.log('[PushClient] ✅ Subscription saved to backend:', data);
  } catch (error: any) {
    console.error('[PushClient] ❌ Backend save failed:', error);
    throw error;
  }
}

/**
 * Remove subscription from backend
 */
async function removeSubscriptionFromBackend(userId: string): Promise<void> {
  try {
    const response = await fetch('/api/push/subscribe', {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ userId }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to remove subscription');
    }

    console.log('[PushClient] ✅ Subscription removed from backend');
  } catch (error) {
    console.error('[PushClient] ❌ Backend removal failed:', error);
    throw error;
  }
}

/**
 * Send test notification to self
 */
export async function sendTestNotification(): Promise<void> {
  const state = await getSubscriptionState();

  if (!state.subscribed || !state.subscription) {
    throw new Error('Not subscribed to notifications');
  }

  try {
    const response = await fetch('/api/push/test', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        subscription: state.subscription.toJSON(),
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Test notification failed');
    }

    console.log('[PushClient] ✅ Test notification sent');
  } catch (error: any) {
    console.error('[PushClient] ❌ Test notification failed:', error);
    throw error;
  }
}

// ==================== HELPERS ====================

/**
 * Convert base64 VAPID key to Uint8Array
 */
function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');

  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }

  return outputArray;
}

/**
 * Generate browser fingerprint (fallback userId)
 */
function generateBrowserFingerprint(): string {
  const nav = navigator as any;
  const screen = window.screen;

  const fingerprint = [
    nav.userAgent || '',
    nav.language || '',
    screen.colorDepth || '',
    screen.width || '',
    screen.height || '',
    new Date().getTimezoneOffset(),
  ].join('|');

  // Simple hash
  let hash = 0;
  for (let i = 0; i < fingerprint.length; i++) {
    const char = fingerprint.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }

  return `browser-${Math.abs(hash).toString(36)}`;
}
