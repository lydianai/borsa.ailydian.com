/**
 * 📨 PUSH NOTIFICATION SERVICE
 * FCM push notifications for trading signals
 *
 * Features:
 * - Send to single device
 * - Send to multiple devices (batch)
 * - Send to all users (broadcast)
 * - Notification templates for signals
 * - Delivery tracking
 * - White-hat compliance: All sends logged
 */

import { getMessaging, isFirebaseAvailable } from './firebase-admin';
import deviceTokenManager from './device-token-manager';
import type * as admin from 'firebase-admin';

// ============================================================================
// TYPES
// ============================================================================

export interface SignalNotification {
  symbol: string;
  signal: 'BUY' | 'SELL' | 'WAIT';
  confidence: number;
  price: number;
  strategy: string;
  reason?: string;
}

export interface NotificationPayload {
  title: string;
  body: string;
  data?: Record<string, string>;
  imageUrl?: string;
}

export interface SendResult {
  success: boolean;
  messageId?: string;
  error?: string;
  invalidTokens?: string[];
}

// ============================================================================
// PUSH NOTIFICATION SERVICE
// ============================================================================

export class PushNotificationService {
  /**
   * Send notification to a single device
   */
  async sendToDevice(
    token: string,
    payload: NotificationPayload
  ): Promise<SendResult> {
    if (!isFirebaseAvailable()) {
      console.warn('[Push] Firebase not available, skipping notification');
      return { success: false, error: 'Firebase not configured' };
    }

    try {
      const messaging = getMessaging();

      const message: admin.messaging.Message = {
        token,
        notification: {
          title: payload.title,
          body: payload.body,
          imageUrl: payload.imageUrl,
        },
        data: payload.data,
        android: {
          priority: 'high',
          notification: {
            sound: 'default',
            channelId: 'trading-signals',
          },
        },
        apns: {
          payload: {
            aps: {
              sound: 'default',
              badge: 1,
            },
          },
        },
        webpush: {
          notification: {
            icon: '/icon-192x192.png',
            badge: '/icon-96x96.png',
            vibrate: [200, 100, 200],
          },
        },
      };

      const messageId = await messaging.send(message);

      // Update last used time
      deviceTokenManager.updateLastUsed(token);

      console.log(`[Push] ✅ Sent to device: ${messageId}`);
      return { success: true, messageId };
    } catch (error: any) {
      console.error('[Push] Send failed:', error.message);

      // Mark token as invalid if it's a registration error
      if (
        error.code === 'messaging/invalid-registration-token' ||
        error.code === 'messaging/registration-token-not-registered'
      ) {
        deviceTokenManager.markTokenInvalid(token);
        return { success: false, error: error.message, invalidTokens: [token] };
      }

      return { success: false, error: error.message };
    }
  }

  /**
   * Send notification to multiple devices (batch)
   */
  async sendToDevices(
    tokens: string[],
    payload: NotificationPayload
  ): Promise<SendResult> {
    if (!isFirebaseAvailable()) {
      console.warn('[Push] Firebase not available, skipping notification');
      return { success: false, error: 'Firebase not configured' };
    }

    if (tokens.length === 0) {
      return { success: true };
    }

    try {
      const messaging = getMessaging();

      const message: admin.messaging.MulticastMessage = {
        tokens,
        notification: {
          title: payload.title,
          body: payload.body,
          imageUrl: payload.imageUrl,
        },
        data: payload.data,
        android: {
          priority: 'high',
          notification: {
            sound: 'default',
            channelId: 'trading-signals',
          },
        },
        apns: {
          payload: {
            aps: {
              sound: 'default',
              badge: 1,
            },
          },
        },
      };

      const response = await messaging.sendEachForMulticast(message);

      // Process responses and mark invalid tokens
      const invalidTokens: string[] = [];

      response.responses.forEach((resp, idx) => {
        if (!resp.success) {
          const error = resp.error;
          if (
            error?.code === 'messaging/invalid-registration-token' ||
            error?.code === 'messaging/registration-token-not-registered'
          ) {
            invalidTokens.push(tokens[idx]);
            deviceTokenManager.markTokenInvalid(tokens[idx]);
          }
        } else {
          // Update last used time for successful sends
          deviceTokenManager.updateLastUsed(tokens[idx]);
        }
      });

      console.log(
        `[Push] ✅ Batch sent: ${response.successCount}/${tokens.length} successful`
      );

      if (invalidTokens.length > 0) {
        console.log(`[Push] Removed ${invalidTokens.length} invalid tokens`);
      }

      return {
        success: response.successCount > 0,
        invalidTokens: invalidTokens.length > 0 ? invalidTokens : undefined,
      };
    } catch (error: any) {
      console.error('[Push] Batch send failed:', error.message);
      return { success: false, error: error.message };
    }
  }

  /**
   * Send notification to a user (all their devices)
   */
  async sendToUser(
    userId: string,
    payload: NotificationPayload
  ): Promise<SendResult> {
    const tokens = deviceTokenManager.getUserTokens(userId);

    if (tokens.length === 0) {
      console.warn(`[Push] No tokens found for user ${userId}`);
      return { success: false, error: 'No tokens registered for user' };
    }

    return await this.sendToDevices(tokens, payload);
  }

  /**
   * Send notification to all devices (broadcast)
   */
  async broadcast(payload: NotificationPayload): Promise<SendResult> {
    const tokens = deviceTokenManager.getAllTokens();

    if (tokens.length === 0) {
      console.warn('[Push] No tokens registered for broadcast');
      return { success: false, error: 'No tokens registered' };
    }

    console.log(`[Push] Broadcasting to ${tokens.length} devices...`);
    return await this.sendToDevices(tokens, payload);
  }

  /**
   * Send trading signal notification
   */
  async sendSignalNotification(
    signal: SignalNotification,
    userIds?: string[]
  ): Promise<SendResult> {
    const payload = this.createSignalPayload(signal);

    // Send to specific users or broadcast
    if (userIds && userIds.length > 0) {
      // Collect tokens from all users
      const tokens: string[] = [];
      for (const userId of userIds) {
        tokens.push(...deviceTokenManager.getUserTokens(userId));
      }

      if (tokens.length === 0) {
        return { success: false, error: 'No tokens found for specified users' };
      }

      return await this.sendToDevices(tokens, payload);
    } else {
      // Broadcast to all users
      return await this.broadcast(payload);
    }
  }

  /**
   * Create notification payload from signal
   */
  private createSignalPayload(signal: SignalNotification): NotificationPayload {
    const emoji = signal.signal === 'BUY' ? '🟢' : signal.signal === 'SELL' ? '🔴' : '⏸️';
    const confidenceEmoji = signal.confidence >= 85 ? '🔥' : signal.confidence >= 70 ? '⚡' : '💡';

    return {
      title: `${emoji} ${signal.signal} Signal - ${signal.symbol}`,
      body: `${confidenceEmoji} Confidence: ${signal.confidence}% | Price: $${signal.price.toLocaleString()} | ${signal.strategy}`,
      data: {
        type: 'trading-signal',
        symbol: signal.symbol,
        signal: signal.signal,
        confidence: signal.confidence.toString(),
        price: signal.price.toString(),
        strategy: signal.strategy,
        timestamp: new Date().toISOString(),
      },
    };
  }

  /**
   * Test notification (for debugging)
   */
  async sendTestNotification(token: string): Promise<SendResult> {
    return await this.sendToDevice(token, {
      title: '🧪 Test Notification',
      body: 'If you see this, push notifications are working!',
      data: {
        type: 'test',
        timestamp: new Date().toISOString(),
      },
    });
  }
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

const pushNotificationService = new PushNotificationService();
export default pushNotificationService;
