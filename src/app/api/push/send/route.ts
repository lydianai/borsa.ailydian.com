/**
 * POST /api/push/send
 * Send push notification (requires authentication)
 *
 * Security:
 * - Requires INTERNAL_SERVICE_TOKEN
 * - Rate limiting via batch size limits
 * - White-hat logging: All sends logged
 */

import { NextRequest, NextResponse } from 'next/server';
import pushNotificationService from '@/lib/push/push-notification-service';
import type { SignalNotification } from '@/lib/push/push-notification-service';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

/**
 * Validate internal service token
 */
function validateServiceToken(request: NextRequest): boolean {
  const token = request.headers.get('x-service-token');
  const expectedToken = process.env.INTERNAL_SERVICE_TOKEN;

  if (!expectedToken) {
    console.warn('[API] INTERNAL_SERVICE_TOKEN not set in environment');
    return false;
  }

  return token === expectedToken;
}

export async function POST(request: NextRequest) {
  try {
    // 1. Validate authentication
    if (!validateServiceToken(request)) {
      console.warn('[API] Push send: Unauthorized request');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // 2. Parse request body
    const body = await request.json();
    const { type, signal, userIds } = body;

    // 3. Validate notification type
    if (!type || typeof type !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid type field' },
        { status: 400 }
      );
    }

    // 4. Handle different notification types
    let result;

    switch (type) {
      case 'signal':
        // Trading signal notification
        if (!signal) {
          return NextResponse.json(
            { error: 'Missing signal data' },
            { status: 400 }
          );
        }

        // Validate signal structure
        const { symbol, signal: signalType, confidence, price, strategy } = signal as SignalNotification;

        if (!symbol || !signalType || !confidence || !price || !strategy) {
          return NextResponse.json(
            { error: 'Invalid signal data structure' },
            { status: 400 }
          );
        }

        result = await pushNotificationService.sendSignalNotification(
          signal,
          userIds
        );
        break;

      case 'test':
        // Test notification (requires token)
        const { token } = body;

        if (!token) {
          return NextResponse.json(
            { error: 'Missing token for test notification' },
            { status: 400 }
          );
        }

        result = await pushNotificationService.sendTestNotification(token);
        break;

      case 'custom':
        // Custom notification
        const { payload, userIds: targetUsers } = body;

        if (!payload || !payload.title || !payload.body) {
          return NextResponse.json(
            { error: 'Invalid payload structure (title and body required)' },
            { status: 400 }
          );
        }

        if (targetUsers && targetUsers.length > 0) {
          // Send to specific users
          const tokens: string[] = [];
          for (const userId of targetUsers) {
            const userTokens = await import('@/lib/push/device-token-manager').then(
              (m) => m.default.getUserTokens(userId)
            );
            tokens.push(...userTokens);
          }

          result = await pushNotificationService.sendToDevices(tokens, payload);
        } else {
          // Broadcast to all
          result = await pushNotificationService.broadcast(payload);
        }
        break;

      default:
        return NextResponse.json(
          { error: `Unknown notification type: ${type}` },
          { status: 400 }
        );
    }

    // 5. Return result
    console.log(`[API] Push notification sent: type=${type}, success=${result.success}`);

    return NextResponse.json({
      success: result.success,
      messageId: result.messageId,
      invalidTokens: result.invalidTokens,
      error: result.error,
    });
  } catch (error: any) {
    console.error('[API] Push send failed:', error.message);

    return NextResponse.json(
      { error: 'Push send failed', message: error.message },
      { status: 500 }
    );
  }
}
