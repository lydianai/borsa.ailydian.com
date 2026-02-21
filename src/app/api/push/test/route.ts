/**
 * POST /api/push/test
 * Send test push notification to validate setup
 *
 * SECURITY:
 * - Rate limiting (max 20 tests per minute per user)
 * - Subscription validation
 * - No authentication required (public endpoint for testing)
 */

import { NextRequest, NextResponse } from 'next/server';
import webPushService from '@/lib/push/web-push-service';
import type { PushSubscription } from 'web-push';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

// Simple in-memory rate limiting for test endpoint
const testRateLimits = new Map<string, { count: number; resetAt: number }>();

function checkRateLimit(ip: string): boolean {
  const now = Date.now();
  const limit = testRateLimits.get(ip);

  if (!limit || now > limit.resetAt) {
    // Reset or create new limit
    testRateLimits.set(ip, {
      count: 1,
      resetAt: now + 60000, // 1 minute
    });
    return true;
  }

  if (limit.count >= 20) {
    return false; // Rate limit exceeded
  }

  limit.count++;
  return true;
}

export async function POST(request: NextRequest) {
  try {
    // 1. Rate limiting
    const ip = request.headers.get('x-forwarded-for') ||
               request.headers.get('x-real-ip') ||
               'unknown';

    if (!checkRateLimit(ip)) {
      return NextResponse.json(
        {
          error: 'Rate limit exceeded',
          message: 'Maximum 20 test notifications per minute. Please wait 60 seconds.',
        },
        { status: 429 }
      );
    }

    // 2. Parse request body
    const body = await request.json();
    const { subscription } = body;

    // 3. Validate subscription
    if (!subscription || typeof subscription !== 'object') {
      return NextResponse.json(
        { error: 'Missing or invalid subscription object' },
        { status: 400 }
      );
    }

    if (!subscription.endpoint || !subscription.keys) {
      return NextResponse.json(
        { error: 'Invalid subscription format' },
        { status: 400 }
      );
    }

    // 4. Send test notification
    const testPayload = {
      title: '🎉 Test Notification',
      body: 'Bildirimler başarıyla aktif! SARDAG Trading Scanner hazır.',
      icon: '/icons/icon-192x192.png',
      badge: '/icons/icon-96x96.png',
      tag: 'test-notification',
      requireInteraction: false,
      data: {
        type: 'test',
        timestamp: Date.now(),
      },
      actions: [
        {
          action: 'close',
          title: '✅ Tamam',
        },
      ],
    };

    await webPushService.sendNotification(
      subscription as PushSubscription,
      testPayload,
      { ttl: 300, urgency: 'normal' } // 5 minutes TTL, normal urgency
    );

    console.log('[API] ✅ Test notification sent successfully');

    return NextResponse.json({
      success: true,
      message: 'Test notification sent',
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[API] ❌ Test notification failed:', error.message);

    // Check for specific error types
    if (error.statusCode === 410) {
      return NextResponse.json(
        {
          error: 'Subscription expired',
          message: 'Please resubscribe to push notifications',
        },
        { status: 410 }
      );
    }

    return NextResponse.json(
      {
        error: 'Test notification failed',
        message: error.message,
      },
      { status: 500 }
    );
  }
}
