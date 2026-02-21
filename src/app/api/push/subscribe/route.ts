/**
 * POST /api/push/subscribe
 * Register Web Push subscription (VAPID)
 *
 * Security:
 * - Subscription validation
 * - User authentication (optional - can use browser fingerprint)
 * - Rate limiting via middleware
 */

import { NextRequest, NextResponse } from 'next/server';
import webPushService from '@/lib/push/web-push-service';
import type { PushSubscription } from 'web-push';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  try {
    // 1. Parse request body
    const body = await request.json();
    const { subscription, userId } = body;

    // 2. Validate subscription object
    if (!subscription || typeof subscription !== 'object') {
      return NextResponse.json(
        { error: 'Missing or invalid subscription object' },
        { status: 400 }
      );
    }

    // Validate subscription structure
    if (!subscription.endpoint || !subscription.keys) {
      return NextResponse.json(
        { error: 'Invalid subscription format. Must include endpoint and keys.' },
        { status: 400 }
      );
    }

    if (!subscription.keys.p256dh || !subscription.keys.auth) {
      return NextResponse.json(
        { error: 'Invalid subscription keys. Must include p256dh and auth.' },
        { status: 400 }
      );
    }

    // 3. Generate user ID if not provided (fallback to browser fingerprint)
    const finalUserId = userId || generateBrowserFingerprint(request);

    // 4. Save subscription (async)
    await webPushService.saveSubscription(finalUserId, subscription as PushSubscription);

    console.log(`[API] ✅ Web Push subscription saved: user=${finalUserId}`);

    const count = await webPushService.getSubscriptionCount();

    return NextResponse.json({
      success: true,
      message: 'Subscription saved successfully',
      userId: finalUserId,
      subscriptionCount: count,
    });
  } catch (error: any) {
    console.error('[API] ❌ Subscription failed:', error.message);

    return NextResponse.json(
      { error: 'Subscription failed', message: error.message },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/push/subscribe
 * Unsubscribe from push notifications
 */
export async function DELETE(request: NextRequest) {
  try {
    const body = await request.json();
    const { userId } = body;

    if (!userId || typeof userId !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid userId' },
        { status: 400 }
      );
    }

    const success = await webPushService.removeSubscription(userId);

    if (!success) {
      return NextResponse.json(
        { error: 'Subscription not found' },
        { status: 404 }
      );
    }

    console.log(`[API] ✅ Web Push unsubscribed: user=${userId}`);

    return NextResponse.json({
      success: true,
      message: 'Unsubscribed successfully',
    });
  } catch (error: any) {
    console.error('[API] ❌ Unsubscribe failed:', error.message);

    return NextResponse.json(
      { error: 'Unsubscribe failed', message: error.message },
      { status: 500 }
    );
  }
}

/**
 * GET /api/push/subscribe
 * Get subscription status
 */
export async function GET(request: NextRequest) {
  try {
    const userId = request.nextUrl.searchParams.get('userId');

    if (!userId) {
      return NextResponse.json(
        { error: 'Missing userId parameter' },
        { status: 400 }
      );
    }

    const subscription = await webPushService.getSubscription(userId);

    return NextResponse.json({
      success: true,
      subscribed: !!subscription,
      subscription: subscription || null,
    });
  } catch (error: any) {
    console.error('[API] ❌ Get subscription failed:', error.message);

    return NextResponse.json(
      { error: 'Get subscription failed', message: error.message },
      { status: 500 }
    );
  }
}

/**
 * Generate browser fingerprint from request headers
 * Used as fallback user ID
 */
function generateBrowserFingerprint(request: NextRequest): string {
  const userAgent = request.headers.get('user-agent') || 'unknown';
  const acceptLanguage = request.headers.get('accept-language') || 'unknown';
  const ip = request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown';

  // Simple hash function
  const str = `${userAgent}-${acceptLanguage}-${ip}`;
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }

  return `browser-${Math.abs(hash).toString(36)}`;
}
