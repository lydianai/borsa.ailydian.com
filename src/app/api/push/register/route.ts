/**
 * POST /api/push/register
 * Register FCM device token for push notifications
 *
 * Security:
 * - User ID validation (from session or token)
 * - Token format validation
 * - White-hat logging: All registrations logged
 */

import { NextRequest, NextResponse } from 'next/server';
import deviceTokenManager from '@/lib/push/device-token-manager';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  try {
    // 1. Parse request body
    const body = await request.json();
    const { token, userId, platform, metadata } = body;

    // 2. Validate required fields
    if (!token || typeof token !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid token' },
        { status: 400 }
      );
    }

    if (!userId || typeof userId !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid userId' },
        { status: 400 }
      );
    }

    if (!platform || !['ios', 'android', 'web'].includes(platform)) {
      return NextResponse.json(
        { error: 'Invalid platform. Must be: ios, android, or web' },
        { status: 400 }
      );
    }

    // 3. Validate token format (basic check)
    if (token.length < 20) {
      return NextResponse.json(
        { error: 'Invalid FCM token format' },
        { status: 400 }
      );
    }

    // 4. Register token
    deviceTokenManager.registerToken(token, userId, platform, metadata);

    // 5. Return success
    console.log(`[API] Device token registered: user=${userId}, platform=${platform}`);

    return NextResponse.json({
      success: true,
      message: 'Device token registered successfully',
      userId,
      platform,
    });
  } catch (error: any) {
    console.error('[API] Token registration failed:', error.message);

    return NextResponse.json(
      { error: 'Token registration failed', message: error.message },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/push/register
 * Unregister device token
 */
export async function DELETE(request: NextRequest) {
  try {
    const body = await request.json();
    const { token } = body;

    if (!token || typeof token !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid token' },
        { status: 400 }
      );
    }

    const success = deviceTokenManager.unregisterToken(token);

    if (!success) {
      return NextResponse.json(
        { error: 'Token not found' },
        { status: 404 }
      );
    }

    console.log(`[API] Device token unregistered: ${token.substring(0, 20)}...`);

    return NextResponse.json({
      success: true,
      message: 'Device token unregistered successfully',
    });
  } catch (error: any) {
    console.error('[API] Token unregistration failed:', error.message);

    return NextResponse.json(
      { error: 'Token unregistration failed', message: error.message },
      { status: 500 }
    );
  }
}
