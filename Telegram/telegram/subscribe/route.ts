/**
 * 🔔 TELEGRAM SUBSCRIPTION API
 * Kullanıcı abonelik yönetimi
 *
 * Features:
 * - Subscribe/unsubscribe users
 * - Get subscriber count
 * - Get subscriber list (dev only)
 * - White-hat compliant
 *
 * Endpoints:
 * - POST /api/telegram/subscribe - Subscribe user
 * - DELETE /api/telegram/subscribe - Unsubscribe user
 * - GET /api/telegram/subscribe - Get subscriber count
 *
 * ⚠️ WHITE-HAT COMPLIANCE:
 * - User can unsubscribe anytime
 * - Only chat ID stored (no personal data)
 * - GDPR/CCPA compliant
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  subscribe,
  unsubscribe,
  isSubscribed,
  getSubscriberCount,
  getAllSubscribers,
} from '@/lib/telegram/notifications';

export const dynamic = 'force-dynamic';

// ============================================================================
// SUBSCRIPTION HANDLERS
// ============================================================================

/**
 * POST /api/telegram/subscribe
 * Kullanıcıyı bildirimlere abone et
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { chatId } = body;

    // Validate chat ID
    if (!chatId || typeof chatId !== 'number' || chatId <= 0) {
      return NextResponse.json(
        {
          error: 'Invalid chatId',
          message: 'chatId must be a positive number',
        },
        { status: 400 }
      );
    }

    // Check if already subscribed
    if (isSubscribed(chatId)) {
      return NextResponse.json({
        success: true,
        message: 'Already subscribed',
        chatId,
        subscriberCount: getSubscriberCount(),
        alreadySubscribed: true,
      });
    }

    // Subscribe user
    const success = subscribe(chatId);

    if (!success) {
      return NextResponse.json(
        {
          error: 'Failed to subscribe',
          message: 'Could not add subscriber',
        },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Subscribed successfully',
      chatId,
      subscriberCount: getSubscriberCount(),
      alreadySubscribed: false,
    });

  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error('[Telegram Subscribe] Error:', error);
    }

    return NextResponse.json(
      {
        error: 'Internal server error',
        message: error.message,
      },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/telegram/subscribe
 * Kullanıcının aboneliğini iptal et
 */
export async function DELETE(request: NextRequest) {
  try {
    const body = await request.json();
    const { chatId } = body;

    // Validate chat ID
    if (!chatId || typeof chatId !== 'number' || chatId <= 0) {
      return NextResponse.json(
        {
          error: 'Invalid chatId',
          message: 'chatId must be a positive number',
        },
        { status: 400 }
      );
    }

    // Unsubscribe user
    const success = unsubscribe(chatId);

    return NextResponse.json({
      success,
      message: success ? 'Unsubscribed successfully' : 'User was not subscribed',
      chatId,
      subscriberCount: getSubscriberCount(),
    });

  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error('[Telegram Unsubscribe] Error:', error);
    }

    return NextResponse.json(
      {
        error: 'Internal server error',
        message: error.message,
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/telegram/subscribe
 * Abone sayısını ve listesini getir
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const includeList = searchParams.get('list') === 'true';

    const nodeEnv = process.env.NODE_ENV as string;

    // Only allow list in development
    if (includeList && nodeEnv === 'production') {
      return NextResponse.json(
        {
          error: 'List not available in production',
          message: 'Subscriber list is only available in development mode',
        },
        { status: 403 }
      );
    }

    const response: any = {
      subscriberCount: getSubscriberCount(),
    };

    if (includeList) {
      response.subscribers = getAllSubscribers();
    }

    return NextResponse.json(response);

  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error('[Telegram Subscribe GET] Error:', error);
    }

    return NextResponse.json(
      {
        error: 'Internal server error',
        message: error.message,
      },
      { status: 500 }
    );
  }
}
