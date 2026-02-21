/**
 * 📡 TELEGRAM WEBHOOK ENDPOINT
 * Telegram'dan gelen mesajları işle
 *
 * Features:
 * - Webhook handler (Grammy)
 * - Secret token validation
 * - Error handling
 * - Serverless-compatible
 * - White-hat compliant
 *
 * Endpoint: POST /api/telegram/webhook
 *
 * ⚠️ WHITE-HAT COMPLIANCE:
 * - Webhook secret validation (security)
 * - No user data logging
 * - Educational purposes only
 */

import { NextRequest, NextResponse } from 'next/server';
import { handleWebhookUpdate } from '@/lib/telegram/bot';
import '@/lib/telegram/handlers'; // Import handlers to register commands

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

// ============================================================================
// WEBHOOK HANDLER
// ============================================================================

/**
 * POST /api/telegram/webhook
 * Handle incoming Telegram updates
 */
export async function POST(request: NextRequest) {
  try {
    // 1. Validate webhook secret (security)
    const secretToken = request.headers.get('x-telegram-bot-api-secret-token');
    const expectedSecret = process.env.TELEGRAM_BOT_WEBHOOK_SECRET;

    if (expectedSecret && secretToken !== expectedSecret) {
      const nodeEnv = process.env.NODE_ENV as string;
      if (nodeEnv !== 'production') {
        console.warn('[Telegram Webhook] Invalid secret token');
      }

      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    // 2. Get request body
    const body = await request.json();

    // 3. Handle update with Grammy
    await handleWebhookUpdate(body);

    // 4. Return success
    return NextResponse.json({ ok: true });

  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error('[Telegram Webhook] Error:', error);
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
 * GET /api/telegram/webhook
 * Webhook status kontrolü
 */
export async function GET() {
  try {
    const { bot } = await import('@/lib/telegram/bot');
    const webhookInfo = await bot.api.getWebhookInfo();

    return NextResponse.json({
      status: 'active',
      url: webhookInfo.url || null,
      pendingUpdates: webhookInfo.pending_update_count || 0,
      lastError: webhookInfo.last_error_message || null,
      lastErrorDate: webhookInfo.last_error_date
        ? new Date(webhookInfo.last_error_date * 1000).toISOString()
        : null,
      maxConnections: webhookInfo.max_connections || null,
      allowedUpdates: webhookInfo.allowed_updates || [],
    });
  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error('[Telegram Webhook GET] Error:', error);
    }

    return NextResponse.json(
      {
        error: 'Failed to get webhook info',
        message: error.message,
      },
      { status: 500 }
    );
  }
}
