/**
 * TEST SIGNAL NOTIFICATION API
 * Manuel Telegram bildirim testi için
 */

import { NextResponse } from 'next/server';
import { notifyNewSignal } from '@/lib/telegram/notifications';

export async function POST(request: Request) {
  try {
    // Test sinyali gönder
    const testSignal = {
      symbol: 'BTCUSDT',
      price: '42,500.00',
      action: 'STRONG_BUY' as const,
      confidence: 95,
      timestamp: Date.now(),
      reason: '✅ PM2 Test - Direct Bot API working!',
      strategy: 'Manual Test from API'
    };

    console.log('[Test Signal] Sending to Telegram...', testSignal);

    const result = await notifyNewSignal(testSignal);

    console.log('[Test Signal] Result:', result);

    if (result.sent > 0) {
      return NextResponse.json({
        success: true,
        message: `✅ Signal sent to ${result.sent} recipient(s)`,
        result
      });
    } else {
      return NextResponse.json({
        success: false,
        message: '❌ No signals sent',
        result,
        debug: {
          chatIds: process.env.TELEGRAM_ALLOWED_CHAT_IDS,
          errors: result.errors
        }
      }, { status: 500 });
    }

  } catch (error: any) {
    console.error('[Test Signal] Error:', error);
    return NextResponse.json({
      success: false,
      error: error.message,
      stack: error.stack
    }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'Test Signal API - Use POST to send test signal',
    usage: 'POST /api/telegram/test-signal',
    info: 'Sends a manual test signal to Telegram using direct bot API'
  });
}
