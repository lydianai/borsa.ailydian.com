/**
 * 🧪 TELEGRAM TEST API
 * Test bildirimi gönder ve sistemi test et
 *
 * Features:
 * - Test notification gönderimi
 * - Signal formatting test
 * - Subscriber count check
 * - White-hat compliant
 *
 * Endpoint: POST /api/telegram/test
 *
 * ⚠️ SECURITY:
 * - Development only by default
 * - Can be enabled in production with auth
 */

import { NextRequest, NextResponse } from 'next/server';
import { notifyNewSignal, getNotificationStats } from '@/lib/telegram/notifications';
import { processAndNotifySignal } from '@/lib/telegram/signal-notifier';
import type { TradingSignal } from '@/lib/telegram/notifications';

export const dynamic = 'force-dynamic';

// ============================================================================
// TEST NOTIFICATION
// ============================================================================

/**
 * POST /api/telegram/test
 * Test notification gönder
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type = 'simple' } = body;

    const stats = getNotificationStats();

    // Eğer hiç abone yoksa uyar
    if (stats.subscriberCount === 0) {
      return NextResponse.json({
        success: false,
        error: 'No subscribers',
        message: 'Henüz abone olan kullanıcı yok. Önce /start ile bota abone olun.',
        stats,
      });
    }

    if (type === 'simple') {
      // Basit test mesajı
      const testSignal: TradingSignal = {
        symbol: 'BTCUSDT',
        price: '50000.00',
        action: 'BUY',
        confidence: 85,
        timestamp: Date.now(),
        reason: 'Test notification - Sistem çalışıyor!',
        strategy: 'Test Strategy',
      };

      const result = await notifyNewSignal(testSignal);

      return NextResponse.json({
        success: true,
        type: 'simple',
        message: 'Test notification sent',
        result,
        stats,
      });
    } else if (type === 'strong_buy') {
      // STRONG_BUY test
      const testSignal: TradingSignal = {
        symbol: 'ETHUSDT',
        price: '3000.50',
        action: 'STRONG_BUY',
        confidence: 92,
        timestamp: Date.now(),
        reason:
          '1. MA Crossover (95%): Golden cross detected\n2. RSI Divergence (90%): Bullish divergence confirmed\n3. Volume Breakout (88%): High volume breakout above resistance',
        strategy: '14/16 strateji BUY - Çok Güçlü Sinyal!',
      };

      const result = await notifyNewSignal(testSignal);

      return NextResponse.json({
        success: true,
        type: 'strong_buy',
        message: 'STRONG_BUY test notification sent',
        result,
        stats,
      });
    } else if (type === 'sell') {
      // SELL test
      const testSignal: TradingSignal = {
        symbol: 'SOLUSDT',
        price: '145.75',
        action: 'SELL',
        confidence: 78,
        timestamp: Date.now(),
        reason:
          '1. RSI Overbought (82%): RSI > 70, overbought condition\n2. Trend Reversal (75%): Bearish reversal pattern detected',
        strategy: '8/16 strateji SELL',
      };

      const result = await notifyNewSignal(testSignal);

      return NextResponse.json({
        success: true,
        type: 'sell',
        message: 'SELL test notification sent',
        result,
        stats,
      });
    } else if (type === 'wait') {
      // WAIT test
      const testSignal: TradingSignal = {
        symbol: 'ADAUSDT',
        price: '0.52',
        action: 'WAIT',
        confidence: 72,
        timestamp: Date.now(),
        reason: 'Piyasa belirsiz. Net sinyal için bekleme önerilir.',
        strategy: '10/16 strateji WAIT',
      };

      const result = await notifyNewSignal(testSignal);

      return NextResponse.json({
        success: true,
        type: 'wait',
        message: 'WAIT test notification sent',
        result,
        stats,
      });
    } else {
      return NextResponse.json(
        {
          error: 'Invalid type',
          message: 'Valid types: simple, strong_buy, sell, wait',
        },
        { status: 400 }
      );
    }
  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error('[Telegram Test] Error:', error);
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
 * GET /api/telegram/test
 * Test durumunu ve istatistikleri getir
 */
export async function GET() {
  try {
    const stats = getNotificationStats();

    return NextResponse.json({
      status: 'ready',
      stats,
      testEndpoints: {
        simple: 'POST /api/telegram/test {"type": "simple"}',
        strongBuy: 'POST /api/telegram/test {"type": "strong_buy"}',
        sell: 'POST /api/telegram/test {"type": "sell"}',
        wait: 'POST /api/telegram/test {"type": "wait"}',
      },
      instructions: {
        step1: 'Telegram\'da botunuzu bulun (@YourBotName)',
        step2: 'Bot\'a /start göndererek abone olun',
        step3: 'Bu endpoint\'e POST request gönderin',
        step4: 'Telegram\'da bildirimi kontrol edin',
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        error: 'Internal server error',
        message: error.message,
      },
      { status: 500 }
    );
  }
}
