/**
 * 🔧 TELEGRAM ADMIN API
 * Bildirim ayarlarını kontrol et ve yönet
 *
 * Features:
 * - Config görüntüleme
 * - Stats görüntüleme
 * - Broadcast mesaj gönderimi
 * - Spam cache temizleme
 *
 * Endpoint: GET /api/telegram/admin
 *
 * ⚠️ SECURITY:
 * - Read-only operations
 * - No sensitive data exposed
 */

import { NextRequest, NextResponse } from 'next/server';
import { TELEGRAM_CONFIG } from '@/lib/telegram/config';
import { getNotificationStats, broadcastMessage } from '@/lib/telegram/notifications';
import { clearSpamCache } from '@/lib/telegram/config';

export const dynamic = 'force-dynamic';

// ============================================================================
// ADMIN ENDPOINTS
// ============================================================================

/**
 * GET /api/telegram/admin
 * Telegram sistem durumunu ve ayarlarını getir
 */
export async function GET(request: NextRequest) {
  try {
    const stats = getNotificationStats();

    return NextResponse.json({
      status: 'active',
      timestamp: new Date().toISOString(),

      // Kullanıcı Ayarları
      config: {
        signalTypes: TELEGRAM_CONFIG.enabledSignalTypes,
        minConfidence: TELEGRAM_CONFIG.minConfidence,
        mode: TELEGRAM_CONFIG.notificationMode,
        strategies: TELEGRAM_CONFIG.enabledStrategies.length === 0
          ? 'All strategies (16)'
          : TELEGRAM_CONFIG.enabledStrategies.join(', '),
        symbolFilter: TELEGRAM_CONFIG.symbolWhitelist.length === 0
          ? 'All symbols'
          : TELEGRAM_CONFIG.symbolWhitelist.join(', '),
        spamPrevention: `${TELEGRAM_CONFIG.minTimeBetweenSameSymbol / 1000 / 60} minutes`,
        dailySummary: TELEGRAM_CONFIG.sendDailySummary
          ? `Yes (${TELEGRAM_CONFIG.dailySummaryHours.join(':')}:00)`
          : 'No',
      },

      // İstatistikler
      stats: {
        subscriberCount: stats.subscriberCount,
        subscribers: stats.subscribers,
      },

      // Test Endpoints
      testEndpoints: {
        simple: 'POST /api/telegram/test {"type": "simple"}',
        strongBuy: 'POST /api/telegram/test {"type": "strong_buy"}',
        sell: 'POST /api/telegram/test {"type": "sell"}',
        wait: 'POST /api/telegram/test {"type": "wait"}',
      },

      // Kullanım Talimatları
      usage: {
        subscribe: 'Telegram\'da @YourBotName\'i bulun ve /start gönderin',
        test: 'POST /api/telegram/test ile test bildirimi gönderin',
        webhook: 'GET /api/telegram/webhook ile webhook durumunu kontrol edin',
        admin: 'GET /api/telegram/admin (bu endpoint) ile ayarları görün',
      },
    });
  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error('[Telegram Admin] Error:', error);
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
 * POST /api/telegram/admin
 * Admin işlemleri (broadcast, cache clear)
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, message } = body;

    if (action === 'broadcast') {
      // Broadcast mesaj gönder
      if (!message) {
        return NextResponse.json(
          { error: 'Message is required for broadcast' },
          { status: 400 }
        );
      }

      const result = await broadcastMessage(message, { parse_mode: 'Markdown' });

      return NextResponse.json({
        success: true,
        action: 'broadcast',
        message: 'Broadcast sent',
        result,
      });
    } else if (action === 'clear_spam_cache') {
      // Spam cache temizle
      clearSpamCache();

      return NextResponse.json({
        success: true,
        action: 'clear_spam_cache',
        message: 'Spam prevention cache cleared',
      });
    } else {
      return NextResponse.json(
        {
          error: 'Invalid action',
          message: 'Valid actions: broadcast, clear_spam_cache',
        },
        { status: 400 }
      );
    }
  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error('[Telegram Admin POST] Error:', error);
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
