/**
 * ⚠️ SYSTEM ALERTS - TELEGRAM NOTIFICATIONS
 * Sistem hata ve durum bildirimleri
 *
 * Özellikler:
 * - Critical system errors
 * - Backend service failures
 * - API health issues
 * - Performance warnings
 * - Security alerts
 */

import { NextResponse } from 'next/server';
import { broadcastMessage } from '@/lib/telegram/notifications';

export const dynamic = 'force-dynamic';

interface SystemAlert {
  type: 'error' | 'warning' | 'info' | 'critical';
  title: string;
  message: string;
  source?: string;
  details?: any;
}

/**
 * Format system alert message for Telegram (TÜRKÇE)
 */
function formatSystemAlert(alert: SystemAlert): string {
  const emoji = {
    error: '❌',
    warning: '⚠️',
    info: 'ℹ️',
    critical: '🚨'
  };

  const priority = {
    error: 'HATA',
    warning: 'UYARI',
    info: 'BİLGİ',
    critical: 'KRİTİK'
  };

  let message = `${emoji[alert.type]} <b>${priority[alert.type]}: ${alert.title}</b>\n\n`;
  message += `${alert.message}\n\n`;

  if (alert.source) {
    message += `📍 Kaynak: ${alert.source}\n`;
  }

  if (alert.details) {
    const detailsStr = typeof alert.details === 'string'
      ? alert.details
      : JSON.stringify(alert.details, null, 2);
    message += `\n<b>Detaylar:</b>\n<pre>${detailsStr}</pre>`;
  }

  message += `\n🕐 ${new Date().toLocaleString('tr-TR', { timeZone: 'Europe/Istanbul' })}`;

  return message;
}

/**
 * GET: Sistem durumu
 */
export async function GET() {
  return NextResponse.json({
    status: 'active',
    description: 'System Alerts - Critical system notifications to Telegram',
    features: [
      'Backend service failures',
      'API health issues',
      'Performance warnings',
      'Security alerts',
      'Critical system errors'
    ],
    alertTypes: {
      critical: 'Immediate attention required (🚨)',
      error: 'System error occurred (❌)',
      warning: 'Potential issue detected (⚠️)',
      info: 'System information (ℹ️)'
    },
    usage: {
      manual: 'POST /api/telegram/system-alerts',
      automated: 'Called by monitoring systems and error handlers',
      example: {
        type: 'error',
        title: 'Backend Service Down',
        message: 'Signal Generator (Port 5004) is not responding',
        source: 'Health Check Monitor',
        details: { port: 5004, lastSeen: '2025-10-26T18:00:00Z' }
      }
    },
    whiteHatRules: [
      '✅ System monitoring only',
      '✅ No user data exposure',
      '✅ Transparent logging',
      '✅ White-hat compliance'
    ]
  });
}

/**
 * POST: Send system alert to Telegram
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();

    const alert: SystemAlert = {
      type: body.type || 'info',
      title: body.title,
      message: body.message,
      source: body.source,
      details: body.details
    };

    // Validate required fields
    if (!alert.title || !alert.message) {
      return NextResponse.json({
        success: false,
        error: 'Missing required fields: title, message'
      }, { status: 400 });
    }

    // Validate alert type
    if (!['error', 'warning', 'info', 'critical'].includes(alert.type)) {
      return NextResponse.json({
        success: false,
        error: 'Invalid alert type. Must be: error, warning, info, critical'
      }, { status: 400 });
    }

    // Format and send to Telegram
    const telegramMessage = formatSystemAlert(alert);

    // ✨ DIREKT TELEGRAM API KULLAN (env var chat IDs)
    const { bot } = await import('@/lib/telegram/bot');
    const chatIds = process.env.TELEGRAM_ALLOWED_CHAT_IDS
      ? process.env.TELEGRAM_ALLOWED_CHAT_IDS.split(',').map((id) => parseInt(id.trim(), 10))
      : [];

    if (chatIds.length === 0) {
      return NextResponse.json({
        success: false,
        error: 'No Telegram chat IDs configured (TELEGRAM_ALLOWED_CHAT_IDS env var)'
      }, { status: 500 });
    }

    let sent = 0;
    let failed = 0;

    for (const chatId of chatIds) {
      try {
        await bot.api.sendMessage(chatId, telegramMessage, { parse_mode: 'HTML' });
        sent++;
      } catch (error: any) {
        failed++;
        console.error(`[System Alerts] Failed to send to ${chatId}:`, error.message);
      }
    }

    console.log(`[System Alerts] ${alert.type.toUpperCase()}: ${alert.title} - Sent: ${sent}, Failed: ${failed}`);

    return NextResponse.json({
      success: true,
      message: 'System alert sent to Telegram',
      stats: {
        sent,
        failed,
        alertType: alert.type,
        title: alert.title
      }
    });

  } catch (error: any) {
    console.error('[System Alerts] Error:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}

/**
 * NOTE: Helper functions moved to @/lib/telegram/system-alerts
 * Import { sendCriticalAlert, sendErrorAlert, sendWarningAlert } from '@/lib/telegram/system-alerts'
 */
