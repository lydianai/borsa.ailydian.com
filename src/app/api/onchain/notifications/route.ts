/**
 * ON-CHAIN NOTIFICATIONS API
 * GET /api/onchain/notifications
 *
 * Returns whale activity notifications based on configured thresholds
 *
 * Query params:
 * - symbols: Comma-separated list of symbols to monitor (optional)
 * - minConfidence: Minimum whale confidence (0-100, default: 60)
 * - minNetflow: Minimum USD netflow (default: 1000000)
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  checkWhaleNotifications,
  getWhaleNotificationSummary,
  monitorSymbolsForWhaleActivity,
  formatForInAppNotification,
} from '@/lib/onchain/whale-notifications';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const symbolsParam = searchParams.get('symbols');
  const minConfidence = parseInt(searchParams.get('minConfidence') || '60', 10);
  const minNetflow = parseInt(searchParams.get('minNetflow') || '1000000', 10);

  try {
    const config = {
      enabled: true,
      minConfidence: Math.max(0, Math.min(100, minConfidence)),
      minNetflow: Math.max(0, minNetflow),
    };

    // Monitor specific symbols if provided
    if (symbolsParam) {
      const symbols = symbolsParam.split(',').map((s) => s.trim().toUpperCase());
      const monitorResults = await monitorSymbolsForWhaleActivity(symbols, config);

      const notifications = Array.from(monitorResults.entries())
        .filter(([_, notification]) => notification !== null)
        .map(([symbol, notification]) => ({
          symbol,
          notification: formatForInAppNotification(notification!),
        }));

      return NextResponse.json({
        success: true,
        data: {
          monitored: symbols,
          notifications,
          count: notifications.length,
        },
      });
    }

    // Get all notifications
    const summary = await getWhaleNotificationSummary(config);

    const formattedNotifications = summary.notifications.map((notification) =>
      formatForInAppNotification(notification)
    );

    return NextResponse.json({
      success: true,
      data: {
        summary: {
          total: summary.total,
          accumulation: summary.byType.accumulation,
          distribution: summary.byType.distribution,
          critical: summary.byType.critical,
        },
        notifications: formattedNotifications,
        config: {
          minConfidence,
          minNetflow,
        },
      },
    });
  } catch (error: any) {
    console.error('[API /onchain/notifications] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch notifications',
        data: null,
      },
      { status: 500 }
    );
  }
}
