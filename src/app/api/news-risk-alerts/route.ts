/**
 * NEWS RISK ALERTS API
 *
 * Kritik haber uyarılarını ve risk skorlarını döndürür
 */

import { NextResponse } from 'next/server';
import { newsRiskAnalyzer } from '@/lib/news-risk-analyzer';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET() {
  try {
    const state = newsRiskAnalyzer.getState();

    // Aktif alertleri getir
    const activeAlerts = newsRiskAnalyzer.getActiveAlerts();

    // Response
    return NextResponse.json({
      success: true,
      data: {
        activeAlerts,
        pauseState: {
          globalPause: state.pauseState.globalPause,
          pausedSymbols: Array.from(state.pauseState.pausedSymbols.entries()).map(
            ([symbol, info]) => ({
              symbol,
              reason: info.reason,
              endsAt: info.endsAt.toISOString(),
            })
          ),
          pauseEndsAt: state.pauseState.pauseEndsAt?.toISOString() || null,
          reason: state.pauseState.reason,
        },
        riskScores: Array.from(state.riskScores.entries()).map(([symbol, score]) => ({
          ...score,
          symbol,
        })),
        recentReductions: state.recentReductions.slice(0, 10), // Son 10
        systemEnabled: state.enabled,
        lastUpdate: state.lastUpdate.toISOString(),
      },
    });
  } catch (error: any) {
    console.error('[NewsRiskAPI] Error:', error.message);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}

/**
 * POST - Alert dismiss etme
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { alertId, action } = body;

    if (action === 'dismiss' && alertId) {
      newsRiskAnalyzer.dismissAlert(alertId);
      return NextResponse.json({
        success: true,
        message: 'Alert dismissed',
      });
    }

    if (action === 'toggleSystem') {
      const { enabled } = body;
      newsRiskAnalyzer.setEnabled(enabled);
      return NextResponse.json({
        success: true,
        message: `System ${enabled ? 'enabled' : 'disabled'}`,
      });
    }

    return NextResponse.json(
      {
        success: false,
        error: 'Invalid action',
      },
      { status: 400 }
    );
  } catch (error: any) {
    console.error('[NewsRiskAPI] POST Error:', error.message);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}
