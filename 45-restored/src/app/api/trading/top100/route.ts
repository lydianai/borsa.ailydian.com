/**
 * TOP 100 COINS COMPREHENSIVE ANALYSIS API
 * Full analysis for Top 100 coins with TA-Lib indicators
 * GET /api/trading/top100?timeframe=1h&limit=10
 */

import { NextRequest, NextResponse } from 'next/server';
import { masterIntegrationService } from '@/services/integration/MasterIntegrationService';
import type { Timeframe } from '@/services/market/BinanceOHLCVService';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';
export const maxDuration = 60; // 60 seconds for processing Top 100

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const timeframe = (searchParams.get('timeframe') || '1h') as Timeframe;
    const limit = Math.min(parseInt(searchParams.get('limit') || '10'), 100);

    // Validate timeframe
    const validTimeframes: Timeframe[] = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'];
    if (!validTimeframes.includes(timeframe)) {
      return NextResponse.json(
        { success: false, error: `Invalid timeframe: ${timeframe}` },
        { status: 400 }
      );
    }

    console.log(`📊 Fetching Top ${limit} comprehensive data [${timeframe}]`);

    const startTime = Date.now();

    // Get comprehensive data for Top N coins
    const data = await masterIntegrationService.getTop100ComprehensiveData(timeframe, limit);

    const processingTime = Date.now() - startTime;

    console.log(`✅ Processed ${data.length} coins in ${processingTime}ms`);

    return NextResponse.json({
      success: true,
      timeframe,
      limit,
      count: data.length,
      data,
      processingTime,
      timestamp: Date.now(),
    });
  } catch (error: any) {
    console.error('❌ Error in Top 100 API:', error.message);
    return NextResponse.json(
      { success: false, error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
