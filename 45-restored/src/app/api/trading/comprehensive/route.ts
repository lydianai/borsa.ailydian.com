/**
 * COMPREHENSIVE MARKET DATA API
 * Full analysis combining CMC + Binance + TA-Lib
 * GET /api/trading/comprehensive?symbol=BTC&timeframes=1h,4h,1d
 */

import { NextRequest, NextResponse } from 'next/server';
import { masterIntegrationService } from '@/services/integration/MasterIntegrationService';
import type { Timeframe } from '@/services/market/BinanceOHLCVService';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol');
    const timeframesParam = searchParams.get('timeframes') || '1h,4h,1d';

    if (!symbol) {
      return NextResponse.json(
        { success: false, error: 'Symbol parameter is required' },
        { status: 400 }
      );
    }

    // Parse timeframes
    const timeframes = timeframesParam.split(',').map(tf => tf.trim() as Timeframe);

    // Validate timeframes
    const validTimeframes: Timeframe[] = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'];
    const invalidTf = timeframes.find(tf => !validTimeframes.includes(tf));
    if (invalidTf) {
      return NextResponse.json(
        { success: false, error: `Invalid timeframe: ${invalidTf}` },
        { status: 400 }
      );
    }

    console.log(`📊 Fetching comprehensive data for ${symbol} [${timeframes.join(', ')}]`);

    // Get comprehensive data
    const data = await masterIntegrationService.getComprehensiveData(symbol, timeframes);

    if (!data) {
      return NextResponse.json(
        { success: false, error: `No data found for symbol: ${symbol}` },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      symbol,
      timeframes,
      data,
      timestamp: Date.now(),
    });
  } catch (error: any) {
    console.error('❌ Error in comprehensive API:', error.message);
    return NextResponse.json(
      { success: false, error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
