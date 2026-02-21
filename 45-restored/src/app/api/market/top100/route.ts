/**
 * TOP 100 COINS API
 * Binance + CoinGecko Hybrid Data
 * Real-time prices from Binance
 * Market rankings from CoinGecko
 */

import { NextRequest, NextResponse } from 'next/server';
import { getMarketDataService } from '@/services/MarketDataService';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const forceRefresh = searchParams.get('refresh') === 'true';

    const marketService = getMarketDataService();

    if (forceRefresh) {
      marketService.clearCache();
    }

    // Top 100 coins (hybrid data)
    const coins = await marketService.getTop100Hybrid();

    // Health check
    const health = await marketService.healthCheck();

    return NextResponse.json({
      success: true,
      data: coins,
      count: coins.length,
      sources: {
        binance: health.binance,
        coingecko: health.coingecko,
      },
      stats: {
        binanceCount: coins.filter(c => c.source === 'binance').length,
        coingeckoCount: coins.filter(c => c.source === 'coingecko').length,
      },
      timestamp: new Date().toISOString(),
      cacheAge: health.lastUpdate,
    });

  } catch (error: any) {
    console.error('Top 100 API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}
