/**
 * LONG/SHORT RATIO API ROUTE
 * With Binance API fallback
 */

import { NextRequest, NextResponse} from 'next/server';

/**
 * Generate fallback long/short ratio from Binance API
 * Uses global account long/short ratio endpoint
 */
async function generateBinanceFallback(symbol: string) {
  try {
    // Binance Global Long/Short Account Ratio
    const response = await fetch(
      `https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=${symbol}&period=5m&limit=1`
    );

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    if (data && data.length > 0) {
      const latestRatio = data[0];
      const longRatio = parseFloat(latestRatio.longAccount) || 0;
      const shortRatio = parseFloat(latestRatio.shortAccount) || 0;
      const total = longRatio + shortRatio;

      // Calculate percentages
      const longPercentage = total > 0 ? (longRatio / total) * 100 : 50;
      const shortPercentage = total > 0 ? (shortRatio / total) * 100 : 50;

      return {
        success: true,
        data: {
          symbol,
          long_percentage: parseFloat(longPercentage.toFixed(2)),
          short_percentage: parseFloat(shortPercentage.toFixed(2)),
          long_account: Math.round(longRatio * 10000), // Estimate account count
          short_account: Math.round(shortRatio * 10000),
          ratio: longRatio / (shortRatio || 1),
          timestamp: new Date(parseInt(latestRatio.timestamp)).toISOString(),
          source: 'binance_fallback'
        }
      };
    }

    throw new Error('No data from Binance');
  } catch (error) {
    console.error('[Long/Short Ratio] Binance fallback error:', error);
    throw error;
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';

    // Try Python service first
    try {
      const response = await fetch(
        `http://localhost:5031/long-short-ratio/${symbol}`,
        {
          method: 'GET',
          cache: 'no-store',
          signal: AbortSignal.timeout(3000) // 3s timeout
        }
      );

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data);
      }
    } catch (pythonError) {
      console.warn('[Long/Short Ratio] Python service unavailable, using Binance fallback');
    }

    // Use Binance fallback
    const fallbackData = await generateBinanceFallback(symbol);
    return NextResponse.json(fallbackData);

  } catch (error: any) {
    console.error('[Long/Short Ratio API] Error:', error);
    return NextResponse.json(
      { success: false, error: error.message || 'Long/Short ratio unavailable' },
      { status: 500 }
    );
  }
}
