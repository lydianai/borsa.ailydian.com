/**
 * FUNDING RATE API ROUTE
 * With Binance API fallback
 */

import { NextRequest, NextResponse } from 'next/server';

/**
 * Generate fallback funding rate history from Binance API
 */
async function generateBinanceFallback(symbol: string, limit: number) {
  try {
    // Binance Funding Rate History endpoint
    const response = await fetch(
      `https://fapi.binance.com/fapi/v1/fundingRate?symbol=${symbol}&limit=${Math.min(limit, 1000)}`
    );

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    if (data && data.length > 0) {
      // Transform to expected format
      const fundingRates = data.map((item: any) => ({
        funding_rate: parseFloat(item.fundingRate),
        funding_time: new Date(item.fundingTime).toISOString(),
        mark_price: parseFloat(item.markPrice) || 0
      }));

      return {
        success: true,
        data: fundingRates.reverse(), // Most recent first
        source: 'binance_fallback'
      };
    }

    throw new Error('No data from Binance');
  } catch (error) {
    console.error('[Funding Rate] Binance fallback error:', error);
    throw error;
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';
    const limit = parseInt(searchParams.get('limit') || '100');

    // Try Python service first
    try {
      const response = await fetch(
        `http://localhost:5031/funding-rate/${symbol}?limit=${limit}`,
        {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          cache: 'no-store',
          signal: AbortSignal.timeout(3000) // 3s timeout
        }
      );

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data);
      }
    } catch (pythonError) {
      console.warn('[Funding Rate] Python service unavailable, using Binance fallback');
    }

    // Use Binance fallback
    const fallbackData = await generateBinanceFallback(symbol, limit);
    return NextResponse.json(fallbackData);

  } catch (error: any) {
    console.error('[Funding Rate API] Error:', error.message);
    return NextResponse.json(
      { success: false, error: error.message || 'Funding rate unavailable' },
      { status: 500 }
    );
  }
}
