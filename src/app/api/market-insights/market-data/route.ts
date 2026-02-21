/**
 * MARKET DATA API ROUTE
 * Provides real-time market data from Binance Futures
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';

    console.log(`[Market Insights - Market Data] Fetching data for ${symbol}`);

    // Use Binance Futures API instead of localhost:5031
    const protocol = request.headers.get('x-forwarded-proto') || 'https';
    const host = request.headers.get('host') || 'localhost:3000';
    const baseUrl = `${protocol}://${host}`;
    const response = await fetch(
      `${baseUrl}/api/binance/futures`,
      { method: 'GET', cache: 'no-store' }
    );

    if (!response.ok) {
      return NextResponse.json(
        { success: false, error: `Binance API error: ${response.status}` },
        { status: response.status }
      );
    }

    const binanceData = await response.json();

    if (!binanceData.success || !binanceData.data.all) {
      return NextResponse.json(
        { success: false, error: 'Invalid Binance data format' },
        { status: 500 }
      );
    }

    // Find requested symbol
    const coin = binanceData.data.all.find((c: any) => c.symbol === symbol);

    if (!coin) {
      return NextResponse.json(
        { success: false, error: `Symbol ${symbol} not found` },
        { status: 404 }
      );
    }

    // Return market data in expected format
    return NextResponse.json({
      success: true,
      data: {
        symbol: coin.symbol,
        price: coin.price,
        change24h: coin.change24h,
        changePercent24h: coin.changePercent24h,
        volume24h: coin.volume24h,
        high24h: coin.high24h,
        low24h: coin.low24h,
        lastUpdate: coin.lastUpdate,
      },
      timestamp: new Date().toISOString(),
    });

  } catch (error: any) {
    console.error('[Market Insights - Market Data Error]:', error);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
