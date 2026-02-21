/**
 * CRYPTO MULTI-STRATEGY ANALYSIS API
 * Returns 8-strategy analysis for any cryptocurrency symbol
 * Same format as traditional-markets multi-strategy
 */

import { NextRequest, NextResponse } from 'next/server';
import { fetchBinanceFuturesData } from '@/lib/binance-data-fetcher';
import { analyzeAssetWithAllStrategies } from '@/lib/analyzers/multi-strategy-traditional';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  try {
    const { symbol: symbolParam } = await params;
    const symbol = symbolParam || 'BTCUSDT';

    console.log(`[Crypto Multi-Strategy] Analyzing ${symbol}...`);

    // Fetch real-time price data with fallback
    const priceData = await fetchBinanceFuturesData();

    if (!priceData.success || !priceData.data || !priceData.data.all) {
      console.error('[Crypto Multi-Strategy] Price data fetch failed:', priceData.error);
      throw new Error(priceData.error || 'Fiyat verisi alınamadı - tüm kaynaklar başarısız');
    }

    // Find the coin data
    const coinData = priceData.data.all.find((c: any) => c.symbol === symbol);

    if (!coinData) {
      console.error(`[Crypto Multi-Strategy] Coin not found: ${symbol}`);
      throw new Error(`Coin not found: ${symbol}`);
    }

    const currentPrice = Number(coinData.price);
    const changePercent24h = Number(coinData.changePercent24h) || 0;

    console.log(`[Crypto Multi-Strategy] ${symbol}: $${currentPrice}, ${changePercent24h}%`);

    // Run 8-strategy analysis
    const analysis = analyzeAssetWithAllStrategies(
      symbol,
      currentPrice,
      changePercent24h,
      'crypto'
    );

    console.log(`[Crypto Multi-Strategy] Analysis complete: ${analysis.overallSignal} (${analysis.overallConfidence}%)`);

    return NextResponse.json({
      success: true,
      data: analysis,
    });
  } catch (error: any) {
    console.error('[Crypto Multi-Strategy] Error:', error.message);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Multi-strategy analysis failed',
      },
      { status: 500 }
    );
  }
}
