/**
 * 📊 BINANCE MULTI-TIMEFRAME API
 *
 * Fetches real price changes for multiple timeframes (1h, 4h, 1d, 1w)
 * for all top futures coins using Binance klines data.
 *
 * Returns: Array of coins with real change percentages for each timeframe
 * Cache: 5 minutes
 */

import { NextRequest, NextResponse } from 'next/server';
import { fetchBinanceFuturesData } from '@/lib/binance-data-fetcher';

export const dynamic = 'force-dynamic';

const MULTI_TF_CACHE_TTL = 5 * 60 * 1000; // 5 minutes
let cache: { data: any; timestamp: number } | null = null;

interface TimeframeData {
  symbol: string;
  price: number;
  change1H: number;
  change4H: number;
  change1D: number;
  change1W: number;
  volume24h: number;
}

async function fetchTimeframeChange(symbol: string, interval: string, hoursBack: number): Promise<number> {
  try {
    const limit = Math.ceil(hoursBack / getIntervalHours(interval)) + 1;
    const response = await fetch(
      `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`
    );

    if (!response.ok) {
      console.error(`[Multi-TF] Failed to fetch ${symbol} ${interval}`);
      return 0;
    }

    const klines = await response.json();

    if (!klines || klines.length < 2) {
      return 0;
    }

    const oldPrice = parseFloat(klines[0][1]); // First candle open
    const currentPrice = parseFloat(klines[klines.length - 1][4]); // Last candle close

    return ((currentPrice - oldPrice) / oldPrice) * 100;
  } catch (error) {
    console.error(`[Multi-TF] Error fetching ${symbol} ${interval}:`, error);
    return 0;
  }
}

function getIntervalHours(interval: string): number {
  const map: { [key: string]: number } = {
    '1h': 1,
    '4h': 4,
    '1d': 24,
    '1w': 168
  };
  return map[interval] || 1;
}

export async function GET(_request: NextRequest) {
  try {
    // Check cache
    const now = Date.now();
    if (cache && (now - cache.timestamp) < MULTI_TF_CACHE_TTL) {
      console.log('[Multi-TF API] Cache hit');
      return NextResponse.json({
        success: true,
        data: cache.data,
        cached: true,
        cacheAge: Math.round((now - cache.timestamp) / 1000)
      });
    }

    console.log('[Multi-TF API] Fetching fresh data...');

    // Use shared data fetcher with fallback support
    const binanceData = await fetchBinanceFuturesData();

    if (!binanceData.success || !binanceData.data) {
      throw new Error('Failed to fetch ticker data from Binance/fallback sources');
    }

    // Get top coins by volume
    const topCoins = binanceData.data.all
      .sort((a, b) => b.volume24h - a.volume24h)
      .slice(0, 50);

    console.log(`[Multi-TF API] Processing ${topCoins.length} top coins...`);

    // Fetch timeframe data for each coin (in batches to avoid rate limits)
    const results: TimeframeData[] = [];

    for (const coin of topCoins) {
      const symbol = coin.symbol;

      // Fetch all timeframes in parallel for this symbol
      const [change1H, change4H, change1D, change1W] = await Promise.all([
        fetchTimeframeChange(symbol, '1h', 1),
        fetchTimeframeChange(symbol, '4h', 4),
        fetchTimeframeChange(symbol, '1d', 24),
        fetchTimeframeChange(symbol, '1w', 168)
      ]);

      results.push({
        symbol,
        price: coin.price,
        change1H: parseFloat(change1H.toFixed(2)),
        change4H: parseFloat(change4H.toFixed(2)),
        change1D: parseFloat(change1D.toFixed(2)),
        change1W: parseFloat(change1W.toFixed(2)),
        volume24h: coin.volume24h
      });

      // Small delay to avoid rate limits (Binance allows ~1200 requests/min)
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    console.log(`[Multi-TF API] Successfully fetched ${results.length} coins with multi-timeframe data`);

    // Update cache
    cache = { data: results, timestamp: now };

    return NextResponse.json({
      success: true,
      data: results,
      cached: false,
      count: results.length
    });

  } catch (error) {
    console.error('[Multi-TF API Error]:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Multi-timeframe data fetch failed'
      },
      { status: 500 }
    );
  }
}
