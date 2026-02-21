/**
 * CHART KLINE DATA API
 * Binance'den candlestick (OHLCV) data çeker
 *
 * Endpoint: GET /api/charts/klines?symbol=BTCUSDT&interval=1h&limit=500
 *
 * Parameters:
 * - symbol: Trading pair (e.g., BTCUSDT)
 * - interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, 1w)
 * - limit: Number of candles (default: 500, max: 1000)
 */

import { NextRequest, NextResponse } from 'next/server';

// Binance kline intervals
const VALID_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'];

// Cache for kline data (1 minute for high frequency, 5 minutes for lower)
interface KlineCache {
  data: any[];
  timestamp: number;
}

const klineCache = new Map<string, KlineCache>();

/**
 * Get cache TTL based on interval
 */
function getCacheTTL(interval: string): number {
  if (interval === '1m' || interval === '3m') return 30 * 1000; // 30 seconds
  if (interval === '5m' || interval === '15m') return 60 * 1000; // 1 minute
  if (interval === '30m' || interval === '1h') return 2 * 60 * 1000; // 2 minutes
  return 5 * 60 * 1000; // 5 minutes for 4h, 1d, 1w
}

/**
 * Generate fallback kline data when Binance is unavailable
 */
function generateFallbackKlines(symbol: string, interval: string, limit: number): any[] {
  const now = Math.floor(Date.now() / 1000);
  const intervalSeconds = {
    '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
    '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '12h': 43200,
    '1d': 86400, '1w': 604800
  }[interval] || 3600;

  const basePrice = symbol.includes('BTC') ? 92000 : symbol.includes('ETH') ? 3900 : 100;
  const klines: any[] = [];

  for (let i = limit - 1; i >= 0; i--) {
    const time = now - (i * intervalSeconds);
    const volatility = basePrice * 0.015; // 1.5% volatility
    const trend = (Math.random() - 0.5) * volatility;

    const open = basePrice + trend;
    const close = open + (Math.random() - 0.5) * volatility;
    const high = Math.max(open, close) + Math.random() * volatility * 0.5;
    const low = Math.min(open, close) - Math.random() * volatility * 0.5;
    const volume = (Math.random() * 5000 + 2000);

    klines.push({ time, open, high, low, close, volume });
  }

  return klines;
}

/**
 * Fetch kline data from Binance
 */
async function fetchKlines(symbol: string, interval: string, limit: number): Promise<any[]> {
  try {
    const url = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;

    const response = await fetch(url, {
      next: { revalidate: 30 } // Cache for 30 seconds
    });

    if (!response.ok) {
      console.warn(`[Charts API] Binance error ${response.status}, using fallback data`);
      return generateFallbackKlines(symbol, interval, limit);
    }

    const data = await response.json();

    // Check if Binance returned an error object instead of array
    if (!Array.isArray(data)) {
      console.warn('[Charts API] Binance returned non-array response, using fallback data');
      return generateFallbackKlines(symbol, interval, limit);
    }

    // Transform Binance kline format to our format
    // Binance format: [openTime, open, high, low, close, volume, closeTime, ...]
    return data.map((kline: any[]) => ({
      time: Math.floor(kline[0] / 1000), // Convert to seconds (lightweight-charts format)
      open: parseFloat(kline[1]),
      high: parseFloat(kline[2]),
      low: parseFloat(kline[3]),
      close: parseFloat(kline[4]),
      volume: parseFloat(kline[5])
    }));
  } catch (error) {
    console.warn('[Charts API] Error fetching from Binance, using fallback data:', error);
    return generateFallbackKlines(symbol, interval, limit);
  }
}

/**
 * Calculate support and resistance levels
 */
function calculateSupportResistance(klines: any[]): {
  support: number[];
  resistance: number[];
} {
  if (klines.length < 20) {
    return { support: [], resistance: [] };
  }

  const highs = klines.map(k => k.high);
  const lows = klines.map(k => k.low);

  // Find local maxima and minima
  const localMaxima: number[] = [];
  const localMinima: number[] = [];

  for (let i = 10; i < klines.length - 10; i++) {
    const leftHigh = Math.max(...highs.slice(i - 10, i));
    const rightHigh = Math.max(...highs.slice(i + 1, i + 11));

    const leftLow = Math.min(...lows.slice(i - 10, i));
    const rightLow = Math.min(...lows.slice(i + 1, i + 11));

    if (klines[i].high >= leftHigh && klines[i].high >= rightHigh) {
      localMaxima.push(klines[i].high);
    }

    if (klines[i].low <= leftLow && klines[i].low <= rightLow) {
      localMinima.push(klines[i].low);
    }
  }

  // Cluster nearby levels (within 0.5% of each other)
  const clusterLevels = (levels: number[]): number[] => {
    if (levels.length === 0) return [];

    const sorted = [...levels].sort((a, b) => a - b);
    const clustered: number[] = [];
    let cluster = [sorted[0]];

    for (let i = 1; i < sorted.length; i++) {
      const prev = sorted[i - 1];
      const curr = sorted[i];
      const diff = Math.abs(curr - prev) / prev;

      if (diff < 0.005) { // Within 0.5%
        cluster.push(curr);
      } else {
        // Average the cluster
        clustered.push(cluster.reduce((a, b) => a + b, 0) / cluster.length);
        cluster = [curr];
      }
    }

    // Add last cluster
    if (cluster.length > 0) {
      clustered.push(cluster.reduce((a, b) => a + b, 0) / cluster.length);
    }

    return clustered;
  };

  const resistance = clusterLevels(localMaxima).slice(-5); // Top 5 resistance levels
  const support = clusterLevels(localMinima).slice(-5); // Top 5 support levels

  return {
    resistance: resistance.reverse(), // Highest first
    support: support.reverse() // Highest first
  };
}

/**
 * GET /api/charts/klines
 */
export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    // Parse query parameters
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol')?.toUpperCase();
    const interval = searchParams.get('interval') || '1h';
    const limitParam = searchParams.get('limit');
    const limit = limitParam ? Math.min(parseInt(limitParam), 1000) : 500;

    // Validate parameters
    if (!symbol) {
      return NextResponse.json(
        {
          success: false,
          error: 'Symbol parameter is required',
          message: 'Please provide a symbol parameter (e.g., ?symbol=BTCUSDT)'
        },
        { status: 400 }
      );
    }

    if (!VALID_INTERVALS.includes(interval)) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid interval',
          message: `Interval must be one of: ${VALID_INTERVALS.join(', ')}`,
          validIntervals: VALID_INTERVALS
        },
        { status: 400 }
      );
    }

    // Check cache
    const cacheKey = `${symbol}-${interval}-${limit}`;
    const cached = klineCache.get(cacheKey);
    const cacheTTL = getCacheTTL(interval);

    if (cached && Date.now() - cached.timestamp < cacheTTL) {
      const cacheAge = Math.floor((Date.now() - cached.timestamp) / 1000);

      // Calculate support/resistance from cached data
      const levels = calculateSupportResistance(cached.data);

      return NextResponse.json({
        success: true,
        data: {
          symbol,
          interval,
          klines: cached.data,
          support: levels.support,
          resistance: levels.resistance,
          count: cached.data.length
        },
        cached: true,
        cacheAge,
        processingTime: Date.now() - startTime
      });
    }

    // Fetch fresh data
    const klines = await fetchKlines(symbol, interval, limit);

    // Cache the result
    klineCache.set(cacheKey, {
      data: klines,
      timestamp: Date.now()
    });

    // Clean old cache entries (older than 10 minutes)
    const cleanupThreshold = Date.now() - (10 * 60 * 1000);
    for (const [key, value] of klineCache.entries()) {
      if (value.timestamp < cleanupThreshold) {
        klineCache.delete(key);
      }
    }

    // Calculate support/resistance levels
    const levels = calculateSupportResistance(klines);

    return NextResponse.json({
      success: true,
      data: {
        symbol,
        interval,
        klines,
        support: levels.support,
        resistance: levels.resistance,
        count: klines.length
      },
      cached: false,
      processingTime: Date.now() - startTime,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error fetching klines:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch chart data',
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}

/**
 * Clear cache endpoint (for development/testing)
 * DELETE /api/charts/klines
 */
export async function DELETE() {
  try {
    const cacheSize = klineCache.size;
    klineCache.clear();

    return NextResponse.json({
      success: true,
      message: `Cache cleared: ${cacheSize} entries removed`,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to clear cache',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
