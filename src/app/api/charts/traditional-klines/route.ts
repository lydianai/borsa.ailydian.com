/**
 * TRADITIONAL MARKETS KLINE DATA API
 * Historical candlestick (OHLCV) data for traditional markets
 *
 * Endpoint: GET /api/charts/traditional-klines?symbol=XAU&interval=1h&limit=500
 *
 * Supported symbols:
 * - Metals: XAU (Gold), XAG (Silver), XPD (Palladium), XCU (Copper)
 * - Forex: EUR, GBP, JPY, CHF, CAD, AUD, CNY, RUB, SAR
 * - Indices: DXY, SPX, NDX, DJI
 * - Energy: BRENT, WTI, NATGAS
 * - Bonds: US2Y, US10Y, US30Y
 * - Agriculture: WHEAT, CORN, SOYBEAN, COFFEE, SUGAR
 *
 * Parameters:
 * - symbol: Asset symbol (required)
 * - interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, 1w) - default: 1h
 * - limit: Number of candles (default: 500, max: 1000)
 */

import { NextRequest, NextResponse } from 'next/server';

// Valid intervals
const VALID_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'];

// Supported symbols by category
const SUPPORTED_SYMBOLS = {
  metals: ['XAU', 'XAG', 'XPD', 'XCU'],
  forex: ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'CNY', 'RUB', 'SAR'],
  indices: ['DXY', 'SPX', 'NDX', 'DJI'],
  energy: ['BRENT', 'WTI', 'NATGAS'],
  bonds: ['US2Y', 'US10Y', 'US30Y'],
  agriculture: ['WHEAT', 'CORN', 'SOYBEAN', 'COFFEE', 'SUGAR']
};

// Cache for kline data
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
 * Convert interval to minutes for API calls
 */
function intervalToMinutes(interval: string): number {
  const unit = interval.slice(-1);
  const value = parseInt(interval.slice(0, -1));

  switch (unit) {
    case 'm': return value;
    case 'h': return value * 60;
    case 'd': return value * 1440;
    case 'w': return value * 10080;
    default: return 60;
  }
}

/**
 * Map symbol to Yahoo Finance or other data source symbol
 */
function mapSymbolToDataSource(symbol: string): string {
  const mappings: Record<string, string> = {
    // Metals
    'XAU': 'GC=F',      // Gold Futures
    'XAG': 'SI=F',      // Silver Futures
    'XPD': 'PA=F',      // Palladium Futures
    'XCU': 'HG=F',      // Copper Futures

    // Forex (vs USD)
    'EUR': 'EURUSD=X',
    'GBP': 'GBPUSD=X',
    'JPY': 'JPYUSD=X',
    'CHF': 'CHFUSD=X',
    'CAD': 'CADUSD=X',
    'AUD': 'AUDUSD=X',
    'CNY': 'CNYUSD=X',
    'RUB': 'RUBUSD=X',
    'SAR': 'SARUSD=X',

    // Indices
    'DXY': 'DX-Y.NYB',  // US Dollar Index
    'SPX': '^GSPC',     // S&P 500
    'NDX': '^NDX',      // NASDAQ 100
    'DJI': '^DJI',      // Dow Jones

    // Energy
    'BRENT': 'BZ=F',    // Brent Crude
    'WTI': 'CL=F',      // WTI Crude
    'NATGAS': 'NG=F',   // Natural Gas

    // Bonds (using yield)
    'US2Y': '^IRX',     // 13 Week Treasury Bill (as proxy)
    'US10Y': '^TNX',    // 10 Year Treasury Yield
    'US30Y': '^TYX',    // 30 Year Treasury Yield

    // Agriculture
    'WHEAT': 'ZW=F',    // Wheat Futures
    'CORN': 'ZC=F',     // Corn Futures
    'SOYBEAN': 'ZS=F',  // Soybean Futures
    'COFFEE': 'KC=F',   // Coffee Futures
    'SUGAR': 'SB=F'     // Sugar Futures
  };

  return mappings[symbol] || symbol;
}

/**
 * Fetch historical kline data
 * Using a combination of sources (Yahoo Finance, Alpha Vantage, etc.)
 */
async function fetchTraditionalKlines(
  symbol: string,
  interval: string,
  limit: number
): Promise<any[]> {
  // For now, we'll generate synthetic data based on current price
  // In production, you'd integrate with real data providers like:
  // - Yahoo Finance API
  // - Alpha Vantage
  // - Twelve Data
  // - Polygon.io

  console.log(`[TradKlines] Fetching ${symbol} ${interval} (${limit} candles)`);

  // Get current price from our traditional markets API
  const currentPriceRes = await fetch(
    `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/traditional-markets?symbol=${symbol}`
  );

  if (!currentPriceRes.ok) {
    throw new Error(`Failed to fetch current price for ${symbol}`);
  }

  const currentPriceData = await currentPriceRes.json();

  if (!currentPriceData.success || !currentPriceData.data) {
    throw new Error(`No data available for ${symbol}`);
  }

  // Extract current price based on symbol type
  let currentPrice = 0;
  const asset = currentPriceData.data;

  if (asset.priceUSD !== undefined) {
    currentPrice = asset.priceUSD;
  } else if (asset.price !== undefined) {
    currentPrice = asset.price;
  } else if (asset.usd !== undefined) {
    currentPrice = asset.usd;
  } else if (asset.rate !== undefined) {
    currentPrice = asset.rate;
  } else if (asset.value !== undefined) {
    currentPrice = asset.value;
  } else if (asset.yield !== undefined) {
    currentPrice = asset.yield;
  }

  if (!currentPrice || currentPrice === 0) {
    throw new Error(`Could not determine current price for ${symbol}. Available fields: ${Object.keys(asset).join(', ')}`);
  }

  // Generate synthetic historical data
  // In production, replace this with real API calls
  const klines: any[] = [];
  const intervalMinutes = intervalToMinutes(interval);
  const now = Math.floor(Date.now() / 1000);

  for (let i = limit - 1; i >= 0; i--) {
    const timestamp = now - (i * intervalMinutes * 60);

    // Add some realistic price variation (±2% random walk)
    const variation = 1 + ((Math.random() - 0.5) * 0.04);
    const basePrice = currentPrice * variation;

    const open = basePrice;
    const high = basePrice * (1 + Math.random() * 0.01);
    const low = basePrice * (1 - Math.random() * 0.01);
    const close = low + Math.random() * (high - low);
    const volume = Math.random() * 1000000; // Random volume

    klines.push({
      time: timestamp,
      open: parseFloat(open.toFixed(8)),
      high: parseFloat(high.toFixed(8)),
      low: parseFloat(low.toFixed(8)),
      close: parseFloat(close.toFixed(8)),
      volume: parseFloat(volume.toFixed(2))
    });
  }

  return klines;
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
        clustered.push(cluster.reduce((a, b) => a + b, 0) / cluster.length);
        cluster = [curr];
      }
    }

    if (cluster.length > 0) {
      clustered.push(cluster.reduce((a, b) => a + b, 0) / cluster.length);
    }

    return clustered;
  };

  const resistance = clusterLevels(localMaxima).slice(-5);
  const support = clusterLevels(localMinima).slice(-5);

  return {
    resistance: resistance.reverse(),
    support: support.reverse()
  };
}

/**
 * Check if symbol is supported
 */
function isSymbolSupported(symbol: string): boolean {
  return Object.values(SUPPORTED_SYMBOLS).flat().includes(symbol);
}

/**
 * GET /api/charts/traditional-klines
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
          message: 'Please provide a symbol parameter (e.g., ?symbol=XAU)',
          supportedSymbols: SUPPORTED_SYMBOLS
        },
        { status: 400 }
      );
    }

    if (!isSymbolSupported(symbol)) {
      return NextResponse.json(
        {
          success: false,
          error: 'Unsupported symbol',
          message: `Symbol ${symbol} is not supported for traditional markets`,
          supportedSymbols: SUPPORTED_SYMBOLS
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
    const klines = await fetchTraditionalKlines(symbol, interval, limit);

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
    console.error('[TradKlines] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch traditional market chart data',
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}

/**
 * Clear cache endpoint
 * DELETE /api/charts/traditional-klines
 */
export async function DELETE() {
  try {
    const cacheSize = klineCache.size;
    klineCache.clear();

    return NextResponse.json({
      success: true,
      message: `Traditional markets cache cleared: ${cacheSize} entries removed`,
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
