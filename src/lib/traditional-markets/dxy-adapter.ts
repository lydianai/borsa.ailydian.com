/**
 * DXY (US DOLLAR INDEX) API ADAPTER
 * Real-time DXY index data - LIVE DATA ONLY
 *
 * Features:
 * - Real-time DXY index from Yahoo Finance
 * - Intraday price data
 * - 24h change calculation
 * - Support/Resistance levels
 * - 15-minute caching
 * - White-hat error handling
 */

import circuitBreakerManager from '../resilience/circuit-breaker';

// ============================================================================
// TYPES
// ============================================================================

export interface DXYData {
  symbol: string;        // 'DXY'
  name: string;          // 'US Dollar Index'
  price: number;         // Current price
  open: number;          // Today's open
  high: number;          // Today's high
  low: number;           // Today's low
  previousClose: number; // Yesterday's close
  change: number;        // Price change
  changePercent: number; // Percentage change
  volume: number;        // Trading volume
  timestamp: Date;
  // Technical levels
  support: number;
  resistance: number;
}

interface CachedDXYData {
  data: DXYData;
  timestamp: number;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const CACHE_TTL = 900000; // 15 minutes

// Yahoo Finance API endpoints (free, no key required)
const API_ENDPOINTS = {
  // Primary: Yahoo Finance quote API
  quote: 'https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB',

  // Backup: Yahoo Finance summary
  summary: 'https://query1.finance.yahoo.com/v10/finance/quoteSummary/DX-Y.NYB',
};

// ============================================================================
// CACHE
// ============================================================================

let cache: CachedDXYData | null = null;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Calculate support/resistance levels using recent high/low
 */
function calculateSupportResistance(data: any): { support: number; resistance: number } {
  const high = data.high || 0;
  const low = data.low || 0;
  const _close = data.close || 0;

  // Simple support/resistance (can be enhanced with more sophisticated algorithms)
  const range = high - low;
  const support = low + range * 0.236; // Fibonacci 23.6%
  const resistance = low + range * 0.764; // Fibonacci 76.4%

  return {
    support: Math.round(support * 1000) / 1000,
    resistance: Math.round(resistance * 1000) / 1000,
  };
}

// ============================================================================
// API FETCHERS (REAL DATA ONLY)
// ============================================================================

/**
 * Fetch DXY data from Yahoo Finance
 */
async function fetchFromYahooFinance(): Promise<DXYData | null> {
  try {
    const response = await fetch(
      `${API_ENDPOINTS.quote}?interval=1d&range=1d&includePrePost=true`,
      {
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'Mozilla/5.0',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Yahoo Finance API failed: ${response.status}`);
    }

    const json = await response.json();

    if (!json.chart?.result?.[0]) {
      throw new Error('Invalid Yahoo Finance response');
    }

    const result = json.chart.result[0];
    const meta = result.meta;
    const quote = result.indicators?.quote?.[0];

    if (!meta || !quote) {
      throw new Error('Missing data in Yahoo Finance response');
    }

    // Extract latest values
    const latestIndex = quote.close.length - 1;
    const close = quote.close[latestIndex];
    const open = quote.open[latestIndex] || meta.regularMarketPrice;
    const high = quote.high[latestIndex] || meta.regularMarketPrice;
    const low = quote.low[latestIndex] || meta.regularMarketPrice;
    const volume = quote.volume[latestIndex] || 0;

    const previousClose = meta.previousClose || meta.chartPreviousClose;
    const change = close - previousClose;
    const changePercent = (change / previousClose) * 100;

    // Calculate support/resistance
    const { support, resistance } = calculateSupportResistance({ high, low, close });

    const dxyData: DXYData = {
      symbol: 'DXY',
      name: 'US Dollar Index',
      price: Math.round(close * 1000) / 1000,
      open: Math.round(open * 1000) / 1000,
      high: Math.round(high * 1000) / 1000,
      low: Math.round(low * 1000) / 1000,
      previousClose: Math.round(previousClose * 1000) / 1000,
      change: Math.round(change * 1000) / 1000,
      changePercent: Math.round(changePercent * 100) / 100,
      volume,
      support,
      resistance,
      timestamp: new Date(result.meta.regularMarketTime * 1000),
    };

    return dxyData;
  } catch (error) {
    console.error('[DXY] Yahoo Finance error:', error);
    return null;
  }
}

/**
 * Fetch DXY data using backup API
 */
async function fetchFromYahooSummary(): Promise<DXYData | null> {
  try {
    const response = await fetch(
      `${API_ENDPOINTS.summary}?modules=price,summaryDetail`,
      {
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'Mozilla/5.0',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Yahoo Summary API failed: ${response.status}`);
    }

    const json = await response.json();
    const price = json.quoteSummary?.result?.[0]?.price;

    if (!price) {
      throw new Error('Missing price data in Yahoo Summary response');
    }

    const close = price.regularMarketPrice?.raw || 0;
    const open = price.regularMarketOpen?.raw || close;
    const high = price.regularMarketDayHigh?.raw || close;
    const low = price.regularMarketDayLow?.raw || close;
    const previousClose = price.regularMarketPreviousClose?.raw || close;
    const change = price.regularMarketChange?.raw || 0;
    const changePercent = price.regularMarketChangePercent?.raw || 0;
    const volume = price.regularMarketVolume?.raw || 0;

    const { support, resistance } = calculateSupportResistance({ high, low, close });

    const dxyData: DXYData = {
      symbol: 'DXY',
      name: 'US Dollar Index',
      price: Math.round(close * 1000) / 1000,
      open: Math.round(open * 1000) / 1000,
      high: Math.round(high * 1000) / 1000,
      low: Math.round(low * 1000) / 1000,
      previousClose: Math.round(previousClose * 1000) / 1000,
      change: Math.round(change * 1000) / 1000,
      changePercent: Math.round(changePercent * 100) / 100,
      volume,
      support,
      resistance,
      timestamp: new Date(price.regularMarketTime * 1000),
    };

    return dxyData;
  } catch (error) {
    console.error('[DXY] Yahoo Summary error:', error);
    return null;
  }
}

/**
 * Fetch DXY data with automatic fallback
 */
async function fetchDXYData(): Promise<DXYData> {
  // Try primary API
  console.log('[DXY] Fetching from Yahoo Finance...');
  let data = await fetchFromYahooFinance();

  // Try backup API if primary fails
  if (!data) {
    console.log('[DXY] Primary failed, trying backup...');
    data = await fetchFromYahooSummary();
  }

  // If both fail, throw error (NO MOCK DATA)
  if (!data) {
    throw new Error('All DXY APIs failed - no data available');
  }

  return data;
}

// ============================================================================
// PUBLIC API
// ============================================================================

/**
 * Get DXY data with caching - REAL-TIME DATA ONLY
 */
export async function getDXYData(forceRefresh: boolean = false): Promise<DXYData> {
  // Check cache
  if (!forceRefresh && cache && Date.now() - cache.timestamp < CACHE_TTL) {
    console.log('[DXY] Returning cached data');
    return cache.data;
  }

  console.log('[DXY] Fetching fresh real-time data...');

  // Fetch with circuit breaker
  const breaker = circuitBreakerManager.getBreaker('dxy-api', {
    failureThreshold: 3,
    successThreshold: 2,
    timeout: 10000,
  });

  try {
    const data = await breaker.execute(
      async () => await fetchDXYData(),
      async () => {
        // Fallback: Return stale cache if available
        if (cache) {
          console.warn('[DXY] Using stale cache as fallback');
          return cache.data;
        }
        throw new Error('No cache available and all APIs failed');
      }
    );

    // Update cache
    cache = {
      data,
      timestamp: Date.now(),
    };

    return data;
  } catch (error: any) {
    console.error('[DXY] Failed to fetch data:', error);

    // Return stale cache or throw
    if (cache) {
      console.warn('[DXY] Returning stale cache due to error');
      return cache.data;
    }

    throw new Error(`Failed to fetch DXY data: ${error.message}`);
  }
}

/**
 * Clear cache
 */
export function clearDXYCache(): void {
  cache = null;
  console.log('[DXY] Cache cleared');
}

/**
 * Get cache statistics
 */
export function getDXYCacheStats() {
  return {
    isCached: cache !== null,
    age: cache ? Date.now() - cache.timestamp : 0,
    ttl: CACHE_TTL,
    isStale: cache ? Date.now() - cache.timestamp > CACHE_TTL : true,
  };
}

/**
 * Get DXY trend (simplified)
 */
export function getDXYTrend(data: DXYData): 'bullish' | 'bearish' | 'neutral' {
  const { price, support, resistance } = data;

  // Simple trend determination
  const range = resistance - support;
  const position = (price - support) / range;

  if (position > 0.7) return 'bullish';
  if (position < 0.3) return 'bearish';
  return 'neutral';
}
