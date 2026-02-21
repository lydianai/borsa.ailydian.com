/**
 * BINANCE HISTORICAL DATA FETCHER
 * Zero-Error implementation with caching
 * Fetches 7d and 30d historical candlestick data
 */

export interface HistoricalKline {
  openTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closeTime: number;
  changePercent: number;
}

export interface HistoricalData {
  symbol: string;
  changePercent7d: number;
  changePercent30d: number;
  volume7d: number;
  volume30d: number;
  volatility7d: number;
  volatility30d: number;
  priceNow: number;
  price7dAgo: number;
  price30dAgo: number;
}

// In-memory cache
const cache: Map<string, { data: HistoricalData; timestamp: number }> = new Map();
const CACHE_DURATION = 300000; // 5 minutes

/**
 * Fetch historical klines from Binance API
 */
async function fetchKlines(
  symbol: string,
  interval: string,
  limit: number
): Promise<any[]> {
  const baseUrl = "https://fapi.binance.com";
  const endpoint = `/fapi/v1/klines`;

  const url = `${baseUrl}${endpoint}?symbol=${symbol}&interval=${interval}&limit=${limit}`;

  try {
    const response = await fetch(url);

    if (!response.ok) {
      console.error(`Binance API error for ${symbol}: ${response.status}`);
      return [];
    }

    const data = await response.json();
    return data;
  } catch (error: any) {
    console.error(`Failed to fetch klines for ${symbol}:`, error.message);
    return [];
  }
}

/**
 * Calculate percentage change between two prices
 */
function calculateChange(priceNow: number, priceBefore: number): number {
  if (!priceBefore || priceBefore === 0) return 0;
  return ((priceNow - priceBefore) / priceBefore) * 100;
}

/**
 * Calculate volatility (standard deviation of returns)
 */
function calculateVolatility(klines: any[]): number {
  if (klines.length < 2) return 0;

  const returns: number[] = [];

  for (let i = 1; i < klines.length; i++) {
    const prevClose = parseFloat(klines[i - 1][4]);
    const currentClose = parseFloat(klines[i][4]);

    if (prevClose > 0) {
      const returnPct = ((currentClose - prevClose) / prevClose) * 100;
      returns.push(returnPct);
    }
  }

  if (returns.length === 0) return 0;

  const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;

  return Math.sqrt(variance);
}

/**
 * Get historical data for a single symbol
 */
export async function getHistoricalData(symbol: string): Promise<HistoricalData | null> {
  const symbolUSDT = symbol.endsWith("USDT") ? symbol : `${symbol}USDT`;

  // Check cache
  const cached = cache.get(symbolUSDT);
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    return cached.data;
  }

  try {
    // Fetch 30 days of daily candles (to get 7d and 30d data)
    const klines30d = await fetchKlines(symbolUSDT, "1d", 31); // 31 days for safety

    if (klines30d.length < 8) {
      // Not enough data
      return null;
    }

    // Extract prices
    const priceNow = parseFloat(klines30d[klines30d.length - 1][4]); // Close price of last candle
    const price7dAgo = parseFloat(klines30d[klines30d.length - 8][4]); // 7 days ago
    const price30dAgo = parseFloat(klines30d[0][4]); // 30 days ago

    // Calculate changes
    const changePercent7d = calculateChange(priceNow, price7dAgo);
    const changePercent30d = calculateChange(priceNow, price30dAgo);

    // Calculate volumes
    const klines7d = klines30d.slice(-8); // Last 7 days + today
    const volume7d = klines7d.reduce((sum, k) => sum + parseFloat(k[5]), 0);
    const volume30d = klines30d.reduce((sum, k) => sum + parseFloat(k[5]), 0);

    // Calculate volatility
    const volatility7d = calculateVolatility(klines7d);
    const volatility30d = calculateVolatility(klines30d);

    const result: HistoricalData = {
      symbol: symbol.replace("USDT", ""),
      changePercent7d,
      changePercent30d,
      volume7d,
      volume30d,
      volatility7d,
      volatility30d,
      priceNow,
      price7dAgo,
      price30dAgo,
    };

    // Update cache
    cache.set(symbolUSDT, { data: result, timestamp: Date.now() });

    return result;
  } catch (error: any) {
    console.error(`Error fetching historical data for ${symbol}:`, error.message);
    return null;
  }
}

/**
 * Get historical data for multiple symbols (batch processing)
 */
export async function getBatchHistoricalData(
  symbols: string[],
  options: {
    batchSize?: number;
    delayMs?: number;
  } = {}
): Promise<Map<string, HistoricalData>> {
  const { batchSize = 10, delayMs = 100 } = options;

  const results = new Map<string, HistoricalData>();

  // Process in batches to avoid rate limiting
  for (let i = 0; i < symbols.length; i += batchSize) {
    const batch = symbols.slice(i, i + batchSize);

    const promises = batch.map((symbol) => getHistoricalData(symbol));
    const batchResults = await Promise.all(promises);

    batchResults.forEach((result, index) => {
      if (result) {
        results.set(batch[index], result);
      }
    });

    // Delay between batches
    if (i + batchSize < symbols.length) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }

  return results;
}

/**
 * Clear cache (useful for testing or forced refresh)
 */
export function clearHistoricalCache(): void {
  cache.clear();
}

/**
 * Get cache statistics
 */
export function getCacheStats(): {
  size: number;
  oldestEntry: number | null;
  newestEntry: number | null;
} {
  if (cache.size === 0) {
    return { size: 0, oldestEntry: null, newestEntry: null };
  }

  let oldest = Infinity;
  let newest = 0;

  cache.forEach((value) => {
    if (value.timestamp < oldest) oldest = value.timestamp;
    if (value.timestamp > newest) newest = value.timestamp;
  });

  return {
    size: cache.size,
    oldestEntry: oldest,
    newestEntry: newest,
  };
}
