/**
 * Market Data Pipeline
 *
 * Real-time and historical market data from Binance
 * White-hat compliant: Public API, rate-limited, ethical usage
 */

import axios from 'axios';

const BINANCE_API_BASE = 'https://api.binance.com';
const BINANCE_API_V3 = `${BINANCE_API_BASE}/api/v3`;

/**
 * Fetch klines (candlestick) data from Binance
 * @param {string} symbol - Trading pair (e.g., BTCUSDT)
 * @param {string} interval - Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
 * @param {number} limit - Number of candles (max 1000)
 * @returns {Promise<Array>} - Array of OHLCV data
 */
export async function fetchKlines(symbol, interval = '1h', limit = 100) {
  try {
    const response = await axios.get(`${BINANCE_API_V3}/klines`, {
      params: {
        symbol: symbol.toUpperCase(),
        interval,
        limit: Math.min(limit, 1000) // Binance max
      },
      timeout: 10000 // 10 second timeout
    });

    // Parse Binance kline format to OHLCV
    return response.data.map(candle => ({
      timestamp: candle[0],
      open: parseFloat(candle[1]),
      high: parseFloat(candle[2]),
      low: parseFloat(candle[3]),
      close: parseFloat(candle[4]),
      volume: parseFloat(candle[5]),
      closeTime: candle[6],
      quoteVolume: parseFloat(candle[7]),
      trades: parseInt(candle[8], 10),
      takerBuyBaseVolume: parseFloat(candle[9]),
      takerBuyQuoteVolume: parseFloat(candle[10])
    }));

  } catch (error) {
    console.error('Binance API error:', {
      symbol,
      interval,
      error: error.message,
      response: error.response?.data
    });

    throw new Error(`Failed to fetch market data: ${error.message}`);
  }
}

/**
 * Fetch current ticker price
 * @param {string} symbol - Trading pair
 * @returns {Promise<Object>} - Current price data
 */
export async function fetchTickerPrice(symbol) {
  try {
    const response = await axios.get(`${BINANCE_API_V3}/ticker/price`, {
      params: { symbol: symbol.toUpperCase() },
      timeout: 5000
    });

    return {
      symbol: response.data.symbol,
      price: parseFloat(response.data.price),
      timestamp: Date.now()
    };

  } catch (error) {
    console.error('Ticker price error:', {
      symbol,
      error: error.message
    });

    throw new Error(`Failed to fetch ticker price: ${error.message}`);
  }
}

/**
 * Fetch 24h ticker statistics
 * @param {string} symbol - Trading pair
 * @returns {Promise<Object>} - 24h statistics
 */
export async function fetch24hStats(symbol) {
  try {
    const response = await axios.get(`${BINANCE_API_V3}/ticker/24hr`, {
      params: { symbol: symbol.toUpperCase() },
      timeout: 5000
    });

    return {
      symbol: response.data.symbol,
      priceChange: parseFloat(response.data.priceChange),
      priceChangePercent: parseFloat(response.data.priceChangePercent),
      weightedAvgPrice: parseFloat(response.data.weightedAvgPrice),
      lastPrice: parseFloat(response.data.lastPrice),
      volume: parseFloat(response.data.volume),
      quoteVolume: parseFloat(response.data.quoteVolume),
      openPrice: parseFloat(response.data.openPrice),
      highPrice: parseFloat(response.data.highPrice),
      lowPrice: parseFloat(response.data.lowPrice),
      timestamp: parseInt(response.data.closeTime, 10)
    };

  } catch (error) {
    console.error('24h stats error:', {
      symbol,
      error: error.message
    });

    throw new Error(`Failed to fetch 24h stats: ${error.message}`);
  }
}

/**
 * Normalize OHLCV data to 0-1 range for ML
 * @param {Array} ohlcv - OHLCV data
 * @returns {Object} - Normalized data and scaling params
 */
export function normalizeOHLCV(ohlcv) {
  if (!ohlcv || ohlcv.length === 0) {
    throw new Error('Empty OHLCV data');
  }

  // Find min/max for normalization
  const closes = ohlcv.map(c => c.close);
  const volumes = ohlcv.map(c => c.volume);

  const priceMin = Math.min(...closes);
  const priceMax = Math.max(...closes);
  const volumeMin = Math.min(...volumes);
  const volumeMax = Math.max(...volumes);

  // Normalize to 0-1 range
  const normalized = ohlcv.map(candle => ({
    timestamp: candle.timestamp,
    open: (candle.open - priceMin) / (priceMax - priceMin),
    high: (candle.high - priceMin) / (priceMax - priceMin),
    low: (candle.low - priceMin) / (priceMax - priceMin),
    close: (candle.close - priceMin) / (priceMax - priceMin),
    volume: (candle.volume - volumeMin) / (volumeMax - volumeMin)
  }));

  return {
    data: normalized,
    scaling: {
      priceMin,
      priceMax,
      volumeMin,
      volumeMax
    }
  };
}

/**
 * Denormalize prediction back to real price
 * @param {number} normalized - Normalized value (0-1)
 * @param {Object} scaling - Scaling parameters
 * @returns {number} - Real price
 */
export function denormalizePrice(normalized, scaling) {
  return normalized * (scaling.priceMax - scaling.priceMin) + scaling.priceMin;
}

/**
 * Extract features for ML model
 * @param {Array} ohlcv - OHLCV data
 * @returns {Array} - Feature matrix
 */
export function extractFeatures(ohlcv) {
  return ohlcv.map(candle => [
    candle.open,
    candle.high,
    candle.low,
    candle.close,
    candle.volume
  ]);
}

/**
 * Market data cache (in-memory)
 * Production: Use Redis
 */
const marketDataCache = new Map();
const CACHE_TTL = 60000; // 1 minute

/**
 * Get market data with caching
 * @param {string} symbol - Trading pair
 * @param {string} interval - Timeframe
 * @param {number} limit - Number of candles
 * @returns {Promise<Array>} - Cached or fresh OHLCV data
 */
export async function getMarketData(symbol, interval = '1h', limit = 100) {
  const cacheKey = `${symbol}-${interval}-${limit}`;
  const cached = marketDataCache.get(cacheKey);

  // Check cache
  if (cached && (Date.now() - cached.timestamp) < CACHE_TTL) {
    console.info('Market data cache HIT:', cacheKey);
    return cached.data;
  }

  // Fetch fresh data
  console.info('Market data cache MISS:', cacheKey);
  const data = await fetchKlines(symbol, interval, limit);

  // Update cache
  marketDataCache.set(cacheKey, {
    data,
    timestamp: Date.now()
  });

  return data;
}

/**
 * Clear cache (for testing)
 */
export function clearMarketDataCache() {
  marketDataCache.clear();
}

/**
 * Get cache statistics
 */
export function getCacheStats() {
  return {
    size: marketDataCache.size,
    keys: Array.from(marketDataCache.keys())
  };
}
