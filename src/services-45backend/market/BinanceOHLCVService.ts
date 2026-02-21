/**
 * BINANCE MULTI-TIMEFRAME OHLCV SERVICE
 * Fetches candlestick data for technical analysis
 * Supports: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
 */

import axios from 'axios';

export type Timeframe = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' | '1M';

export interface OHLCVCandle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closeTime: number;
  quoteVolume: number;
  trades: number;
  takerBuyBaseVolume: number;
  takerBuyQuoteVolume: number;
}

export interface MultiTimeframeData {
  symbol: string;
  timeframes: {
    [key in Timeframe]?: OHLCVCandle[];
  };
  lastUpdate: number;
}

export class BinanceOHLCVService {
  private readonly baseUrl = 'https://api.binance.com/api/v3';
  private cache: Map<string, { data: OHLCVCandle[]; timestamp: number }> = new Map();
  private readonly cacheDuration = 60000; // 1 minute cache for most timeframes

  /**
   * Get OHLCV data for a single symbol and timeframe
   */
  async getOHLCV(
    symbol: string,
    timeframe: Timeframe,
    limit: number = 500
  ): Promise<OHLCVCandle[]> {
    const cacheKey = `${symbol}-${timeframe}`;
    const cached = this.cache.get(cacheKey);

    // Return cached data if fresh
    if (cached && Date.now() - cached.timestamp < this.getCacheDuration(timeframe)) {
      return cached.data;
    }

    try {
      const response = await axios.get(`${this.baseUrl}/klines`, {
        params: {
          symbol: symbol.toUpperCase(),
          interval: timeframe,
          limit: Math.min(limit, 1000), // Binance max is 1000
        },
        timeout: 10000,
      });

      const candles = this.parseKlines(response.data);

      // Update cache
      this.cache.set(cacheKey, {
        data: candles,
        timestamp: Date.now(),
      });

      console.log(`✅ Fetched ${candles.length} ${timeframe} candles for ${symbol}`);
      return candles;
    } catch (error: any) {
      console.error(`❌ Error fetching OHLCV for ${symbol}:`, error.message);

      // Return stale cache if available
      if (cached) {
        console.log('⚠️  Using stale cache');
        return cached.data;
      }

      return [];
    }
  }

  /**
   * Get OHLCV data for multiple timeframes simultaneously
   */
  async getMultiTimeframeOHLCV(
    symbol: string,
    timeframes: Timeframe[],
    limit: number = 500
  ): Promise<MultiTimeframeData> {
    const result: MultiTimeframeData = {
      symbol,
      timeframes: {},
      lastUpdate: Date.now(),
    };

    // Fetch all timeframes in parallel
    const promises = timeframes.map(async (tf) => {
      const data = await this.getOHLCV(symbol, tf, limit);
      result.timeframes[tf] = data;
    });

    await Promise.all(promises);

    return result;
  }

  /**
   * Get OHLCV for multiple symbols (batch request)
   */
  async getBatchOHLCV(
    symbols: string[],
    timeframe: Timeframe,
    limit: number = 500
  ): Promise<Map<string, OHLCVCandle[]>> {
    const results = new Map<string, OHLCVCandle[]>();

    // Fetch in parallel with rate limiting
    const batchSize = 10; // Process 10 symbols at a time
    for (let i = 0; i < symbols.length; i += batchSize) {
      const batch = symbols.slice(i, i + batchSize);
      const promises = batch.map(async (symbol) => {
        const data = await this.getOHLCV(symbol, timeframe, limit);
        results.set(symbol, data);
      });

      await Promise.all(promises);

      // Small delay between batches to avoid rate limits
      if (i + batchSize < symbols.length) {
        await this.delay(200);
      }
    }

    return results;
  }

  /**
   * Get latest candle (current price action)
   */
  async getLatestCandle(symbol: string, timeframe: Timeframe): Promise<OHLCVCandle | null> {
    try {
      const candles = await this.getOHLCV(symbol, timeframe, 1);
      return candles.length > 0 ? candles[candles.length - 1] : null;
    } catch (error) {
      console.error(`Error getting latest candle for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Get historical data for a specific date range
   */
  async getHistoricalOHLCV(
    symbol: string,
    timeframe: Timeframe,
    startTime: number,
    endTime: number
  ): Promise<OHLCVCandle[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/klines`, {
        params: {
          symbol: symbol.toUpperCase(),
          interval: timeframe,
          startTime,
          endTime,
          limit: 1000,
        },
        timeout: 15000,
      });

      return this.parseKlines(response.data);
    } catch (error: any) {
      console.error(`Error fetching historical data for ${symbol}:`, error.message);
      return [];
    }
  }

  /**
   * Get aggregated trade data
   */
  async getAggregatedTrades(symbol: string, limit: number = 500): Promise<any[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/aggTrades`, {
        params: {
          symbol: symbol.toUpperCase(),
          limit: Math.min(limit, 1000),
        },
        timeout: 10000,
      });

      return response.data;
    } catch (error: any) {
      console.error(`Error fetching aggregated trades for ${symbol}:`, error.message);
      return [];
    }
  }

  /**
   * Get order book (market depth)
   */
  async getOrderBook(symbol: string, limit: number = 100): Promise<any> {
    try {
      const response = await axios.get(`${this.baseUrl}/depth`, {
        params: {
          symbol: symbol.toUpperCase(),
          limit: Math.min(limit, 5000),
        },
        timeout: 10000,
      });

      return {
        symbol,
        lastUpdateId: response.data.lastUpdateId,
        bids: response.data.bids.map((bid: string[]) => ({
          price: parseFloat(bid[0]),
          quantity: parseFloat(bid[1]),
        })),
        asks: response.data.asks.map((ask: string[]) => ({
          price: parseFloat(ask[0]),
          quantity: parseFloat(ask[1]),
        })),
      };
    } catch (error: any) {
      console.error(`Error fetching order book for ${symbol}:`, error.message);
      return null;
    }
  }

  /**
   * Get 24h ticker data
   */
  async get24hTicker(symbol: string): Promise<any> {
    try {
      const response = await axios.get(`${this.baseUrl}/ticker/24hr`, {
        params: {
          symbol: symbol.toUpperCase(),
        },
        timeout: 10000,
      });

      return response.data;
    } catch (error: any) {
      console.error(`Error fetching 24h ticker for ${symbol}:`, error.message);
      return null;
    }
  }

  /**
   * Get all tickers (all symbols)
   */
  async getAllTickers(): Promise<any[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/ticker/24hr`, {
        timeout: 15000,
      });

      return response.data;
    } catch (error: any) {
      console.error('Error fetching all tickers:', error.message);
      return [];
    }
  }

  /**
   * Parse Binance klines response
   */
  private parseKlines(klines: any[]): OHLCVCandle[] {
    return klines.map((k) => ({
      timestamp: k[0],
      open: parseFloat(k[1]),
      high: parseFloat(k[2]),
      low: parseFloat(k[3]),
      close: parseFloat(k[4]),
      volume: parseFloat(k[5]),
      closeTime: k[6],
      quoteVolume: parseFloat(k[7]),
      trades: k[8],
      takerBuyBaseVolume: parseFloat(k[9]),
      takerBuyQuoteVolume: parseFloat(k[10]),
    }));
  }

  /**
   * Get cache duration based on timeframe
   */
  private getCacheDuration(timeframe: Timeframe): number {
    switch (timeframe) {
      case '1m':
        return 30000; // 30 seconds
      case '5m':
        return 60000; // 1 minute
      case '15m':
        return 120000; // 2 minutes
      case '30m':
        return 180000; // 3 minutes
      case '1h':
        return 300000; // 5 minutes
      case '4h':
        return 600000; // 10 minutes
      case '1d':
        return 1800000; // 30 minutes
      case '1w':
      case '1M':
        return 3600000; // 1 hour
      default:
        return 60000; // 1 minute default
    }
  }

  /**
   * Delay utility
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Clear cache
   */
  clearCache() {
    this.cache.clear();
    console.log('🗑️  OHLCV cache cleared');
  }

  /**
   * Get cache stats
   */
  getCacheStats() {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
    };
  }

  /**
   * Calculate VWAP (Volume Weighted Average Price) from candles
   */
  calculateVWAP(candles: OHLCVCandle[]): number {
    if (candles.length === 0) return 0;

    let totalPV = 0; // Price * Volume
    let totalVolume = 0;

    for (const candle of candles) {
      const typicalPrice = (candle.high + candle.low + candle.close) / 3;
      totalPV += typicalPrice * candle.volume;
      totalVolume += candle.volume;
    }

    return totalVolume > 0 ? totalPV / totalVolume : 0;
  }

  /**
   * Calculate ATR (Average True Range) from candles
   */
  calculateATR(candles: OHLCVCandle[], period: number = 14): number {
    if (candles.length < period + 1) return 0;

    const trueRanges: number[] = [];

    for (let i = 1; i < candles.length; i++) {
      const current = candles[i];
      const previous = candles[i - 1];

      const tr = Math.max(
        current.high - current.low,
        Math.abs(current.high - previous.close),
        Math.abs(current.low - previous.close)
      );

      trueRanges.push(tr);
    }

    // Calculate SMA of true ranges
    const recentTRs = trueRanges.slice(-period);
    return recentTRs.reduce((sum, tr) => sum + tr, 0) / period;
  }

  /**
   * Detect support/resistance levels from candles
   */
  detectSupportResistance(candles: OHLCVCandle[], tolerance: number = 0.02): {
    support: number[];
    resistance: number[];
  } {
    if (candles.length < 20) {
      return { support: [], resistance: [] };
    }

    const _lows = candles.map((c) => c.low);
    const _highs = candles.map((c) => c.high);

    const support: number[] = [];
    const resistance: number[] = [];

    // Find local minima (support) and maxima (resistance)
    for (let i = 2; i < candles.length - 2; i++) {
      const current = candles[i];

      // Check if it's a local minimum (support)
      if (
        current.low < candles[i - 1].low &&
        current.low < candles[i - 2].low &&
        current.low < candles[i + 1].low &&
        current.low < candles[i + 2].low
      ) {
        support.push(current.low);
      }

      // Check if it's a local maximum (resistance)
      if (
        current.high > candles[i - 1].high &&
        current.high > candles[i - 2].high &&
        current.high > candles[i + 1].high &&
        current.high > candles[i + 2].high
      ) {
        resistance.push(current.high);
      }
    }

    // Group similar levels
    const groupLevels = (levels: number[]): number[] => {
      if (levels.length === 0) return [];

      const sorted = [...levels].sort((a, b) => a - b);
      const grouped: number[] = [sorted[0]];

      for (let i = 1; i < sorted.length; i++) {
        const lastGroup = grouped[grouped.length - 1];
        const diff = Math.abs(sorted[i] - lastGroup) / lastGroup;

        if (diff > tolerance) {
          grouped.push(sorted[i]);
        }
      }

      return grouped;
    };

    return {
      support: groupLevels(support),
      resistance: groupLevels(resistance),
    };
  }
}

// Singleton instance
export const binanceOHLCVService = new BinanceOHLCVService();
