/**
 * COIN LIST SERVICE
 * Fetches and caches 522+ USDT perpetual contracts from Binance Futures
 *
 * Features:
 * - Fetches all USDT perpetual symbols
 * - Caches list with TTL
 * - Filters by volume/status
 * - White-hat compliance: All API calls logged
 */

import circuitBreakerManager from '../resilience/circuit-breaker';

// ============================================================================
// TYPES
// ============================================================================

export interface CoinInfo {
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  status: string;
  pricePrecision: number;
  quantityPrecision: number;
}

export interface CoinListCache {
  symbols: string[];
  coinInfo: CoinInfo[];
  timestamp: number;
  count: number;
}

// ============================================================================
// COIN LIST SERVICE
// ============================================================================

export class CoinListService {
  private cache: CoinListCache | null = null;
  private cacheTTL: number = 3600000; // 1 hour
  private circuitBreaker = circuitBreakerManager.getBreaker('binance-api', {
    failureThreshold: 3,
    successThreshold: 2,
    timeout: 30000,
    monitoringPeriod: 60000,
  });

  /**
   * Get all USDT perpetual symbols from Binance Futures
   */
  async getAllSymbols(forceRefresh: boolean = false): Promise<string[]> {
    // Check cache
    if (!forceRefresh && this.cache && Date.now() - this.cache.timestamp < this.cacheTTL) {
      console.log(`[CoinList] Using cached list (${this.cache.count} symbols)`);
      return this.cache.symbols;
    }

    // Fetch from API with circuit breaker
    try {
      const coinInfo = await this.circuitBreaker.execute(
        async () => {
          return await this.fetchFromBinance();
        },
        async () => {
          // Fallback: Return cached list even if expired
          if (this.cache) {
            console.warn('[CoinList] Using stale cache as fallback');
            return this.cache.coinInfo;
          }
          throw new Error('No cache available');
        }
      );

      // Extract symbols
      const symbols = coinInfo.map((coin) => coin.symbol);

      // Update cache
      this.cache = {
        symbols,
        coinInfo,
        timestamp: Date.now(),
        count: symbols.length,
      };

      console.log(`[CoinList] ✅ Fetched ${symbols.length} USDT perpetual symbols`);
      return symbols;
    } catch (error: any) {
      console.error('[CoinList] Failed to fetch symbols:', error.message);

      // Return cached list if available
      if (this.cache) {
        console.warn('[CoinList] Returning stale cache due to error');
        return this.cache.symbols;
      }

      throw error;
    }
  }

  /**
   * Fetch coin list from Binance Futures API
   */
  private async fetchFromBinance(): Promise<CoinInfo[]> {
    const baseUrl = process.env.BINANCE_BASE || 'https://fapi.binance.com';
    const url = `${baseUrl}/fapi/v1/exchangeInfo`;

    console.log(`[CoinList] Fetching from ${url}...`);

    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; SardagAI/1.0)',
      },
      signal: AbortSignal.timeout(10000), // 10 second timeout
    });

    if (!response.ok) {
      throw new Error(`Binance API returned ${response.status}`);
    }

    const data = await response.json();

    // Filter USDT perpetual contracts that are trading
    const usdtPerpetuals: CoinInfo[] = data.symbols
      .filter((s: any) => {
        return (
          s.quoteAsset === 'USDT' &&
          s.contractType === 'PERPETUAL' &&
          s.status === 'TRADING'
        );
      })
      .map((s: any) => ({
        symbol: s.symbol,
        baseAsset: s.baseAsset,
        quoteAsset: s.quoteAsset,
        status: s.status,
        pricePrecision: s.pricePrecision,
        quantityPrecision: s.quantityPrecision,
      }));

    return usdtPerpetuals;
  }

  /**
   * Get top N symbols by 24h volume
   */
  async getTopSymbolsByVolume(limit: number = 100): Promise<string[]> {
    try {
      const baseUrl = process.env.BINANCE_BASE || 'https://fapi.binance.com';
      const url = `${baseUrl}/fapi/v1/ticker/24hr`;

      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; SardagAI/1.0)',
        },
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) {
        throw new Error(`Binance API returned ${response.status}`);
      }

      const tickers = await response.json();

      // Filter USDT pairs and sort by volume
      const usdtTickers = tickers
        .filter((t: any) => t.symbol.endsWith('USDT'))
        .sort((a: any, b: any) => parseFloat(b.quoteVolume) - parseFloat(a.quoteVolume))
        .slice(0, limit)
        .map((t: any) => t.symbol);

      console.log(`[CoinList] ✅ Top ${limit} symbols by volume fetched`);
      return usdtTickers;
    } catch (error: any) {
      console.error('[CoinList] Failed to fetch top symbols:', error.message);

      // Fallback: Return first N from cached list
      const allSymbols = await this.getAllSymbols();
      return allSymbols.slice(0, limit);
    }
  }

  /**
   * Get coin info by symbol
   */
  getCoinInfo(symbol: string): CoinInfo | null {
    if (!this.cache) {
      return null;
    }

    return this.cache.coinInfo.find((c) => c.symbol === symbol) || null;
  }

  /**
   * Get cache stats
   */
  getCacheStats(): {
    cached: boolean;
    count: number;
    age: number;
    expiresIn: number;
  } {
    if (!this.cache) {
      return {
        cached: false,
        count: 0,
        age: 0,
        expiresIn: 0,
      };
    }

    const age = Date.now() - this.cache.timestamp;
    const expiresIn = Math.max(0, this.cacheTTL - age);

    return {
      cached: true,
      count: this.cache.count,
      age,
      expiresIn,
    };
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    console.log('[CoinList] Cache cleared');
    this.cache = null;
  }
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

const coinListService = new CoinListService();
export default coinListService;
