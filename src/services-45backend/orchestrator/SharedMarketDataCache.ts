import axios from 'axios';
import { EventEmitter } from 'events';

export interface CachedMarketData {
  symbol: string;
  price: number;
  volume: number;
  high24h: number;
  low24h: number;
  change24h: number;
  timestamp: number;
  marketCap?: number;
  indicators?: {
    rsi?: number;
    macd?: { value: number; signal: number; histogram: number };
    bollingerBands?: { upper: number; middle: number; lower: number };
    ema?: { short: number; long: number };
    sma?: { short: number; long: number };
    vwap?: number;
    atr?: number;
  };
}

export interface CacheConfig {
  updateInterval: number;
  binanceApiUrl: string;
  talibServiceUrl: string;
  maxCacheSize: number;
  enableAutoUpdate: boolean;
}

export class SharedMarketDataCache extends EventEmitter {
  private cache: Map<string, CachedMarketData> = new Map();
  private updateInterval: NodeJS.Timeout | null = null;
  private config: CacheConfig;
  private isRunning: boolean = false;
  private subscribedSymbols: Set<string> = new Set();

  constructor(config: Partial<CacheConfig> = {}) {
    super();
    this.config = {
      updateInterval: config.updateInterval || 2000,
      binanceApiUrl: config.binanceApiUrl || process.env.BINANCE_API_URL || 'https://api.binance.com/api/v3',
      talibServiceUrl: config.talibServiceUrl || 'http://localhost:5005',
      maxCacheSize: config.maxCacheSize || 1000,
      enableAutoUpdate: config.enableAutoUpdate !== undefined ? config.enableAutoUpdate : true
    };
  }

  subscribe(symbol: string): void {
    this.subscribedSymbols.add(symbol);
    console.log(`[MarketDataCache] Subscribed to ${symbol}`);
  }

  unsubscribe(symbol: string): void {
    this.subscribedSymbols.delete(symbol);
    this.cache.delete(symbol);
    console.log(`[MarketDataCache] Unsubscribed from ${symbol}`);
  }

  subscribeMultiple(symbols: string[]): void {
    symbols.forEach(symbol => this.subscribe(symbol));
  }

  get(symbol: string): CachedMarketData | undefined {
    return this.cache.get(symbol);
  }

  getAll(): Map<string, CachedMarketData> {
    return new Map(this.cache);
  }

  set(symbol: string, data: CachedMarketData): void {
    if (this.cache.size >= this.config.maxCacheSize && !this.cache.has(symbol)) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(symbol, data);
    this.emit('data:updated', { symbol, data });
  }

  async fetchMarketData(symbol: string): Promise<CachedMarketData | null> {
    try {
      const binanceSymbol = symbol.replace('/', '');
      
      const [tickerResponse, klineResponse] = await Promise.all([
        axios.get(`${this.config.binanceApiUrl}/ticker/24hr`, {
          params: { symbol: binanceSymbol },
          timeout: 5000
        }),
        axios.get(`${this.config.binanceApiUrl}/klines`, {
          params: {
            symbol: binanceSymbol,
            interval: '1m',
            limit: 100
          },
          timeout: 5000
        }).catch(() => null)
      ]);

      if (!tickerResponse.data) {
        throw new Error(`No ticker data for ${symbol}`);
      }

      const ticker = tickerResponse.data;

      const marketData: CachedMarketData = {
        symbol,
        price: parseFloat(ticker.lastPrice),
        volume: parseFloat(ticker.volume),
        high24h: parseFloat(ticker.highPrice),
        low24h: parseFloat(ticker.lowPrice),
        change24h: parseFloat(ticker.priceChangePercent),
        timestamp: Date.now()
      };

      if (klineResponse?.data && Array.isArray(klineResponse.data)) {
        const closes = klineResponse.data.map((k: any) => parseFloat(k[4]));
        const volumes = klineResponse.data.map((k: any) => parseFloat(k[5]));

        try {
          const indicators = await this.fetchIndicators(symbol, closes, volumes);
          if (indicators) {
            marketData.indicators = indicators;
          }
        } catch (error) {
          console.warn(`[MarketDataCache] Failed to fetch indicators for ${symbol}:`, error);
        }
      }

      this.set(symbol, marketData);
      return marketData;
    } catch (error) {
      console.error(`[MarketDataCache] Error fetching data for ${symbol}:`, error);
      this.emit('error', { symbol, error });
      return null;
    }
  }

  private async fetchIndicators(
    symbol: string,
    closes: number[],
    volumes: number[]
  ): Promise<CachedMarketData['indicators'] | null> {
    try {
      const response = await axios.post(
        `${this.config.talibServiceUrl}/indicators`,
        {
          symbol,
          closes,
          volumes,
          indicators: ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'ATR']
        },
        { timeout: 3000 }
      );

      if (response.data?.success && response.data?.indicators) {
        const ind = response.data.indicators;
        return {
          rsi: ind.RSI?.[ind.RSI.length - 1],
          macd: ind.MACD ? {
            value: ind.MACD.macd[ind.MACD.macd.length - 1],
            signal: ind.MACD.signal[ind.MACD.signal.length - 1],
            histogram: ind.MACD.histogram[ind.MACD.histogram.length - 1]
          } : undefined,
          bollingerBands: ind.BBANDS ? {
            upper: ind.BBANDS.upper[ind.BBANDS.upper.length - 1],
            middle: ind.BBANDS.middle[ind.BBANDS.middle.length - 1],
            lower: ind.BBANDS.lower[ind.BBANDS.lower.length - 1]
          } : undefined,
          ema: ind.EMA ? {
            short: ind.EMA.short[ind.EMA.short.length - 1],
            long: ind.EMA.long[ind.EMA.long.length - 1]
          } : undefined,
          sma: ind.SMA ? {
            short: ind.SMA.short[ind.SMA.short.length - 1],
            long: ind.SMA.long[ind.SMA.long.length - 1]
          } : undefined,
          atr: ind.ATR?.[ind.ATR.length - 1]
        };
      }

      return null;
    } catch (error) {
      return null;
    }
  }

  async fetchBatch(symbols: string[]): Promise<Map<string, CachedMarketData>> {
    const results = new Map<string, CachedMarketData>();

    const promises = symbols.map(async (symbol) => {
      const data = await this.fetchMarketData(symbol);
      if (data) {
        results.set(symbol, data);
      }
    });

    await Promise.all(promises);
    return results;
  }

  private async updateLoop(): Promise<void> {
    if (!this.isRunning) return;

    try {
      const symbols = Array.from(this.subscribedSymbols);
      
      if (symbols.length === 0) {
        return;
      }

      console.log(`[MarketDataCache] Updating ${symbols.length} symbols...`);

      await this.fetchBatch(symbols);

      this.emit('batch:updated', {
        count: symbols.length,
        timestamp: Date.now()
      });
    } catch (error) {
      console.error('[MarketDataCache] Update loop error:', error);
      this.emit('error', { context: 'updateLoop', error });
    }
  }

  start(): void {
    if (this.isRunning) {
      console.warn('[MarketDataCache] Already running');
      return;
    }

    this.isRunning = true;
    console.log('[MarketDataCache] Starting...');

    if (this.config.enableAutoUpdate) {
      this.updateInterval = setInterval(() => {
        this.updateLoop();
      }, this.config.updateInterval);
    }

    console.log('[MarketDataCache] Started');
    this.emit('started');
  }

  stop(): void {
    if (!this.isRunning) {
      console.warn('[MarketDataCache] Not running');
      return;
    }

    this.isRunning = false;

    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }

    console.log('[MarketDataCache] Stopped');
    this.emit('stopped');
  }

  clear(): void {
    this.cache.clear();
    this.emit('cleared');
  }

  getStats(): {
    cacheSize: number;
    subscribedSymbols: number;
    isRunning: boolean;
    updateInterval: number;
  } {
    return {
      cacheSize: this.cache.size,
      subscribedSymbols: this.subscribedSymbols.size,
      isRunning: this.isRunning,
      updateInterval: this.config.updateInterval
    };
  }

  async warmup(symbols: string[]): Promise<void> {
    console.log(`[MarketDataCache] Warming up cache with ${symbols.length} symbols...`);
    
    this.subscribeMultiple(symbols);
    await this.fetchBatch(symbols);
    
    console.log(`[MarketDataCache] Warmup complete. Cached ${this.cache.size} symbols.`);
  }
}

let cacheInstance: SharedMarketDataCache | null = null;

export function getMarketDataCache(): SharedMarketDataCache {
  if (!cacheInstance) {
    cacheInstance = new SharedMarketDataCache();
  }
  return cacheInstance;
}
