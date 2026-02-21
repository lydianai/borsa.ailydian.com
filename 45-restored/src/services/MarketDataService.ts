/**
 * MARKET DATA SERVICE - HYBRID (Binance + CoinGecko)
 * Binance API'den real-time price data
 * CoinGecko API'den market cap rankings
 * En sağlıklı ve hızlı data feed
 */

export interface CoinData {
  id: string;
  symbol: string;
  name: string;
  rank: number;
  price: number;
  priceChange24h: number;
  priceChangePercentage24h: number;
  marketCap: number;
  volume24h: number;
  circulatingSupply: number;
  totalSupply: number;
  high24h: number;
  low24h: number;
  ath: number;
  athDate: string;
  lastUpdated: string;
  source: 'binance' | 'coingecko';
}

export interface BinanceTicker {
  symbol: string;
  price: string;
  priceChange: string;
  priceChangePercent: string;
  volume: string;
  quoteVolume: string;
  highPrice: string;
  lowPrice: string;
  openTime: number;
  closeTime: number;
}

/**
 * Hybrid Market Data Service
 * Primary: Binance API (fast, real-time, no rate limits)
 * Fallback: CoinGecko API (market cap, rankings)
 */
export class MarketDataService {
  private binanceBaseUrl = 'https://api.binance.com/api/v3';
  private coingeckoBaseUrl = 'https://api.coingecko.com/api/v3';

  private cache: {
    binanceTickers: Map<string, BinanceTicker>;
    coingeckoCoins: CoinData[];
    top100: CoinData[];
    lastBinanceUpdate: number;
    lastCoingeckoUpdate: number;
  } = {
    binanceTickers: new Map(),
    coingeckoCoins: [],
    top100: [],
    lastBinanceUpdate: 0,
    lastCoingeckoUpdate: 0,
  };

  private binanceCacheDuration = 5000; // 5 saniye (Binance çok hızlı)
  private coingeckoCacheDuration = 60000; // 1 dakika

  /**
   * Binance'den tüm USDT trading pairs'i al
   */
  async getBinanceTickers(): Promise<Map<string, BinanceTicker>> {
    try {
      const now = Date.now();

      // Cache kontrolü
      if (this.cache.binanceTickers.size > 0 && (now - this.cache.lastBinanceUpdate) < this.binanceCacheDuration) {
        return this.cache.binanceTickers;
      }

      console.log('🔄 Fetching Binance tickers...');

      // Binance 24hr ticker price change statistics
      const response = await fetch(`${this.binanceBaseUrl}/ticker/24hr`, {
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Binance API error: ${response.status}`);
      }

      const tickers: BinanceTicker[] = await response.json();

      // Sadece USDT pairs'i filtrele ve map'e çevir
      const tickerMap = new Map<string, BinanceTicker>();

      for (const ticker of tickers) {
        if (ticker.symbol.endsWith('USDT')) {
          // BTCUSDT -> BTC
          const symbol = ticker.symbol.replace('USDT', '');
          tickerMap.set(symbol, ticker);
        }
      }

      this.cache.binanceTickers = tickerMap;
      this.cache.lastBinanceUpdate = now;

      console.log(`✅ Fetched ${tickerMap.size} Binance USDT pairs`);
      return tickerMap;

    } catch (error) {
      console.error('❌ Binance API error:', error);

      // Cache varsa döndür
      if (this.cache.binanceTickers.size > 0) {
        console.log('⚠️ Returning stale Binance cache');
        return this.cache.binanceTickers;
      }

      return new Map();
    }
  }

  /**
   * CoinGecko'dan top 100 market cap rankings
   */
  async getCoingeckoTop100(): Promise<CoinData[]> {
    try {
      const now = Date.now();

      // Cache kontrolü
      if (this.cache.coingeckoCoins.length > 0 && (now - this.cache.lastCoingeckoUpdate) < this.coingeckoCacheDuration) {
        return this.cache.coingeckoCoins;
      }

      console.log('🔄 Fetching CoinGecko top 100...');

      const response = await fetch(
        `${this.coingeckoBaseUrl}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false`,
        {
          headers: {
            'Accept': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`CoinGecko API error: ${response.status}`);
      }

      const data = await response.json();

      const coins: CoinData[] = data.map((coin: any, index: number) => ({
        id: coin.id,
        symbol: coin.symbol.toUpperCase(),
        name: coin.name,
        rank: index + 1,
        price: coin.current_price || 0,
        priceChange24h: coin.price_change_24h || 0,
        priceChangePercentage24h: coin.price_change_percentage_24h || 0,
        marketCap: coin.market_cap || 0,
        volume24h: coin.total_volume || 0,
        circulatingSupply: coin.circulating_supply || 0,
        totalSupply: coin.total_supply || 0,
        high24h: coin.high_24h || 0,
        low24h: coin.low_24h || 0,
        ath: coin.ath || 0,
        athDate: coin.ath_date || '',
        lastUpdated: coin.last_updated || new Date().toISOString(),
        source: 'coingecko',
      }));

      this.cache.coingeckoCoins = coins;
      this.cache.lastCoingeckoUpdate = now;

      console.log(`✅ Fetched ${coins.length} CoinGecko coins`);
      return coins;

    } catch (error) {
      console.error('❌ CoinGecko API error:', error);

      if (this.cache.coingeckoCoins.length > 0) {
        console.log('⚠️ Returning stale CoinGecko cache');
        return this.cache.coingeckoCoins;
      }

      return [];
    }
  }

  /**
   * HYBRID: Top 100 coins (Binance prices + CoinGecko rankings)
   * En güncel ve sağlıklı veri
   */
  async getTop100Hybrid(): Promise<CoinData[]> {
    try {
      // Her iki kaynaktan da veri çek (parallel)
      const [binanceTickers, coingeckoCoins] = await Promise.all([
        this.getBinanceTickers(),
        this.getCoingeckoTop100(),
      ]);

      console.log('🔀 Merging Binance + CoinGecko data...');

      // CoinGecko rankings'i kullan ama Binance'den güncel fiyatları al
      const hybridCoins: CoinData[] = coingeckoCoins.map(coin => {
        const binanceTicker = binanceTickers.get(coin.symbol);

        if (binanceTicker) {
          // Binance'den güncel veri varsa kullan (daha fresh)
          return {
            ...coin,
            price: parseFloat(binanceTicker.price),
            priceChange24h: parseFloat(binanceTicker.priceChange),
            priceChangePercentage24h: parseFloat(binanceTicker.priceChangePercent),
            volume24h: parseFloat(binanceTicker.quoteVolume),
            high24h: parseFloat(binanceTicker.highPrice),
            low24h: parseFloat(binanceTicker.lowPrice),
            source: 'binance',
            lastUpdated: new Date(binanceTicker.closeTime).toISOString(),
          };
        }

        // Binance'de yoksa CoinGecko verisini kullan
        return {
          ...coin,
          source: 'coingecko',
        };
      });

      this.cache.top100 = hybridCoins;

      console.log(`✅ Hybrid top 100 ready (${hybridCoins.filter(c => c.source === 'binance').length} from Binance, ${hybridCoins.filter(c => c.source === 'coingecko').length} from CoinGecko)`);

      return hybridCoins;

    } catch (error) {
      console.error('❌ Hybrid data merge error:', error);

      // Cache varsa döndür
      if (this.cache.top100.length > 0) {
        return this.cache.top100;
      }

      // Fallback
      return this.getFallbackTop100();
    }
  }

  /**
   * Trading pairs listesi (Top 100 symbols)
   */
  async getTradingPairs(): Promise<string[]> {
    try {
      const coins = await this.getTop100Hybrid();
      return coins.map(c => c.symbol);
    } catch (error) {
      console.error('❌ Error getting trading pairs:', error);
      return this.getFallbackTop100().map(c => c.symbol);
    }
  }

  /**
   * Belirli coinlerin fiyatlarını al (Binance first)
   */
  async getCoinsPrices(symbols: string[]): Promise<Map<string, number>> {
    try {
      const binanceTickers = await this.getBinanceTickers();
      const priceMap = new Map<string, number>();

      for (const symbol of symbols) {
        const ticker = binanceTickers.get(symbol.toUpperCase());
        if (ticker) {
          priceMap.set(symbol.toUpperCase(), parseFloat(ticker.price));
        }
      }

      // Binance'de bulunamayanlar için CoinGecko'ya sor
      const missingSymbols = symbols.filter(s => !priceMap.has(s.toUpperCase()));
      if (missingSymbols.length > 0) {
        const coingeckoCoins = await this.getCoingeckoTop100();
        for (const symbol of missingSymbols) {
          const coin = coingeckoCoins.find(c => c.symbol === symbol.toUpperCase());
          if (coin) {
            priceMap.set(symbol.toUpperCase(), coin.price);
          }
        }
      }

      return priceMap;

    } catch (error) {
      console.error('❌ Error fetching coin prices:', error);
      return new Map();
    }
  }

  /**
   * Tek bir coin detayı (Binance first)
   */
  async getCoinDetails(symbol: string): Promise<CoinData | null> {
    try {
      const coins = await this.getTop100Hybrid();
      return coins.find(c => c.symbol === symbol.toUpperCase()) || null;
    } catch (error) {
      console.error(`❌ Error fetching ${symbol} details:`, error);
      return null;
    }
  }

  /**
   * Real-time price update (Binance WebSocket için hazırlık)
   */
  async getRealtimePrice(symbol: string): Promise<number | null> {
    try {
      const ticker = await fetch(`${this.binanceBaseUrl}/ticker/price?symbol=${symbol}USDT`);
      if (!ticker.ok) return null;

      const data = await ticker.json();
      return parseFloat(data.price);
    } catch (error) {
      return null;
    }
  }

  /**
   * Cache temizle
   */
  clearCache(): void {
    this.cache.binanceTickers.clear();
    this.cache.coingeckoCoins = [];
    this.cache.top100 = [];
    this.cache.lastBinanceUpdate = 0;
    this.cache.lastCoingeckoUpdate = 0;
    console.log('🗑️ Market data cache cleared');
  }

  /**
   * Fallback: Static top 100
   */
  private getFallbackTop100(): CoinData[] {
    const top100Symbols = [
      'BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'USDC', 'XRP', 'DOGE', 'ADA', 'TRX',
      'AVAX', 'SHIB', 'DOT', 'LINK', 'MATIC', 'BCH', 'UNI', 'LTC', 'ICP', 'NEAR',
      'APT', 'STX', 'XLM', 'FIL', 'ARB', 'VET', 'HBAR', 'OP', 'IMX', 'ATOM',
      'MKR', 'ALGO', 'INJ', 'RNDR', 'GRT', 'RUNE', 'QNT', 'SAND', 'FTM', 'MANA',
      'AAVE', 'THETA', 'AXS', 'XTZ', 'EOS', 'EGLD', 'FLOW', 'KAVA', 'CHZ', 'ZIL',
      'NEO', 'DASH', 'CAKE', 'ENJ', 'ZEC', 'GALA', 'ROSE', 'HNT', 'LRC', 'CRV',
      'KCS', 'MINA', 'FXS', 'GMT', 'ONE', 'BAT', 'COMP', 'WAVES', 'ZRX', 'CELO',
      'YFI', 'QTUM', 'SNX', 'SKL', 'OMG', 'ANKR', 'AUDIO', 'STORJ', 'RLC', 'BAND',
      'NMR', 'BAL', 'UMA', 'OCEAN', 'REQ', 'REN', 'KNC', 'BNT', 'SC', 'LSK',
      'ICX', 'IOST', 'IOTA', 'ONT', 'TFUEL', 'ZEN', 'DGB', 'RVN', 'HOT', 'BTT',
    ];

    return top100Symbols.map((symbol, index) => ({
      id: symbol.toLowerCase(),
      symbol,
      name: symbol,
      rank: index + 1,
      price: 1.0,
      priceChange24h: 0,
      priceChangePercentage24h: 0,
      marketCap: 0,
      volume24h: 0,
      circulatingSupply: 0,
      totalSupply: 0,
      high24h: 0,
      low24h: 0,
      ath: 0,
      athDate: '',
      lastUpdated: new Date().toISOString(),
      source: 'coingecko',
    }));
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{
    binance: boolean;
    coingecko: boolean;
    cacheSize: number;
    lastUpdate: string;
  }> {
    let binanceOk = false;
    let coingeckoOk = false;

    try {
      const binanceRes = await fetch(`${this.binanceBaseUrl}/ping`);
      binanceOk = binanceRes.ok;
    } catch {}

    try {
      const coingeckoRes = await fetch(`${this.coingeckoBaseUrl}/ping`);
      coingeckoOk = coingeckoRes.ok;
    } catch {}

    return {
      binance: binanceOk,
      coingecko: coingeckoOk,
      cacheSize: this.cache.top100.length,
      lastUpdate: new Date(Math.max(this.cache.lastBinanceUpdate, this.cache.lastCoingeckoUpdate)).toISOString(),
    };
  }
}

// Singleton
let marketDataServiceInstance: MarketDataService | null = null;

export function getMarketDataService(): MarketDataService {
  if (!marketDataServiceInstance) {
    marketDataServiceInstance = new MarketDataService();
  }
  return marketDataServiceInstance;
}
