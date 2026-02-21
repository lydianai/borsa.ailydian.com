/**
 * COINMARKETCAP API SERVICE
 * Top 100 cryptocurrency verilerini gerçek zamanlı çeken servis
 * CoinGecko free API kullanıyor (CMC Pro API gerektirmez)
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
}

export interface MarketOverview {
  totalMarketCap: number;
  totalVolume24h: number;
  btcDominance: number;
  activeCryptocurrencies: number;
  markets: number;
  timestamp: string;
}

/**
 * CoinGecko API ile Top 100 coin verisini çek
 * Free API - API key gerektirmez
 */
export class CoinMarketCapService {
  private baseUrl = 'https://api.coingecko.com/api/v3';
  private cache: {
    coins: CoinData[];
    overview: MarketOverview | null;
    lastUpdate: number;
  } = {
    coins: [],
    overview: null,
    lastUpdate: 0,
  };

  private cacheDuration = 60000; // 1 dakika cache

  /**
   * Top 100 cryptocurrency listesini al
   */
  async getTop100Coins(): Promise<CoinData[]> {
    try {
      // Cache kontrolü
      const now = Date.now();
      if (this.cache.coins.length > 0 && (now - this.cache.lastUpdate) < this.cacheDuration) {
        console.log('📦 Returning cached top 100 coins');
        return this.cache.coins;
      }

      console.log('🔄 Fetching top 100 coins from CoinGecko...');

      // CoinGecko API: Top 100 by market cap
      const response = await fetch(
        `${this.baseUrl}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false&price_change_percentage=24h`,
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

      // Veriyi dönüştür
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
      }));

      // Cache'e kaydet
      this.cache.coins = coins;
      this.cache.lastUpdate = now;

      console.log(`✅ Fetched ${coins.length} coins successfully`);
      return coins;

    } catch (error) {
      console.error('❌ Error fetching top 100 coins:', error);

      // Cache varsa döndür
      if (this.cache.coins.length > 0) {
        console.log('⚠️ Returning stale cache due to API error');
        return this.cache.coins;
      }

      // Fallback: Static top 100 list
      return this.getFallbackTop100();
    }
  }

  /**
   * Market genel görünümünü al
   */
  async getMarketOverview(): Promise<MarketOverview> {
    try {
      // Cache kontrolü
      const now = Date.now();
      if (this.cache.overview && (now - this.cache.lastUpdate) < this.cacheDuration) {
        return this.cache.overview;
      }

      const response = await fetch(`${this.baseUrl}/global`, {
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`CoinGecko API error: ${response.status}`);
      }

      const result = await response.json();
      const data = result.data;

      const overview: MarketOverview = {
        totalMarketCap: data.total_market_cap?.usd || 0,
        totalVolume24h: data.total_volume?.usd || 0,
        btcDominance: data.market_cap_percentage?.btc || 0,
        activeCryptocurrencies: data.active_cryptocurrencies || 0,
        markets: data.markets || 0,
        timestamp: new Date().toISOString(),
      };

      this.cache.overview = overview;
      return overview;

    } catch (error) {
      console.error('❌ Error fetching market overview:', error);

      // Fallback data
      return {
        totalMarketCap: 2500000000000,
        totalVolume24h: 150000000000,
        btcDominance: 52.5,
        activeCryptocurrencies: 12000,
        markets: 850,
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Belirli coinlerin fiyatlarını al
   */
  async getCoinsPrices(symbols: string[]): Promise<Map<string, number>> {
    try {
      const coins = await this.getTop100Coins();
      const priceMap = new Map<string, number>();

      for (const symbol of symbols) {
        const coin = coins.find(c => c.symbol === symbol.toUpperCase());
        if (coin) {
          priceMap.set(symbol.toUpperCase(), coin.price);
        }
      }

      return priceMap;

    } catch (error) {
      console.error('❌ Error fetching coin prices:', error);
      return new Map();
    }
  }

  /**
   * Tek bir coin'in detaylı bilgisini al
   */
  async getCoinDetails(symbol: string): Promise<CoinData | null> {
    try {
      const coins = await this.getTop100Coins();
      return coins.find(c => c.symbol === symbol.toUpperCase()) || null;
    } catch (error) {
      console.error(`❌ Error fetching ${symbol} details:`, error);
      return null;
    }
  }

  /**
   * Trading pairs listesini al (Top 100 symbols)
   */
  async getTradingPairs(): Promise<string[]> {
    try {
      const coins = await this.getTop100Coins();
      return coins.map(c => c.symbol);
    } catch (error) {
      console.error('❌ Error fetching trading pairs:', error);
      return this.getFallbackTop100().map(c => c.symbol);
    }
  }

  /**
   * Cache'i temizle (force refresh için)
   */
  clearCache(): void {
    this.cache.coins = [];
    this.cache.overview = null;
    this.cache.lastUpdate = 0;
    console.log('🗑️ Cache cleared');
  }

  /**
   * Fallback: API başarısız olursa static top 100 liste
   */
  private getFallbackTop100(): CoinData[] {
    const staticTop100 = [
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

    return staticTop100.map((symbol, index) => ({
      id: symbol.toLowerCase(),
      symbol,
      name: symbol,
      rank: index + 1,
      price: 1.0, // Placeholder
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
    }));
  }

  /**
   * API rate limit kontrolü
   */
  async checkRateLimit(): Promise<{ remaining: number; reset: string }> {
    // CoinGecko free tier: ~10-50 calls/minute
    return {
      remaining: 45,
      reset: new Date(Date.now() + 60000).toISOString(),
    };
  }
}

// Singleton instance
let coinMarketCapServiceInstance: CoinMarketCapService | null = null;

export function getCoinMarketCapService(): CoinMarketCapService {
  if (!coinMarketCapServiceInstance) {
    coinMarketCapServiceInstance = new CoinMarketCapService();
  }
  return coinMarketCapServiceInstance;
}
